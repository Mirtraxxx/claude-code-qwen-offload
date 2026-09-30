"""Live console for the ExLlamaV3 / Strata servers.

Two console styles, chosen once at import time:
- "panel" (default): a live panel pinned at the bottom of the console, redrawn
  in place, plus one permanent line per finished request in the scrollback.
  Used only when stdout is a tty and ANSI/VT processing can be enabled.
- "lines": a status line every second (one request) or a block every 2 s
  (several requests), all scrolling. Forced with the env var CONSOLE_STYLE=lines.
Standard library only, Python 3.11.
"""

import atexit
import ctypes
import hashlib
import json
import os
import shutil
import sys
import threading
import time

CONSOLE_EVERY = float(os.environ.get("CONSOLE_EVERY", "1.0") or 1.0)   # status line period, one request
BLOCK_EVERY = float(os.environ.get("BOARD_EVERY", "2.0") or 2.0)       # block period, 2+ requests
PANEL_EVERY = float(os.environ.get("PANEL_EVERY", "0.5") or 0.5)       # panel redraw period, panel mode
WIDTH = 120                         # no printed line may be longer; cut long lines to WIDTH-3 chars + "..."

_OUT_LOCK = threading.RLock()         # guards the raw streams and the panel lines on screen
_PANEL: list[str] = []                # the panel lines currently on screen
_RAW_OUT = sys.stdout
_RAW_ERR = sys.stderr
_WAITED: dict[int, float] = {}        # task -> seconds spent queued, stored by Waiting.done()
_REDRAW = [False]                     # set when a request starts/ends so the panel redraws on the next tick


def _cols() -> int:
    return shutil.get_terminal_size((WIDTH, 30)).columns


class _PanelStream:
    """sys.stdout / sys.stderr in panel mode: text passes through, the panel is erased and redrawn around it."""

    def __init__(self, raw):
        self.raw = raw
        self._buf = ""

    def write(self, s):
        with _OUT_LOCK:
            if not _PANEL and not self._buf:
                self.raw.write(s)
                return len(s)
            self._buf += s
            if "\n" in self._buf:
                i = self._buf.rfind("\n")
                text, self._buf = self._buf[: i + 1], self._buf[i + 1:]
                _through(self.raw, text)
            return len(s)

    def flush(self):
        self.raw.flush()

    def __getattr__(self, name):
        return getattr(self.raw, name)


def _panel_rows() -> int:
    """Screen rows the panel takes now. Each line was cut to fit the width it was drawn at, but if the window has
    been narrowed since, the terminal re-wraps a long line over several rows."""
    c = max(1, _cols())
    return sum(max(1, -(-len(ln) // c)) for ln in _PANEL)


def _erase() -> None:
    # Assumes _OUT_LOCK is held. Cursor to the start of the first panel line, clear to end of screen.
    n = _panel_rows()
    if n:
        _RAW_OUT.write(f"\x1b[{n}F\x1b[J")
        _RAW_OUT.flush()


def _draw(lines: list[str]) -> None:
    # Assumes _OUT_LOCK is held. Cut every line to cols-1 so no line ever wraps.
    c = _cols()
    lines = [ln[: c - 1] for ln in lines]
    for ln in lines:
        _RAW_OUT.write(ln + "\x1b[K\n")
    _RAW_OUT.flush()
    _PANEL[:] = lines


def _through(raw, text: str) -> None:
    # Assumes _OUT_LOCK is held (the panel stream calls this).
    _erase()
    raw.write(text)
    raw.flush()
    _draw(_PANEL)


def _set_panel(lines: list[str]) -> None:
    # Replace the panel in place, no flicker. The ticker must NOT hold _LC_LOCK here.
    with _OUT_LOCK:
        old = _panel_rows()
        if old:
            _RAW_OUT.write(f"\x1b[{old}F")
        c = _cols()
        lines = [ln[: c - 1] for ln in lines]
        for ln in lines:
            _RAW_OUT.write(ln + "\x1b[K\n")
        _RAW_OUT.write("\x1b[J")
        if lines and not old:
            _RAW_OUT.write("\x1b[?25l")   # hide the cursor while the panel is shown
        elif old and not lines:
            _RAW_OUT.write("\x1b[?25h")   # show it again when the panel is gone
        _RAW_OUT.flush()
        _PANEL[:] = lines


def _cleanup_panel() -> None:
    with _OUT_LOCK:
        if _PANEL:
            _RAW_OUT.write(f"\x1b[{_panel_rows()}F\x1b[J")
        _RAW_OUT.write("\x1b[?25h")
        _RAW_OUT.flush()
        _PANEL[:] = []


def _detect_style() -> str:
    if os.environ.get("CONSOLE_STYLE", "panel").strip().lower() == "lines":
        return "lines"
    if not sys.stdout.isatty():
        return "lines"
    if os.name == "nt":
        try:
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-11)
            m = ctypes.c_uint32()
            if not k.GetConsoleMode(h, ctypes.byref(m)):
                return "lines"
            if not k.SetConsoleMode(h, m.value | 0x0004):
                return "lines"
        except Exception:
            return "lines"
    return "panel"


_STYLE = _detect_style()   # "panel" or "lines", decided once at import time

if _STYLE == "panel":
    sys.stdout = _PanelStream(_RAW_OUT)
    sys.stderr = _PanelStream(_RAW_ERR)
    atexit.register(_cleanup_panel)

_LC_LOCK = threading.Lock()         # guards _LIVE, the counters and printing
_LIVE: dict[int, object] = {}       # id(entry) -> LlamaConsole or Waiting, while it is live
_PP_RATE = [None]                   # last measured prompt-reading speed (tok/s) of a finished request with >= 512 fresh tokens
_WRITTEN = [0]                      # tokens written by finished/aborted requests, ever
_TASK_SEQ = [0]

ON_FINISH = None                    # optional hook, called after a done line: ON_FINISH(con, n_cache, decode_ms, n_gen, finish, note)
ON_ABORT = None                     # optional hook, called after an ended line: ON_ABORT(con, n_gen)


def next_task() -> int:
    """Return 0, 1, 2, ... thread safe."""
    with _LC_LOCK:
        t = _TASK_SEQ[0]
        _TASK_SEQ[0] += 1
        return t


def _dur(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 10.0:
        return f"{sec:.1f} s"
    if sec < 60.0:
        return f"{int(sec)} s"
    if sec < 3600.0:
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"
    return f"{int(sec // 3600)}h{int((sec % 3600) // 60):02d}m"


def _clip(line: str) -> str:
    if len(line) > WIDTH:
        return line[: WIDTH - 3] + "..."
    return line


def _emit(lines: list[str]) -> None:
    # Assumes _LC_LOCK is held. Code already holding the lock prints through here.
    for ln in lines:
        print(_clip(ln), flush=True)


def lc_print(msg: str, extra: list[str] | None = None) -> None:
    # Never call this while holding _LC_LOCK (plain Lock, not reentrant).
    with _LC_LOCK:
        _emit([msg] + list(extra or []))


_WORKERS: dict[str, dict] = {}      # conversation key -> {"name": "A"}


def worker_for(messages) -> dict:
    """The worker letter of a client conversation, keyed by its first user message, so a coding agent keeps its
    letter across the many requests of its session (a context compaction may give it a new one)."""
    first = ""
    for msg in messages or []:
        if msg.get("role") == "user":
            c = msg.get("content") or ""
            first = c if isinstance(c, str) else json.dumps(c, sort_keys=True, default=str)[:20000]
            break
    key = hashlib.sha1(first.encode("utf-8", "replace")).hexdigest()
    with _LC_LOCK:
        w = _WORKERS.get(key)
        if w is None:
            if len(_WORKERS) >= 256:
                _WORKERS.clear()
            n = len(_WORKERS) if not _WORKERS else max(v["n"] for v in _WORKERS.values()) + 1
            w = _WORKERS[key] = {"n": n, "name": chr(65 + n % 26) + ("" if n < 26 else str(n // 26))}
        return w


def _close(con) -> bool:
    """Mark a console closed; True only for the one caller that closed it (finish/abort/error race)."""
    with con.lock:
        if con.closed:
            return False
        con.closed = True
        return True


def _label(task: int, worker) -> str:
    name = worker["name"] if worker else "-"
    return f"{'#' + str(task) + ' ' + name:<8}"


class Waiting:
    """A request that is queued and not running yet."""

    def __init__(self, task: int, worker: dict | None = None, note: str = ""):
        self.task = task
        self.worker = worker
        self.note = note
        self.t0 = time.time()
        with _LC_LOCK:
            _LIVE[id(self)] = self
            _REDRAW[0] = True
            if _STYLE != "panel":
                msg = f"{_label(task, worker)} waiting"
                if note:
                    msg += f"  {note}"
                _emit([msg])
        _ensure_ticker()

    def done(self) -> None:
        """Unregister from _LIVE (idempotent). Prints nothing."""
        with _LC_LOCK:
            _LIVE.pop(id(self), None)
            _REDRAW[0] = True
            if self.task not in _WAITED:
                _WAITED[self.task] = time.time() - self.t0
                if len(_WAITED) > 1000:
                    _WAITED.clear()

    def status(self) -> str:
        return f"{_label(self.task, self.worker)} waiting  {_dur(time.time() - self.t0)}"

    def panel_row(self, now: float) -> str:
        row = f"  {_label(self.task, self.worker)} queued  {_dur(now - self.t0)}"
        if self.note:
            row += f"  {self.note}"
        return row


class LlamaConsole:
    def __init__(self, task: int, n_ctx: int, n_prompt: int, draft_name: str | None = None, poll=None,
                 every: float | None = None, slot: int = 0, chunk_note: str = "", worker: dict | None = None):
        self.task = task
        self.n_ctx = n_ctx
        self.n_prompt = n_prompt
        self.draft_name = draft_name
        self.poll = poll
        self.every = every
        self.slot = slot
        self.chunk_note = chunk_note
        self.worker = worker
        self.meta = ""
        self.budget_hit = False
        self.phase = None
        self.n_cache = None
        self.cache_exact = False
        self.pos = None
        self.pp_rate = None
        self.pp_t = None
        self.prompt_ms = None
        self.t_dec0 = None
        self.n_gen = 0
        self.t_first = None
        self.closed = False
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.win = None
        self._inst = 0.0
        self.cancel_note = ""
        self._samples: list[tuple[float, int]] = []
        self._panel_rate = 0.0

    def launch(self) -> "LlamaConsole":
        with _LC_LOCK:
            _LIVE[id(self)] = self
            _REDRAW[0] = True
            if _STYLE != "panel":
                _emit([f"{_label(self.task, self.worker)} start    {self.n_prompt:,} tok prompt"])
        _ensure_ticker()
        return self

    def set_cache(self, n_cache: int, exact: bool = True) -> None:
        if self.cache_exact and not exact:
            return
        self.n_cache = n_cache
        self.cache_exact = exact or self.cache_exact

    def progress(self, pos: int, rate: float, t_report: float | None = None) -> None:
        self.pos = pos
        self.pp_rate = rate
        self.pp_t = t_report or time.time()

    def prompt_done(self, n_cache: int, prompt_ms: float, t_dec0: float | None = None) -> None:
        self.set_cache(n_cache, True)
        if self.prompt_ms is not None:
            return
        self.prompt_ms = prompt_ms
        self.t_dec0 = t_dec0 or time.time()
        self.win = (self.t_dec0, 0)

    def tokens(self, n_gen: int, t_first: float | None = None) -> None:
        self.n_gen = max(self.n_gen, n_gen)
        if self.t_first is None and t_first:
            self.t_first = t_first

    def token(self) -> None:
        self.n_gen += 1
        if self.t_first is None:
            self.t_first = time.time()

    def live(self) -> dict:
        now = time.time()
        cache = self.n_cache or 0
        fresh = max(0, self.n_prompt - cache)
        pms = self.prompt_ms if self.prompt_ms is not None else (now - self.t0) * 1000
        dms = (now - self.t_dec0) * 1000 if self.t_dec0 else 0.0
        return {
            "cache_n": cache,
            "prompt_n": fresh,
            "prompt_ms": round(pms, 2),
            "prompt_per_second": round(1000 * fresh / max(pms, 1e-6), 2),
            "predicted_n": self.n_gen,
            "predicted_ms": round(dms, 2),
            "predicted_per_second": round(1000 * self.n_gen / dms, 2) if dms > 0 else 0.0,
        }

    def status(self, now: float) -> str | None:
        if self.closed:
            return None
        label = _label(self.task, self.worker)
        if self.t_dec0 is None:
            self._inst = 0.0
            cache = self.n_cache or 0
            fresh = max(1, self.n_prompt - cache)
            cached = f"  ({cache:,} cached)" if cache else ""
            if self.pos is not None and self.n_cache is not None and self.pp_rate:
                done = min(fresh, max(0, self.pos - cache))
                left = max(0.0, (fresh - done) / self.pp_rate - (now - self.pp_t))
                pct = 100.0 * done / fresh
                return (f"{label} reading  {done:,}/{fresh:,} tok {pct:3.0f}%  "
                        f"{self.pp_rate:,.0f} t/s  ~{_dur(left)} left{cached}")
            est = ""
            if _PP_RATE[0]:
                est = f", ~{_dur(max(0, fresh / _PP_RATE[0] - (now - self.t0)))} left (est.)"
            return f"{label} reading  {fresh:,} tok  {_dur(now - self.t0)} so far{est}{cached}"
        dec = now - self.t_dec0
        if self.n_gen == 0 and dec < 1.0:
            self._inst = 0.0
            return f"{label} writing  first token..."
        avg = self.n_gen / dec if dec > 0 else 0.0
        tw, nw = self.win if self.win else (self.t_dec0, 0)
        dt = now - tw
        inst = (self.n_gen - nw) / dt if dt > 0 else 0.0
        self._inst = inst
        self.win = (now, self.n_gen)
        s = (f"{label} writing  {self.n_gen:,} tok  {avg:,.1f} t/s  "
            f"(now {inst:,.0f})  {_dur(dec)}")
        return s

    def inst_rate(self) -> float:
        """The 'now' rate computed at the last status() call; 0 while reading."""
        return self._inst

    def panel_row(self, now: float) -> str | None:
        """The live panel row for this request; built under _LC_LOCK by the ticker."""
        if self.closed:
            return None
        row = f"  {_label(self.task, self.worker)} "
        if self.t_dec0 is None:
            self._panel_rate = 0.0
            cache = self.n_cache or 0
            fresh = max(1, self.n_prompt - cache)
            cached = f"  +{cache:,} cached" if cache else ""
            if self.pos is not None and self.n_cache is not None and self.pp_rate:
                done = min(fresh, max(0, self.pos - cache))
                left = max(0.0, (fresh - done) / self.pp_rate - (now - self.pp_t))
                pct = 100.0 * done / fresh
                return row + (f"pp   {done:,}/{fresh:,}  {pct:.0f}%  "
                              f"{self.pp_rate:,.0f} t/s  ~{_dur(left)} left{cached}")
            est = f"  ~{_dur(max(0, fresh / _PP_RATE[0] - (now - self.t0)))} left (est.)" if _PP_RATE[0] else ""
            return row + f"pp   {fresh:,} tok  {_dur(now - self.t0)} so far{est}{cached}"
        dec = now - self.t_dec0
        if self.n_gen == 0 and dec < 1.0:
            self._panel_rate = 0.0
            return row + "gen  starting..."
        self._samples.append((now, self.n_gen))
        cutoff = now - 2.5
        self._samples = [s for s in self._samples if s[0] >= cutoff]
        if len(self._samples) >= 2:
            rate = (self.n_gen - self._samples[0][1]) / (now - self._samples[0][0])
        else:
            rate = self.n_gen / dec if dec > 0 else 0.0
        self._panel_rate = rate
        return row + f"gen  {self.n_gen:,} tok  {rate:,.0f} t/s  {_dur(dec)}"

    def finish(self, prompt_ms, n_cache, decode_ms, n_gen, finish="stop", draft_accepted=None, draft_total=None, note=""):
        if not _close(self):
            return
        with _LC_LOCK:
            _LIVE.pop(id(self), None)
            _REDRAW[0] = True
            _WRITTEN[0] += n_gen
            fresh = max(0, self.n_prompt - n_cache)
            if fresh >= 512 and prompt_ms > 0:
                _PP_RATE[0] = 1000 * fresh / prompt_ms
            waited = _WAITED.pop(self.task, None)
            D = f" | draft {100 * draft_accepted / draft_total:.0f}%" if draft_accepted is not None and draft_total is not None and draft_total > 0 else ""
            N = f" | {note}" if note else ""
            F = "" if finish == "stop" else f" | {finish}"
            msg = None
            if _STYLE == "panel":
                if fresh == 0:
                    P = f"0 +{n_cache:,} cached" if n_cache else "0"
                elif fresh >= 64 and prompt_ms > 0:
                    P = f"{fresh:,} @ {1000 * fresh / prompt_ms:,.0f} t/s"
                    if n_cache:
                        P += f" +{n_cache:,} cached"
                else:
                    P = f"{fresh:,}" + (f" +{n_cache:,} cached" if n_cache else "")
                G = f"{n_gen:,} @ {1000 * n_gen / decode_ms:,.1f} t/s" if n_gen > 0 and decode_ms > 0 else f"{n_gen:,}"
                T = f" | {_dur(time.time() - self.t0)}"
                Q = f" | waited {_dur(waited)}" if waited is not None and waited >= 0.5 else ""
                msg = f"{time.strftime('%H:%M:%S')}  {_label(self.task, self.worker)} pp {P} | gen {G}{Q}{D}{F}{N}{T}"
            else:
                if fresh == 0:
                    R = f"0 ({n_cache:,} cached)"
                else:
                    C = f", {n_cache:,} cached" if n_cache else ""
                    rate = 1000 * fresh / prompt_ms if prompt_ms > 0 else 0.0
                    R = f"{fresh:,} @ {rate:,.0f} t/s ({_dur(prompt_ms / 1000)}{C})"
                if n_gen == 0:
                    W = "0"
                elif decode_ms <= 0:
                    W = f"{n_gen:,}"
                else:
                    W = f"{n_gen:,} @ {1000 * n_gen / decode_ms:,.1f} t/s ({_dur(decode_ms / 1000)})"
                _emit([f"{_label(self.task, self.worker)} done     read {R} | wrote {W}{D}{F}{N}"])
        if msg:
            lc_print(msg)
        if ON_FINISH:
            try:
                ON_FINISH(self, n_cache, decode_ms, n_gen, finish, note)
            except Exception:
                pass

    def cancelled(self, detail: str = "") -> None:
        if _STYLE == "panel":
            self.cancel_note = detail or "client gone"
        else:
            lc_print(f"{_label(self.task, self.worker)} cancel   {detail or 'client gone'}")

    def abort(self, n_gen: int | None = None) -> None:
        if not _close(self):
            return
        ng = n_gen if n_gen is not None else self.n_gen
        with _LC_LOCK:
            _LIVE.pop(id(self), None)
            _REDRAW[0] = True
            _WRITTEN[0] += ng
            _WAITED.pop(self.task, None)
            msg = None
            if _STYLE == "panel":
                msg = (f"{time.strftime('%H:%M:%S')}  {_label(self.task, self.worker)} "
                       f"ended after {ng:,} tok written ({self.cancel_note or 'client gone'})")
            else:
                _emit([f"{_label(self.task, self.worker)} ended    {ng:,} tok written, no final numbers (client gone)"])
        if msg:
            lc_print(msg)
        if ON_ABORT:
            try:
                ON_ABORT(self, ng)
            except Exception:
                pass

    def error(self, msg: str) -> None:
        if not _close(self):
            return
        with _LC_LOCK:
            _LIVE.pop(id(self), None)
            _REDRAW[0] = True
            _WAITED.pop(self.task, None)
            line = None
            if _STYLE == "panel":
                line = f"{time.strftime('%H:%M:%S')}  {_label(self.task, self.worker)} ERROR {msg}"
            else:
                _emit([f"{_label(self.task, self.worker)} ERROR    {msg}"])
        if line:
            lc_print(line)


_TICKER_ON = [False]


def _ensure_ticker() -> None:
    with _LC_LOCK:
        if not _TICKER_ON[0]:
            _TICKER_ON[0] = True
            threading.Thread(target=_tick_loop, daemon=True).start()


def _print_block(consoles: list, waiters: list, now: float, span_tasks: set) -> None:
    entries = sorted(consoles + waiters, key=lambda e: e.task)
    lines = []
    write_total = 0.0
    read_total = 0.0
    for e in entries:
        s = e.status(now) if isinstance(e, LlamaConsole) else e.status()
        if s is None:
            continue
        lines.append("   " + s)
        span_tasks.add(e.task)
        if isinstance(e, LlamaConsole):
            if e.t_dec0 is not None:
                write_total += e.inst_rate()
            elif e.pp_rate:
                read_total += e.pp_rate
    head = f"{len(consoles)} running"
    if len(waiters):
        head += f", {len(waiters)} waiting"
    speeds = []
    if any(isinstance(e, LlamaConsole) and e.t_dec0 is not None for e in consoles):
        speeds.append(f"writing {write_total:,.0f} t/s total")
    if read_total > 0:
        speeds.append(f"reading {read_total:,.0f} t/s")
    msg = f"== {len(consoles) + len(waiters)} requests: {head}" + "".join(f" | {x}" for x in speeds) + " =="
    lc_print(msg, extra=lines)


def _tick_loop() -> None:
    mode = "none"
    last_print = 0.0
    last_panel = 0.0
    span_t0 = 0.0
    span_written0 = 0
    span_tasks: set[int] = set()
    busy = None                 # panel mode: a busy period that had 2+ requests at once; summarised once all is idle
    idle_since = None
    while True:
        time.sleep(0.1)
        now = time.time()
        with _LC_LOCK:
            snap = list(_LIVE.values())
        for e in snap:  # poll outside the lock: it may call abort(), which takes the lock
            if isinstance(e, LlamaConsole) and e.poll:
                try:
                    e.poll(e)
                except Exception:
                    pass
        with _LC_LOCK:
            snap = list(_LIVE.values())
        consoles = [e for e in snap if isinstance(e, LlamaConsole) and not e.closed]
        waiters = [e for e in snap if isinstance(e, Waiting)]
        n = len(consoles) + len(waiters)
        prev_mode = mode
        if n >= 2:
            if mode != "multi":
                mode = "multi"
                span_t0 = now
                span_written0 = _WRITTEN[0] + sum(c.n_gen for c in consoles)
                span_tasks = set()
                last_print = 0.0  # print the first block right away
        elif n == 1:
            mode = "single"
        else:
            mode = "none"
        if _STYLE == "panel":
            if busy is None and n >= 2:
                busy = {"t0": now, "w0": _WRITTEN[0] + sum(c.n_gen for c in consoles), "tasks": set(), "peak": 0}
            if busy is not None:
                busy["tasks"].update(e.task for e in consoles + waiters)
                busy["peak"] = max(busy["peak"], n)
                if n:
                    idle_since = None
                elif idle_since is None:
                    idle_since = now
                elif now - idle_since >= 2.0:   # quiet for 2 s: the busy period is over
                    el = max(idle_since - busy["t0"], 1e-6)
                    tok = _WRITTEN[0] - busy["w0"]
                    lc_print(f"{time.strftime('%H:%M:%S', time.localtime(idle_since))}  ── busy {_dur(el)}, "
                             f"up to {busy['peak']} at once: {len(busy['tasks'])} requests, {tok:,} tok written, "
                             f"{tok / el:,.0f} t/s combined")
                    busy, idle_since = None, None
        elif prev_mode == "multi" and n <= 1:
            el = max(now - span_t0, 1e-6)
            tok = _WRITTEN[0] + sum(c.n_gen for c in consoles) - span_written0
            lc_print(f"== back to one request: {len(span_tasks)} requests over {_dur(el)}, "
                     f"{tok:,} tok written, {tok / el:,.0f} t/s combined ==" if n else
                     f"== all done: {len(span_tasks)} requests over {_dur(el)}, "
                     f"{tok:,} tok written, {tok / el:,.0f} t/s combined ==")
            mode, last_print = "none", 0.0
        if _STYLE == "panel":
            if n == 0:
                if _PANEL:
                    _set_panel([])
                continue
            if _REDRAW[0] or now - last_panel >= PANEL_EVERY:
                last_panel = now
                _REDRAW[0] = False
                with _LC_LOCK:
                    rows = []
                    pp_total = 0.0
                    gen_total = 0.0
                    for e in sorted(consoles + waiters, key=lambda e: e.task):
                        r = e.panel_row(now)
                        if r is None:
                            continue
                        rows.append(r)
                        span_tasks.add(e.task)
                        if isinstance(e, LlamaConsole):
                            if e.t_dec0 is not None:
                                gen_total += e._panel_rate
                            elif e.pp_rate:
                                pp_total += e.pp_rate
                    head = f"── LIVE ── {len(consoles)} running"
                    if len(waiters):
                        head += f", {len(waiters)} queued"
                    if pp_total > 0:
                        head += f" ── pp {pp_total:,.0f} t/s"
                    if gen_total > 0:
                        head += f" ── gen {gen_total:,.0f} t/s"
                    head += " "
                    head += "─" * max(0, _cols() - 1 - len(head))
                _set_panel([head] + rows)
            continue
        if n >= 2:
            if now - last_print >= BLOCK_EVERY:
                last_print = now
                _print_block(consoles, waiters, now, span_tasks)
        elif n == 1:
            entry = consoles[0] if consoles else waiters[0]
            # skip the first CONSOLE_EVERY s after registration: short requests show only start + done
            if now - last_print >= CONSOLE_EVERY and now - entry.t0 >= CONSOLE_EVERY:
                s = entry.status(now) if isinstance(entry, LlamaConsole) else entry.status()
                if s:
                    last_print = now
                    lc_print(s)


def _demo():
    def run_a():
        con = LlamaConsole(next_task(), 131072, 20000, "MTP", worker={"name": "A"}).launch()
        con.set_cache(12000)
        for i in range(4):  # 8,000 fresh tokens at ~2,500 t/s
            time.sleep(0.8)
            con.progress(12000 + (i + 1) * 2000, 2500.0, time.time())
        con.prompt_done(12000, 3200)
        con.phase = "thinking"
        for i in range(1, 4):  # ~70 t/s for 3 s
            time.sleep(1)
            con.tokens(i * 70, time.time())
        con.finish(3200, 12000, 3000, 210, "stop", 150, 190)

    def run_bc(name: str, finish_it: bool):
        con = LlamaConsole(next_task(), 131072, 2000, None, worker={"name": name}).launch()
        con.set_cache(0)
        for i in range(1, 4):  # read a 2,000 tok prompt for 1 s
            time.sleep(0.34)
            con.progress(2000 * i // 3, 2000.0, time.time())
        con.prompt_done(0, 1000)
        for i in range(1, 6):  # write ~40 t/s for 5 s
            time.sleep(1)
            con.tokens(i * 40, time.time())
        if finish_it:
            con.finish(1000, 0, 5000, 200, "stop")
        else:
            con.cancelled("client gone")
            con.abort()

    def run_d():
        w = Waiting(next_task(), {"name": "D"}, "2 slots busy")
        time.sleep(3)
        w.done()
        con = LlamaConsole(next_task(), 131072, 0, None, worker={"name": "D"}).launch()
        con.prompt_done(0, 0)
        for i in range(1, 4):  # write for 3 s
            time.sleep(1)
            con.tokens(i * 40, time.time())
        con.finish(0, 0, 3000, 120, "stop")

    ta = threading.Thread(target=run_a)
    ta.start()
    ta.join()
    threads = [threading.Thread(target=run_bc, args=("B", True)),
               threading.Thread(target=run_bc, args=("C", False)),
               threading.Thread(target=run_d)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(3)  # let the ticker print the "back to one request" summary


if __name__ == "__main__":
    _demo()
