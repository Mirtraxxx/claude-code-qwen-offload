import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import threading
import traceback
import time
import uuid
import urllib.request
import webbrowser
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import AutoTokenizer

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler, ComboSampler
from exllamav3.cache import CacheLayer_quant, CacheLayer_fp16
from exllamav3.generator.sampler.custom import (
    CustomSampler,
    SS_Argmax,
    SS_MinP,
    SS_PresFreqP,
    SS_RepP,
    SS_Sample,
    SS_Temperature,
    SS_TopK,
    SS_TopP,
)


class SS_OutputOnlyPresFreqP(SS_PresFreqP):
    """
    OpenAI / llama.cpp / vLLM compatible presence & frequency penalty step:
    In ExLlamaV3, `state.past_ids` contains `cat([prompt_ids, generated_ids])`.
    Standard `SS_PresFreqP(sustain_range=1024)` scans backwards into the last 1,024 tokens
    of the INPUT PROMPT / TOOL OUTPUT, subtracting 1.5 logits from filenames (`Username`,
    `myproject.html`), URLs, and code identifiers before the assistant even types them!
    This step sets `sustain_range = min(max_range, num_generated_tokens)` so ONLY newly
    generated tokens in the current response are penalized, and disables penalty inside
    `<tool_call>` blocks (`active_flag[0] = False`) so code/paths are never mutated.
    """

    def __init__(
        self,
        prompt_tokens: int,
        active_flag: list[bool],
        pres_p: float = 0.0,
        freq_p: float = 0.0,
        max_range: int = 1024,
    ):
        super().__init__(pres_p=pres_p, freq_p=freq_p, sustain_range=max_range, decay_range=0)
        self.prompt_tokens = prompt_tokens
        self.active_flag = active_flag
        self.max_range = max_range

    def run(self, state):
        if state.past_ids is None or not self.active_flag[0] or (self.pres_p == 0.0 and self.freq_p == 0.0):
            return
        cur_gen_len = int(state.past_ids.shape[-1]) - self.prompt_tokens
        if cur_gen_len <= 0:
            return
        self.sustain_range = min(self.max_range, cur_gen_len)
        self.decay_range = 0
        super().run(state)


def build_qwen_sampler(
    prompt_tokens: int,
    active_flag: list[bool],
    rep_p: float,
    pres_p: float,
    freq_p: float,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> CustomSampler:
    stack = [
        SS_RepP(rep_p, 1024, 256),
        SS_OutputOnlyPresFreqP(prompt_tokens, active_flag, pres_p=pres_p, freq_p=freq_p, max_range=1024),
    ]
    if temperature <= 0.01 or top_k == 1:
        stack.append(SS_Argmax())
    else:
        stack.extend([
            SS_Temperature(temperature),
            SS_MinP(min_p),
            SS_TopK(top_k),
            SS_TopP(top_p),
            SS_Sample(),
        ])
    return CustomSampler(stack)

app = FastAPI(title="ExLlamaV3 + DFlash2 OpenAI Server (Hermes Compatible)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BatchPump:
    """One generator loop, many requests. ExLlama batches every job already in the queue."""

    def __init__(self):
        self.generator = None
        self.slots = 1
        self.queues: dict[int, asyncio.Queue] = {}
        self.active = 0
        self.task: asyncio.Task | None = None

    def configure(self, generator, slots: int):
        self.generator = generator
        self.slots = max(1, int(slots))

    def _ensure(self):
        if self.task is None:
            self.task = asyncio.get_running_loop().create_task(self._pump())

    def _announce(self, previous: int, active: int):
        if active >= 2 and previous < 2:
            print("\n=========================================================================", flush=True)
            print(f"  CONCURRENCY RUNNING  {active}/{self.slots} slots", flush=True)
            print("=========================================================================\n", flush=True)
        elif previous >= 2 and active < 2:
            print(f"\n  concurrency dropped to {active}/{self.slots}\n", flush=True)

    async def _pump(self):
        while True:
            if self.generator is None or self.generator.num_remaining_jobs() == 0:
                await asyncio.sleep(0.01)
                continue
            try:
                results = self.generator.iterate()
            except Exception as exc:
                traceback.print_exc()
                err = {"stage": "error", "eos": True, "error": str(exc)}
                for q in list(self.queues.values()):
                    q.put_nowait(err)
                await asyncio.sleep(0.05)
                continue
            for res in results or []:
                q = self.queues.get(res.get("serial"))
                if q is not None:
                    q.put_nowait(res)
            await asyncio.sleep(0)

    async def events(self, job):
        self._ensure()
        q: asyncio.Queue = asyncio.Queue()
        serial = None
        previous = self.active
        self.active += 1
        self._announce(previous, self.active)
        try:
            serial = self.generator.enqueue(job)
            self.queues[serial] = q
            while True:
                res = await q.get()
                yield res
                if res.get("stage") == "error":
                    raise RuntimeError(res.get("error") or "generation failed")
                if res.get("eos"):
                    return
        finally:
            if serial is not None:
                self.queues.pop(serial, None)
                gen = self.generator
                if job in gen.pending_jobs or job in gen.active_jobs:
                    try:
                        gen.cancel(job)
                    except Exception:
                        pass
            previous = self.active
            self.active = max(0, self.active - 1)
            self._announce(previous, self.active)


def _live_tag(res: dict | None = None) -> str:
    batch: BatchPump = STATE["batch"]
    active = batch.active
    serial = "?" if res is None else res.get("serial", "?")
    mode = "CONCURRENT" if active > 1 else "single"
    return f"{mode} {active}/{batch.slots} job {serial}"


def _snippet(messages) -> str:
    for msg in reversed(messages):
        role = msg.get("role")
        if role not in ("user", "tool"):
            continue
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        content = " ".join(content.split())
        if not content:
            continue
        if len(content) > 80:
            content = content[:79] + "…"
        return f"{role}: {content}"
    return "no recent text"


def _next_req_id() -> int:
    n = int(STATE.get("req_seq", -1)) + 1
    STATE["req_seq"] = n
    return n


# ------------------------------------------------------------------------------------------------ console
# llama-server style console with honest numbers (the same helper in Strata's serve/server.py and in
# exl3_openai_server.py, so both windows read and count alike):
#   * prefill progress and "prompt eval" count only FRESH prompt tokens; reused (cached) tokens are shown apart,
#     never inside a speed
#   * decode counts and times come from the engine (Strata: T lines and DONE; ExLlamaV3: job.new_tokens,
#     time_first_token, and the final result's new_tokens / time_generate), never from re-tokenized text
#   * a ticker prints about every CONSOLE_EVERY seconds; token counts are only what the engine has reported, the
#     elapsed time is live and the "left" estimate is marked with ~
#   * no timestamps, ASCII only
_LC_LOCK = threading.Lock()
CONSOLE_EVERY = float(os.environ.get("CONSOLE_EVERY", "1.0") or 1.0)


def _lc_dur(sec: float) -> str:
    """12 s, 3m05s, 1h02m."""
    sec = max(0, int(round(sec)))
    if sec < 60:
        return f"{sec} s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def lc_print(msg: str, level: str = "I", extra: list[str] | None = None):
    """One console record; `extra` lines follow unprefixed (like llama-server's timing block)."""
    with _LC_LOCK:
        print(f"{level} {msg}", flush=True)
        for line in extra or []:
            print(line, flush=True)


class LlamaConsole:
    """One request (llama-server "task") in the console.  Feed it what the engine reports: set_cache(), progress(),
    prompt_done(), tokens(); or give it `poll(con)`, which the ticker calls before each line to read live engine
    state.  finish() prints the final timing block and the release line."""

    def __init__(self, task: int, n_ctx: int, n_prompt: int, draft_name: str | None = None, poll=None,
                 every: float | None = None, slot: int = 0, chunk_note: str = ""):
        self.task, self.n_ctx, self.n_prompt, self.draft_name, self.poll = task, n_ctx, n_prompt, draft_name, poll
        self.every = CONSOLE_EVERY if every is None else every
        self.chunk_note = chunk_note          # e.g. "the engine reports once per 2048-token chunk"
        self.pre = f"id {slot:2d} | task {task}"
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.n_cache, self.cache_exact, self._cache_shown = None, False, None
        self.pos = None                       # absolute prompt position last reported by the engine
        self.pp_rate = None                   # fresh tokens / s, as reported
        self.pp_t = None                      # when that report arrived (wall clock)
        self.prompt_ms = None
        self.t_dec0 = None                    # decode clock start (wall clock), the engine's own where known
        self.n_gen = 0
        self.t_first = None                   # first generated token (wall clock)
        self.win = None
        self.closed = False
        self.stop_evt = threading.Event()
        self.thread = None

    # ---------------------------------------------------------------- lifecycle
    def launch(self):
        lc_print(f"slot launch_slot_: {self.pre} | processing task")
        lc_print(f"slot update_slots: {self.pre} | new prompt, n_ctx_slot = {self.n_ctx}, "
                 f"n_prompt_tokens = {self.n_prompt}")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _stop_ticker(self):
        self.stop_evt.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)

    # ---------------------------------------------------------------- engine facts
    def set_cache(self, n_cache: int, exact: bool = True):
        """Prompt tokens reused from the cache (not evaluated again)."""
        with self.lock:
            if self.cache_exact and not exact:
                return
            self.n_cache, self.cache_exact = int(n_cache), exact or self.cache_exact
            show = self.n_cache != self._cache_shown          # print once; again only if the value changes
            if show:
                self._cache_shown = self.n_cache
        if show:
            how = "reused from cache" if exact else "reused from cache, inferred from the engine's rate"
            lc_print(f"slot update_slots: {self.pre} | n_past = {self.n_cache} ({how}), "
                     f"n_tokens to eval = {max(0, self.n_prompt - self.n_cache)}")

    def progress(self, pos: int, rate: float, t_report: float | None = None):
        """The engine reached absolute prompt position `pos` at `t_report` (wall clock, default now), having read
        fresh tokens at `rate` tok/s up to then."""
        with self.lock:
            self.pos, self.pp_rate = int(pos), float(rate)
            self.pp_t = time.time() if t_report is None else float(t_report)

    def prompt_done(self, n_cache: int, prompt_ms: float, t_dec0: float | None = None):
        """Prefill finished; the decode clock starts at t_dec0 (wall clock; now if not given)."""
        self.set_cache(n_cache, True)
        with self.lock:
            if self.prompt_ms is not None:
                return
            self.prompt_ms = float(prompt_ms)
            self.t_dec0 = time.time() if t_dec0 is None else float(t_dec0)
            self.win = (self.t_dec0, 0)
            fresh = max(0, self.n_prompt - self.n_cache)
            cached = f" ({self.n_cache:,} cached)" if self.n_cache else ""
            took = (f"{self.prompt_ms / 1000.0:.2f} s" if self.prompt_ms < 60000.0
                    else _lc_dur(self.prompt_ms / 1000.0))
        lc_print(f"slot update_slots: {self.pre} | prompt done: {fresh:,} / {fresh:,} tokens in {took}, "
                 f"{1000.0 * fresh / max(self.prompt_ms, 1e-6):.0f} t/s{cached}")

    def tokens(self, n_gen: int, t_first: float | None = None):
        with self.lock:
            self.n_gen = max(self.n_gen, int(n_gen))
            if self.t_first is None and t_first is not None:
                self.t_first = float(t_first)

    def token(self):
        with self.lock:
            self.n_gen += 1
            if self.t_first is None:
                self.t_first = time.time()

    # ---------------------------------------------------------------- live numbers
    def live(self) -> dict:
        """llama.cpp-named live timings for API chunks (prompt_n = fresh tokens, cache_n = reused)."""
        with self.lock:
            cache = self.n_cache or 0
            fresh = max(0, self.n_prompt - cache)
            pms = self.prompt_ms if self.prompt_ms is not None else (time.time() - self.t0) * 1000.0
            dms = (time.time() - self.t_dec0) * 1000.0 if self.t_dec0 else 0.0
            return {"cache_n": cache, "prompt_n": fresh, "prompt_ms": round(pms, 2),
                    "prompt_per_second": round(1000.0 * fresh / max(pms, 1e-6), 2),
                    "predicted_n": self.n_gen, "predicted_ms": round(dms, 2),
                    "predicted_per_second": round(1000.0 * self.n_gen / dms, 2) if dms > 0 else 0.0}

    def _line(self) -> str | None:
        now = time.time()
        with self.lock:
            if self.t_dec0 is None:                                  # prefill
                # counts: the engine's last report (never interpolated); elapsed: live; left: ~ from the rate
                cache = self.n_cache or 0
                cached = f" ({cache:,} cached)" if cache else ""
                fresh_total = max(1, self.n_prompt - cache)
                el = _lc_dur(now - self.t0)
                if self.pos is None or self.n_cache is None:
                    note = f", {self.chunk_note}" if self.chunk_note else ""
                    return (f"slot update_slots: {self.pre} | prompt processing: {fresh_total:,} tokens to read, "
                            f"{el} elapsed{note}{cached}")
                done = min(fresh_total, max(0, self.pos - cache))
                left = ""
                if self.pp_rate > 0:
                    left = f", ~{_lc_dur(max(0.0, (fresh_total - done) / self.pp_rate - (now - self.pp_t)))} left"
                return (f"slot update_slots: {self.pre} | prompt processing: {done:,} / {fresh_total:,} tokens "
                        f"({100.0 * done / fresh_total:.1f}%), {self.pp_rate:.0f} t/s, {el} elapsed{left}{cached}")
            t_w, n_w = self.win
            dt = now - t_w
            inst = (self.n_gen - n_w) / dt if dt > 0 else 0.0
            self.win = (now, self.n_gen)
            dec = now - self.t_dec0
            avg = self.n_gen / dec if dec > 0 else 0.0
            return (f"slot print_timing: {self.pre} | n_gen = {self.n_gen}, tg = {avg:.2f} t/s, "
                    f"tg_1s = {inst:.2f} t/s")

    def _run(self):
        step = min(0.1, self.every) if self.poll is not None else self.every   # poll often, print every `every`
        next_print = time.time() + self.every
        while not self.stop_evt.wait(step):
            if self.poll is not None:
                try:
                    self.poll(self)
                except Exception:
                    pass
            if time.time() < next_print:
                continue
            next_print += self.every
            line = self._line()
            if line and not self.stop_evt.is_set():
                lc_print(line)

    # ---------------------------------------------------------------- end of a request
    def finish(self, prompt_ms: float, n_cache: int, decode_ms: float, n_gen: int, finish: str = "stop",
               draft_accepted: int | None = None, draft_total: int | None = None, note: str = ""):
        """The engine's final numbers: prompt_ms / decode_ms / n_gen / draft counts exactly as it reported them."""
        self._stop_ticker()
        with self.lock:
            if self.closed:
                return
            self.closed = True
            ttft = "-" if self.t_first is None else f"{(self.t_first - self.t0) * 1000.0:.2f} ms"
        fresh = max(0, self.n_prompt - int(n_cache))
        pms, dms, ng = float(prompt_ms), float(decode_ms), int(n_gen)
        block = [
            f"prompt eval time = {pms:10.2f} ms / {fresh:5d} tokens ({pms / max(fresh, 1):8.2f} ms per token, "
            f"{1000.0 * fresh / max(pms, 1e-6):8.2f} tokens per second)",
            f"       eval time = {dms:10.2f} ms / {ng:5d} tokens ({dms / max(ng, 1):8.2f} ms per token, "
            f"{1000.0 * ng / max(dms, 1e-6):8.2f} tokens per second)",
            f"      total time = {pms + dms:10.2f} ms / {fresh + ng:5d} tokens",
        ]
        if draft_total is not None and draft_accepted is not None and self.draft_name:
            rate = draft_accepted / draft_total if draft_total else 0.0
            block.append(f"{self.draft_name} draft acceptance rate = {rate:0.5f} ({draft_accepted:5d} accepted / "
                         f"{draft_total:5d} generated)")
        block.append(f"     prompt cache = {int(n_cache)} of {self.n_prompt} prompt tokens reused, ttft = {ttft}, finish = {finish}"
                     + (f", {note}" if note else ""))
        lc_print(f"slot print_timing: {self.pre} |", extra=block)
        lc_print(f"slot      release: {self.pre} | stop processing: n_past = {self.n_prompt + ng}, truncated = 0")

    def cancelled(self, detail: str = ""):
        lc_print(f"srv          stop: cancel task, id_task = {self.task}" + (f" ({detail})" if detail else ""), "W")

    def abort(self, n_gen: int | None = None):
        """The request ended without final numbers (client gone): release with what is known."""
        self._stop_ticker()
        with self.lock:
            if self.closed:
                return
            self.closed = True
            ng = self.n_gen if n_gen is None else int(n_gen)
        lc_print(f"slot      release: {self.pre} | stop processing: n_past = {self.n_prompt + ng}, truncated = 0")

    def error(self, msg: str):
        self._stop_ticker()
        with self.lock:
            self.closed = True
        lc_print(f"srv    send_error: task id = {self.task}, error: {msg}", "E")


# ExLlamaV3 side of the console: every number is read from the generator's own job state (kv_position, cached
# pages/tokens, new_tokens, time_first_prefill / time_first_token) or its final result (new_tokens, time_prefill,
# time_generate, cached_tokens, accepted/rejected draft tokens).  A requeued job keeps its serial number, so the
# live job is looked up by serial.
def _exl3_live_job(job):
    serial = getattr(job, "serial_number", None)
    if serial is None:
        return job, False
    gen = STATE.get("generator")
    for j in list(getattr(gen, "active_jobs", []) or []) + list(getattr(gen, "pending_jobs", []) or []):
        if getattr(j, "serial_number", None) == serial:
            return j, False
    return job, True


def _exl3_poll(job, prompt_tokens: int):
    gone_since = [None]
    last = [None, None]                  # (kv position, when it last moved)

    def poll(con):
        j, gone = _exl3_live_job(job)
        if gone:                         # finished, cancelled or failed; finish()/abort() normally follow at once
            if gone_since[0] is None:
                gone_since[0] = time.time()
            elif time.time() - gone_since[0] > 5.0:
                con.abort()              # nobody closed it (an error path): release, stop ticking
            return
        gone_since[0] = None
        if j.time_first_prefill is None:
            return
        if con.t_dec0 is None:
            cached = min(prompt_tokens, j.cached_pages * 256 + j.cached_tokens)
            con.set_cache(cached)
            if j.time_first_token is None:
                pos = sum(sq.kv_position for sq in j.sequences)
                if pos != last[0]:
                    last[0], last[1] = pos, time.time()
                el = last[1] - j.time_first_prefill
                if pos > cached and el > 0:      # rate up to the moment the position last moved
                    con.progress(pos, (pos - cached) / el, t_report=last[1])
                return
            con.prompt_done(cached, (j.time_first_token - j.time_first_prefill) * 1000.0, t_dec0=j.time_first_token)
        con.tokens(max(0, int(getattr(j, "rq_new_tokens", 0) or 0) + int(j.new_tokens)), t_first=j.time_first_token)

    return poll


def exl3_console(job, req_id: int, prompt_tokens: int) -> LlamaConsole:
    draft = "DFlash2" if STATE.get("draft_model") is not None else None
    return LlamaConsole(req_id, int(STATE.get("cache_size", 0) or 0), prompt_tokens, draft,
                        poll=_exl3_poll(job, prompt_tokens)).launch()


def exl3_finish(con: LlamaConsole, last_res: dict, job, prompt_tokens: int, finish: str):
    """The final block, from the generator's end-of-stream result."""
    j, _ = _exl3_live_job(job)
    cached = int(last_res.get("cached_tokens", min(prompt_tokens, j.cached_pages * 256 + j.cached_tokens)))
    t_pp = float(last_res.get("time_prefill", 0.0) or 0.0)
    t_gen = float(last_res.get("time_generate", 0.0) or 0.0)
    n = int(last_res.get("new_tokens", con.n_gen))
    acc, rej = last_res.get("accepted_draft_tokens"), last_res.get("rejected_draft_tokens")
    if con.prompt_ms is None:            # a request shorter than one console tick
        con.prompt_done(cached, t_pp * 1000.0)
    con.tokens(n)
    con.finish(t_pp * 1000.0, cached, t_gen * 1000.0, n, finish,
               None if acc is None else int(acc), None if acc is None or rej is None else int(acc) + int(rej))


STATE: dict[str, Any] = {
    "model": None,
    "vision_model": None,
    "draft_model": None,
    "tokenizer": None,
    "hf_tokenizer": None,
    "generator": None,
    "batch": BatchPump(),
    "lock": asyncio.Lock(),
    "default_temp": 0.6,
    "default_reasoning_effort": "low",
}

MODEL_ALIASES = [
    "qwen3.8-27b",
    "swift-1.5-qwen3.8-27b",
    "Swift-1.5-Qwen3.8-27B-Uncensored",
    "Swift-1.5-Qwen3.8-27B-Uncensored-EXL3-3.75bpw",
    "swift-qwen3.8-27b",
    "swift-qwen-27b",
    "Swift-Qwen3.8-27B",
    "Swift-Qwen3.8-27B-EXL3",
    "Swift-Qwen3.8-27B-Uncensored",
    "default",
]


def load_pil_image(url_or_data: str) -> Image.Image:
    if url_or_data.startswith("data:image"):
        header, b64data = url_or_data.split(",", 1)
        raw = base64.b64decode(b64data)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    elif url_or_data.startswith("http://") or url_or_data.startswith("https://"):
        req = urllib.request.Request(url_or_data, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    else:
        path = url_or_data.replace("file:///", "").replace("file://", "")
        return Image.open(path).convert("RGB")


def coerce_param_value(val_str: str, param_schema: dict[str, Any]) -> Any:
    val_stripped = val_str.strip()
    ptype = param_schema.get("type")
    if isinstance(ptype, list):
        ptype = next((t for t in ptype if t != "null"), ptype[0] if ptype else "string")
    if ptype == "integer":
        try:
            return int(val_stripped)
        except Exception:
            pass
    elif ptype == "number":
        try:
            return float(val_stripped)
        except Exception:
            pass
    elif ptype == "boolean":
        if val_stripped.lower() in ("true", "1", "yes"):
            return True
        if val_stripped.lower() in ("false", "0", "no"):
            return False
    elif ptype in ("object", "array"):
        try:
            return json.loads(val_stripped)
        except Exception:
            pass
    return val_str


def parse_tool_calls(text: str, tools: Optional[list[dict[str, Any]]] = None) -> tuple[str, list[dict[str, Any]]]:
    """Extract <tool_call>...</tool_call> blocks (both Qwen3.8 XML <function=...> and JSON formats)."""
    if "<tool_call>" not in text:
        return text, []

    schema_map: dict[str, dict[str, Any]] = {}
    if tools:
        for t in tools:
            fn = t.get("function") or {}
            fname = fn.get("name")
            if fname:
                schema_map[fname] = (fn.get("parameters") or {}).get("properties") or {}

    tool_calls: list[dict[str, Any]] = []
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    for match in pattern.finditer(text):
        inner = match.group(1).strip()
        fn_match = re.search(r"<function=([^\s>]+)>\s*(.*?)\s*</function>", inner, re.DOTALL)
        if fn_match:
            fn_name = fn_match.group(1).strip()
            fn_body = fn_match.group(2)
            args_dict: dict[str, Any] = {}
            props = schema_map.get(fn_name, {})
            for p_match in re.finditer(r"<parameter=([^\s>]+)>\n?(.*?)\n?</parameter>", fn_body, re.DOTALL):
                p_name = p_match.group(1).strip()
                p_val = p_match.group(2)
                args_dict[p_name] = coerce_param_value(p_val, props.get(p_name, {}))
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps(args_dict, ensure_ascii=False),
                },
            })
        else:
            try:
                parsed = json.loads(inner)
                fn_name = parsed.get("name") or (parsed.get("function") or {}).get("name")
                fn_args = parsed.get("arguments") or (parsed.get("function") or {}).get("arguments") or {}
                if isinstance(fn_args, dict):
                    fn_args = json.dumps(fn_args, ensure_ascii=False)
                if fn_name:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {"name": fn_name, "arguments": str(fn_args)},
                    })
            except Exception:
                pass

    cleaned_text = pattern.sub("", text).strip()
    return cleaned_text, tool_calls


def prepare_messages_and_images(raw_messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Convert OpenAI messages + extract PIL images into ExLlamaV3 vision embeddings (with SHA-256 MMEmbedding cache so KV cache never invalidates across turns)."""
    tokenizer = STATE["tokenizer"]
    vision_model = STATE["vision_model"]
    img_cache: dict[str, Any] = STATE.setdefault("image_cache", {})
    image_embeddings = []
    cleaned_messages = []

    for msg in raw_messages:
        m = dict(msg)
        role = m.get("role", "user")
        if role == "developer":
            m["role"] = "system"

        # Ensure reasoning_content is always preserved for Qwen3.8 chat_template.jinja
        if role == "assistant" and not m.get("reasoning_content"):
            alt_reasoning = m.get("reasoning") or m.get("reasoning_text")
            if isinstance(alt_reasoning, str) and alt_reasoning:
                m["reasoning_content"] = alt_reasoning

        # Normalize tool_calls arguments from JSON string -> dict for Qwen3.8 chat_template.jinja
        if m.get("tool_calls") and isinstance(m["tool_calls"], list):
            new_tcs = []
            for tc in m["tool_calls"]:
                tc_copy = dict(tc)
                if "function" in tc_copy and isinstance(tc_copy["function"], dict):
                    fn_copy = dict(tc_copy["function"])
                    args = fn_copy.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn_copy["arguments"] = json.loads(args) if args.strip() else {}
                        except Exception:
                            fn_copy["arguments"] = {}
                    tc_copy["function"] = fn_copy
                new_tcs.append(tc_copy)
            m["tool_calls"] = new_tcs

        content = m.get("content")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    text_parts.append(str(item))
                    continue
                itype = item.get("type", "")
                if itype == "text" or "text" in item:
                    text_parts.append(item.get("text", ""))
                elif itype == "image_url" or "image_url" in item or "image" in item:
                    img_obj = item.get("image_url") or item.get("image")
                    url = img_obj.get("url") if isinstance(img_obj, dict) else str(img_obj)
                    if vision_model is not None and url:
                        try:
                            cache_key = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()
                            ie = img_cache.get(cache_key)
                            if ie is None:
                                pil_img = load_pil_image(url)
                                ie = vision_model.get_image_embeddings(tokenizer=tokenizer, image=pil_img)
                                if len(img_cache) >= 64:
                                    img_cache.pop(next(iter(img_cache)))
                                img_cache[cache_key] = ie
                            image_embeddings.append(ie)
                            text_parts.append(ie.text_alias)
                        except Exception as e:
                            print(f" [!] Vision load warning: {e}")
            m["content"] = "\n".join(text_parts)
        elif content is None:
            m["content"] = ""

        cleaned_messages.append(m)

    # Ensure only the first message has role=='system' (Qwen3.8 jinja requirement)
    if len(cleaned_messages) > 1:
        sys_parts = []
        non_sys = []
        for i, msg in enumerate(cleaned_messages):
            if msg["role"] == "system":
                if msg.get("content"):
                    sys_parts.append(str(msg["content"]))
            else:
                non_sys.append(msg)
        if sys_parts:
            cleaned_messages = [{"role": "system", "content": "\n\n".join(sys_parts)}] + non_sys

    return cleaned_messages, image_embeddings


@app.get("/health")
async def health():
    return {"status": "ok"}


def _build_props_payload(model_param: Optional[str] = None) -> dict[str, Any]:
    has_vision = bool(STATE.get("vision_model") is not None)
    has_draft = bool(STATE.get("draft_model") is not None)
    slots = max(1, int(STATE["batch"].slots))
    n_ctx = int(STATE.get("cache_size", 131072))
    model_path = str(STATE.get("model_path") or "Swift-1.5-Qwen3.8-27B-Uncensored-EXL3-3.75bpw")
    model_alias = str(model_param or STATE.get("model_alias") or "qwen3.8-27b")
    hf_tok = STATE.get("hf_tokenizer")
    chat_tmpl = getattr(hf_tok, "chat_template", "") or ""
    return {
        "role": "model",
        "model_path": model_path,
        "model_alias": model_alias,
        "total_slots": slots,
        "modalities": {
            "vision": has_vision,
            "audio": False,
            "video": False,
        },
        "default_generation_settings": {
            "id": 0,
            "id_task": 0,
            "n_ctx": n_ctx,
            "speculative": has_draft,
            "is_processing": bool(STATE["batch"].active > 0),
            "params": {
                "n_predict": -1,
                "temperature": float(STATE.get("default_temp", 0.6)),
                "top_k": int(STATE.get("default_top_k", 20)),
                "top_p": float(STATE.get("default_top_p", 0.95)),
                "min_p": float(STATE.get("default_min_p", 0.0)),
                "repeat_penalty": float(STATE.get("default_rep_p", 1.0)),
                "presence_penalty": float(STATE.get("default_pres_p", 1.5)),
                "frequency_penalty": float(STATE.get("default_freq_p", 0.0)),
                "reasoning_effort": str(STATE.get("default_reasoning_effort", "xhigh")),
            },
        },
        "chat_template": chat_tmpl if isinstance(chat_tmpl, str) else "",
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "build_info": "ExLlamaV3 + DFlash2 (llama-server WebUI compatible)",
        "webui_settings": {},
    }


@app.get("/props")
@app.get("/v1/props")
async def get_props(model: Optional[str] = None, autoload: Optional[str] = None):
    return _build_props_payload(model)


@app.get("/slots")
@app.get("/v1/slots")
async def get_slots():
    slots = max(1, int(STATE["batch"].slots))
    n_ctx = int(STATE.get("cache_size", 131072))
    has_draft = bool(STATE.get("draft_model") is not None)
    return [
        {
            "id": i,
            "id_task": i,
            "n_ctx": n_ctx,
            "speculative": has_draft,
            "is_processing": bool(i < STATE["batch"].active),
        }
        for i in range(slots)
    ]


@app.post("/v1/streams/lookup")
async def streams_lookup():
    return []


@app.post("/v1/chat/completions/control")
async def chat_completions_control():
    return {"status": "ok"}


@app.get("/models")
@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    has_vision = bool(STATE.get("vision_model") is not None)
    items = [
        {
            "id": alias,
            "model": alias,
            "name": alias,
            "object": "model",
            "created": now,
            "owned_by": "exllamav3-dflash2",
            "status": {"value": "loaded"},
            "modalities": {
                "vision": has_vision,
                "audio": False,
                "video": False,
            },
        }
        for alias in MODEL_ALIASES
    ]
    return {
        "object": "list",
        "data": items,
        "models": items,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    raw_messages = body.get("messages", [])
    tools = body.get("tools") or None
    stream = bool(body.get("stream", False))

    # Handle llama-server WebUI preEncode (n_predict: 0 or max_tokens: 0) immediately
    raw_n_predict = body.get("n_predict")
    raw_max_tokens = body.get("max_tokens") if body.get("max_tokens") is not None else body.get("max_completion_tokens")
    if (raw_n_predict is not None and int(raw_n_predict) == 0) or (raw_max_tokens is not None and int(raw_max_tokens) == 0):
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "qwen3.8-27b"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    req_max_toks = int(raw_max_tokens or raw_n_predict or 16384)
    if req_max_toks < 0:
        req_max_toks = 16384
    max_tokens = max(req_max_toks, 16384) if tools else req_max_toks

    req_temp = body.get("temperature")
    if req_temp is not None and float(req_temp) == 0.0:
        temp = 0.0
    else:
        temp = STATE.get("default_temp", 1.0)

    top_p = STATE.get("default_top_p", 0.95)
    top_k = STATE.get("default_top_k", 20)
    min_p = STATE.get("default_min_p", 0.0)
    pres_p = STATE.get("default_pres_p", 1.5)
    freq_p = STATE.get("default_freq_p", 0.0)
    rep_p = STATE.get("default_rep_p", 1.0)
    # CRITICAL: Do NOT enable SS_DRY (dry_multiplier > 0) by default for coding/agents!
    # ExLlamaV3's SS_DRY has no sequence breakers by default and bans repeating 2+ token
    # sequences from the last 1024 tokens (including prompt/tool outputs), which forces the model
    # to mutate filenames/paths (e.g. 'Username' -> 'Usrname'/'Usernmae', 'myproject' -> 'myprojject').
    dry_mult = float(body.get("dry_multiplier") or 0.0)

    # Thinking / reasoning effort controlled by the launched .bat mode (--mode thinking vs --mode instruct)
    chat_kwargs = body.get("chat_template_kwargs") or {}
    if not STATE.get("default_enable_thinking", True):
        enable_thinking = False
    else:
        enable_thinking = chat_kwargs.get("enable_thinking", True)

    reasoning_effort = body.get("reasoning_effort") or chat_kwargs.get("reasoning_effort") or STATE["default_reasoning_effort"]
    if reasoning_effort not in ("low", "medium", "xhigh"):
        reasoning_effort = STATE["default_reasoning_effort"]

    reasoning_budget = int(body.get("reasoning_budget") or STATE.get("default_reasoning_budget", 2048))
    reasoning_budget_msg = STATE.get(
        "default_reasoning_budget_msg",
        " ... reasoning budget reached, finalize response now.",
    )
    preserve_thinking = STATE.get("default_reasoning_preserve", True)

    cleaned_messages, image_embeddings = prepare_messages_and_images(raw_messages)

    hf_tok = STATE["hf_tokenizer"]
    tokenizer = STATE["tokenizer"]
    generator = STATE["generator"]

    prompt_str = hf_tok.apply_chat_template(
        cleaned_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=preserve_thinking,
    )

    if image_embeddings:
        input_ids = tokenizer.encode(prompt_str, encode_special_tokens=True, embeddings=image_embeddings)
    else:
        input_ids = tokenizer.encode(prompt_str, encode_special_tokens=True)

    prompt_tokens = int(input_ids.shape[-1])
    penalty_active = [True]
    sampler = build_qwen_sampler(
        prompt_tokens=prompt_tokens,
        active_flag=penalty_active,
        rep_p=rep_p,
        pres_p=pres_p,
        freq_p=freq_p,
        temperature=0.0 if temp <= 0.01 else temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_ts = int(time.time())
    model_name = body.get("model", "qwen3.8-27b")
    req_id = _next_req_id()
    req_note = _snippet(cleaned_messages)
    batch: BatchPump = STATE["batch"]

    if not stream:
        async with STATE["lock"]:
            job = Job(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                stop_conditions=generator.model.config.eos_token_id_list,
                sampler=sampler,
                embeddings=image_embeddings if image_embeddings else None,
            )
            con = exl3_console(job, req_id, prompt_tokens)

            full_text = ""
            last_res = {}
            gen_start_t = None
            win_t = None
            win_tokens = 0
            cur_tokens = 0
            sec_idx = 0
            pp_start_t = time.perf_counter()
            pp_chunk_t = pp_start_t
            pp_prev_pos = 0
            cached_toks_init = 0
            prefill_done_logged = False
            budget_enforced = False

            async for res in STATE["batch"].events(job):
                    stage = res.get("stage")
                    if stage == "started":
                        pp_start_t = time.perf_counter()
                        pp_chunk_t = pp_start_t
                        cached_toks_init = min(prompt_tokens, job.cached_pages * 256 + job.cached_tokens)
                        pp_prev_pos = cached_toks_init
                        continue

                    if stage == "prefill":
                        now_pp = time.perf_counter()
                        curr_p = int(res.get("curr_progress", 0))
                        max_p = max(1, int(res.get("max_progress", prompt_tokens)))
                        cur_cached = min(prompt_tokens, job.cached_pages * 256 + job.cached_tokens)
                        if cur_cached > cached_toks_init:
                            cached_toks_init = cur_cached
                            pp_prev_pos = max(pp_prev_pos, cached_toks_init)
                        chunk_toks = max(0, curr_p - pp_prev_pos)
                        chunk_dt = max(now_pp - pp_chunk_t, 1e-4)
                        if chunk_toks > 0 and (chunk_dt >= 0.5 or curr_p >= max_p):
                            t_elapsed = max(0.01, now_pp - pp_start_t)
                            pp_prev_pos = curr_p
                            pp_chunk_t = now_pp
                        continue

                    if stage != "streaming":
                        continue

                    if not prefill_done_logged:
                        prefill_done_logged = True
                        if job.time_first_token and job.time_first_prefill:
                            t_pp = max(1e-4, float(job.time_first_token - job.time_first_prefill))
                            first_step_dt = max(0.002, time.time() - float(job.time_first_token))
                        else:
                            t_pp = max(1e-4, time.perf_counter() - pp_start_t)
                            first_step_dt = 0.015
                        gen_start_t = time.perf_counter() - first_step_dt
                        win_t = gen_start_t

                    piece = res.get("text", "")
                    if piece:
                        step_toks = max(1, len(tokenizer.encode(piece, add_bos=False)[0]))
                        cur_tokens += step_toks
                        win_tokens += step_toks
                        full_text += piece
                        if (
                            enable_thinking
                            and reasoning_budget > 0
                            and not budget_enforced
                            and "</think>" not in full_text
                            and cur_tokens >= reasoning_budget
                        ):
                            budget_enforced = True
                            job.constrain_output_now(f"{reasoning_budget_msg}\n</think>\n\n")
                        if penalty_active[0] and (("<tool_call>" in full_text) or (tools and "</think>" in full_text)):
                            penalty_active[0] = False
                        now_t = time.perf_counter()
                        dt = now_t - win_t
                        if dt >= 3.0:
                            sec_idx += 1
                            avg_tps = cur_tokens / max(now_t - gen_start_t, 1e-4)
                            inst_tps = win_tokens / max(dt, 1e-4)
                            win_t = now_t
                            win_tokens = 0
                    if res.get("eos"):
                        last_res = res
                        break

        new_tokens = int(last_res.get("new_tokens", cur_tokens))
        t_gen = float(last_res.get("time_generate", 0.01))
        t_prefill = float(last_res.get("time_prefill", 0.01))
        cached_final = int(last_res.get("cached_tokens", min(prompt_tokens, job.cached_pages * 256 + job.cached_tokens)))
        eval_toks = max(1, prompt_tokens - cached_final)
        dacc = int(last_res.get("accepted_draft_tokens", 0))
        drej = int(last_res.get("rejected_draft_tokens", 0))
        tot_d = dacc + drej
        acc_pct = (100.0 * dacc / tot_d) if tot_d > 0 else 0.0
        rounds = max(1, new_tokens - dacc)
        tok_step = new_tokens / rounds
        pp_tps = eval_toks / max(t_prefill, 1e-5)
        pp_ms_tok = (t_prefill * 1000.0) / eval_toks
        tg_tps = new_tokens / max(t_gen, 1e-5)
        tg_ms_tok = (t_gen * 1000.0) / max(new_tokens, 1)
        tot_ms = (t_prefill + t_gen) * 1000.0

        exl3_finish(con, last_res, job, prompt_tokens, "length" if new_tokens >= max_tokens else "stop")

        reasoning_text = None
        answer_text = full_text
        if enable_thinking and "</think>" in full_text:
            parts = full_text.split("</think>", 1)
            reasoning_text = parts[0].replace("<think>", "").strip()
            answer_text = parts[1].strip()
        elif enable_thinking and full_text.startswith("<think>"):
            answer_text = full_text.replace("<think>", "").strip()

        cleaned_answer, tool_calls = parse_tool_calls(answer_text, tools=tools)
        finish_reason = "tool_calls" if tool_calls else ("length" if new_tokens >= max_tokens else "stop")

        msg_payload: dict[str, Any] = {
            "role": "assistant",
            "content": cleaned_answer if (cleaned_answer or not tool_calls) else None,
        }
        if reasoning_text:
            msg_payload["reasoning_content"] = reasoning_text
            msg_payload["reasoning"] = reasoning_text
        if tool_calls:
            msg_payload["tool_calls"] = tool_calls

        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": msg_payload,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": new_tokens,
                "total_tokens": prompt_tokens + new_tokens,
                "cached_tokens": cached_final,
                "prompt_tokens_details": {"cached_tokens": cached_final},
            },
        })

    # STREAMING MODE (SSE)
    async def event_generator():
        cur_tokens = 0
        con = None
        try:
            async with STATE["lock"]:
                job = Job(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    stop_conditions=generator.model.config.eos_token_id_list,
                    sampler=sampler,
                    embeddings=image_embeddings if image_embeddings else None,
                )
                con = exl3_console(job, req_id, prompt_tokens)
                # Initial role chunk
                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first_chunk)}\n\n"

                in_thinking = bool(enable_thinking)
                buffer = ""
                full_text = ""
                in_tool_call = False
                last_res = {}

                gen_start_t = None
                think_end_t = None
                think_tokens = 0
                win_t = None
                win_tokens = 0
                sec_idx = 0
                pp_start_t = time.perf_counter()
                pp_chunk_t = pp_start_t
                pp_prev_pos = 0
                cached_toks_init = 0
                prefill_done_logged = False
                budget_enforced = False

                async for res in STATE["batch"].events(job):
                    stage = res.get("stage")
                    if stage == "started":
                        pp_start_t = time.perf_counter()
                        pp_chunk_t = pp_start_t
                        cached_toks_init = min(prompt_tokens, job.cached_pages * 256 + job.cached_tokens)
                        pp_prev_pos = cached_toks_init
                        continue

                    if stage == "prefill":
                        now_pp = time.perf_counter()
                        curr_p = int(res.get("curr_progress", 0))
                        max_p = max(1, int(res.get("max_progress", prompt_tokens)))
                        cur_cached = min(prompt_tokens, job.cached_pages * 256 + job.cached_tokens)
                        if cur_cached > cached_toks_init:
                            cached_toks_init = cur_cached
                            pp_prev_pos = max(pp_prev_pos, cached_toks_init)
                        chunk_toks = max(0, curr_p - pp_prev_pos)
                        chunk_dt = max(now_pp - pp_chunk_t, 1e-4)
                        if chunk_toks > 0 and (chunk_dt >= 0.5 or curr_p >= max_p):
                            t_elapsed = max(0.01, now_pp - pp_start_t)
                            pp_prev_pos = curr_p
                            pp_chunk_t = now_pp
                        continue

                    if stage != "streaming":
                        continue

                    if not prefill_done_logged:
                        prefill_done_logged = True
                        if job.time_first_token and job.time_first_prefill:
                            t_pp = max(1e-4, float(job.time_first_token - job.time_first_prefill))
                            first_step_dt = max(0.002, time.time() - float(job.time_first_token))
                        else:
                            t_pp = max(1e-4, time.perf_counter() - pp_start_t)
                            first_step_dt = 0.015
                        gen_start_t = time.perf_counter() - first_step_dt
                        win_t = gen_start_t

                    piece = res.get("text", "")
                    if piece:
                        step_toks = max(1, len(tokenizer.encode(piece, add_bos=False)[0]))
                        cur_tokens += step_toks
                        win_tokens += step_toks
                        if (
                            in_thinking
                            and reasoning_budget > 0
                            and not budget_enforced
                            and "</think>" not in (buffer + piece)
                            and cur_tokens >= reasoning_budget
                        ):
                            budget_enforced = True
                            job.constrain_output_now(f"{reasoning_budget_msg}\n</think>\n\n")
                        now_t = time.perf_counter()
                        dt = now_t - win_t
                        if dt >= 3.0:
                            sec_idx += 1
                            avg_tps = cur_tokens / max(now_t - gen_start_t, 1e-4)
                            inst_tps = win_tokens / max(dt, 1e-4)
                            win_t = now_t
                            win_tokens = 0
                            if in_tool_call:
                                yield ": keepalive\n\n"

                        full_text += piece
                        buffer += piece

                        if in_thinking:
                            if "</think>" in buffer:
                                think_end_t = time.perf_counter()
                                think_tokens = cur_tokens
                                think_part, rest = buffer.split("</think>", 1)
                                think_part = think_part.replace("<think>", "")
                                if think_part:
                                    chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_ts,
                                        "model": model_name,
                                        "choices": [{"index": 0, "delta": {"reasoning_content": think_part, "reasoning": think_part}, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                in_thinking = False
                                if tools:
                                    penalty_active[0] = False
                                buffer = rest.lstrip("\r\n")
                            elif len(buffer) > 12:
                                emit = buffer[:-8].replace("<think>", "")
                                buffer = buffer[-8:]
                                if emit:
                                    chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_ts,
                                        "model": model_name,
                                        "choices": [{"index": 0, "delta": {"reasoning_content": emit, "reasoning": emit}, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"

                        if not in_thinking and not in_tool_call:
                            if "<tool_call>" in buffer:
                                before_tc, tc_rest = buffer.split("<tool_call>", 1)
                                if before_tc:
                                    chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_ts,
                                        "model": model_name,
                                        "choices": [{"index": 0, "delta": {"content": before_tc}, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                in_tool_call = True
                                penalty_active[0] = False
                                buffer = "<tool_call>" + tc_rest
                            elif len(buffer) > 16:
                                emit = buffer[:-12]
                                buffer = buffer[-12:]
                                if emit:
                                    chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_ts,
                                        "model": model_name,
                                        "choices": [{"index": 0, "delta": {"content": emit}, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"

                    if res.get("eos"):
                        last_res = res
                        break

            # Flush remaining buffer
            if in_thinking and buffer:
                emit = buffer.replace("<think>", "").replace("</think>", "")
                if emit:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"reasoning_content": emit, "reasoning": emit}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                buffer = ""

            answer_after_think = full_text.split("</think>", 1)[1] if "</think>" in full_text else full_text
            _, tool_calls = parse_tool_calls(answer_after_think, tools=tools)

            if not in_tool_call and buffer:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": buffer}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            if tool_calls:
                tc_deltas = []
                for idx, tc in enumerate(tool_calls):
                    tc_deltas.append({
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    })
                tc_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(tc_chunk)}\n\n"

            end_t = time.perf_counter()
            new_tokens = int(last_res.get("new_tokens", cur_tokens))
            t_gen = float(last_res.get("time_generate", max(end_t - (gen_start_t or end_t), 0.01)))
            t_prefill = float(last_res.get("time_prefill", 0.01))
            cached_final = int(last_res.get("cached_tokens", min(prompt_tokens, job.cached_pages * 256 + job.cached_tokens)))
            eval_toks = max(1, prompt_tokens - cached_final)
            dacc = int(last_res.get("accepted_draft_tokens", 0))
            drej = int(last_res.get("rejected_draft_tokens", 0))
            tot_d = dacc + drej
            acc_pct = (100.0 * dacc / tot_d) if tot_d > 0 else 0.0
            rounds = max(1, new_tokens - dacc)
            tok_step = new_tokens / rounds
            pp_tps = eval_toks / max(t_prefill, 1e-5)
            pp_ms_tok = (t_prefill * 1000.0) / eval_toks
            tg_tps = new_tokens / max(t_gen, 1e-5)
            tg_ms_tok = (t_gen * 1000.0) / max(new_tokens, 1)
            tot_ms = (t_prefill + t_gen) * 1000.0

            exl3_finish(con, last_res, job, prompt_tokens,
                        "tool_calls" if tool_calls else ("length" if new_tokens >= max_tokens else "stop"))

            finish_reason = "tool_calls" if tool_calls else ("length" if new_tokens >= max_tokens else "stop")
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": new_tokens,
                    "total_tokens": prompt_tokens + new_tokens,
                    "cached_tokens": cached_final,
                    "prompt_tokens_details": {"cached_tokens": cached_final},
                },
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            if con is not None:
                con.cancelled()
                con.abort()
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")


WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")
if os.path.isdir(WEBUI_DIR):
    @app.get("/", include_in_schema=False)
    async def webui_index():
        return FileResponse(os.path.join(WEBUI_DIR, "index.html"), media_type="text/html; charset=utf-8")

    app.mount("/", StaticFiles(directory=WEBUI_DIR, html=True), name="webui")


class QuietPollingFilter(logging.Filter):
    """Suppress high-frequency WebUI polling & static asset requests from cluttering generation telemetry."""
    QUIET_PREFIXES = (
        "GET /props",
        "GET /v1/props",
        "GET /slots",
        "GET /v1/slots",
        "GET /health",
        "GET /v1/models",
        "GET /models",
        "POST /v1/streams/lookup",
        "GET /_app/",
        "GET /favicon",
        "GET /apple-",
        "GET /manifest",
        "GET /sw.js",
        "GET /workbox-",
        "GET /pwa-",
        "GET /maskable-",
        "GET /version.json",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for p in self.QUIET_PREFIXES:
            if p in msg:
                return False
        return True


def main():
    parser = argparse.ArgumentParser()
    from exllamav3 import model_init
    model_init.add_args(
        parser,
        cache=True,
        default_cache_size=98304,
        add_sampling_args=False,
        add_draft_model_args=True,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mode", type=str, choices=["thinking", "instruct"], default="thinking")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", type=str, default="xhigh")
    parser.add_argument("--reasoning-budget", type=int, default=2048)
    parser.add_argument(
        "--reasoning-budget-message",
        type=str,
        default=" ... reasoning budget reached, finalize response now.",
    )
    parser.add_argument("--reasoning-preserve", action="store_true", default=True)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--open-browser", action="store_true", help="Open http://host:port/ in default browser when ready")
    args = parser.parse_args()

    is_instruct = (args.mode == "instruct") or args.no_thinking
    model_basename = os.path.basename(os.path.normpath(args.model_dir))
    if model_basename and model_basename not in MODEL_ALIASES:
        MODEL_ALIASES.insert(1, model_basename)

    STATE["model_path"] = model_basename or args.model_dir
    STATE["model_alias"] = "qwen3.8-27b"
    STATE["cache_size"] = int(args.cache_size)
    STATE["default_enable_thinking"] = not is_instruct
    STATE["default_temp"] = args.temp if args.temp is not None else (0.7 if is_instruct else 1.0)
    STATE["default_top_p"] = args.top_p if args.top_p is not None else (0.80 if is_instruct else 0.95)
    STATE["default_top_k"] = args.top_k
    STATE["default_min_p"] = args.min_p
    STATE["default_pres_p"] = args.presence_penalty
    STATE["default_rep_p"] = args.repetition_penalty
    STATE["default_freq_p"] = args.frequency_penalty
    STATE["default_reasoning_effort"] = args.reasoning_effort
    STATE["default_reasoning_budget"] = 0 if is_instruct else args.reasoning_budget
    STATE["default_reasoning_budget_msg"] = args.reasoning_budget_message
    STATE["default_reasoning_preserve"] = args.reasoning_preserve

    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)

    if not args.no_vision:
        try:
            print(" -- Loading 6-bit Vision Tower...")
            vision_model = Model.from_config(config, component="vision")
            vision_model.load(progressbar=True)
            STATE["vision_model"] = vision_model
        except Exception as e:
            print(f" [!] Vision tower skipped: {e}")

    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        draft_model=draft_model,
        draft_cache=draft_cache,
    )

    STATE["model"] = model
    STATE["draft_model"] = draft_model
    STATE["tokenizer"] = tokenizer
    STATE["hf_tokenizer"] = hf_tokenizer
    STATE["generator"] = generator

    vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
    mode_label = "INSTRUCT (Non-Thinking)" if is_instruct else f"THINKING (effort={args.reasoning_effort}, budget={args.reasoning_budget}t, preserve=True)"
    print("=========================================================================")
    local_url = f"http://127.0.0.1:{args.port}"
    if args.host in ("0.0.0.0", "::"):
        print(f"  ExLlamaV3 + DFlash2 Server Ready on {local_url}/v1 (All interfaces / Wi-Fi: port {args.port})")
        print(f"  Web Chat UI  : {local_url}/  (Official llama.cpp WebUI)")
    else:
        print(f"  ExLlamaV3 + DFlash2 Server Ready on http://{args.host}:{args.port}/v1")
        print(f"  Web Chat UI  : http://{args.host}:{args.port}/  (Official llama.cpp WebUI)")
    print(f"  Active Mode  : {mode_label}")
    print(f"  Sampler (Gov): temp={STATE['default_temp']}, top_p={STATE['default_top_p']}, top_k={STATE['default_top_k']}, min_p={STATE['default_min_p']}, pres_p={STATE['default_pres_p']}")
    slots = max(1, int(args.autosplit_max_batch_size))
    STATE["lock"] = asyncio.Semaphore(slots)
    STATE["batch"].configure(generator, slots)

    print(f"  Context Pool : {args.cache_size:,} tokens ({args.cache_quant}-bit KV Cache)")
    print(f"  Concurrency  : {slots} slots  (this window prints CONCURRENCY RUNNING when 2 or more overlap)")
    print(f"  Vision Tower : {'Loaded (6-bit)' if STATE['vision_model'] else 'Disabled'}")
    print(f"  VRAM Usage   : {vram_alloc:.2f} GiB allocated / {vram_res:.2f} GiB reserved")
    print("=========================================================================")

    logging.getLogger("uvicorn.access").addFilter(QuietPollingFilter())

    if args.open_browser:
        browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{browser_host}:{args.port}/")).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
