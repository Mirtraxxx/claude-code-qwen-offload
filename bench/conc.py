"""Concurrency check for the local model server: 1, 2 and 3 simultaneous streaming requests.
Token counts come from the final chunk's usage (DFlash sends several tokens per chunk).

Usage: python conc.py <prose|code> 1 2 3   (the numbers are how many requests to run at once, one round each)"""
import json, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPTS = {
    "prose": "Write a 400-word grim short story about a mercenary in a flooded crypt. No title.",
    "code": ("Write a complete Python module implementing an LRU cache class with get, put, delete, resize and "
             "a thread-safe variant, with type hints and docstrings, followed by a pytest test suite covering "
             "every method and the eviction order. Output only code."),
}
MODE = "prose"


def one(i, out, t0):
    body = {"model": "qwen3.8-27b", "stream": True, "max_tokens": 700, "temperature": 0.7,
            "messages": [{"role": "user", "content": PROMPTS[MODE]}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    first, toks, finish, err = None, 0, None, None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                c = (d.get("choices") or [{}])[0]
                if c.get("delta", {}).get("content") and first is None:
                    first = time.time()
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
                if d.get("usage"):
                    toks = d["usage"]["completion_tokens"]
    except Exception as e:
        err = repr(e)
    end = time.time()
    gen = end - first if first else 0
    out[i] = {"ttft": (first - t0) if first else None, "gen_s": gen, "toks": toks,
              "tok_s": toks / gen if gen else 0, "total": end - t0, "finish": finish, "err": err}


def run(k):
    out, t0 = {}, time.time()
    ts = [threading.Thread(target=one, args=(i, out, t0)) for i in range(k)]
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.time() - t0
    agg = sum(o["toks"] for o in out.values()) / wall
    print(f"\n== {MODE}, {k} at once: wall {wall:.1f}s, aggregate {agg:.0f} tok/s")
    for i in sorted(out):
        o = out[i]
        if not o["finish"]:
            print(f"  req {i + 1}: FAILED after {o['total']:.1f}s (no finish_reason) {o['err'] or ''}")
            continue
        print(f"  req {i + 1}: first token {o['ttft']:.2f}s, {o['toks']} tok in {o['gen_s']:.1f}s "
              f"= {o['tok_s']:.0f} tok/s, done at {o['total']:.1f}s ({o['finish']})")
    sys.stdout.flush()


MODE = sys.argv[1] if len(sys.argv) > 1 else "prose"
for k in [int(x) for x in (sys.argv[2:] or ["1", "2", "3"])]:
    run(k)
