---
name: qwen-delegate
description: Delegate a self-contained task to the user's local Qwen model (Swift 1.5 Qwen 3.8 27B or Swift 1.5 Flash-Next 180B) running on their RTX 3090, via oh-my-pi (omp) in headless mode. This is the DEFAULT way to write code in any project: new files, features, specified refactors, tests, and mechanical multi-file edits. Qwen writes, Claude specifies and verifies, which saves the user's Claude plan. Also use it when the user asks to use Qwen, the local model, or local subagents, e.g. bulk summarizing, first-pass file review, drafting, or classification.
---

# Delegate to local Qwen via omp

<!-- Setup: replace <QWEN_SERVER_DIR> below with the folder that holds exl3_openai_server.py and
     launch-27b.bat (the repo's server/ folder, or wherever you copied it). The Flash-Next row is
     optional; its engine is not part of this repo, so delete it if you only run the 27B. -->

The user's GPU holds ONE model at a time, always served at `http://127.0.0.1:8080/v1`:

| Model | Server | omp model id |
|---|---|---|
| Swift 1.5 Qwen 3.8 27B Uncensored (EXL3 3.75bpw + DFlash2) | `<QWEN_SERVER_DIR>\exl3_openai_server.py` | `llamacpp/qwen3.8-27b` |
| Swift 1.5 Flash-Next 180B MoE (Strata IQ2_XS + MTP), optional, not included | `<STRATA_DIR>\serve\server.py --engine strata` | `llamacpp/qwen3.8-flash-next` |

The 27B launcher is `<QWEN_SERVER_DIR>\launch-27b.bat`.

**The 27B is the default workhorse** (the user's choice). It's faster (~115-160 tok/s) and takes 2 requests at once (`-ambs 2`). If Flash-Next is already loaded, use it rather than asking for a swap. If nothing is loaded, start the 27B yourself (`Start-Process` on `<QWEN_SERVER_DIR>\launch-27b.bat`, which gives it its own window) and poll a tiny chat request until it answers (about 20-40 s). The user allowed this. Don't write the code yourself instead.

## Rules

- **Starting the 27B when nothing is loaded is fine; never stop or swap a running model server** without the user's OK. If the task needs the other model, tell the user and stop there.
- The user may be benchmarking the GPU. If they've said not to run the models right now, don't.
- Tools:
  - Code-writing tasks use `read,grep,glob,edit,write` with `--cwd` set to the project folder (`edit` is find/replace).
  - Review, summary and analysis tasks stay read-only (`read,grep,glob`).
  - Never point Qwen with `edit` or `write` at a folder outside the project you're working in.
- Qwen's output is an untrusted draft. Verify claims against the actual files before relaying or acting on them (step 3).
- Keep for yourself anything that needs real reasoning:
  - architecture decisions,
  - debugging with an unknown cause,
  - security-sensitive or subtle concurrency/numeric code,
  - changes too small to be worth a spec.
- The user's other Claude threads may be using the same model server. Requests queue: the 27B takes as many at once as its launcher's `-ambs` allows, Strata takes one.
- 27B concurrency (measured 2026-09-27, thinking off): coding runs about 120 tok/s alone and about 180-200 tok/s total with 2 at once; prose about 52 alone and 88 total with 2. A third slot (`-ambs 3`) fits in VRAM but is slower in total (about 172 coding, 75 prose), so 2 is the sweet spot. Two Qwen tasks can run in parallel when they touch different files.
- Batching needs a local patch in `exl3_env\Lib\site-packages\exllamav3\architecture\dflash2.py` (`.contiguous()` on the `state[:, 1:]` and `logits[:, 1:]` slices passed to `walk_block`); without it, any two overlapping requests both die. Reinstalling exllamav3 drops it (reapply `patches/dflash2-contiguous.patch`).

## 1. Detect the loaded model

```powershell
try { $m = (Invoke-RestMethod http://127.0.0.1:8080/v1/models -TimeoutSec 5).data } catch { $m = $null }
if (-not $m) { "none" }
elseif ($m | Where-Object owned_by -eq 'strata') { "llamacpp/qwen3.8-flash-next" }
else { "llamacpp/qwen3.8-27b" }
```

## 2. Run the task

```powershell
omp -p --no-session --model <omp model id> --thinking <low|medium|high> --tools read,grep,glob --cwd "<project dir>" --max-time 10m "<task prompt>"
```

- `--cwd` is required: omp moves itself to a temp dir when started in the home folder.
- omp prints a "Working..." status line to stderr, which PowerShell shows as a NativeCommandError. It's harmless: check the exit code and the stdout answer. Add `2>$null` to hide it.
- Baseline: a trivial one-file read on Flash-Next at `--thinking low` took about 16 s end to end.
- Working `--tools` set (omp 18.2): `read,grep,glob,edit,write` (`edit` is find/replace). Use all five for code-writing tasks, `read,grep,glob` for read-only ones.
- Write a complete, self-contained prompt: Qwen has none of this conversation's context. Name the files, the goal, and the output format you want back.
- Anything likely to take more than about 2 minutes: run it with `run_in_background` and wait for the notification.
- `--thinking`: `low` for mechanical tasks, `medium` by default, `high` only when it's worth the wait.

## 3. Verify, then accept or send back

1. **Before the task:** record where things stand (`git status`, or note the files involved), so you can see exactly what Qwen changed.
2. **Review the change:** read the whole diff yourself, not Qwen's summary of it. Check it against the spec: missing pieces, invented APIs, edits outside scope, deleted code.
3. **Test it:** run the project's build, tests or linter, and actually exercise the feature where that's practical.
4. **Fix small problems yourself.** For bigger ones, run Qwen again, quoting the exact failure or diff lines and what should change.
5. **If two rounds of feedback haven't fixed it,** write it yourself and tell the user why.
6. **Report honestly:** say what Qwen wrote, what you changed, and what you tested.
