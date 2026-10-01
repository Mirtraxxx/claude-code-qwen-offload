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
  - Code-writing tasks use `read,grep,glob,edit,write` with `--cwd` set to the project folder.
  - Review, summary and analysis tasks stay read-only (`read,grep`).
  - Never point Qwen with `write` at a folder outside the project you're working in.
- Qwen's output is an untrusted draft. Verify claims against the actual files before relaying or acting on them (step 3).
- Keep for yourself anything that needs real reasoning:
  - architecture decisions,
  - debugging with an unknown cause,
  - security-sensitive or subtle concurrency/numeric code,
  - changes too small to be worth a spec.
- The user's other Claude threads may be using the same model server. Requests queue: the 27B takes as many at once as its launcher's `-ambs` allows, Strata takes one.
- 27B concurrency (measured 2026-09-27, thinking off): coding runs about 120 tok/s alone and about 180-200 tok/s total with 2 at once; prose about 52 alone and 88 total with 2. A third slot (`-ambs 3`) fits in VRAM but is slower in total (about 172 coding, 75 prose), so 2 is the sweet spot. Two Qwen tasks can run in parallel when they touch different files.
- Batching needs a local patch in `exl3_env\Lib\site-packages\exllamav3\architecture\dflash2.py` (`.contiguous()` on the `state[:, 1:]` and `logits[:, 1:]` slices passed to `walk_block`); without it, any two overlapping requests both die. Reinstalling exllamav3 drops it (reapply `patches/dflash2-contiguous.patch`).
- The server also patches Python 3.11's Windows accept loop (`_keep_accepting_on_windows` in `exl3_openai_server.py`, 2026-09-27). Without it, one client dropping a connection at the wrong moment closes port 8080 for good while the process keeps generating: omp then fails with "Unable to connect". If that symptom shows up again, check the patch is still there.

## Worker setup (tuned 2026-09-27; check it's still in place if workers misbehave)

- **Check where omp's config really lives.** If the env var `PI_CODING_AGENT_DIR` is set, omp reads that folder, not `~/.omp/agent`. Change it with `omp config set <key> <value>` from bash (PowerShell strips the quotes from JSON arrays) and read it back with `omp config get <key>`.
- `models.yml` providers `llamacpp`, `lmstudio` and `unsloth` need `compat: qwenTemplateReasoningEffort: true`. Without it omp sends no `reasoning_effort` at all, so `--thinking low` silently runs at the server's default xhigh with unlimited thinking.
- Compaction (`config.yml`):
  - `thresholdTokens: 40000`, so two workers fit the 128k pool with output room.
  - `keepRecentTokens: 8000`. At the default of 20000, workers re-hit the threshold 2-3 turns after each compaction.
  - `methodOrder: [handoff, soft, shake]`. The default `snapcompact` archives history as bitmap images of text that the 27B can't read back, so workers thrash: compact, re-read, compact.
  - A handoff is an LLM call and takes 1-3 min at `low`.
- The server (`exl3_openai_server.py`):
  - `--effort-budgets low=4096,medium=8192` caps thinking per requested effort.
  - `--max-chunk-size 512` keeps one worker's prompt reading from stalling the other to ~3 tok/s.
  - Its console (`console_view.py`) pins a live panel at the bottom (one row per running or queued request, with pp and t/s) and leaves one line per finished request above it; `CONSOLE_STYLE=lines` brings back the old scrolling lines.
  - It aborts a job when its client disconnects.
  - It appends one line per request to `<QWEN_SERVER_DIR>\requests.log`: worker, effort, thinking budget, prompt/cached/out tokens, finish reason, and `THINK-CAP-HIT` or `ABORTED`. Read this log to see what workers actually did.
- Measured with these settings (two workers at once, 16-file read-and-write jobs): both finished in 16-18 min with 3 handoffs each. There were no truncated replies, and replies ran at 23-96 tok/s (median 43).
- **Brief shape for anything bigger than a few files:**
  - Tell Qwen to work incrementally: finish and save each piece before reading more.
  - At most 3 file reads per turn.
  - `write` only creates new files; changes to an existing file go through `edit`. After a compaction, a worker used `write` to regenerate its own output file from memory and silently dropped half its findings.
  - A job whose working set is bigger than ~35k tokens can't hold it all and must write as it goes.

## 1. Detect the loaded model

```powershell
try { $m = (Invoke-RestMethod http://127.0.0.1:8080/v1/models -TimeoutSec 5).data } catch { $m = $null }
if (-not $m) { "none" }
elseif ($m | Where-Object { $_.owned_by -eq 'strata' -or $_.id -match 'iq2|flash' }) { "llamacpp/qwen3.8-flash-next" }   # Strata's /v1/models has no owned_by any more (2026-09-29)
else { "llamacpp/qwen3.8-27b" }
```

## 2. Run the task

**Code-writing jobs: use the wrapper.** Write the brief to a file, then run:

```bash
node "$HOME/.claude/skills/qwen-delegate/qwen-run.mjs" --cwd "<project dir>" --brief "<brief file>" --task "<ledger label>" --files a.js,b.css [--thinking medium] [--max-time 30m] [--reruns 1]
```

What the wrapper does:
- Detects the model and prepends house rules to the brief: work incrementally, at most 3 reads per turn, `write` only for new files, touch only the listed files, never delete code the task doesn't ask to change (and list any deletions), and remove the old copy when replacing a function.
- Snapshots the project, runs omp with stdin closed (otherwise `omp -p` hangs waiting for piped input), and diffs the result into `%TEMP%\qwen-runs\<time>-<task>.diff`.
- Flags `EMPTY-ANSWER`, `TIMEOUT`, `EXIT-n`, `NO-CHANGES`, `OUT-OF-SCOPE:` and `MISSING:`.
- Also flags `REMOVED-CODE:n` (imports, defs/classes/functions, `<style>`/`<script>`/`<link>` lines removed from a listed file and not re-added) and `DUPLICATE-DEF:` (a function or top-level Python def that now exists twice: the old copy was left behind). It prints each line under the answer. Qwen's most common bug is deleting a line it wasn't asked to touch (6 times by 2026-09-30, several would have crashed), so check every listed removal.
- Appends the ledger line to `<cwd>/qwen/ledger.csv` when that file exists. `diff_chars` counts only the files in `--files`; changes elsewhere in the folder (a parallel job, your own edits) are noted but not credited. **Always pass `--files`**: without it every changed file counts, which inflated the 2026-09-28/30 ledgers by up to 3x on parallel runs.
- Exit codes: 0 clean, 1 flags, 2 bad args, 3 no server, 4 omp missing.
- The snapshot skips images, audio, 3D models, archives, model weights, `node_modules`, venvs, dot-folders and browser profiles (any folder with a `Local State` or `prefs.js`), since Qwen only writes text. Above 20,000 remaining files it stops, compares only the `--files` and flags `SNAPSHOT-CAPPED` (out-of-scope edits go unseen then: check the folder yourself).
- Use `--dry-run` to see the command, and `--no-ledger` for throwaway runs. `--dry-run` also prints the snapshot's file count.

Then verify the diff file yourself (step 3). The flags are a first filter, not the review. Anything longer than about 2 minutes goes in `run_in_background`.

Read-only jobs (review, summary) can call omp directly:

```powershell
omp -p --no-session --model <omp model id> --thinking <low|medium|high> --tools read,grep --cwd "<project dir>" --max-time 10m "<task prompt>"
```

- `--cwd` is required: omp moves itself to a temp dir when started in the home folder.
- omp prints a "Working..." status line to stderr, which PowerShell shows as a NativeCommandError. It's harmless: check the exit code and the stdout answer. Add `2>$null` to hide it.
- Baseline: a trivial one-file read on Flash-Next at `--thinking low` took about 16 s end to end.
- Code-writing tasks use `--tools read,grep,glob,edit,write` (`edit` is find/replace, so Qwen can make targeted changes to big files). Other valid names (omp 18.2): `goal, init_experiment, run_experiment, log_experiment, update_notes`.
- Write a complete, self-contained prompt: Qwen has none of this conversation's context. Name the files, the goal, and the output format you want back.
- Anything likely to take more than about 2 minutes: run it with `run_in_background` and wait for the notification.
- `--thinking`: `low` by default (the server caps it at 4,096 thinking tokens), `medium` (8,192) for tricky logic. `high` maps to the server's uncapped xhigh: only use it when it's worth a very long wait. The Swift finetune can think through a whole 16k reply at xhigh and leave no answer.

## 3. Verify, then accept or send back

1. **Before the task:** record where things stand (`git status`, or note the files involved), so you can see exactly what Qwen changed.
2. **Review the change:** read the whole diff yourself, not Qwen's summary of it. Check it against the spec: missing pieces, invented APIs, edits outside scope, deleted code.
3. **Test it:** run the project's build, tests or linter, and actually exercise the feature where that's practical.
4. **Fix small problems yourself.** For bigger ones, run Qwen again, quoting the exact failure or diff lines and what should change.
5. **If two rounds of feedback haven't fixed it,** write it yourself and tell the user why.
6. **Report honestly:** say what Qwen wrote, what you changed, and what you tested.
7. **Log it:** `qwen-run.mjs` appends the line itself; add what you fixed to its notes column. Otherwise append one line to the project's `qwen/ledger.csv` (`date,task,brief_chars,diff_chars,reruns,notes`; diff chars = added lines of Qwen's diff, or the drafted file's size). The user wants a token-savings recap from this ledger at each project stage: when a work session stops, or when a stage or the project is done-ish. Recap: tokens ≈ chars/3.5 for code, /4 for prose; net saved = Qwen's output minus briefs; name the ratio, the reruns, and which tasks paid off.
