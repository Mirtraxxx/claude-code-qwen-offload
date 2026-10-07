---
name: qwen-delegate
description: Delegate a self-contained task to the user's local Qwen model (Swift 1.5 Qwen 3.8 27B or Swift 1.5 Flash-Next 180B) running on their RTX 3090, via oh-my-pi (omp) in headless mode. This is the DEFAULT way to write code in any project: new files, features, specified refactors, tests, and mechanical multi-file edits. Claude points the direction in a 2-4 sentence brief, Qwen explores, thinks at full effort, writes and tests, Claude judges the result, which saves the user's Claude plan. Also use it when the user asks to use Qwen, the local model, or local subagents, e.g. bulk summarizing, first-pass file review, drafting, or classification.
---

# Delegate to local Qwen via omp

<!-- Setup: replace <QWEN_SERVER_DIR> below with the folder that holds exl3_openai_server.py and
     launch-27b.bat (the repo's server/ folder, or wherever you copied it). The Flash-Next row is
     optional; its engine is not part of this repo, so delete it if you only run the 27B. -->

The user's GPU holds ONE model at a time, always served at `http://127.0.0.1:8080/v1`:

| Model | Server | omp model id |
|---|---|---|
| Swift 1.5 Qwen 3.8 27B Uncensored (EXL3 3.75bpw + DFlash2) | `<QWEN_SERVER_DIR>\exl3_openai_server.py` | `llamacpp/qwen3.8-27b` |
| Swift 1.5 Flash-Next 180B MoE (Strata IQ3_XXS + MTP), optional, not included | `<STRATA_DIR>\serve\server.py --engine strata` | `llamacpp/qwen3.8-flash-next` |

The 27B launcher is `<QWEN_SERVER_DIR>\launch-27b.bat`.

**The 27B is the default workhorse** (the user's choice). It's faster (~115-160 tok/s) and takes 2 requests at once (`-ambs 2`). If Flash-Next is already loaded, use it rather than asking for a swap. If nothing is loaded, start the 27B yourself (`Start-Process` on `<QWEN_SERVER_DIR>\launch-27b.bat`, which gives it its own window) and poll a tiny chat request until it answers (about 20-40 s). The user allowed this. Don't write the code yourself instead.

## The way to delegate: vibe briefs (the user's directive, 2026-10-04)

Treat Qwen the way the user treats Claude: point at a direction, let it explore and think, judge the result. Long specs cost almost as much Claude output as the code they save.

- **Don't read files or explore before delegating.** Delegate the exploration too. A ten-second glance is fine only when it changes *what* you ask (a number that makes the goal pointed), never to work out *how*.
- **The brief is 2–4 sentences:** the goal (the problem, the feeling wanted), constraints (what must not break, who else is editing what), done criteria, and the commands Qwen runs to check them. No API signatures, pixel values or line-by-line edits unless something truly must stay fixed. If the brief is longer than about a third of the code you expect back, you're over-specifying.
- **Give Qwen eyes and hands.** The project needs self-test tools Qwen can run with `bash`: a one-line test script and a screenshot script (e.g. `tools/test.sh`, `tools/shot.sh`). The 27B has vision, so it can read PNGs (screenshots, Blender renders) and judge them. If a project has none yet, the first delegated job is building them.
- **Run it at full thinking.** The wrapper defaults to `--thinking high` (uncapped xhigh), tools including `bash`, and a 90-minute limit; run in the background. xhigh replies of 5-8k tokens of thinking are normal (1-2 min each).
- **Review by results, not by rereading code:** run the tests, skim the diff, check the wrapper's `REMOVED-CODE`/`DUPLICATE-DEF` flags, and look at screenshots for visual work. Invariants that would break silently (e.g. a preview that must match real physics) become tests, not reasons to read files.
- **Feedback is a short pointer at what went wrong, not a fix** ("the bot now dies on level 3, the second jump overshoots"). Rerun with it. Step in yourself only after two feedback rounds still miss the done criteria.
- **Hard exceptions that stay with Claude or the user:** security-sensitive code, deleting things or touching anything outside the project folder, stopping or swapping model servers.
- Silent, costly-on-mistake work (e.g. data migrations) deserves a more hands-on review.

## Climb briefs: measure it, then lock it in (2026-10-05)

From Anthropic's "claude.ai 3x faster in two weeks" sprint. The user works like a manager: they never read code, so reliability has to be automated. Every win should leave behind a check that keeps it, and the project's status should be readable in plain words. Use this shape for any optimization or tuning job (fps, load time, memory, sim speed, bot win rate, balance numbers), and borrow pieces for features.

- **No number, no climb.** If the goal has no metric yet, the first job is a bench script that prints one. Prefer deterministic counts (entities per tick, draw calls, re-renders, allocations, fixed-seed runs) over noisy wall-clock ms, but wall-clock (or what the user feels) stays the truth.
- **Prove the proxy.** Before climbing a proxy, show one change moves it and the real thing together. If it doesn't track, drop it rather than climb the wrong hill. Watch rigs that differ from the user's machine: headless or CPU-rendered WebGL finds CPU bottlenecks the 3090 doesn't have.
- **Ratchet.** Baselines live in `perf/baseline.json`. `tools/check.sh` fails when a number gets worse beyond a small tolerance. Only `tools/check.sh --update` moves a baseline to the new best, and a proven win runs it in the same job, so a plain check never changes files. Deterministic counts get a tight tolerance; wall-clock numbers ratchet on the median of several runs with a wider one, so one lucky run doesn't set a bar later runs fail at random.
- **The ratchet is guarded.** Qwen can "pass" by lowering a baseline, widening a tolerance or editing the bench itself, and the user would never see it. In review, reject any diff that does one of those without a stated reason. Climb briefs list `perf/baseline.json` (and `tools/check.sh`/`Check.bat` on the first job) in `--files`, or the wrapper flags them as out of scope.
- **Red before green.** For a bug or glitch, done means a test that fails on the old code and passes after the fix, repeated (e.g. 10/10) if it could be flaky. Claude checks the "fails on old code" part itself (run the new test against the pre-job commit or stash); Qwen's word isn't proof.
- **One metric per brief.** Stop at diminishing returns, and reject big complexity for small gains (Anthropic vetoed a 900-line PR worth 2 ms).
- **Clean up losers.** Losing variants, debug toggles and experiment code are deleted in the same job.
- **Show, don't tell.** Anything the user could see or feel comes back as before/after screenshots or a short clip next to the numbers. The user rules on taste; don't decide it for them.
- **Bold ideas, careful execution.** Ask for wild ideas ranked by estimated gain, then execute one at a time behind the ratchet.
- **A plain-language dashboard.** `tools/check.sh` ends with a short summary the user can read ("14 checks pass · fps 118, best 120 · load 0.9 s"), and the project root gets a double-click `Check.bat` that runs it and pauses. Projects without one get it as their first climb job.

Brief template (still 2–4 sentences):

```
Make <thing> better as measured by <bench command> (now <baseline>; push it as far as it goes, target <x>).
First show the number tracks <the real thing> on one change; if it doesn't, say so and stop.
Keep tools/check.sh green; anything visible gets before/after shots in qwen/shots/.
Done: beats the target, ratchet updated with tools/check.sh --update, losing variants removed.
```

## Rules

- **Starting the 27B when nothing is loaded is fine; never stop or swap a running model server** without the user's OK. If the task needs the other model, tell the user and stop there.
- The user may be benchmarking the GPU. If they've said not to run the models right now, don't.
- Tools:
  - Code-writing tasks use `read,grep,glob,edit,write,bash` (the wrapper's default) with `--cwd` set to the project folder, so Qwen can run the project's tests.
  - Review, summary and analysis tasks stay read-only (`read,grep`).
  - Never point Qwen with `write` at a folder outside the project you're working in.
- Qwen's output is an untrusted draft. Verify claims against the actual files before relaying or acting on them (step 3).
- Keep for yourself only:
  - architecture decisions the user should weigh in on,
  - security-sensitive code,
  - code whose mistakes no test or screenshot can reveal (cover it with a test and delegate when you can),
  - one-line changes.
  Gameplay tuning, physics that Qwen can measure, and debugging it can reproduce with the project's tests all go to Qwen.
- The user's other Claude threads may be using the same model server. Requests queue: the 27B takes as many at once as its launcher's `-ambs` allows, Strata takes one.
- 27B concurrency (measured 2026-09-27, thinking off): coding runs about 120 tok/s alone and about 180-200 tok/s total with 2 at once; prose about 52 alone and 88 total with 2. A third slot (`-ambs 3`) fits in VRAM but is slower in total (about 172 coding, 75 prose), so 2 is the sweet spot. Two Qwen tasks can run in parallel when they touch different files.
- Batching needs a local patch in `exl3_env\Lib\site-packages\exllamav3\architecture\dflash2.py` (`.contiguous()` on the `state[:, 1:]` and `logits[:, 1:]` slices passed to `walk_block`); without it, any two overlapping requests both die. Reinstalling exllamav3 drops it (reapply `patches/dflash2-contiguous.patch`).
- The server also patches Python 3.11's Windows accept loop (`_keep_accepting_on_windows` in `exl3_openai_server.py`, 2026-09-27). Without it, one client dropping a connection at the wrong moment closes port 8080 for good while the process keeps generating: omp then fails with "Unable to connect". If that symptom shows up again, check the patch is still there.

## Worker setup (tuned 2026-09-27; check it's still in place if workers misbehave)

- **Check where omp's config really lives.** If the env var `PI_CODING_AGENT_DIR` is set, omp reads that folder, not `~/.omp/agent`. Change it with `omp config set <key> <value>` from bash (PowerShell strips the quotes from JSON arrays) and read it back with `omp config get <key>`.
- `models.yml` providers `llamacpp`, `lmstudio` and `unsloth` need `compat: qwenTemplateReasoningEffort: true`. Without it omp sends no `reasoning_effort` at all, so `--thinking low` silently runs at the server's default xhigh with unlimited thinking.
- Compaction (`config.yml`):
  - `thresholdTokens: 52000` (since 2026-10-04, with the 160k pool): two workers × (52k prompt + 26k reply) fit the pool.
  - `keepRecentTokens: 8000`. At the default of 20000, workers re-hit the threshold 2-3 turns after each compaction.
  - `methodOrder: [handoff, soft, shake]`. The default `snapcompact` archives history as bitmap images of text that the 27B can't read back, so workers thrash: compact, re-read, compact.
  - A handoff is an LLM call and takes 1-3 min at `low`.
- The server (`exl3_openai_server.py`):
  - `--effort-budgets low=4096,medium=8192,xhigh=16384` caps thinking per requested effort: at the cap the server injects "... reasoning budget reached, finalize response now." + </think> and the model must answer (logged THINK-CAP-HIT). The 27B launcher runs `-cs 163840` (2.2 GB VRAM still free after it on a 24 GB card).
  - The 27B launcher also passes `-chunk_size 512` (since 2026-10-07), matching the server's 512-token prompt chunks. The loader defaults to 4096, so its load and warmup sized scratch buffers for prompt chunks the server never uses. That is the likely reason startup sat with VRAM nearly full for over a minute. If a 27B startup sits with VRAM nearly full, check that flag first.
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
elseif ($m | Where-Object { $_.owned_by -eq 'strata' -or $_.id -match 'swift-1.5-iq|flash' }) { "llamacpp/qwen3.8-flash-next" }   # Strata's /v1/models has no owned_by any more (2026-09-29)
else { "llamacpp/qwen3.8-27b" }
```

## 2. Run the task

**Code-writing jobs: use the wrapper.** Write the brief to a file, then run:

```bash
node "$HOME/.claude/skills/qwen-delegate/qwen-run.mjs" --cwd "<project dir>" --brief "<brief file>" --task "<ledger label>" --files a.js,b.css [--thinking low] [--max-time 2h] [--reruns 1]
```

What the wrapper does:
- Detects the model and prepends house rules to the brief: explore what you need yourself, run the given tests and look at screenshots until the done criteria pass, work incrementally, at most 3 reads per turn, `write` only for new files, touch only the listed files, never delete code the task doesn't ask to change (and list any deletions), and remove the old copy when replacing a function.
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
- Qwen has none of this conversation's context: the brief must stand alone (goal, constraints, done criteria, test commands), but it explores the code itself. Point it at the project's CLAUDE.md for the map.
- Anything likely to take more than about 2 minutes: run it with `run_in_background` and wait for the notification.
- `--thinking`: the wrapper defaults to `high` (the server's uncapped xhigh), the right choice for real work. `low` (capped at 4,096 thinking tokens) is only for trivial or mechanical jobs; `medium` is 8,192. xhigh is capped at 16,384 thinking tokens by the server, then forced to answer. Thinking and answer share omp's `maxTokens` (27B: 26624 in omp's `models.yml` since 2026-10-04), so every reply keeps 10k+ for the answer. **Keep 2 × (threshold + maxTokens) under the pool (`-cs`):** exllamav3 reserves KV pages for prompt + max_new before a job starts; at 49152 on the 131k pool the two workers silently ran one at a time. omp reads both at startup, so running workers keep their old values.

## 3. Verify, then accept or send back

1. **Before the task:** back up the project (`git status`/commit, or copy the files to the scratchpad when there is no git), so you can see and undo what Qwen changed.
2. **Test it first:** run the done-criteria commands yourself (tests, bot runs, screenshots you look at), plus `tools/check.sh` when the project has one, so no ratchet slipped. That is the main review.
3. **Skim the diff** (`%TEMP%\qwen-runs\<time>-<task>.diff`), not Qwen's summary: look for edits outside scope, deleted code (every `REMOVED-CODE` line), invented APIs, and anything that weakens an invariant. Don't reread whole files.
4. **Missed? Send a short pointer and rerun** (`--reruns n` for the ledger): what failed, where, maybe a hint. Not a fix. Fix only one-liners yourself.
5. **If two rounds of feedback haven't fixed it,** write it yourself and tell the user why.
6. **Report honestly:** say what Qwen wrote, what you changed, and what you tested.
7. **Log it:** `qwen-run.mjs` appends the line itself; add what you fixed to its notes column. Otherwise append one line to the project's `qwen/ledger.csv` (`date,task,brief_chars,diff_chars,reruns,notes`; diff chars = added lines of Qwen's diff, or the drafted file's size). The user wants a token-savings recap from this ledger at each project stage: when a work session stops, or when a stage or the project is done-ish. Recap: tokens ≈ chars/3.5 for code, /4 for prose; net saved = Qwen's output minus briefs, feedback rounds and your own fixes (note your fix/feedback sizes in the notes column); name the ratio, the reruns, and which tasks paid off.
