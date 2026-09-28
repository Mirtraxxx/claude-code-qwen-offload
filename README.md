# Claude Code + local Qwen: offload the coding, keep the judgment

This repo is a working setup for using **Claude Code as the architect and reviewer** while a
**local Qwen model writes the code**. Claude writes a self-contained spec, hands it to the local
model through [oh-my-pi (omp)](https://github.com/can1357/oh-my-pi) in headless mode, then reads
the diff, runs the tests and fixes or sends back what's wrong. The point is to make a Claude plan
last longer: the bulk output tokens (the code itself) come from your own GPU.

What's in here:

| Path | What it is |
|---|---|
| `skill/qwen-delegate/SKILL.md` | Claude Code skill: when and how to delegate to Qwen, and how to verify the result |
| `skill/qwen-delegate/qwen-run.mjs` | Wrapper that runs one coding job: adds house rules to the brief, diffs the result, flags problems, logs the ledger |
| `claude-md/CLAUDE.md.snippet` | Policy section for `~/.claude/CLAUDE.md` that makes delegation the default |
| `server/exl3_openai_server.py` | OpenAI-compatible FastAPI server on exllamav3, with DFlash2 speculative decoding, request batching, thinking caps and a live worker board |
| `server/launch-27b.bat` | Windows launcher for the 27B with the recommended settings (`-ambs 2`) |
| `server/requirements.txt` | Python deps for the server (versions from the reference machine) |
| `patches/dflash2-contiguous.patch` | exllamav3 1.5.1 fix needed for batching with DFlash2 |
| `omp/models.yml`, `omp/config.yml` | omp provider/model entry for the local server, and optional role mapping |
| `bench/conc.py` | Throughput benchmark with 1, 2 or 3 concurrent requests |
| `ledger.example.csv` | Header for a per-job savings ledger (see below) |

## Hardware and model

- Windows 11, one **RTX 3090 (24 GB)**.
- Model: **Swift 1.5 Qwen 3.8 27B Uncensored, EXL3 3.75bpw**:
  <https://huggingface.co/tiktits/Swift-1.5-Qwen3.8-27B-Uncensored-EXL3-3.75bpw> (14.5 GiB).
- Draft model for speculative decoding: an EXL3 5.0bpw quant of the DFlash2 drafter
  [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
  (`Qwen3.8-27B-DFlash2-EXL3-5.0bpw`, about 1.5 GB). Use an existing EXL3 quant of it, or make one
  with exllamav3's converter.
- 131k context with a 4-bit KV cache fits alongside two concurrent requests.

### How the model was made

The full recipe is on the [model card](https://huggingface.co/tiktits/Swift-1.5-Qwen3.8-27B-Uncensored-EXL3-3.75bpw). In short:

- **Base:** UkisAI's Swift 1.5, a Qwen3.8-27B fine-tune (GSPO reinforcement learning plus on-policy distillation)
  that uses about 58% fewer reasoning tokens and ends its thinking on its own, so no reasoning budget is needed.
- **Uncensoring:** refusal-direction abliteration (Arditi et al., 2024). A layer-38 refusal direction taken from 400
  AdvBench vs 400 Alpaca prompts is projected out (rank-1, float32) of 131 residual-writing tensors before
  quantization, using tools from `ajgazin/Swift-Qwen3.8-27B-Uncensored-MTP` and `orcarouter/Qwen3.8-27B-Uncensored`.
- **EXL3 quant** (`SC_3.75bpw_H5_V6_MTP4`): 3.75 bpw backbone, 5.0 bpw LM head, 6.0 bpw vision tower, 4.0 bpw MTP
  head, BF16 embeddings kept on the CPU; 14.52 GiB total.
- **Sampling** (from the card): coding at temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5; reasoning at
  temperature 1.0, top_p 0.95, top_k 20.
- **Model license:** Swift Open License v1.0 (personal, research and educational use, and commercial use under $1M
  yearly revenue); base Qwen3.8-27B is Apache 2.0. This repo's own files are public domain (see `LICENSE`).

The author also runs a Swift 1.5 Flash-Next 180B MoE through a separate engine (Strata). That setup
isn't included here; the skill still mentions it as an optional second model and works fine
without it.

## Setup

### 1. exllamav3 environment

Reference versions: Python 3.11, torch 2.10.0+cu128, **exllamav3 1.5.1**
(`exllamav3-1.5.1+cu128.torch2.10.0` prebuilt wheel), fastapi 0.141.1, uvicorn 0.53.0,
transformers 5.17.0. With [uv](https://docs.astral.sh/uv/), from the `server/` folder:

```powershell
uv venv exl3_env --python 3.11
uv pip install --python exl3_env\Scripts\python.exe torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python exl3_env\Scripts\python.exe "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.1/exllamav3-1.5.1+cu128.torch2.10.0-cp311-cp311-win_amd64.whl"
uv pip install --python exl3_env\Scripts\python.exe -r requirements.txt
```

Download the models:

```powershell
hf download tiktits/Swift-1.5-Qwen3.8-27B-Uncensored-EXL3-3.75bpw --local-dir "$env:USERPROFILE\models\Swift-1.5-Qwen3.8-27B-Uncensored-EXL3-3.75bpw"
# and put the DFlash2 EXL3 draft model in "$env:USERPROFILE\models\Qwen3.8-27B-DFlash2-EXL3-5.0bpw"
```

### 2. Apply the batching patch

With DFlash2 drafting, exllamav3 1.5.1 crashes as soon as two requests overlap: both die with a
`.view()` error on a non-contiguous slice. The patch adds `.contiguous()` to the `state[:, 1:]`
and `logits[:, 1:]` slices passed to `walk_block` in `exllamav3/architecture/dflash2.py`.

```powershell
cd server\exl3_env\Lib\site-packages
git apply <repo>\patches\dflash2-contiguous.patch
```

It's a one-line change, so you can also make it by hand. Reinstalling or upgrading exllamav3
drops it; check whether newer versions still need it before applying.

### 3. Launch the server

Run `server\launch-27b.bat`. It uses `EXL3_PYTHON`, `MODEL_DIR` and `DRAFT_DIR` if set, otherwise
the defaults at the top of the file (the venv next to the script and `%USERPROFILE%\models\...`).
Key flags: `-ambs 2` (two concurrent requests, the throughput sweet spot), `-cs 131072 -cq 4,4`
(131k context, 4-bit KV cache). The API is at `http://127.0.0.1:8080/v1` and model id
`qwen3.8-27b`. The server serves the llama.cpp web UI at `/` if you drop its static build into
`server/webui/` (optional).

What the server adds on top of plain exllamav3 (all on by default):

- **Thinking caps per effort** (`--effort-budgets low=4096,medium=8192`): a request that asks for
  `low` or `medium` effort without its own thinking budget gets capped there. `high` maps to
  uncapped `xhigh`.
- **Short prompt chunks** (`--max-chunk-size 512`): while one request reads a long prompt, the
  other keeps generating instead of stalling to ~3 tok/s.
- **Worker board:** while 2+ requests overlap, the console prints one line every 3 s with each
  worker's state (queued, reading prompt, writing, or tools/idle), its speed and tokens so far, and
  the total. A "worker" is one client conversation, keyed by its first user message.
- **Disconnect abort:** a job stops within about a second when its client goes away (uvicorn
  otherwise keeps generating for a dead connection).
- **`requests.log`** next to the script: one line per request with worker, effort, thinking
  budget, prompt/cached/output tokens, finish reason, and `THINK-CAP-HIT` or `ABORTED`.

The server has **no authentication** and will fetch image URLs or read local image paths sent in
requests, so it binds to `127.0.0.1` by default. Only set `HOST=0.0.0.0` on a network you trust.

Quick check:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models
```

### 4. Install omp and add the model

Install oh-my-pi (Windows: `irm https://omp.sh/install.ps1 | iex`, or `bun install -g @oh-my-pi/pi-coding-agent`;
see its README for other options). Tested with **omp 18.2.11**.

Merge `omp/models.yml` into `%USERPROFILE%\.omp\agent\models.yml`. Optionally copy
`omp/config.yml` to `%USERPROFILE%\.omp\agent\config.yml` so every omp role uses the local model
and omp's own subagent fan-out is off. (If `PI_CODING_AGENT_DIR` is set, omp uses that folder
instead of `%USERPROFILE%\.omp\agent`.) Two settings matter even if you skip the rest:

- `qwenTemplateReasoningEffort: true` in the provider's `compat` (already in `omp/models.yml`).
  Without it omp sends no effort level at all, so `--thinking low` silently runs at the server's
  default: xhigh, with unlimited thinking.
- The `compaction` block in `omp/config.yml`. It keeps each worker under about 40k tokens so two
  fit in the 131k cache, and it uses LLM-written handoffs instead of omp's default `snapcompact`,
  which stores history as images the 27B can't read back.

The reference machine also points omp's shell at Git Bash in
`%USERPROFILE%\.omp\agent\settings.json`: `{ "shellPath": "C:\\Program Files\\Git\\bin\\bash.exe" }`.

### 5. Install the Claude Code skill and policy

```powershell
Copy-Item -Recurse skill\qwen-delegate "$env:USERPROFILE\.claude\skills\"
```

In the installed `SKILL.md`, replace `<QWEN_SERVER_DIR>` with the folder that holds the server and
launcher, `<SKILL_DIR>` with the skill's folder, and `<OMP_CONFIG_DIR>` with omp's config folder
(and delete the Flash-Next row if you don't run it). The wrapper needs Node 22+ and Git on PATH. Then paste
`claude-md/CLAUDE.md.snippet` into `~/.claude/CLAUDE.md`, again replacing `<QWEN_SERVER_DIR>`.

## What a delegation looks like

Claude writes a complete brief to a file (Qwen has none of the conversation's context: name the
files, the goal and the expected output) and runs the wrapper:

```bash
node qwen-run.mjs --cwd <project> --brief brief.md --task "add save slots" --files js/save.js,js/ui.js
```

The wrapper:

- detects the loaded model;
- prepends house rules to the brief: work incrementally, read at most 3 files per turn, use
  `write` only for new files, touch only the listed files;
- runs omp with stdin closed (`omp -p` otherwise hangs waiting for piped input);
- snapshots the project before and after, and writes a diff to `%TEMP%\qwen-runs\`;
- flags `EMPTY-ANSWER`, `TIMEOUT`, `EXIT-n`, `NO-CHANGES`, `OUT-OF-SCOPE:` and `MISSING:`;
- appends a line to `<project>/qwen/ledger.csv` if that file exists.

It exits 0 when clean and 1 when anything was flagged. Read-only jobs can call omp directly:

```powershell
omp -p --no-session --model llamacpp/qwen3.8-27b --thinking low --tools read,grep,glob --cwd <project> "<brief>"
```

Then Claude reads the diff, runs the build and tests, and either accepts, fixes small things itself,
or reruns Qwen quoting the exact failure.

## Tips

- **`--cwd` is required.** omp moves itself to a temp dir when started from the home folder.
- **`--thinking low` by default** (capped at 4,096 thinking tokens), `medium` (8,192) for tricky
  logic. At uncapped `high`/xhigh the model can spend a whole 16k reply thinking and output nothing.
- **Big jobs must write as they go.** A job whose working set is bigger than about 35k tokens
  can't hold it all between compactions: brief it to finish and save each piece before reading
  more. After one compaction, a worker used `write` to regenerate its own output file from memory
  and silently dropped half its findings. That's why the house rules restrict `write` to new files.
- **Tools:** `read,grep,glob,edit,write` for code-writing tasks (`edit` is find/replace);
  `read,grep,glob` for review and analysis. Never give it write access outside the project.
- **Treat the output as an untrusted draft.** Always review the diff yourself (not Qwen's summary
  of it) and run the tests.
- omp prints a "Working..." status line to stderr; PowerShell shows it as a NativeCommandError.
  It's harmless: check the exit code and stdout.
- With `-ambs 2`, two Qwen jobs can run in parallel when they touch different files.

## Throughput (RTX 3090, thinking off)

Measured with `bench/conc.py` (`python conc.py <prose|code> 1 2 3`), 700-token responses:

| Concurrent requests | Coding tok/s total | Prose tok/s total |
|---|---|---|
| 1 | 115–121 | ~51 |
| 2 (`-ambs 2`, recommended) | 178–198 | ~88 |
| 3 (`-ambs 3`, 24.1/24.6 GB VRAM) | 166–174 | ~75 |

Real coding jobs run at about 130 tok/s single. Code is faster than prose, most likely because
DFlash2's drafts get accepted more often on predictable text. A third slot fits in VRAM but lowers total throughput.

**Two long agent jobs at once** (thinking low, with the compaction settings above): two 16-file
read-and-write jobs both finished in 16–18 minutes, with 3 handoffs each and no truncated replies.
Replies ran at 23–96 tok/s (median 43), lower than the benchmark because both workers keep
re-reading long prompts and each handoff takes 1–3 minutes.

## Does it actually save Claude tokens?

**Method.** Count the characters Qwen wrote into the project (code diffs plus drafted files).
Subtract the characters Claude wrote in briefs to get that output. Design docs don't count, because
Claude writes those either way. Convert at about 3.5 chars per token for code and about 4 for prose.

**Measured** after the first stretch of a real project (a JavaScript browser game prototype):

- Qwen wrote about 72k characters (~20k tokens): ~30k of feature code and game content across
  several tasks, ~17.5k and ~15.8k in two CSS files, and ~8.9k in a content draft (names, dialogue
  lines).
- Claude wrote 8 briefs, about 21k characters (~5.5k tokens).
- **Net: about 14–15k Claude output tokens saved, a return of about 3.5×.** Output tokens cost
  about 5× input tokens, and reviewing a diff mostly costs input (reading), not output.

**Caveats.**

- In Qwen's favour: Claude would also have spent thinking tokens writing that code itself, and
  those aren't counted.
- Against Qwen: reruns (one run spent its whole budget thinking and produced nothing) and
  follow-up fixes Claude had to make.

**Lessons.**

- Big, mechanical output pays best. The CSS and content drafts returned about 10×.
- A small task whose brief is about as long as its diff saves nothing, e.g. a 2.7k-char brief for a
  ~3k-char diff. It's only worth delegating if Claude can do other work in parallel meanwhile.
- Keep a ledger: one CSV line per Qwen job with task, brief chars, diff chars and reruns.
  `ledger.example.csv` has the header (`date,task,brief_chars,diff_chars,reruns,notes`).
