@echo off
title Swift 1.5 Qwen 3.8 27B Uncensored [THINKING MODE - Official Guide] (EXL3 3.75bpw H5 + DFlash2)

rem ---------------------------------------------------------------------------------------------
rem  Paths: set these as environment variables, or edit the defaults below.
rem    EXL3_PYTHON  python.exe of the exllamav3 venv (default: exl3_env next to this .bat)
rem    MODEL_DIR    Swift-1.5-Qwen3.8-27B-Uncensored-EXL3-3.75bpw (target model)
rem    DRAFT_DIR    Qwen3.8-27B-DFlash2-EXL3-5.0bpw (DFlash2 draft model)
rem  HOST is 127.0.0.1 (this machine only). The server has no authentication and can read local
rem  image paths, so only set HOST=0.0.0.0 on a network you trust.
rem ---------------------------------------------------------------------------------------------
if not defined EXL3_PYTHON set "EXL3_PYTHON=%~dp0exl3_env\Scripts\python.exe"
if not defined MODEL_DIR   set "MODEL_DIR=%USERPROFILE%\models\Swift-1.5-Qwen3.8-27B-Uncensored-EXL3-3.75bpw"
if not defined DRAFT_DIR   set "DRAFT_DIR=%USERPROFILE%\models\Qwen3.8-27B-DFlash2-EXL3-5.0bpw"
if not defined HOST        set "HOST=127.0.0.1"
if not defined PORT        set "PORT=8080"

echo =========================================================================
echo   Swift 1.5 Qwen 3.8 27B Uncensored (SC_3.75bpw_H5_V6) + DFlash2
echo   - Mode:       THINKING MODE (enable_thinking=true, effort=xhigh)
echo   - Reasoning:  Native Swift 1.5 RL/OPD (--reasoning-budget 0, --reasoning-preserve)
echo   - Think caps: low 4k, medium 8k, xhigh 16k tokens, then forced to answer
echo   - Official:   temp=0.6 (Coding/DFlash2), top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0
echo   - Target:     Swift-1.5-Qwen3.8-27B-Uncensored-EXL3-3.75bpw (14.52 GiB, 5-bit Head, 6-bit Vision)
echo   - Drafter:    Qwen3.8-27B-DFlash2-EXL3-5.0bpw (1.47 GB)
echo   - Context:    163,840 tokens (-cs 163840, 4,4-bit Hadamard KV Cache)
echo   - Batching:   2 concurrent requests (-ambs 2)
echo   - Load chunk: 512 tokens (-chunk_size 512, same as the server's prompt chunk, so warmup doesn't size buffers for 4096)
echo   - Web Chat:   http://127.0.0.1:%PORT%/
echo   - API:        http://127.0.0.1:%PORT%/v1
echo   - Host:       %HOST%
echo =========================================================================
echo.

"%EXL3_PYTHON%" "%~dp0exl3_openai_server.py" ^
  -m "%MODEL_DIR%" ^
  -dm "%DRAFT_DIR%" ^
  -cs 163840 ^
  -cq 4,4 ^
  -ambs 2 ^
  -chunk_size 512 ^
  --mode thinking ^
  --reasoning-effort xhigh ^
  --reasoning-budget 0 ^
  --effort-budgets low=4096,medium=8192,xhigh=16384 ^
  --reasoning-preserve ^
  --temp 0.6 ^
  --top-p 0.95 ^
  --top-k 20 ^
  --min-p 0.0 ^
  --presence-penalty 0.0 ^
  --repetition-penalty 1.0 ^
  --host %HOST% ^
  --port %PORT% ^
  --open-browser

pause
