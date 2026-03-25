@echo off
REM deploy.bat — Native bootstrapper for Coral TPU Detection Skill (Windows)
REM
REM Installs ai-edge-litert in a Python venv and verifies Edge TPU connectivity.
REM Called by Aegis skill-runtime-manager during installation.
REM
REM Prerequisites:
REM   - Python 3.9+ installed and on PATH
REM   - libedgetpu installed (https://github.com/google-coral/libedgetpu/releases)
REM
REM Exit codes:
REM   0 = success (Edge TPU detected)
REM   1 = fatal error
REM   2 = partial success (no TPU, CPU fallback)

setlocal EnableDelayedExpansion

set SKILL_DIR=%~dp0
set VENV_DIR=%SKILL_DIR%.venv
set LOG_PREFIX=[coral-tpu-deploy]

REM ─── Step 1: Find Python ───────────────────────────────────────────────────

echo %LOG_PREFIX% Checking for Python 3.9+... >&2

where python >nul 2>&1
if errorlevel 1 (
    echo {"event": "error", "stage": "python", "message": "Python not found. Install Python 3.9+"}
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo %LOG_PREFIX% Found Python %PY_VER% >&2

REM ─── Step 2: Create virtual environment ────────────────────────────────────

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo {"event": "progress", "stage": "venv", "message": "Creating virtual environment..."}
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo {"event": "error", "stage": "venv", "message": "Failed to create virtual environment"}
        exit /b 1
    )
)

set VENV_PIP=%VENV_DIR%\Scripts\pip.exe
set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe

echo %LOG_PREFIX% Virtual environment ready >&2

REM ─── Step 3: Install dependencies ─────────────────────────────────────────

echo {"event": "progress", "stage": "install", "message": "Installing ai-edge-litert and dependencies..."}
"%VENV_PIP%" install --upgrade pip -q 2>&1
"%VENV_PIP%" install -r "%SKILL_DIR%requirements.txt" -q 2>&1

echo %LOG_PREFIX% Dependencies installed >&2
echo {"event": "progress", "stage": "install", "message": "Dependencies installed"}

REM ─── Step 4: Check libedgetpu ──────────────────────────────────────────────

echo %LOG_PREFIX% Checking for libedgetpu... >&2
echo {"event": "progress", "stage": "driver", "message": "Checking for Edge TPU driver..."}

where edgetpu.dll >nul 2>&1
if errorlevel 1 (
    echo %LOG_PREFIX% WARNING: edgetpu.dll not found. >&2
    echo %LOG_PREFIX% Download the runtime from: >&2
    echo %LOG_PREFIX%   https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip >&2
    echo %LOG_PREFIX% Extract and run install.bat >&2
    echo {"event": "progress", "stage": "driver", "message": "libedgetpu not found — install from Google releases"}
)

REM ─── Step 5: Probe for Edge TPU ───────────────────────────────────────────

echo {"event": "progress", "stage": "probe", "message": "Checking for Edge TPU devices..."}

set TPU_FOUND=false
"%VENV_PYTHON%" "%SKILL_DIR%scripts\tpu_probe.py" > "%TEMP%\tpu_probe_result.json" 2>nul
if not errorlevel 1 (
    set TPU_FOUND=true
)

REM ─── Step 6: Complete ──────────────────────────────────────────────────────

set RUN_CMD=%VENV_PYTHON% %SKILL_DIR%scripts\detect.py

if "%TPU_FOUND%"=="true" (
    echo {"event": "complete", "status": "success", "accelerator_found": true, "run_command": "%RUN_CMD%", "message": "Coral TPU skill installed — Edge TPU ready"}
    echo %LOG_PREFIX% Done! Edge TPU ready. >&2
    exit /b 0
) else (
    echo {"event": "complete", "status": "partial", "accelerator_found": false, "run_command": "%RUN_CMD%", "message": "Coral TPU skill installed — no TPU detected (CPU fallback)"}
    echo %LOG_PREFIX% Done with warning: no TPU detected. >&2
    exit /b 2
)
