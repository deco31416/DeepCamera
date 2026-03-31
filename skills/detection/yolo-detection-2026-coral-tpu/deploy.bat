@echo off
REM deploy.bat — Coral TPU Detection Skill installer for Windows
REM
REM What this does:
REM   1. Downloads + installs the Edge TPU runtime (edgetpu.dll) via UAC
REM   2. Creates a Python virtual environment (Python 3.9–3.11 recommended)
REM   3. Installs ai-edge-litert and image processing deps
REM   4. Verifies the compiled yolo26n_edgetpu.tflite model is present
REM   5. Probes for an Edge TPU device
REM
REM Note: pycoral is NOT used. detect.py uses ai-edge-litert directly,
REM       which supports Python 3.9–3.13 and does not require pycoral.
REM
REM Exit codes:
REM   0 = success (TPU detected and ready)
REM   1 = fatal error
REM   2 = partial success (no TPU detected, CPU fallback available)

setlocal enabledelayedexpansion

set "SKILL_DIR=%~dp0"
set "LOG_PREFIX=[coral-tpu-deploy]"

REM Ensure we run inside the skill folder
cd /d "%SKILL_DIR%"

echo %LOG_PREFIX% Platform: Windows 1>&2
echo {"event": "progress", "stage": "platform", "message": "Windows installer starting..."}

REM ─── Step 1: Edge TPU Runtime (UAC elevated install + local bundle) ─────────
REM Two-pronged approach:
REM   A) Run install.bat elevated  → registers WinUSB driver + edgetpu.dll in System32
REM   B) Copy DLLs to local lib\   → Python os.add_dll_directory() picks them up
REM Approach B always happens. Approach A adds the USB device driver (needed for hardware).

REM Check for VC++ 2019 Redistributable (required by edgetpu.dll)
echo %LOG_PREFIX% Checking for Visual C++ 2019 redistributable... 1>&2
powershell -NoProfile -Command "if (Test-Path 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64') { exit 0 } else { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% Installing Visual C++ 2019 Redistributable... 1>&2
    echo {"event": "progress", "stage": "platform", "message": "Installing Visual C++ 2019 Redistributable (required for edgetpu.dll)..."}
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://aka.ms/vs/16/release/vc_redist.x64.exe' -OutFile '%TEMP%\vc_redist.x64.exe' -UseBasicParsing"
    "%TEMP%\vc_redist.x64.exe" /install /quiet /norestart
)

echo %LOG_PREFIX% Downloading Edge TPU runtime... 1>&2
echo {"event": "progress", "stage": "platform", "message": "Downloading Google Edge TPU runtime (edgetpu.dll)..."}

set "TMP_DIR=%TEMP%\coral_tpu_install_%RANDOM%"
mkdir "%TMP_DIR%"
cd /d "%TMP_DIR%"

powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip' -OutFile 'edgetpu_runtime_20221024.zip' -UseBasicParsing"
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: Failed to download Edge TPU runtime. Check internet connectivity. 1>&2
    echo {"event": "error", "stage": "platform", "message": "Download failed - check internet connectivity"}
    cd /d "%SKILL_DIR%"
    rmdir /S /Q "%TMP_DIR%" 2>nul
    exit /b 1
)

powershell -NoProfile -Command "Expand-Archive -Path 'edgetpu_runtime_20221024.zip' -DestinationPath '.' -Force"
cd edgetpu_runtime

echo %LOG_PREFIX% Prompting for Administrator rights to install system driver... 1>&2
echo {"event": "progress", "stage": "platform", "message": "A UAC prompt will appear. Approve it to install the Coral WinUSB driver system-wide."}

REM Write 'N' to a file to suppress the max-frequency clock-speed prompt inside install.bat
echo N> "%TMP_DIR%\clock_answer.txt"

REM Run install.bat elevated. We redirect from our pre-written answer file.
powershell -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList "/c install.bat < '%TMP_DIR%\clock_answer.txt'" -WorkingDirectory '%TMP_DIR%\edgetpu_runtime' -Verb RunAs -Wait" 2>nul

if %errorlevel% neq 0 (
    echo %LOG_PREFIX% System-wide driver install skipped (UAC declined). Will use local DLL bundle. 1>&2
    echo {"event": "progress", "stage": "platform", "message": "UAC declined - using local DLL bundle. Hardware TPU requires the system driver; re-install to retry."}
) else (
    echo %LOG_PREFIX% System-wide Edge TPU driver installed. 1>&2
    echo {"event": "progress", "stage": "platform", "message": "Coral WinUSB driver installed system-wide."}
)

REM Always copy DLLs to local lib\ — Python 3.8+ os.add_dll_directory() uses this.
REM This works even if the UAC install above was skipped.
if not exist "%SKILL_DIR%lib" mkdir "%SKILL_DIR%lib"
copy /Y "libedgetpu\direct\x64_windows\edgetpu.dll" "%SKILL_DIR%lib\edgetpu.dll" >nul 2>&1
copy /Y "third_party\libusb_win\libusb-1.0.dll" "%SKILL_DIR%lib\libusb-1.0.dll" >nul 2>&1

if not exist "%SKILL_DIR%lib\edgetpu.dll" (
    echo %LOG_PREFIX% ERROR: Could not extract edgetpu.dll from runtime zip. 1>&2
    echo {"event": "error", "stage": "platform", "message": "Failed to extract edgetpu.dll - runtime zip may be corrupt"}
    cd /d "%SKILL_DIR%"
    rmdir /S /Q "%TMP_DIR%" 2>nul
    exit /b 1
)

echo %LOG_PREFIX% edgetpu.dll bundled to lib\ for Python DLL search. 1>&2
echo {"event": "progress", "stage": "platform", "message": "Edge TPU DLLs ready."}

cd /d "%SKILL_DIR%"
rmdir /S /Q "%TMP_DIR%" 2>nul

REM ─── Step 2: Find Python ─────────────────────────────────────────────────────
REM ai-edge-litert supports Python 3.9–3.13. We prefer the system default.
REM If only Python 3.12+ is available, it still works (no pycoral needed).

set "PYTHON_CMD="

REM Try common Python launchers in preference order
for %%P in (python python3 py) do (
    if not defined PYTHON_CMD (
        %%P --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=%%P"
        )
    )
)

if not defined PYTHON_CMD (
    echo %LOG_PREFIX% ERROR: Python not found on PATH. 1>&2
    echo {"event": "error", "stage": "python", "message": "Python not found — install Python 3.9-3.11 from python.org and re-run"}
    exit /b 1
)

REM Get Python version for info only (not blocking — ai-edge-litert works 3.9-3.13)
for /f "tokens=2" %%V in ('!PYTHON_CMD! --version 2^>^&1') do set "PY_VERSION=%%V"
echo %LOG_PREFIX% Python version: !PY_VERSION! 1>&2
echo {"event": "progress", "stage": "python", "message": "Using Python !PY_VERSION!"}

REM ─── Step 3: Create virtual environment ──────────────────────────────────────

set "VENV_DIR=%SKILL_DIR%venv"
echo %LOG_PREFIX% Creating virtual environment at %VENV_DIR%... 1>&2
echo {"event": "progress", "stage": "build", "message": "Creating Python virtual environment..."}

!PYTHON_CMD! -m venv "%VENV_DIR%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo %LOG_PREFIX% ERROR: Failed to create virtual environment. 1>&2
    echo {"event": "error", "stage": "build", "message": "venv creation failed"}
    exit /b 1
)

REM ─── Step 4: Install Python dependencies ─────────────────────────────────────
REM ai-edge-litert: LiteRT runtime with Edge TPU delegate support (Python 3.9-3.13)
REM numpy + Pillow: image processing

echo %LOG_PREFIX% Upgrading pip... 1>&2
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo %LOG_PREFIX% Installing dependencies (ai-edge-litert, numpy, Pillow)... 1>&2
echo {"event": "progress", "stage": "build", "message": "Installing ai-edge-litert and image processing libraries..."}

"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%SKILL_DIR%requirements.txt" --quiet
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: pip install failed. 1>&2
    echo {"event": "error", "stage": "build", "message": "pip install requirements.txt failed"}
    exit /b 1
)

echo %LOG_PREFIX% Dependencies installed. 1>&2
echo {"event": "progress", "stage": "build", "message": "Python dependencies installed successfully."}

REM ─── Step 5: Verify compiled EdgeTPU model ────────────────────────────────────
REM The yolo26n_edgetpu.tflite is pre-compiled via docker/compile.sh and committed
REM to the git repository. deploy.bat does NOT compile it — that requires Linux.

echo %LOG_PREFIX% Checking for compiled EdgeTPU model... 1>&2

set "MODEL_FOUND=false"
set "MODEL_FILE="

REM Accept either naming convention from edgetpu_compiler output
for %%M in (
    "%SKILL_DIR%models\yolo26n_int8_edgetpu.tflite"
    "%SKILL_DIR%models\yolo26n_edgetpu.tflite"
    "%SKILL_DIR%models\yolo26n_320_edgetpu.tflite"
) do (
    if exist %%M (
        set "MODEL_FOUND=true"
        set "MODEL_FILE=%%~M"
    )
)

if "!MODEL_FOUND!"=="false" (
    echo %LOG_PREFIX% WARNING: No pre-compiled EdgeTPU model found in models\. 1>&2
    echo {"event": "progress", "stage": "model", "message": "No EdgeTPU model found — will fall back to CPU inference (SSD MobileNet)"}
) else (
    echo %LOG_PREFIX% Found model: !MODEL_FILE! 1>&2
    echo {"event": "progress", "stage": "model", "message": "Edge TPU model ready: yolo26n_edgetpu.tflite"}
)

REM Download SSD MobileNet as a universal CPU fallback so the skill is unconditionally functional
if not exist "%SKILL_DIR%models\ssd_mobilenet_v2_coco_quant_postprocess.tflite" (
    echo %LOG_PREFIX% Downloading SSD MobileNet CPU fallback model... 1>&2
    if not exist "%SKILL_DIR%models" mkdir "%SKILL_DIR%models"
    powershell -NoProfile -Command ^
      "Invoke-WebRequest -Uri 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite' -OutFile '%SKILL_DIR%models\ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite' -UseBasicParsing" 2>nul
    powershell -NoProfile -Command ^
      "Invoke-WebRequest -Uri 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite' -OutFile '%SKILL_DIR%models\ssd_mobilenet_v2_coco_quant_postprocess.tflite' -UseBasicParsing" 2>nul
)

REM ─── Step 6: Probe for Edge TPU devices ──────────────────────────────────────

echo %LOG_PREFIX% Probing for Edge TPU devices... 1>&2
echo {"event": "progress", "stage": "probe", "message": "Checking for Coral USB Accelerator..."}

set "TPU_FOUND=false"
set "PROBE_JSON="

for /f "delims=" %%I in ('"%VENV_DIR%\Scripts\python.exe" "%SKILL_DIR%scripts\tpu_probe.py" 2^>nul') do (
    set "PROBE_JSON=%%I"
)

echo !PROBE_JSON! | findstr /C:"\"available\": true" >nul 2>&1
if %errorlevel% equ 0 (
    set "TPU_FOUND=true"
    echo %LOG_PREFIX% Edge TPU detected. 1>&2
    echo {"event": "progress", "stage": "probe", "message": "Coral USB Accelerator detected and ready."}
) else (
    echo %LOG_PREFIX% No Edge TPU detected (device may not be plugged in). 1>&2
    echo {"event": "progress", "stage": "probe", "message": "No Edge TPU detected. Plug in the Coral USB Accelerator and restart the skill."}
)

REM ─── Step 7: Done ────────────────────────────────────────────────────────────

if "!TPU_FOUND!"=="true" (
    echo {"event": "complete", "status": "success", "tpu_found": true, "message": "Coral TPU skill installed and Edge TPU is ready."}
    exit /b 0
) else (
    echo {"event": "complete", "status": "partial", "tpu_found": false, "message": "Coral TPU skill installed. Plug in your Coral USB Accelerator to enable hardware acceleration."}
    exit /b 0
)
