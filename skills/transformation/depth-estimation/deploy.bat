@echo off
REM deploy.bat — Windows deployment for Depth Estimation (Privacy) skill
REM Creates venv, installs PyTorch + CUDA dependencies, verifies GPU detection.
REM
REM The Aegis deployment agent calls this on Windows instead of deploy.sh.

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "MODELS_DIR=%USERPROFILE%\.aegis-ai\models\feature-extraction"

echo === Depth Estimation (Privacy) — Windows Setup ===

REM ── Create venv ───────────────────────────────────────────────────
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Is Python installed?
        exit /b 1
    )
)

set "PIP=%VENV_DIR%\Scripts\pip.exe"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

REM Upgrade pip
"%PIP%" install --upgrade pip --quiet

echo.
echo === Windows — PyTorch backend (CUDA/CPU) ===
echo Installing PyTorch dependencies...
"%PIP%" install --quiet -r "%SCRIPT_DIR%requirements.txt"

if errorlevel 1 (
    echo ERROR: pip install failed. Check requirements.txt and network connectivity.
    exit /b 1
)

echo [OK] PyTorch dependencies installed

REM ── Verify installation ───────────────────────────────────────────
"%PYTHON%" -c "import torch, cv2, numpy, PIL; from depth_anything_v2.dpt import DepthAnythingV2; cuda = 'YES' if torch.cuda.is_available() else 'NO'; gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print(f'[OK] Verified: torch={torch.__version__}, CUDA={cuda}, GPU={gpu}')"

if errorlevel 1 (
    echo WARNING: Verification failed. Some packages may not be installed correctly.
    echo Trying minimal verification...
    "%PYTHON%" -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')"
)

echo.
echo === Setup complete ===

endlocal
