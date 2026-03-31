@echo off
setlocal enabledelayedexpansion
title Aegis-AI WSL Coral TPU Deployer

echo ===================================================
echo   Aegis-AI Windows WSL Coral TPU Deployment 
echo ===================================================
echo.
echo This script will install the Edge TPU dependencies 
echo natively inside the Windows Subsystem for Linux (WSL).
echo It utilizes usbipd-win to map the Coral USB to the 
echo Linux Kernel, ensuring maximum stability.
echo.

:: 1. Verify wsl exists
where wsl >nul 2>nul
if %errorlevel% neq 0 (
    call :ColorText 0C "ERROR: Windows Subsystem for Linux (WSL) is not installed."
    echo Please install WSL by running 'wsl --install' in an Administrator terminal.
    exit /b 1
)

:: 2. Verify usbipd exists
where usbipd >nul 2>nul
if %errorlevel% neq 0 (
    call :ColorText 0E "WARNING: usbipd is not installed. Please install it:"
    echo Run: winget install usbipd -e
    exit /b 1
)

:: 3. Inform about hardware binding
echo [1/4] Ensuring hardware is bound...
echo Note: Hardware IDs 18d1:9302 and 1a6e:089a must be bound to usbipd.
echo If they are not bound yet, please run 'usbipd bind' as Administrator.

:: 4. Get the WSL path to the current directory
set "DIR_PATH=%~dp0"
set "DIR_PATH=%DIR_PATH:\=/%"
set "DIR_PATH=%DIR_PATH:C:=/mnt/c%"
set "DIR_PATH=%DIR_PATH:~0,-1%"

:: 5. Install Dependencies inside WSL
echo.
echo [2/4] Initializing WSL Python 3.9 environment...
wsl -u root -e bash -c "apt-get update && apt-get install -y software-properties-common curl wget libusb-1.0-0 && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.9 python3.9-venv python3.9-distutils"
if %errorlevel% neq 0 (
    call :ColorText 0C "ERROR: Failed to install Python 3.9 in WSL. Ensure you have internet access and WSL is running Ubuntu."
    exit /b 1
)

:: 6. Create Virtual Env
echo.
echo [3/4] Creating Virtual Environment...
wsl -e bash -c "cd '%DIR_PATH%' && python3.9 -m venv wsl_venv"
if %errorlevel% neq 0 (
    call :ColorText 0C "ERROR: Failed to create venv."
    exit /b 1
)

:: 7. Install Python Packages and EdgeTPU Lib
echo.
echo [4/4] Installing tflite-runtime and Coral TPU drivers...
wsl -e bash -c "cd '%DIR_PATH%' && source wsl_venv/bin/activate && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && python3.9 -m pip install tflite-runtime==2.14.0 numpy pillow"
wsl -u root -e bash -c "cd '%DIR_PATH%' && wget -qO libedgetpu.deb https://packages.cloud.google.com/apt/pool/coral-edgetpu-stable/libedgetpu1-max_16.0_amd64_0ac21f1924dd4b125d5cfc5f6d0e4a5e.deb && dpkg -x libedgetpu.deb ext && cp ext/usr/lib/x86_64-linux-gnu/libedgetpu.so.1.0 libedgetpu.so.1 && rm -rf ext libedgetpu.deb"

echo.
call :ColorText 0A "SUCCESS: Windows WSL Deployment Complete!"
echo.
echo Aegis-AI is ready to trigger the detection node natively on WSL!
echo You can safely close this terminal.
exit /b 0

:ColorText
<nul set /p ".=%DEL%" > "%~2"
findstr /v /a:%1 /R "^$" "%~2" nul
del "%~2" > nul 2>&1
echo.
goto :eof
