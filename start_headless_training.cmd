@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0analysis\timing\start_headless_training.ps1" %*
if errorlevel 1 (
    echo.
    echo Training could not be started. See the error above.
    pause
    exit /b 1
)
