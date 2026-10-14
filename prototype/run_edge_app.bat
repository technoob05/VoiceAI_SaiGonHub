@echo off
REM ======================================================================
REM  run_edge_app.bat
REM  One-click launcher for the VoiceAI SaiGonHub Edge S2ST DESKTOP app.
REM  Opens the 100%-on-device translator in a native window (pywebview).
REM
REM  Usage: double-click this file, or run it from a terminal.
REM  Note : first launch warms the models (~60-90 s) before the window opens.
REM ======================================================================

REM Run from this script's own folder regardless of where it was invoked.
cd /d "%~dp0"

echo [run_edge_app] starting VoiceAI SaiGonHub Edge desktop app...
echo [run_edge_app] warming models (~60-90s) before the window appears...

python "%~dp0desktop_app.py"

if errorlevel 1 (
    echo.
    echo [run_edge_app] ERROR: the app exited with an error ^(code %errorlevel%^).
    echo [run_edge_app] Read the message above. Press any key to close.
    pause >nul
    exit /b %errorlevel%
)

exit /b 0
