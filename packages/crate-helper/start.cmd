@echo off
rem Double-click launcher. Real work happens in start.ps1: it installs any
rem missing prerequisites, fetches yt-dlp/ffmpeg, then runs the server.
rem Leave this window open while downloading.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
echo.
echo Helper stopped.
pause
