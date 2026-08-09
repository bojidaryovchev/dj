@echo off
rem Start the DJ Crate helper. Leave this window open while downloading.
cd /d "%~dp0"
rem -u so progress and status lines appear immediately rather than buffering.
python -u server.py
pause
