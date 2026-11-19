@echo off
REM Double-click to build / update the app data from your NIFTY_WEEKLY folder.
REM The first time it asks for the folder path (e.g. D:\NIFTY_WEEKLY) and remembers it.
cd /d "%~dp0"
python build_data.py %*
pause
