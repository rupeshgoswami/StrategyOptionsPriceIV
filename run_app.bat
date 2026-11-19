@echo off
REM Double-click to open the Strategy Analyzer in your browser.
cd /d "%~dp0"
python -m streamlit run app.py
pause
