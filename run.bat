@echo off
REM ========================================
REM Ship Value Prediction - Streamlit Launcher
REM ========================================

echo.
echo ====================================
echo   Ship Value Prediction - Streamlit
echo ====================================
echo.

cd /d "%~dp0"

REM Launch Streamlit app
echo Launching Streamlit application...
echo Opening http://localhost:8501
echo.

python -m streamlit run app.py

pause
