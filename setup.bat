@echo off
echo ===================================================
echo   ATS Resume Analyzer - Teammate Setup Script
echo ===================================================

:: Check for Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10 or higher from python.org and try again.
    pause
    exit /b
)

echo [1/3] Creating virtual environment (.venv)...
python -m venv .venv

echo [2/3] Activating virtual environment...
call .venv\Scripts\activate

echo [3/3] Installing required packages...
pip install -r requirements.txt

echo.
echo ===================================================
echo   Setup Complete!
echo ===================================================
echo.
echo Your environment is ready. To test the pipeline, run:
echo.
echo   call .venv\Scripts\activate
echo   python run_submission.py
echo.
echo Or test with custom paths:
echo   python run_submission.py --candidates "data\candidates.jsonl" --jd "data\job_description.docx"
echo.
pause
