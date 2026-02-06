@echo off
REM ChatKnime Backend Setup Script
REM Run this script on a new machine to configure the environment

echo ============================================================
echo ChatKnime Backend Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Create venv if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

REM Check for .env
if not exist ".env" (
    echo.
    echo ============================================================
    echo IMPORTANT: Create .env file
    echo ============================================================
    echo.
    echo Copy .env.example to .env and set your values:
    echo.
    echo   copy .env.example .env
    echo.
    echo Required settings:
    echo   GOOGLE_CLOUD_PROJECT=your-project-id
    echo.
    echo Then run:
    echo   gcloud auth application-default login
    echo.
    echo ============================================================
) else (
    echo .env file found.
)

echo.
echo Setup complete!
echo.
echo To use:
echo   .\transpile.bat path\to\workflow.knwf
echo.
pause
