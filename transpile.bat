@echo off
REM ChatKnime Transpiler - Wrapper Script
REM Usage: transpile.bat arquivo.knwf

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Resolve input file to absolute path BEFORE changing directory
set INPUT_FILE=%~f1

REM Check if file was provided
if "%INPUT_FILE%"=="" (
    echo Usage: transpile.bat arquivo.knwf
    echo.
    echo Examples:
    echo   transpile.bat meu_fluxo.knwf
    echo   transpile.bat C:\caminho\para\fluxo.knwf
    exit /b 1
)

REM Check if file exists
if not exist "%INPUT_FILE%" (
    echo Error: File not found: %INPUT_FILE%
    exit /b 1
)

REM Run the Python transpiler from backend directory
pushd "%SCRIPT_DIR%backend"
call venv\Scripts\python.exe transpile.py "%INPUT_FILE%" %2 %3 %4 %5
set RESULT=%ERRORLEVEL%
popd

exit /b %RESULT%
