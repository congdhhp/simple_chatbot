@echo off
REM Simple CLI Chatbot - Windows Launcher
REM This script activates the virtual environment and starts the chatbot

echo 🤖 Starting Simple CLI Chatbot...

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run setup.py first:
    echo   python setup.py
    pause
    exit /b 1
)

REM Activate virtual environment and start chatbot
call venv\Scripts\activate.bat
python chatbot.py %*

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo ❌ Chatbot exited with an error
    pause
)
