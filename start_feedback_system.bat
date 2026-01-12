@echo off
REM Activate virtual environment if you have one (uncomment and set path if needed)
REM call venv\Scripts\activate

REM Start Django server
REM Start Django server in a new window and keep it running
start "Django Server" cmd /k "cd training_feedback_system && python manage.py runserver"

REM Wait a few seconds for the server to start
REM Wait up to 15 seconds for the server to start
setlocal enabledelayedexpansion
set COUNT=0
:CHECK_SERVER
timeout /t 2 >nul
set /a COUNT+=1
curl --silent http://127.0.0.1:8000 >nul 2>&1
if errorlevel 1 (
	if !COUNT! lss 8 goto CHECK_SERVER
)

REM Open admin login page in Chrome
start chrome http://127.0.0.1:8000/login/

echo Feedback System started. Admin login page opened in Chrome.
pause
