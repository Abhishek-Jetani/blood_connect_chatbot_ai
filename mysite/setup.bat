@echo off
REM Blood Connect Chatbot - One Command Setup (Windows)
REM Run this script to setup everything at once

echo.
echo 🚀 Blood Connect Chatbot - Setup Script
echo ========================================
echo.

REM Check if in correct directory
if not exist "manage.py" (
    echo ❌ Error: manage.py not found!
    echo Make sure you run this from the mysite directory
    pause
    exit /b 1
)

REM Step 1: Install dependencies
echo 📦 Step 1: Installing dependencies...
pip install -r requirements.txt >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Dependencies installed
) else (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Step 2: Create migrations
echo 📝 Step 2: Creating migrations...
python manage.py makemigrations chatbot >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Migrations created
) else (
    echo ❌ Failed to create migrations
    pause
    exit /b 1
)

REM Step 3: Apply migrations
echo 🔄 Step 3: Applying migrations...
python manage.py migrate >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Database tables created
) else (
    echo ❌ Failed to apply migrations
    pause
    exit /b 1
)

REM Step 4: Verify setup
echo ✔️  Step 4: Verifying setup...
python manage.py check >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Setup verification passed
) else (
    echo ❌ Setup verification failed
    pause
    exit /b 1
)

echo.
echo ✨ Setup Complete! ✨
echo.
echo Next steps:
echo   1. Run: python manage.py runserver
echo   2. Open: http://localhost:8000/chatbot/
echo   3. Start chatting!
echo.
echo Need help? Read START_HERE.md
echo.
pause
