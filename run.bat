@echo off
REM Semantic Colorization Project - Run Script (Windows)

setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

REM Function to print colored output (simplified for Windows)
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

REM Check if virtual environment exists
if not exist "venv" (
    echo %WARNING% Virtual environment not found. Creating one...
    python -m venv venv
    echo %SUCCESS% Virtual environment created.
)

REM Function to activate virtual environment
if not defined VIRTUAL_ENV (
    echo %INFO% Activating virtual environment...
    call venv\Scripts\activate
    echo %SUCCESS% Virtual environment activated.
)

REM Handle commands
if "%1"=="setup" goto :setup
if "%1"=="install" goto :install
if "%1"=="dirs" goto :dirs
if "%1"=="test" goto :test
if "%1"=="train" goto :train
if "%1"=="evaluate" goto :evaluate
if "%1"=="gui" goto :gui
if "%1"=="clean" goto :clean
goto :help

:setup
echo %INFO% Setting up Semantic Colorization Project...
call :dirs
call :install
echo %SUCCESS% Setup completed! You can now run: run.bat train
goto :eof

:install
echo %INFO% Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo %SUCCESS% Dependencies installed.
goto :eof

:dirs
echo %INFO% Creating directory structure...
mkdir data\raw 2>nul
mkdir data\processed\images 2>nul
mkdir data\processed\masks 2>nul
mkdir data\external 2>nul
mkdir outputs\checkpoints 2>nul
mkdir outputs\results 2>nul
mkdir outputs\logs 2>nul
mkdir src 2>nul
mkdir tests 2>nul
mkdir notebooks 2>nul

REM Create .gitkeep files
echo. > data\raw\.gitkeep
echo. > data\processed\images\.gitkeep
echo. > data\processed\masks\.gitkeep
echo. > data\external\.gitkeep
echo. > outputs\checkpoints\.gitkeep
echo. > outputs\results\.gitkeep
echo. > outputs\logs\.gitkeep

echo %SUCCESS% Directory structure created.
goto :eof

:test
echo %INFO% Running tests...
python -m pytest tests/ -v
goto :eof

:train
echo %INFO% Starting model training...
shift
python src_train.py %*
goto :eof

:evaluate
echo %INFO% Starting model evaluation...
shift
python src_evaluate.py %*
goto :eof

:gui
echo %INFO% Starting GUI application...
python src_gui.py
goto :eof

:clean
echo %INFO% Cleaning up...
rmdir /s /q __pycache__ 2>nul
rmdir /s /q .pytest_cache 2>nul
del .coverage 2>nul
for /r . %%i in (*.pyc) do del "%%i" 2>nul
for /r . %%i in (*.pyo) do del "%%i" 2>nul
echo %SUCCESS% Cleanup completed.
goto :eof

:help
echo Semantic Colorization Project - Run Script (Windows)
echo.
echo Usage: run.bat [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   setup         - Set up the project (install deps, create dirs)
echo   install       - Install dependencies
echo   dirs          - Create directory structure
echo   test          - Run tests
echo   train         - Train the model
echo   evaluate      - Evaluate the model
echo   gui           - Launch GUI application
echo   clean         - Clean up temporary files
echo   help          - Show this help message
echo.
echo Examples:
echo   run.bat setup                                    # Complete setup
echo   run.bat train --epochs 50 --batch-size 16       # Train with custom params
echo   run.bat evaluate --checkpoint best_model.pth     # Evaluate model
echo   run.bat gui                                      # Launch GUI
echo.
goto :eof
