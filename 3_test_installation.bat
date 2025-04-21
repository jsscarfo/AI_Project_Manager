@echo off
setlocal enabledelayedexpansion

echo BeeAI Framework Installation - Step 3
echo ===================================
echo Testing Installation
echo.

:: Check if we need to cd to the python directory
set PYTHON_DIR=C:\Users\jssca\CascadeProjects\Development Project Manager\ai_project_manager_v4\V5\python
if exist "%PYTHON_DIR%" (
    echo Changing to directory: %PYTHON_DIR%
    cd /d "%PYTHON_DIR%"
)

:: Run the installation test
echo Running installation test...
python test_installation.py
if %errorlevel% neq 0 (
    echo Test failed! Please check the output above.
    pause
    exit /b 1
)

echo.
echo Test complete! Your BeeAI Framework installation is working correctly.
echo.
echo If you want to try additional tests, you can run:
echo python examples/minimal_test.py
echo.

pause 