@echo off
setlocal enabledelayedexpansion

echo BeeAI Framework Installation - Step 1
echo ===================================
echo.

:: Set the absolute path to V5/python directory
set INSTALL_DIR=C:\Users\jssca\CascadeProjects\Development Project Manager\ai_project_manager_v4\V5\python

:: Check if directory exists
if not exist "%INSTALL_DIR%" (
    echo Error: Installation directory not found: %INSTALL_DIR%
    echo Please update the INSTALL_DIR variable in this script with the correct path.
    pause
    exit /b 1
)

:: Navigate to the directory
echo Changing to directory: %INSTALL_DIR%
cd /d "%INSTALL_DIR%"
if %errorlevel% neq 0 (
    echo Failed to navigate to the installation directory.
    pause
    exit /b 1
)

:: Install the package in development mode
echo Installing BeeAI Framework in development mode...
call pip install -e .
if %errorlevel% neq 0 (
    echo Installation failed! Please check for errors above.
    pause
    exit /b 1
)
echo Installation successful!
echo.

:: Run the installation test
echo Running installation test...
python examples/minimal_test.py
if %errorlevel% neq 0 (
    echo Test failed! Please check the output above.
) else (
    echo Test completed successfully!
)
echo.

echo Step 1 complete! Please run 2_configure_api_keys.bat to configure API keys.
pause 