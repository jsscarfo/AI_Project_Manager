@echo off
setlocal enabledelayedexpansion

echo BeeAI Framework Clean Installation
echo =================================
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

:: Configure API keys
echo Configuring API keys...

:: Configure Grok API
echo Setting up Grok API...
call beeai env add LLM_MODEL=grok-1
if %errorlevel% neq 0 goto :config_error

call beeai env add LLM_API_BASE=https://api.x.ai/v1
if %errorlevel% neq 0 goto :config_error

call beeai env add LLM_API_KEY=xai-oovnDvFZmpZMVaGgJj3ZB4YaVXuNyr6DdwFtUF0rsPOWQ5mMoGiRXcOp5MX9G7C3U0Nr4pvAZmiAO8rW
if %errorlevel% neq 0 goto :config_error

echo Grok API configuration complete!
echo.

:: Configure OpenRouter API
echo Setting up OpenRouter API...
call beeai env add LLM_MODEL=google/gemini-2.0-pro-exp-02-05:free
if %errorlevel% neq 0 goto :config_error

call beeai env add LLM_API_BASE=https://openrouter.ai/api/v1
if %errorlevel% neq 0 goto :config_error

call beeai env add LLM_API_KEY=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
if %errorlevel% neq 0 goto :config_error

echo OpenRouter API configuration complete!
echo.

:: List configured providers
echo Listing configured providers...
call beeai provider list
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

echo All done! BeeAI Framework is now installed and configured.
pause
exit /b 0

:config_error
echo Configuration failed! Please check the output above.
pause
exit /b 1 