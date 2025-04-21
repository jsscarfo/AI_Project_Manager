@echo off
setlocal enabledelayedexpansion

echo BeeAI Framework Installation - Step 2
echo ===================================
echo Configuring API Keys
echo.

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

echo Step 2 complete! You can now run the test_installation.py script.
echo.
echo python test_installation.py
echo.

echo Installation and configuration complete!
pause
exit /b 0

:config_error
echo Configuration failed! Please check if the beeai command is available.
echo.
echo If you just installed BeeAI, you may need to open a new command prompt
echo window for the beeai command to be recognized in your PATH.
echo.
pause
exit /b 1 