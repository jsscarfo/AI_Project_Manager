@echo off
echo BeeAI Framework Clean Installation
echo =================================
echo.

:: Navigate to the V5/python directory
cd /V5/python/

:: Install the package in development mode
echo Installing BeeAI Framework in development mode...
pip install -e .
if %errorlevel% neq 0 (
    echo Installation failed!
    pause
    exit /b 1
)
echo Installation successful!
echo.

:: Configure API keys
echo Configuring API keys...

:: Configure Grok API
echo Setting up Grok API...
beeai env add LLM_MODEL=grok-1
beeai env add LLM_API_BASE=https://api.x.ai/v1
beeai env add LLM_API_KEY=xai-oovnDvFZmpZMVaGgJj3ZB4YaVXuNyr6DdwFtUF0rsPOWQ5mMoGiRXcOp5MX9G7C3U0Nr4pvAZmiAO8rW
echo Grok API configuration complete!
echo.

:: Configure OpenRouter API
echo Setting up OpenRouter API...
beeai env add LLM_MODEL=google/gemini-2.0-pro-exp-02-05:free
beeai env add LLM_API_BASE=https://openrouter.ai/api/v1
beeai env add LLM_API_KEY=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
echo OpenRouter API configuration complete!
echo.

:: List configured providers
echo Listing configured providers...
beeai provider list
echo.

:: Run the installation test
echo Running installation test...
python examples/minimal_test.py
echo.

echo All done! BeeAI Framework is now installed and configured.
pause 