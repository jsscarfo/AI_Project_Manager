#!/usr/bin/env python
"""
Tests for LLM Provider Adapters.

This module contains tests for the various LLM provider adapters
and the LLM provider manager with fallback capabilities.
"""

import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any

# Import components to test
from llm_adapter import (
    LLMRequest, 
    LLMResponse, 
    OpenAIAdapter, 
    AnthropicAdapter,
    LLMProviderFactory,
    LLMProviderManager
)


# Tests for LLMRequest
def test_llm_request_defaults():
    """Test that LLMRequest has the expected default values."""
    request = LLMRequest(prompt="Test prompt")
    
    assert request.prompt == "Test prompt"
    assert request.system_prompt is None
    assert request.max_tokens is None
    assert request.temperature == 0.7
    assert request.stream is False
    assert request.stop_sequences is None
    assert request.metadata is None


# Tests for OpenAIAdapter with mocked API calls
@pytest.mark.asyncio
async def test_openai_adapter_generate():
    """Test OpenAIAdapter's generate method with a mocked API call."""
    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "id": "test-id",
        "object": "chat.completion",
        "created": 1677721715,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test response"
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 6,
            "total_tokens": 11
        }
    })
    
    # Create mock for ClientSession
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.post.return_value.__aenter__.return_value = mock_response
    
    # Patch aiohttp.ClientSession to return our mock
    with patch('aiohttp.ClientSession', return_value=mock_session):
        adapter = OpenAIAdapter(api_key="test-api-key", model="gpt-4")
        result = await adapter.generate(
            prompt="Test prompt",
            system_prompt="Test system prompt"
        )
        
        # Verify the result
        assert result == "This is a test response"
        
        # Verify the API was called correctly
        assert mock_session.post.called
        args, kwargs = mock_session.post.call_args
        assert kwargs["json"]["messages"][0]["role"] == "system"
        assert kwargs["json"]["messages"][0]["content"] == "Test system prompt"
        assert kwargs["json"]["messages"][1]["role"] == "user"
        assert kwargs["json"]["messages"][1]["content"] == "Test prompt"


@pytest.mark.asyncio
async def test_openai_adapter_generate_with_details():
    """Test OpenAIAdapter's generate_with_details method."""
    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "id": "test-id",
        "object": "chat.completion",
        "created": 1677721715,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test response"
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 6,
            "total_tokens": 11
        }
    })
    
    # Create mock for ClientSession
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.post.return_value.__aenter__.return_value = mock_response
    
    # Patch aiohttp.ClientSession to return our mock
    with patch('aiohttp.ClientSession', return_value=mock_session):
        adapter = OpenAIAdapter(api_key="test-api-key", model="gpt-4")
        request = LLMRequest(
            prompt="Test prompt",
            system_prompt="Test system prompt",
            temperature=0.5
        )
        
        result = await adapter.generate_with_details(request)
        
        # Verify the result
        assert result.text == "This is a test response"
        assert result.provider == "openai"
        assert result.model == "gpt-4"
        assert result.finish_reason == "stop"
        assert result.usage is not None
        assert result.metadata is not None
        assert result.metadata["input_tokens"] == 5
        assert result.metadata["output_tokens"] == 6


@pytest.mark.asyncio
async def test_openai_adapter_error_handling():
    """Test OpenAIAdapter error handling."""
    # Create a mock response with an error
    mock_response = AsyncMock()
    mock_response.status = 400
    mock_response.text = AsyncMock(return_value=json.dumps({
        "error": {
            "message": "Invalid API key",
            "type": "invalid_request_error",
            "code": "invalid_api_key"
        }
    }))
    
    # Create mock for ClientSession
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.post.return_value.__aenter__.return_value = mock_response
    
    # Patch aiohttp.ClientSession to return our mock
    with patch('aiohttp.ClientSession', return_value=mock_session):
        adapter = OpenAIAdapter(api_key="invalid-key", model="gpt-4")
        
        # Test that the expected exception is raised
        with pytest.raises(Exception) as excinfo:
            await adapter.generate(prompt="Test prompt")
        
        # Verify the error message
        assert "OpenAI API error: 400" in str(excinfo.value)


# Tests for AnthropicAdapter with mocked API calls
@pytest.mark.asyncio
async def test_anthropic_adapter_generate():
    """Test AnthropicAdapter's generate method with a mocked API call."""
    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "id": "test-id",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "This is a test response from Claude"
            }
        ],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8
        }
    })
    
    # Create mock for ClientSession
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.post.return_value.__aenter__.return_value = mock_response
    
    # Patch aiohttp.ClientSession to return our mock
    with patch('aiohttp.ClientSession', return_value=mock_session):
        adapter = AnthropicAdapter(api_key="test-api-key", model="claude-3-opus-20240229")
        result = await adapter.generate(
            prompt="Test prompt",
            system_prompt="Test system prompt"
        )
        
        # Verify the result
        assert result == "This is a test response from Claude"
        
        # Verify the API was called correctly
        assert mock_session.post.called
        args, kwargs = mock_session.post.call_args
        assert kwargs["json"]["messages"][0]["role"] == "user"
        assert kwargs["json"]["messages"][0]["content"] == "Test prompt"
        assert kwargs["json"]["system"] == "Test system prompt"


# Tests for LLMProviderFactory
def test_llm_provider_factory():
    """Test LLMProviderFactory creates the right adapters."""
    # Test creating OpenAI adapter
    openai_adapter = LLMProviderFactory.create_provider("openai", {
        "api_key": "test-openai-key",
        "model": "gpt-4"
    })
    assert isinstance(openai_adapter, OpenAIAdapter)
    assert openai_adapter.api_key == "test-openai-key"
    assert openai_adapter.model == "gpt-4"
    
    # Test creating Anthropic adapter
    anthropic_adapter = LLMProviderFactory.create_provider("anthropic", {
        "api_key": "test-anthropic-key",
        "model": "claude-3-opus-20240229"
    })
    assert isinstance(anthropic_adapter, AnthropicAdapter)
    assert anthropic_adapter.api_key == "test-anthropic-key"
    assert anthropic_adapter.model == "claude-3-opus-20240229"
    
    # Test unsupported provider
    with pytest.raises(ValueError) as excinfo:
        LLMProviderFactory.create_provider("unsupported", {})
    assert "Unsupported provider type" in str(excinfo.value)


# Tests for LLMProviderManager
@pytest.mark.asyncio
async def test_llm_provider_manager():
    """Test LLMProviderManager with mocked providers."""
    # Create mock adapters
    mock_openai = AsyncMock()
    mock_openai.generate.return_value = "Response from OpenAI"
    
    mock_anthropic = AsyncMock()
    mock_anthropic.generate.return_value = "Response from Anthropic"
    
    # Create provider manager
    providers = {
        "openai": mock_openai,
        "anthropic": mock_anthropic
    }
    manager = LLMProviderManager(providers, default_provider="openai")
    
    # Test using default provider
    result = await manager.generate(prompt="Test prompt")
    assert result == "Response from OpenAI"
    assert mock_openai.generate.called
    
    # Test specifying a provider
    result = await manager.generate(prompt="Test prompt", provider_name="anthropic")
    assert result == "Response from Anthropic"
    assert mock_anthropic.generate.called


@pytest.mark.asyncio
async def test_llm_provider_manager_fallback():
    """Test LLMProviderManager fallback mechanism."""
    # Create mock adapters
    mock_openai = AsyncMock()
    mock_openai.generate.side_effect = Exception("OpenAI API error")
    
    mock_anthropic = AsyncMock()
    mock_anthropic.generate.return_value = "Fallback response from Anthropic"
    
    # Create provider manager
    providers = {
        "openai": mock_openai,
        "anthropic": mock_anthropic
    }
    manager = LLMProviderManager(providers, default_provider="openai")
    
    # Test fallback
    result = await manager.generate(prompt="Test prompt")
    assert result == "Fallback response from Anthropic"
    assert mock_openai.generate.called
    assert mock_anthropic.generate.called
    
    # Test without fallback
    with pytest.raises(Exception) as excinfo:
        await manager.generate(prompt="Test prompt", fallback=False)
    assert "All providers failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_llm_provider_manager_with_details():
    """Test LLMProviderManager's generate_with_details method."""
    # Create mock response
    mock_response = LLMResponse(
        text="Test response",
        provider="openai",
        model="gpt-4"
    )
    
    # Create mock adapters
    mock_openai = AsyncMock()
    mock_openai.generate_with_details.return_value = mock_response
    
    # Create provider manager
    providers = {
        "openai": mock_openai
    }
    manager = LLMProviderManager(providers, default_provider="openai")
    
    # Test generate_with_details
    request = LLMRequest(prompt="Test prompt")
    result = await manager.generate_with_details(request)
    
    assert result.text == "Test response"
    assert result.provider == "openai"
    assert result.model == "gpt-4"
    assert mock_openai.generate_with_details.called


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 