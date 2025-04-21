#!/usr/bin/env python
"""
LLM Provider Adapters for Sequential Thinking.

This module implements adapter interfaces and concrete implementations
for different LLM providers to be used with sequential thinking.
"""

import logging
import json
import time
import abc
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic
import asyncio
import aiohttp
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMRequest(BaseModel):
    """Base model for LLM request parameters."""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    """Base model for LLM response data."""
    text: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    provider: str
    created_at: float = Field(default_factory=time.time)
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMAdapter(abc.ABC):
    """Abstract base class for LLM provider adapters."""
    
    @abc.abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt to guide the LLM
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abc.abstractmethod
    async def generate_with_details(self, 
                                  request: LLMRequest) -> LLMResponse:
        """
        Generate text with detailed response information.
        
        Args:
            request: The LLM request parameters
            
        Returns:
            Detailed LLM response
        """
        pass
    
    @abc.abstractmethod
    async def generate_stream(self, 
                            request: LLMRequest) -> 'AsyncGenerator[str, None]':
        """
        Generate text as a stream of tokens.
        
        Args:
            request: The LLM request parameters
            
        Yields:
            Token strings as they are generated
        """
        pass


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        self.api_key = api_key
        self.model = model
        self.api_base = "https://api.openai.com/v1"
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text from OpenAI.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        
        response = await self.generate_with_details(request)
        return response.text
    
    async def generate_with_details(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text with detailed response.
        
        Args:
            request: LLM request parameters
            
        Returns:
            Detailed LLM response
        """
        try:
            async with aiohttp.ClientSession() as session:
                messages = []
                
                # Add system message if provided
                if request.system_prompt:
                    messages.append({"role": "system", "content": request.system_prompt})
                
                # Add user message
                messages.append({"role": "user", "content": request.prompt})
                
                # Prepare request payload
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": request.temperature or 0.7,
                }
                
                if request.max_tokens:
                    payload["max_tokens"] = request.max_tokens
                
                if request.stop_sequences:
                    payload["stop"] = request.stop_sequences
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    generation_time = time.time() - start_time
                    
                    completion = data["choices"][0]["message"]["content"]
                    
                    return LLMResponse(
                        text=completion,
                        finish_reason=data["choices"][0].get("finish_reason"),
                        usage=data.get("usage"),
                        model=data.get("model", self.model),
                        provider="openai",
                        request_id=data.get("id"),
                        metadata={
                            "generation_time": generation_time,
                            "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                            "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
                        }
                    )
        
        except Exception as e:
            logger.error(f"Error generating from OpenAI: {str(e)}")
            raise
    
    async def generate_stream(self, request: LLMRequest):
        """
        Generate text as a stream of tokens.
        
        Args:
            request: LLM request parameters
            
        Yields:
            Token strings as they are generated
        """
        try:
            async with aiohttp.ClientSession() as session:
                messages = []
                
                # Add system message if provided
                if request.system_prompt:
                    messages.append({"role": "system", "content": request.system_prompt})
                
                # Add user message
                messages.append({"role": "user", "content": request.prompt})
                
                # Prepare request payload
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": request.temperature or 0.7,
                    "stream": True
                }
                
                if request.max_tokens:
                    payload["max_tokens"] = request.max_tokens
                
                if request.stop_sequences:
                    payload["stop"] = request.stop_sequences
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                    
                    # Process SSE stream
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data: "):
                            json_str = line[6:]  # Remove "data: " prefix
                            try:
                                data = json.loads(json_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse SSE line: {line}")
        
        except Exception as e:
            logger.error(f"Error in streaming from OpenAI: {str(e)}")
            raise


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API."""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        """
        Initialize Anthropic adapter.
        
        Args:
            api_key: Anthropic API key
            model: Model to use
        """
        self.api_key = api_key
        self.model = model
        self.api_base = "https://api.anthropic.com/v1"
        self.anthropic_version = "2023-06-01"
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text from Anthropic Claude.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        
        response = await self.generate_with_details(request)
        return response.text
    
    async def generate_with_details(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text with detailed response from Anthropic.
        
        Args:
            request: LLM request parameters
            
        Returns:
            Detailed LLM response
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare request payload
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": request.prompt}],
                    "temperature": request.temperature or 0.7,
                }
                
                if request.system_prompt:
                    payload["system"] = request.system_prompt
                
                if request.max_tokens:
                    payload["max_tokens"] = request.max_tokens
                
                headers = {
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key,
                    "anthropic-version": self.anthropic_version
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.api_base}/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Anthropic API error: {response.status} - {error_text}")
                        raise Exception(f"Anthropic API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    generation_time = time.time() - start_time
                    
                    # Extract completion from response
                    completion = data["content"][0]["text"]
                    
                    return LLMResponse(
                        text=completion,
                        finish_reason=data.get("stop_reason"),
                        usage={"input_tokens": data.get("usage", {}).get("input_tokens", 0),
                               "output_tokens": data.get("usage", {}).get("output_tokens", 0)},
                        model=data.get("model", self.model),
                        provider="anthropic",
                        request_id=data.get("id"),
                        metadata={
                            "generation_time": generation_time,
                            "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                            "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                        }
                    )
        
        except Exception as e:
            logger.error(f"Error generating from Anthropic: {str(e)}")
            raise
    
    async def generate_stream(self, request: LLMRequest):
        """
        Generate text as a stream from Anthropic.
        
        Args:
            request: LLM request parameters
            
        Yields:
            Token strings as they are generated
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare request payload
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": request.prompt}],
                    "temperature": request.temperature or 0.7,
                    "stream": True
                }
                
                if request.system_prompt:
                    payload["system"] = request.system_prompt
                
                if request.max_tokens:
                    payload["max_tokens"] = request.max_tokens
                
                headers = {
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key,
                    "anthropic-version": self.anthropic_version
                }
                
                async with session.post(
                    f"{self.api_base}/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Anthropic API error: {response.status} - {error_text}")
                        raise Exception(f"Anthropic API error: {response.status} - {error_text}")
                    
                    # Process the stream
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            json_str = line[6:]  # Remove "data: " prefix
                            try:
                                data = json.loads(json_str)
                                if data.get("type") == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if "text" in delta:
                                        yield delta["text"]
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse SSE line: {line}")
        
        except Exception as e:
            logger.error(f"Error in streaming from Anthropic: {str(e)}")
            raise


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> LLMAdapter:
        """
        Create an LLM provider adapter.
        
        Args:
            provider_type: Type of provider ("openai", "anthropic", etc.)
            config: Provider configuration
            
        Returns:
            LLM adapter instance
        """
        if provider_type.lower() == "openai":
            return OpenAIAdapter(
                api_key=config.get("api_key"),
                model=config.get("model", "gpt-4")
            )
        elif provider_type.lower() == "anthropic":
            return AnthropicAdapter(
                api_key=config.get("api_key"),
                model=config.get("model", "claude-3-opus-20240229")
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")


class LLMProviderManager:
    """
    Manager for LLM providers with fallback capabilities.
    
    This class manages multiple LLM providers and implements fallback
    strategies when the primary provider fails.
    """
    
    def __init__(self, providers: Dict[str, LLMAdapter], default_provider: str):
        """
        Initialize LLM provider manager.
        
        Args:
            providers: Dictionary of named providers
            default_provider: Name of the default provider
        """
        self.providers = providers
        self.default_provider = default_provider
        
        if default_provider not in providers:
            raise ValueError(f"Default provider '{default_provider}' not found in providers")
    
    async def generate(self, 
                     prompt: str, 
                     system_prompt: Optional[str] = None,
                     provider_name: Optional[str] = None,
                     fallback: bool = True,
                     **kwargs) -> str:
        """
        Generate text using the specified or default provider with fallback.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            provider_name: Name of the provider to use (or default if None)
            fallback: Whether to try other providers if the first fails
            **kwargs: Additional parameters for the provider
            
        Returns:
            Generated text
        """
        # Start with the specified provider or default
        primary_provider = provider_name or self.default_provider
        provider_order = [primary_provider]
        
        # If fallback is enabled, add other providers to the list
        if fallback:
            provider_order.extend([p for p in self.providers.keys() if p != primary_provider])
        
        # Try providers in order
        last_error = None
        for provider_name in provider_order:
            try:
                provider = self.providers.get(provider_name)
                if not provider:
                    logger.warning(f"Provider '{provider_name}' not found, skipping")
                    continue
                
                return await provider.generate(prompt, system_prompt, **kwargs)
            
            except Exception as e:
                logger.error(f"Error with provider '{provider_name}': {str(e)}")
                last_error = e
        
        # If we're here, all providers failed
        raise Exception(f"All providers failed. Last error: {str(last_error)}")
    
    async def generate_with_details(self, 
                                  request: LLMRequest,
                                  provider_name: Optional[str] = None,
                                  fallback: bool = True) -> LLMResponse:
        """
        Generate text with detailed response, with fallback support.
        
        Args:
            request: LLM request parameters
            provider_name: Name of the provider to use (or default if None)
            fallback: Whether to try other providers if the first fails
            
        Returns:
            Detailed LLM response
        """
        # Start with the specified provider or default
        primary_provider = provider_name or self.default_provider
        provider_order = [primary_provider]
        
        # If fallback is enabled, add other providers to the list
        if fallback:
            provider_order.extend([p for p in self.providers.keys() if p != primary_provider])
        
        # Try providers in order
        last_error = None
        for provider_name in provider_order:
            try:
                provider = self.providers.get(provider_name)
                if not provider:
                    logger.warning(f"Provider '{provider_name}' not found, skipping")
                    continue
                
                # Get response
                response = await provider.generate_with_details(request)
                
                # Add fallback information if this wasn't the primary provider
                if provider_name != primary_provider:
                    if not response.metadata:
                        response.metadata = {}
                    response.metadata["fallback_from"] = primary_provider
                
                return response
            
            except Exception as e:
                logger.error(f"Error with provider '{provider_name}': {str(e)}")
                last_error = e
        
        # If we're here, all providers failed
        raise Exception(f"All providers failed. Last error: {str(last_error)}") 