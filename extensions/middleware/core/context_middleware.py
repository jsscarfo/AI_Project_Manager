"""
Contextual Enhancement Middleware for BeeAI Framework

This module provides a unified middleware layer that enhances LLM context
by integrating vector memory, knowledge graph, and ACRS capabilities.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
import logging
from pydantic import BaseModel, Field

# BeeAI imports
from beeai_framework.backend.chat import ChatModel, ChatModelParameters
from beeai_framework.backend.message import Message, SystemMessage, UserMessage
from beeai_framework.emitter import Emitter

logger = logging.getLogger(__name__)

class ContextProviderConfig(BaseModel):
    """Configuration for a context provider."""
    enabled: bool = True
    priority: int = 100
    max_latency_ms: int = 500
    weight: float = 1.0

class ContextProvider:
    """Base class for context providers (vector memory, knowledge graph, etc.)."""
    
    def __init__(self, config: Optional[ContextProviderConfig] = None):
        self.config = config or ContextProviderConfig()
        self.emitter = Emitter.root().child(
            namespace=["middleware", "context", self.__class__.__name__.lower()],
            creator=self,
        )
    
    async def get_context(self, query: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve context relevant to the query.
        
        Args:
            query: The query to retrieve context for
            metadata: Additional metadata for the query
            
        Returns:
            List of context items (dictionaries with content and metadata)
        """
        raise NotImplementedError("Context providers must implement get_context")
    
    async def add_context(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add content to the context store.
        
        Args:
            content: The content to store
            metadata: Metadata for the content
            
        Returns:
            Identifier for the stored content
        """
        raise NotImplementedError("Context providers must implement add_context")

class ContextualEnhancementMiddleware:
    """
    Middleware that enhances LLM prompts with contextual information.
    
    This middleware intercepts requests to LLMs and enhances the prompt
    with relevant context from multiple providers.
    """
    
    def __init__(
        self,
        providers: List[ContextProvider] = None,
        enabled: bool = True,
        max_tokens: int = 2000,
        max_latency_ms: int = 500,
    ):
        self.providers = providers or []
        self.enabled = enabled
        self.max_tokens = max_tokens
        self.max_latency_ms = max_latency_ms
        self.emitter = Emitter.root().child(
            namespace=["middleware", "contextual_enhancement"],
            creator=self,
        )
        
        logger.info(f"Initialized ContextualEnhancementMiddleware with {len(self.providers)} providers")
    
    def add_provider(self, provider: ContextProvider) -> None:
        """Add a context provider to the middleware."""
        self.providers.append(provider)
        logger.info(f"Added context provider: {provider.__class__.__name__}")
    
    def wrap_llm(self, model: ChatModel) -> ChatModel:
        """
        Wrap a ChatModel with context enhancement.
        
        This returns a proxy object that intercepts calls to the LLM's create method
        and enhances messages with context before passing them to the actual LLM.
        
        Args:
            model: The ChatModel to wrap
            
        Returns:
            Enhanced ChatModel
        """
        original_create = model.create
        middleware = self
        
        async def enhanced_create(messages: List[Message], **kwargs):
            if not middleware.enabled:
                return await original_create(messages=messages, **kwargs)
            
            enhanced_messages = await middleware.enhance_messages(messages)
            self.emitter.emit("enhanced", {"original_count": len(messages), "enhanced_count": len(enhanced_messages)})
            
            return await original_create(messages=enhanced_messages, **kwargs)
        
        # Replace the create method with our enhanced version
        model.create = enhanced_create  # type: ignore
        return model
    
    async def enhance_messages(self, messages: List[Message]) -> List[Message]:
        """
        Enhance messages with contextual information.
        
        Args:
            messages: The original messages
            
        Returns:
            Enhanced messages with additional context
        """
        if not self.providers or not messages:
            return messages
        
        # Extract the user query from the last user message
        user_messages = [m for m in messages if isinstance(m, UserMessage)]
        if not user_messages:
            return messages
        
        query = user_messages[-1].text
        metadata = self._extract_context_metadata(messages)
        
        try:
            # Get context from all providers with timeout
            start_time = asyncio.get_event_loop().time()
            context_items = await self._gather_context(query, metadata)
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if not context_items:
                logger.debug("No context retrieved for query")
                return messages
            
            # Create a context message to insert
            formatted_context = self._format_context(context_items)
            context_message = SystemMessage(f"Relevant context for your task:\n\n{formatted_context}")
            
            # Insert the context before the last user message
            result = []
            for i, message in enumerate(messages):
                if i == len(messages) - 1 and isinstance(message, UserMessage):
                    result.append(context_message)
                result.append(message)
            
            logger.info(f"Enhanced query with {len(context_items)} context items in {elapsed_ms:.2f}ms")
            self.emitter.emit("context_added", {
                "items_count": len(context_items),
                "latency_ms": elapsed_ms,
                "token_estimate": len(formatted_context) // 4  # Rough estimate
            })
            
            return result
        except Exception as e:
            logger.error(f"Error enhancing context: {e}", exc_info=True)
            # Graceful degradation - return original messages
            return messages
    
    async def _gather_context(self, query: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather context from all providers with timeout."""
        context_tasks = []
        
        for provider in self.providers:
            if provider.config.enabled:
                task = asyncio.create_task(provider.get_context(query, metadata))
                context_tasks.append(task)
        
        # Create a timeout task
        timeout_ms = min(self.max_latency_ms, max(p.config.max_latency_ms for p in self.providers if p.config.enabled))
        
        # Gather results with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*context_tasks, return_exceptions=True),
                timeout=timeout_ms / 1000.0
            )
            
            # Process results, handling exceptions
            all_context = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Provider {self.providers[i].__class__.__name__} failed: {result}")
                    continue
                all_context.extend(result)
            
            # Sort by relevance and other factors
            return self._prioritize_context(all_context)
        except asyncio.TimeoutError:
            logger.warning(f"Context gathering timed out after {timeout_ms}ms")
            # Return partial results from completed tasks
            all_context = []
            for task in context_tasks:
                if task.done() and not task.exception():
                    all_context.extend(task.result())
            return self._prioritize_context(all_context)
    
    def _prioritize_context(self, context_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize context items based on relevance and recency."""
        # Simple prioritization by score
        sorted_items = sorted(
            context_items, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )
        
        # Apply token budget - rough estimate
        result = []
        token_count = 0
        token_budget = self.max_tokens
        
        for item in sorted_items:
            content = item.get('content', '')
            # Estimate tokens (4 chars per token is rough approximation)
            tokens = len(content) // 4
            if token_count + tokens <= token_budget:
                result.append(item)
                token_count += tokens
            else:
                break
        
        return result
    
    def _format_context(self, context_items: List[Dict[str, Any]]) -> str:
        """Format context items into a string for insertion in the prompt."""
        formatted = []
        
        for i, item in enumerate(context_items):
            content = item.get('content', '')
            source = item.get('metadata', {}).get('source', 'unknown')
            category = item.get('metadata', {}).get('category', '')
            
            # Format as Markdown for better readability
            formatted.append(f"[{i+1}] {category}/{source}:\n{content}\n")
        
        return "\n".join(formatted)
    
    def _extract_context_metadata(self, messages: List[Message]) -> Dict[str, Any]:
        """Extract metadata from conversation context."""
        # This can be enhanced to extract project info, etc.
        return {
            "message_count": len(messages),
            "conversation_length": sum(len(m.text) for m in messages)
        } 