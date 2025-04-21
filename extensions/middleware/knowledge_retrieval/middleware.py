#!/usr/bin/env python
"""
Knowledge Retrieval Middleware Integration.

This module implements the middleware for integrating knowledge retrieval
with Sequential Thinking, providing contextual knowledge during reasoning steps.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from pydantic import BaseModel, Field

from beeai_framework.vector.base import VectorMemoryProvider
from beeai_framework.middleware.base import Middleware, MiddlewareRequest, MiddlewareResponse

from .core import KnowledgeRetrievalProcessor, KnowledgeRetrievalSettings, KnowledgeRetrievalResult

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeRetrievalRequest(BaseModel):
    """Model for knowledge retrieval middleware request."""
    prompt: str = Field(..., description="The prompt to process")
    task_type: str = Field("general", description="The type of task being performed")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Filter for knowledge metadata")
    max_results: Optional[int] = Field(None, description="Maximum number of results to return")
    trace_id: Optional[str] = Field(None, description="Trace ID for logging")


class KnowledgeRetrievalResponse(BaseModel):
    """Model for knowledge retrieval middleware response."""
    enhanced_prompt: str = Field(..., description="The prompt enhanced with knowledge context")
    knowledge_entries: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved knowledge entries")
    trace_id: str = Field(..., description="Trace ID for reference")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


class ContextEnhancementProvider:
    """
    Provider for enhancing context in Sequential Thinking middleware.
    
    Implements the interface expected by the Sequential Thinking middleware
    for context enhancement during reasoning steps.
    """
    
    def __init__(
        self,
        knowledge_processor: KnowledgeRetrievalProcessor
    ):
        """
        Initialize the context enhancement provider.
        
        Args:
            knowledge_processor: Processor for knowledge retrieval
        """
        self.knowledge_processor = knowledge_processor
    
    async def enhance_context(
        self,
        prompt: str,
        thought_number: int,
        previous_thoughts: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance context with relevant knowledge for the current thinking step.
        
        This method is called by the Sequential Thinking middleware to enhance
        the context for each thinking step.
        
        Args:
            prompt: The current prompt
            thought_number: Current thought number
            previous_thoughts: Previous thoughts in the reasoning chain
            context: Additional context
            
        Returns:
            Dictionary containing enhanced prompt and metrics
        """
        task_context = context or {}
        start_time = time.time()
        
        # Extract task type for filtering
        task_type = task_context.get("task_type", "general")
        metadata_filter = task_context.get("metadata_filter", {})
        
        # Retrieve knowledge based on prompt and previous thoughts
        retrieval_result = await self.knowledge_processor.retrieve_knowledge(
            query=prompt,
            task_context=task_context,
            previous_thoughts=previous_thoughts,
            step_number=thought_number,
            metadata_filter=metadata_filter
        )
        
        # Check if we have any results
        if not retrieval_result.has_results:
            logger.debug(f"No knowledge entries found for step {thought_number}")
            return {
                "enhanced_prompt": prompt,
                "context_metrics": {
                    "entries_found": 0,
                    "retrieval_time": retrieval_result.metrics.get("retrieval_time", 0),
                    "quality_score": 0.0,
                    "sources": []
                }
            }
        
        # Create enhanced prompt with knowledge context
        enhanced_prompt = self._create_enhanced_prompt(prompt, retrieval_result, thought_number)
        
        # Extract sources for metrics
        sources = list(retrieval_result.metrics.get("sources", {}).keys())
        
        # Calculate a simple quality score based on number of entries and their importance
        avg_importance = sum(
            entry.metadata.get("importance", 0.5) 
            for entry in retrieval_result.entries
        ) / len(retrieval_result.entries)
        
        quality_score = min(1.0, (len(retrieval_result.entries) / 5) * avg_importance)
        
        # Prepare context metrics
        context_metrics = {
            "entries_found": len(retrieval_result.entries),
            "retrieval_time": retrieval_result.metrics.get("retrieval_time", 0),
            "quality_score": quality_score,
            "sources": sources,
            "categories": list(retrieval_result.metrics.get("categories", {}).keys())
        }
        
        return {
            "enhanced_prompt": enhanced_prompt,
            "context_metrics": context_metrics,
            "knowledge_entries": [
                {
                    "content": entry.content,
                    "metadata": entry.metadata
                }
                for entry in retrieval_result.entries
            ]
        }
    
    def _create_enhanced_prompt(
        self,
        original_prompt: str,
        retrieval_result: KnowledgeRetrievalResult,
        thought_number: int
    ) -> str:
        """
        Create an enhanced prompt with knowledge context.
        
        Args:
            original_prompt: Original prompt
            retrieval_result: Result of knowledge retrieval
            thought_number: Current thought number
            
        Returns:
            Enhanced prompt string
        """
        # If we have a pre-formatted context string, use it
        if retrieval_result.context_str:
            # Adapt formatting based on thought number
            if thought_number == 1:
                # For first thought, put context first
                return f"{retrieval_result.context_str}\n\n{original_prompt}"
            else:
                # For later thoughts, put original prompt first
                return f"{original_prompt}\n\nAdditional context:\n{retrieval_result.context_str}"
        
        # Otherwise, format it ourselves
        knowledge_context = []
        for i, entry in enumerate(retrieval_result.entries, 1):
            category = entry.metadata.get("category", "general")
            importance = entry.metadata.get("importance", 0.5)
            
            # Only include highly important entries for clarity
            if importance >= 0.6:
                knowledge_context.append(f"{i}. [{category.upper()}] {entry.content}")
        
        if not knowledge_context:
            return original_prompt
        
        context_str = "\n".join(knowledge_context)
        
        # Adapt formatting based on thought number
        if thought_number == 1:
            return f"Relevant knowledge:\n{context_str}\n\n{original_prompt}"
        else:
            return f"{original_prompt}\n\nRelevant knowledge for this step:\n{context_str}"


class KnowledgeRetrievalMiddleware(Middleware):
    """
    Middleware for knowledge retrieval and sequential thinking integration.
    
    This middleware handles knowledge retrieval requests and provides
    integration with the Sequential Thinking middleware through the
    ContextEnhancementProvider.
    """
    
    def __init__(
        self,
        vector_provider: VectorMemoryProvider,
        settings: Optional[KnowledgeRetrievalSettings] = None,
        sequential_middleware: Optional[Any] = None
    ):
        """
        Initialize the knowledge retrieval middleware.
        
        Args:
            vector_provider: Provider for vector memory operations
            settings: Optional settings for knowledge retrieval
            sequential_middleware: Optional sequential thinking middleware for integration
        """
        self.vector_provider = vector_provider
        self.settings = settings or KnowledgeRetrievalSettings()
        self.sequential_middleware = sequential_middleware
        
        # Initialize the knowledge retrieval processor
        self.knowledge_processor = KnowledgeRetrievalProcessor(
            vector_provider=self.vector_provider,
            settings=self.settings
        )
        
        # Initialize context enhancement provider for sequential thinking integration
        self.context_provider = ContextEnhancementProvider(
            knowledge_processor=self.knowledge_processor
        )
        
        # Initialize the middleware with the sequential thinking middleware if provided
        if sequential_middleware:
            self._integrate_with_sequential_thinking()
    
    def _integrate_with_sequential_thinking(self):
        """
        Integrate this middleware with the Sequential Thinking middleware.
        
        This method configures the Sequential Thinking middleware to use
        this middleware's context enhancement provider.
        """
        try:
            # Set the context refinement processor
            if hasattr(self.sequential_middleware, 'context_refinement_processor'):
                self.sequential_middleware.context_refinement_processor = self.context_provider
            
            # If using a processor directly, update its context provider
            if hasattr(self.sequential_middleware, 'processor') and hasattr(self.sequential_middleware.processor, 'context_provider'):
                self.sequential_middleware.processor.context_provider = self.context_provider
                
            logger.info("Successfully integrated Knowledge Retrieval with Sequential Thinking middleware")
        except Exception as e:
            logger.error(f"Error integrating with Sequential Thinking middleware: {str(e)}")
    
    async def set_sequential_middleware(self, middleware: Any):
        """
        Set the Sequential Thinking middleware for integration.
        
        Args:
            middleware: Sequential Thinking middleware instance
        """
        self.sequential_middleware = middleware
        self._integrate_with_sequential_thinking()
    
    async def process(
        self,
        request: MiddlewareRequest
    ) -> MiddlewareResponse:
        """
        Process a middleware request.
        
        Args:
            request: The middleware request
            
        Returns:
            Middleware response
        """
        start_time = time.time()
        
        # Parse request
        try:
            retrieval_request = KnowledgeRetrievalRequest(
                prompt=request.prompt,
                task_type=request.context.get("task_type", "general") if request.context else "general",
                context=request.context,
                metadata_filter=request.context.get("metadata_filter") if request.context else None,
                max_results=request.context.get("max_results") if request.context else None,
                trace_id=request.trace_id
            )
        except Exception as e:
            logger.error(f"Error parsing request: {str(e)}")
            return MiddlewareResponse(
                prompt=request.prompt,
                success=False,
                error=f"Invalid request: {str(e)}",
                processing_time=time.time() - start_time
            )
        
        # Define the task context
        task_context = retrieval_request.context or {}
        
        # Retrieve knowledge
        try:
            retrieval_result = await self.knowledge_processor.retrieve_knowledge(
                query=retrieval_request.prompt,
                task_context=task_context,
                metadata_filter=retrieval_request.metadata_filter,
                max_results=retrieval_request.max_results or self.settings.max_results
            )
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return MiddlewareResponse(
                prompt=request.prompt,
                success=False,
                error=f"Knowledge retrieval error: {str(e)}",
                processing_time=time.time() - start_time
            )
        
        # If no results, return original prompt
        if not retrieval_result.has_results:
            return MiddlewareResponse(
                prompt=request.prompt,
                success=True,
                context=request.context,
                metadata={
                    "knowledge_retrieval": {
                        "entries_found": 0,
                        "retrieval_time": retrieval_result.metrics.get("retrieval_time", 0)
                    }
                },
                processing_time=time.time() - start_time
            )
        
        # Create enhanced prompt
        enhanced_prompt = f"{request.prompt}\n\n{retrieval_result.context_str}"
        
        # Prepare response
        return MiddlewareResponse(
            prompt=enhanced_prompt,
            success=True,
            context=request.context,
            metadata={
                "knowledge_retrieval": {
                    "entries_found": len(retrieval_result.entries),
                    "retrieval_time": retrieval_result.metrics.get("retrieval_time", 0),
                    "query": retrieval_result.query,
                    "sources": retrieval_result.metrics.get("sources", {}),
                    "categories": retrieval_result.metrics.get("categories", {})
                }
            },
            processing_time=time.time() - start_time
        ) 