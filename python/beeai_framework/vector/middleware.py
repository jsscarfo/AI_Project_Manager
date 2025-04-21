#!/usr/bin/env python
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contextual Enhancement Middleware Module

This module provides middleware components that enhance LLM interactions
with contextual information retrieved from the vector memory system.
"""

import logging
import asyncio
import re
import json
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import time
import uuid

from beeai_framework.middleware.base import Middleware, MiddlewareConfig, MiddlewareContext
from beeai_framework.errors import FrameworkError
from beeai_framework.vector.base import ContextResult, VectorMemoryProvider, ContextMetadata
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage, SystemMessage

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from beeai_framework.vector.knowledge_capture import KnowledgeCaptureProcessor
else:
    # Create a dummy type for runtime
    class KnowledgeCaptureProcessor:
        """Stub for KnowledgeCaptureProcessor to avoid circular imports."""
        pass

from beeai_framework.vector.sequential_thinking_integration import (
    SequentialKnowledgeIntegration,
    IntegrationConfig,
    KnowledgeRetrievalConfig
)

logger = logging.getLogger(__name__)


class ContextualEnhancementConfig(MiddlewareConfig):
    """Configuration for the contextual enhancement middleware."""
    
    # Context selection settings
    max_context_items: int = Field(default=5, description="Maximum number of context items to include")
    min_similarity_threshold: float = Field(default=0.7, description="Minimum similarity score threshold for including context")
    
    # Task type mapping for hierarchical selection
    task_type_mapping: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "high_level_planning": ["domain", "techstack"],
            "code_implementation": ["techstack", "project"],
            "debugging": ["project"],
            "research": ["domain", "techstack"],
            "explanation": ["domain", "techstack", "project"],
        },
        description="Mapping of task types to appropriate hierarchical levels"
    )
    
    # Context formatting
    context_preamble: str = Field(
        default="### Relevant Context:\n\n",
        description="Text to prepend to the context section"
    )
    context_item_prefix: str = Field(
        default="--- Context Item ({level}, {source}) ---\n",
        description="Prefix template for each context item"
    )
    context_item_suffix: str = Field(
        default="\n\n",
        description="Suffix for each context item"
    )
    
    # Smart filtering
    enable_smart_filtering: bool = Field(default=True, description="Enable smart filtering of context based on content")
    content_based_filter_threshold: float = Field(default=0.8, description="Threshold for content-based filtering")
    
    # Sequential thinking integration
    detect_reasoning_step: bool = Field(default=True, description="Detect reasoning step from the request")
    reasoning_step_patterns: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "problem_definition": [
                r"define.*problem", r"clarify.*requirements", r"understand.*task"
            ],
            "task_breakdown": [
                r"break.*down", r"subtasks", r"step.*by.*step"
            ],
            "solution_design": [
                r"design.*solution", r"architectural", r"approach"
            ],
            "implementation": [
                r"implement", r"code", r"develop"
            ],
            "testing": [
                r"test", r"verify", r"validate"
            ],
            "refinement": [
                r"refine", r"improve", r"optimize"
            ],
        },
        description="Regex patterns to match reasoning steps in requests"
    )
    
    # Advanced options
    enable_context_scoring: bool = Field(default=True, description="Enable scoring and ranking of context items")
    context_token_budget: Optional[int] = Field(default=None, description="Maximum token budget for context")
    extract_keywords: bool = Field(default=True, description="Extract keywords from the request for better context retrieval")
    default_levels: List[str] = Field(
        default=["domain", "techstack", "project"],
        description="Default levels to include when task type is unknown"
    )
    include_as_system_message: bool = Field(
        default=True,
        description="Whether to include context as a system message"
    )
    use_sequential_thinking: bool = Field(
        default=False,
        description="Whether to use sequential thinking for context enhancement"
    )
    metadata_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default metadata filters to apply when retrieving context"
    )


class ContextualEnhancementMiddleware(Middleware):
    """
    Middleware that enhances LLM interactions with contextual information.
    
    This middleware:
    1. Analyzes incoming requests to determine the task type and reasoning step
    2. Retrieves relevant context from the vector memory based on the analysis
    3. Filters and ranks the context based on relevance and importance
    4. Enhances the request with the selected context
    5. Integrates with sequential thinking by providing appropriate context for each step
    
    The middleware implements the Selective Contextual Retrieval system that
    provides AI agents with precisely the information they need for their specific tasks.
    """
    
    def __init__(
        self, 
        vector_provider: VectorMemoryProvider,
        config: Optional[ContextualEnhancementConfig] = None,
        keyword_extractor: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initialize the contextual enhancement middleware.
        
        Args:
            vector_provider: The vector memory provider for context retrieval
            config: Configuration for the middleware
            keyword_extractor: Optional function to extract keywords from text
        """
        super().__init__(config or ContextualEnhancementConfig())
        self.config = config or ContextualEnhancementConfig()
        self.vector_provider = vector_provider
        self.keyword_extractor = keyword_extractor
    
    async def process(self, context: MiddlewareContext) -> MiddlewareContext:
        """
        Process the request context through this middleware.
        
        This enhances the context with relevant information from the vector memory.
        
        Args:
            context: The request context to process
            
        Returns:
            The enhanced context
        """
        logger.debug("Processing request through contextual enhancement middleware")
        
        # Extract request details
        request = context.request
        
        # Skip processing if this is not an LLM request
        if not self._is_llm_request(request):
            logger.debug("Not an LLM request, skipping contextual enhancement")
            return context
        
        try:
            # Get the query text from the request
            query_text = self._extract_query_text(request)
            if not query_text:
                logger.debug("No query text found in request, skipping contextual enhancement")
                return context
            
            # Analyze the request to determine task type and reasoning step
            task_type, reasoning_step = await self._analyze_request(query_text)
            
            # Determine appropriate knowledge levels based on task type
            levels = self._get_knowledge_levels(task_type)
            
            # Extract keywords if enabled
            keywords = []
            if self.config.extract_keywords and self.keyword_extractor:
                keywords = self.keyword_extractor(query_text)
                
            # Enhance query with keywords if available
            enhanced_query = query_text
            if keywords:
                enhanced_query = f"{query_text} {' '.join(keywords)}"
            
            # Retrieve context based on the query and levels
            context_results = await self.vector_provider.get_context(
                query=enhanced_query,
                metadata_filter={"level": {"$in": levels}},
                limit=self.config.max_context_items * 2,  # Get more than needed for filtering
            )
            
            # Apply smart filtering if enabled
            if self.config.enable_smart_filtering:
                context_results = self._apply_smart_filtering(context_results, query_text, reasoning_step)
            
            # Limit to max items after filtering
            context_results = context_results[:self.config.max_context_items]
            
            # Format context for inclusion in the request
            formatted_context = self._format_context(context_results)
            
            # Enhance the request with the context
            enhanced_request = self._enhance_request_with_context(request, formatted_context)
            
            # Update the context with the enhanced request
            context.request = enhanced_request
            
            # Add metadata about the enhancement
            context.enhance_metadata({
                "contextual_enhancement": {
                    "task_type": task_type,
                    "reasoning_step": reasoning_step,
                    "levels": levels,
                    "context_count": len(context_results),
                }
            })
            
            logger.debug(f"Enhanced request with {len(context_results)} context items")
            
            return context
            
        except Exception as e:
            logger.error(f"Error in contextual enhancement: {str(e)}")
            # Don't fail the request if enhancement fails, just continue without enhancement
            return context
    
    def _is_llm_request(self, request: Any) -> bool:
        """Determine if this is an LLM request that should be enhanced with context."""
        # Check for ChatModel.create or similar LLM request patterns
        if hasattr(request, "messages") and isinstance(request.messages, list):
            return True
        
        # Check for other request types that might benefit from context
        if hasattr(request, "prompt") and isinstance(request.prompt, str):
            return True
            
        return False
    
    def _extract_query_text(self, request: Any) -> str:
        """Extract the query text from the request for context retrieval."""
        # Handle different request formats
        if hasattr(request, "messages") and isinstance(request.messages, list):
            # Extract from the last user message
            for message in reversed(request.messages):
                if hasattr(message, "role") and message.role == "user":
                    return message.content
                if hasattr(message, "content") and isinstance(message.content, str):
                    return message.content
        
        if hasattr(request, "prompt") and isinstance(request.prompt, str):
            return request.prompt
            
        # Default fallback
        return str(request)
    
    async def _analyze_request(self, query_text: str) -> Tuple[str, Optional[str]]:
        """
        Analyze the request to determine the task type and reasoning step.
        
        Args:
            query_text: The query text from the request
            
        Returns:
            Tuple of (task_type, reasoning_step)
        """
        # Default task type
        task_type = "general"
        
        # Detect task type based on content
        lower_query = query_text.lower()
        
        if re.search(r"plan|architect|design|overview|strategy", lower_query):
            task_type = "high_level_planning"
        elif re.search(r"code|implement|develop|programming|write.*function", lower_query):
            task_type = "code_implementation"
        elif re.search(r"debug|error|fix|issue|problem|not.*working", lower_query):
            task_type = "debugging"
        elif re.search(r"research|find|learn|information|knowledge", lower_query):
            task_type = "research"
        elif re.search(r"explain|describe|clarify|understand", lower_query):
            task_type = "explanation"
            
        # Detect reasoning step if enabled
        reasoning_step = None
        if self.config.detect_reasoning_step:
            for step, patterns in self.config.reasoning_step_patterns.items():
                if any(re.search(pattern, lower_query, re.IGNORECASE) for pattern in patterns):
                    reasoning_step = step
                    break
        
        return task_type, reasoning_step
    
    def _get_knowledge_levels(self, task_type: str) -> List[str]:
        """
        Determine the appropriate knowledge levels based on task type.
        
        Args:
            task_type: The detected task type
            
        Returns:
            List of knowledge levels to include
        """
        # Get levels from the mapping, or use defaults
        return self.config.task_type_mapping.get(task_type, self.config.default_levels)
    
    def _apply_smart_filtering(
        self, 
        context_results: List[ContextResult], 
        query_text: str,
        reasoning_step: Optional[str] = None
    ) -> List[ContextResult]:
        """
        Apply smart filtering to the context results.
        
        This removes redundant or less relevant context based on content
        similarity and the current reasoning step.
        
        Args:
            context_results: Original context results
            query_text: The query text
            reasoning_step: Optional reasoning step
            
        Returns:
            Filtered context results
        """
        if not context_results:
            return []
            
        # Filter by similarity threshold
        filtered_results = [
            result for result in context_results
            if result.score >= self.config.min_similarity_threshold
        ]
        
        # If we have a reasoning step, prioritize content relevant to that step
        if reasoning_step:
            # Adjust scores based on relevance to the reasoning step
            for result in filtered_results:
                if reasoning_step == "problem_definition" and "requirement" in result.content.lower():
                    result.score += 0.1
                elif reasoning_step == "task_breakdown" and "step" in result.content.lower():
                    result.score += 0.1
                elif reasoning_step == "solution_design" and "architecture" in result.content.lower():
                    result.score += 0.1
                elif reasoning_step == "implementation" and "code" in result.content.lower():
                    result.score += 0.1
                elif reasoning_step == "testing" and "test" in result.content.lower():
                    result.score += 0.1
                
            # Resort by updated scores
            filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        # Remove redundant content
        unique_results = []
        content_hashes = set()
        
        for result in filtered_results:
            # Simple content fingerprinting for similarity detection
            content_words = set(result.content.lower().split()[:50])
            content_hash = hash(frozenset(content_words))
            
            # Check if we already have very similar content
            if content_hash not in content_hashes:
                unique_results.append(result)
                content_hashes.add(content_hash)
        
        return unique_results
    
    def _format_context(self, context_results: List[ContextResult]) -> str:
        """
        Format the context results for inclusion in the request.
        
        Args:
            context_results: The context results to format
            
        Returns:
            Formatted context text
        """
        if not context_results:
            return ""
            
        formatted_parts = [self.config.context_preamble]
        
        for result in context_results:
            # Format the item prefix with metadata
            item_prefix = self.config.context_item_prefix.format(
                level=result.metadata.level,
                source=result.metadata.source,
                category=result.metadata.category or "general",
                score=f"{result.score:.2f}",
            )
            
            # Add this context item
            formatted_parts.append(f"{item_prefix}{result.content}{self.config.context_item_suffix}")
            
        return "".join(formatted_parts)
    
    def _enhance_request_with_context(self, request: Any, formatted_context: str) -> Any:
        """
        Enhance the request with the formatted context.
        
        This modifies the request to include the context in the appropriate location.
        
        Args:
            request: The original request
            formatted_context: The formatted context text
            
        Returns:
            Enhanced request
        """
        if not formatted_context:
            return request
            
        # Create a copy of the request to modify
        enhanced_request = request
        
        # Handle different request types differently
        if hasattr(request, "messages") and isinstance(request.messages, list):
            # For chat completion requests, add system message with context
            # Make a deep copy of the messages
            messages_copy = request.messages.copy()
            
            # Look for an existing system message to update
            system_message_found = False
            for i, message in enumerate(messages_copy):
                if hasattr(message, "role") and message.role == "system":
                    # Update existing system message
                    if hasattr(message, "content"):
                        message.content = f"{message.content}\n\n{formatted_context}"
                        system_message_found = True
                        break
            
            if not system_message_found:
                # Add a new system message at the beginning
                from beeai_framework.backend.message import SystemMessage
                messages_copy.insert(0, SystemMessage(content=formatted_context))
            
            # Update the request with the modified messages
            enhanced_request.messages = messages_copy
            
        elif hasattr(request, "prompt") and isinstance(request.prompt, str):
            # For completion requests, prepend the context to the prompt
            enhanced_request.prompt = f"{formatted_context}\n\n{request.prompt}"
            
        return enhanced_request
    
    async def get_context_for_step(
        self,
        task: str,
        step: str,
        current_context: Optional[Dict[str, Any]] = None,
        previous_steps: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get context specifically for a sequential thinking step.
        
        This method is designed to integrate with sequential thinking processes,
        providing the appropriate context for each step in the reasoning process.
        
        Args:
            task: The overall task
            step: The current reasoning step
            current_context: Optional current context
            previous_steps: Optional results from previous steps
            
        Returns:
            Context dictionary for this step
        """
        # Determine knowledge levels based on step
        levels = self.config.task_type_mapping.get("general", self.config.default_levels)
        
        # Adjust query based on step and previous steps
        query = task
        if step == "problem_definition":
            query = f"define problem for: {task}"
        elif step == "task_breakdown":
            query = f"break down task: {task}"
        elif step == "solution_design":
            query = f"design solution for: {task}"
        elif step == "implementation":
            query = f"implement: {task}"
        elif step == "testing":
            query = f"test: {task}"
        
        # Include previous steps in the query if available
        if previous_steps:
            # Extract text from previous steps (assuming they have a content or text attribute)
            previous_text = []
            for prev in previous_steps:
                if hasattr(prev, "content"):
                    previous_text.append(prev.content)
                elif hasattr(prev, "text"):
                    previous_text.append(prev.text)
                else:
                    previous_text.append(str(prev))
            
            # Add a condensed version of previous steps to the query
            combined_previous = " ".join(previous_text)
            if len(combined_previous) > 1000:
                combined_previous = combined_previous[:1000] + "..."
            query = f"{query}\nPrevious steps: {combined_previous}"
        
        # Retrieve context based on the query and levels
        context_results = await self.vector_provider.get_context(
            query=query,
            metadata_filter={"level": {"$in": levels}},
            limit=self.config.max_context_items,
        )
        
        # Format context
        formatted_context = self._format_context(context_results)
        
        # Combine with current context if provided
        result_context = current_context.copy() if current_context else {}
        result_context["retrieved_context"] = formatted_context
        result_context["step"] = step
        
        return result_context

    def _format_context_as_system_message(self, context_items: List[Dict[str, Any]]) -> SystemMessage:
        """Format context items as a system message.
        
        Args:
            context_items: List of context items
            
        Returns:
            System message containing formatted context
        """
        context_text = "Relevant context from knowledge base:\n\n"
        
        for idx, item in enumerate(context_items, 1):
            content = item["content"]
            metadata = item.get("metadata", {})
            score = item.get("score", 0)
            
            source = metadata.get("source", "unknown")
            category = metadata.get("category", "general")
            
            # Format the context item
            context_text += f"[{idx}] {category.upper()} ({source}):\n{content}\n\n"
        
        context_text += (
            "Use the above context to help answer the user's question when relevant. "
            "Do not mention that you're using this context unless asked directly."
        )
        
        return SystemMessage(content=context_text)

class KnowledgeRetrievalMiddleware(Middleware):
    """
    Middleware for integrating knowledge retrieval with sequential thinking.
    
    This middleware enhances requests with relevant knowledge from the vector store
    based on the reasoning step in the sequential thinking process, while optimizing
    context window usage and implementing feedback-based retrieval improvements.
    """
    
    def __init__(
        self, 
        vector_provider: VectorMemoryProvider,
        knowledge_capture_processor: Optional[KnowledgeCaptureProcessor] = None,
        config: Optional[MiddlewareConfig] = None,
        integration_config: Optional[IntegrationConfig] = None,
        retrieval_config: Optional[KnowledgeRetrievalConfig] = None,
        token_estimator: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize the knowledge retrieval middleware.
        
        Args:
            vector_provider: Provider for vector storage
            knowledge_capture_processor: Optional processor for knowledge capture
            config: Middleware configuration
            integration_config: Configuration for knowledge-sequential thinking integration
            retrieval_config: Configuration for knowledge retrieval
            token_estimator: Optional function to estimate tokens in a string
        """
        super().__init__(config)
        
        # Initialize the integration component
        self.integration = SequentialKnowledgeIntegration(
            vector_provider=vector_provider,
            knowledge_capture_processor=knowledge_capture_processor,
            config=integration_config,
            retrieval_config=retrieval_config
        )
        
        # Token estimator for context window optimization
        self.token_estimator = token_estimator or self._default_token_estimator
        
        # Track step history to optimize context across steps
        self.step_history = {}
        
        logger.info("Initialized KnowledgeRetrievalMiddleware")
    
    async def process(self, context: MiddlewareContext) -> MiddlewareContext:
        """
        Process a request through the knowledge retrieval middleware.
        
        This middleware enhances sequential thinking requests with relevant
        context from the vector store based on the reasoning step.
        
        Args:
            context: The middleware context
            
        Returns:
            Updated middleware context
        """
        request = context.request
        
        # Check if this is a sequential thinking request
        if not self._is_sequential_thinking_request(request):
            return context
            
        # Extract step information, system prompt, and user prompt
        step_info, system_prompt, user_prompt = self._extract_request_data(request)
        
        if not step_info:
            return context
        
        # Add trace_id to step_info for consistent tracking
        trace_id = self._get_trace_id(request, context)
        step_info["trace_id"] = trace_id
        
        # Get previous steps if available
        previous_steps = self._get_previous_steps(trace_id, step_info.get("thought_number", 1))
            
        # Enhance the step with knowledge retrieval
        try:
            enhanced_data = await self.integration.enhance_step(
                step_info=step_info,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Optimize context window usage based on token estimates
            optimized_data = self._optimize_context_window(
                enhanced_data=enhanced_data,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                previous_steps=previous_steps
            )
            
            # Update the request with enhanced prompts
            self._update_request_with_enhancements(request, optimized_data)
            
            # Store relevant information for post-processing
            metadata = {
                "knowledge_retrieval": {
                    "applied": optimized_data.get("context_applied", False),
                    "context_items_count": len(optimized_data.get("context_items", [])),
                    "step_type": optimized_data.get("step_type"),
                    "step_number": step_info.get("thought_number", 1),
                    "trace_id": trace_id,
                    "optimized": optimized_data.get("optimized", False),
                    "token_usage": optimized_data.get("token_usage", {}),
                    "context_items": optimized_data.get("context_items", [])
                }
            }
            
            # Add metadata to context for post-processing
            context.enhance_metadata(metadata)
            
            logger.debug(
                f"Enhanced sequential thinking step {step_info.get('thought_number', 1)} "
                f"with {len(optimized_data.get('context_items', []))} context items "
                f"({optimized_data.get('token_usage', {}).get('context_tokens', 0)} tokens)"
            )
            
        except Exception as e:
            logger.error(f"Error enhancing step with knowledge retrieval: {str(e)}")
            # Continue processing without enhancement
        
        return context
    
    async def post_process(self, context: MiddlewareContext) -> MiddlewareContext:
        """
        Post-process a response after it has been generated.
        
        This captures knowledge from the response and collects feedback on 
        context relevance to improve future retrievals.
        
        Args:
            context: The middleware context
            
        Returns:
            Updated middleware context
        """
        if not context.response_generated:
            return context
            
        # Check if this was a sequential thinking request that we enhanced
        retrieval_metadata = context.metadata.get("knowledge_retrieval")
        if not retrieval_metadata or not retrieval_metadata.get("applied", False):
            return context
            
        # Extract step result and process it
        try:
            step_result = self._extract_step_result(context.response)
            original_request = context.request
            
            if not step_result:
                return context
                
            # Add to step history for tracking
            self._update_step_history(
                trace_id=retrieval_metadata.get("trace_id"),
                step_number=retrieval_metadata.get("step_number", 1),
                step_result=step_result,
                context_items=retrieval_metadata.get("context_items", [])
            )
            
            # Process the step result to capture knowledge and collect feedback
            result = await self.integration.process_step_result(
                step_result=step_result,
                original_request=original_request,
                enhanced_request=retrieval_metadata
            )
            
            # Add feedback and knowledge capture results to response metadata
            if result:
                context.enhance_metadata({
                    "knowledge_retrieval_result": {
                        "knowledge_captured": result.get("knowledge_captured", False),
                        "feedback_collected": result.get("feedback_collected", False),
                        "next_step_suggestions": result.get("next_step_suggestions", []),
                        "context_quality_score": result.get("context_quality_score", 0)
                    }
                })
                
            logger.debug(
                f"Processed step result for step {retrieval_metadata.get('step_number', 1)} "
                f"({retrieval_metadata.get('step_type', 'unknown')}) with "
                f"quality score {result.get('context_quality_score', 0):.2f}"
            )
                
        except Exception as e:
            logger.error(f"Error in post-processing step result: {str(e)}")
        
        return context
    
    def _is_sequential_thinking_request(self, request: Any) -> bool:
        """
        Check if the request is a sequential thinking request.
        
        Args:
            request: The request to check
            
        Returns:
            True if this is a sequential thinking request, False otherwise
        """
        # Check for common sequential thinking request properties
        if hasattr(request, "thought_number") or isinstance(request, dict) and "thought_number" in request:
            return True
            
        # Check for sequential thinking in request structure
        if hasattr(request, "metadata") and getattr(request.metadata, "task_type", "") == "sequential_thinking":
            return True
            
        return False
    
    def _extract_request_data(self, request: Any) -> Tuple[Optional[Dict[str, Any]], str, str]:
        """
        Extract step information, system prompt, and user prompt from the request.
        
        Args:
            request: The request to extract data from
            
        Returns:
            Tuple of (step_info, system_prompt, user_prompt)
        """
        step_info = None
        system_prompt = ""
        user_prompt = ""
        
        # Handle dictionary-style request
        if isinstance(request, dict):
            step_info = {
                "thought_number": request.get("thought_number", 1),
                "total_thoughts": request.get("total_thoughts", 5),
                "step_type": request.get("step_type")
            }
            system_prompt = request.get("system_prompt", "")
            user_prompt = request.get("prompt", "") or request.get("user_prompt", "")
            
        # Handle object-style request
        elif hasattr(request, "thought_number"):
            step_info = {
                "thought_number": getattr(request, "thought_number", 1),
                "total_thoughts": getattr(request, "total_thoughts", 5),
                "step_type": getattr(request, "step_type", None)
            }
            system_prompt = getattr(request, "system_prompt", "")
            user_prompt = getattr(request, "prompt", "") or getattr(request, "user_prompt", "")
            
        return step_info, system_prompt, user_prompt
    
    def _update_request_with_enhancements(self, request: Any, enhanced_data: Dict[str, Any]) -> None:
        """
        Update the request with enhanced prompts from knowledge retrieval.
        
        Args:
            request: The request to update
            enhanced_data: Enhanced data from knowledge retrieval
        """
        # Update dictionary-style request
        if isinstance(request, dict):
            request["system_prompt"] = enhanced_data.get("system_prompt", request.get("system_prompt", ""))
            if "prompt" in request:
                request["prompt"] = enhanced_data.get("user_prompt", request.get("prompt", ""))
            elif "user_prompt" in request:
                request["user_prompt"] = enhanced_data.get("user_prompt", request.get("user_prompt", ""))
                
        # Update object-style request
        elif hasattr(request, "system_prompt"):
            setattr(request, "system_prompt", enhanced_data.get("system_prompt", getattr(request, "system_prompt", "")))
            if hasattr(request, "prompt"):
                setattr(request, "prompt", enhanced_data.get("user_prompt", getattr(request, "prompt", "")))
            elif hasattr(request, "user_prompt"):
                setattr(request, "user_prompt", enhanced_data.get("user_prompt", getattr(request, "user_prompt", "")))
    
    def _extract_step_result(self, response: Any) -> Optional[Dict[str, Any]]:
        """
        Extract step result from the response.
        
        Args:
            response: The response to extract from
            
        Returns:
            Step result dictionary or None if not found
        """
        # Handle dictionary-style response
        if isinstance(response, dict):
            if "thought" in response or "thought_number" in response:
                return response
                
        # Handle object-style response
        elif hasattr(response, "thought"):
            return {
                "thought": getattr(response, "thought", ""),
                "thought_number": getattr(response, "thought_number", 1),
                "next_thought_needed": getattr(response, "next_thought_needed", False),
                "total_thoughts": getattr(response, "total_thoughts", 1)
            }
        
        return None
            
    def _get_trace_id(self, request: Any, context: MiddlewareContext) -> str:
        """
        Get or create a trace ID for tracking sequential thinking steps.
        
        Args:
            request: The request object
            context: Middleware context
            
        Returns:
            Trace ID string
        """
        # Try to get from request
        if isinstance(request, dict) and "trace_id" in request:
            return request["trace_id"]
        elif hasattr(request, "trace_id") and getattr(request, "trace_id"):
            return getattr(request, "trace_id")
            
        # Try to get from context metadata
        if context.metadata and "trace_id" in context.metadata:
            return context.metadata["trace_id"]
            
        # Generate a new trace ID
        return str(uuid.uuid4())
    
    def _get_previous_steps(self, trace_id: str, current_step: int) -> List[Dict[str, Any]]:
        """
        Get previous steps for a trace to provide continuity.
        
        Args:
            trace_id: The trace ID
            current_step: Current step number
            
        Returns:
            List of previous step data
        """
        if trace_id not in self.step_history:
            return []
            
        # Get steps before the current step
        return [step for step in self.step_history[trace_id] 
                if step["step_number"] < current_step]
    
    def _update_step_history(
        self, 
        trace_id: str, 
        step_number: int, 
        step_result: Dict[str, Any],
        context_items: List[Dict[str, Any]]
    ) -> None:
        """
        Update step history for a trace.
        
        Args:
            trace_id: The trace ID
            step_number: Step number
            step_result: Step result data
            context_items: Context items used for this step
        """
        if trace_id not in self.step_history:
            self.step_history[trace_id] = []
            
        # Add step data to history
        self.step_history[trace_id].append({
            "step_number": step_number,
            "thought": step_result.get("thought", ""),
            "context_items": context_items,
            "timestamp": time.time()
        })
        
        # Limit history size (keep last 10 traces)
        if len(self.step_history) > 10:
            oldest_trace = min(self.step_history.keys(), 
                              key=lambda t: min(step["timestamp"] for step in self.step_history[t]))
            del self.step_history[oldest_trace]
    
    def _optimize_context_window(
        self, 
        enhanced_data: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        previous_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize context window usage to balance comprehensiveness with relevance.
        
        Args:
            enhanced_data: Enhanced data from knowledge retrieval
            system_prompt: Original system prompt
            user_prompt: Original user prompt
            previous_steps: Previous steps in the trace
            
        Returns:
            Optimized enhanced data
        """
        result = enhanced_data.copy()
        result["optimized"] = False
        
        if not enhanced_data.get("context_applied", False):
            return result
            
        # Calculate token usage
        enhanced_system_prompt = enhanced_data.get("system_prompt", system_prompt)
        enhanced_user_prompt = enhanced_data.get("user_prompt", user_prompt)
        
        system_tokens = self.token_estimator(enhanced_system_prompt)
        user_tokens = self.token_estimator(enhanced_user_prompt)
        
        # Calculate tokens from context items
        context_items = enhanced_data.get("context_items", [])
        context_tokens = sum(self.token_estimator(item.get("content", "")) 
                            for item in context_items)
        
        # Calculate total tokens
        total_tokens = system_tokens + user_tokens
        
        # Set token usage stats
        result["token_usage"] = {
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "context_tokens": context_tokens,
            "total_tokens": total_tokens
        }
        
        # Define token budget (could be model-specific)
        token_budget = 3500  # Conservative budget for 4K context models
        
        # If within budget, return as is
        if total_tokens <= token_budget:
            return result
            
        # Need to optimize - prioritize by relevance and recency
        logger.debug(f"Optimizing context window for step {enhanced_data.get('step_type')} - "
                    f"current tokens: {total_tokens}, budget: {token_budget}")
        
        # Sort context items by similarity score
        sorted_items = sorted(
            context_items, 
            key=lambda x: x.get("similarity", 0), 
            reverse=True
        )
        
        # Calculate token reduction needed
        reduction_needed = total_tokens - token_budget + 100  # Add buffer
        
        # Start removing least relevant items until we're under budget
        removed_tokens = 0
        optimized_items = []
        
        for item in sorted_items:
            item_tokens = self.token_estimator(item.get("content", ""))
            
            if removed_tokens < reduction_needed:
                # Skip this item to reduce tokens
                removed_tokens += item_tokens
                continue
                
            optimized_items.append(item)
        
        # If we removed all items, keep at least the most relevant one
        if not optimized_items and sorted_items:
            optimized_items = [sorted_items[0]]
        
        # Rebuild enhanced prompts with optimized items
        if len(optimized_items) < len(context_items):
            # Update context in prompts
            result["context_items"] = optimized_items
            
            # Recalculate context string
            optimized_context = self._format_optimized_context(optimized_items)
            
            # Reapply to prompts
            if enhanced_data.get("apply_to_system_prompt", True):
                result["system_prompt"] = self._apply_context_to_prompt(
                    system_prompt, optimized_context
                )
            else:
                result["user_prompt"] = self._apply_context_to_prompt(
                    user_prompt, optimized_context
                )
            
            # Update token usage
            optimized_context_tokens = sum(self.token_estimator(item.get("content", "")) 
                                         for item in optimized_items)
            
            result["token_usage"] = {
                "system_tokens": self.token_estimator(result["system_prompt"]),
                "user_tokens": self.token_estimator(result["user_prompt"]),
                "context_tokens": optimized_context_tokens,
                "total_tokens": self.token_estimator(result["system_prompt"]) + 
                               self.token_estimator(result["user_prompt"]),
                "removed_tokens": removed_tokens
            }
            
            result["optimized"] = True
            
            logger.debug(
                f"Optimized context window: removed {len(context_items) - len(optimized_items)} "
                f"items, saved {removed_tokens} tokens"
            )
        
        return result
    
    def _format_optimized_context(self, context_items: List[Dict[str, Any]]) -> str:
        """
        Format optimized context items into a unified context string.
        
        Args:
            context_items: Optimized context items
            
        Returns:
            Formatted context string
        """
        if not context_items:
            return ""
            
        parts = ["### Relevant Context:"]
        
        for i, item in enumerate(context_items, 1):
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            
            source = metadata.get("source", "unknown")
            importance = metadata.get("importance", 0)
            
            parts.append(f"[{i}] {source.capitalize()} (relevance: {item.get('similarity', 0):.2f}):")
            parts.append(content)
            parts.append("")
        
        return "\n".join(parts)
    
    def _apply_context_to_prompt(self, prompt: str, context: str) -> str:
        """
        Apply context to a prompt in a way that preserves the original prompt.
        
        Args:
            prompt: Original prompt
            context: Context to apply
            
        Returns:
            Combined prompt with context
        """
        if not context:
            return prompt
            
        return f"{prompt}\n\n{context}"
    
    def _default_token_estimator(self, text: str) -> int:
        """
        Default token estimator function (approximate).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        # Simple approximation: 4 characters per token
        return len(text) // 4 + 1 