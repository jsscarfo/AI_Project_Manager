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
Knowledge Capture Module

This module provides functionality for automatically extracting, categorizing,
and storing knowledge from LLM interactions for future contextual retrieval.
"""

import logging
import re
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from pydantic import BaseModel, Field, model_validator, root_validator
from dataclasses import dataclass, field

from beeai_framework.errors import FrameworkError
from beeai_framework.vector.base import ContextMetadata, VectorMemoryProvider
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, UserMessage, SystemMessage, Message
from beeai_framework.middleware.base import Middleware, MiddlewareConfig, MiddlewareContext

logger = logging.getLogger(__name__)


class ExtractedKnowledgeItem(BaseModel):
    """A single piece of extracted knowledge."""
    
    content: str = Field(description="The knowledge content")
    source: str = Field(description="Source of the knowledge")
    category: str = Field(description="Category of the knowledge")
    level: str = Field(description="Hierarchical level (domain, techstack, project)")
    confidence: float = Field(description="Confidence score (0-1)")
    importance: float = Field(description="Importance score (0-1)")
    timestamp: str = Field(description="Timestamp when knowledge was extracted")
    related_entities: Optional[List[str]] = Field(default=None, description="Related entities or concepts")


class KnowledgeCaptureSetting(BaseModel):
    """Setting for knowledge capture control."""
    
    enabled: bool = Field(default=True, description="Whether knowledge capture is enabled")
    importance_threshold: float = Field(
        default=0.6, 
        description="Minimum importance score to capture knowledge (0-1)"
    )
    extraction_temperature: float = Field(
        default=0.2,
        description="Temperature for the extraction model"
    )
    min_token_length: int = Field(
        default=50,
        description="Minimum token length for knowledge to be captured"
    )
    max_token_length: int = Field(
        default=1000,
        description="Maximum token length for knowledge to be captured"
    )
    content_types: List[str] = Field(
        default=["code_snippet", "concept", "explanation", "best_practice"],
        description="Types of content to capture"
    )


class KnowledgeEntry(BaseModel):
    """Knowledge entry to be stored in vector memory."""
    
    content: str = Field(..., description="Content of the knowledge entry")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for the knowledge entry"
    )
    
    @root_validator(skip_on_failure=True)
    def check_metadata_structure(cls, values):
        """Validate that required metadata fields are present."""
        metadata = values.get("metadata", {})
        
        # Ensure required fields
        required_fields = ["source", "category", "level", "importance"]
        for field in required_fields:
            if field not in metadata:
                if field == "source":
                    metadata[field] = "chat_conversation"
                elif field == "category":
                    metadata[field] = "general"
                elif field == "level":
                    metadata[field] = "concept"
                elif field == "importance":
                    metadata[field] = 0.7
        
        values["metadata"] = metadata
        return values


class KnowledgeExtractor(BaseModel):
    """Extracts knowledge from assistant responses."""
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    extraction_prompt: str = Field(
        default=(
            "Extract key knowledge from the following messages that would be valuable "
            "to store for future reference. Focus on:\n"
            "1. Technical explanations\n"
            "2. Code patterns and examples\n"
            "3. Problem-solving approaches\n"
            "4. Best practices\n"
            "5. Conceptual frameworks\n\n"
            "For each knowledge item, provide:\n"
            "- The content (exact quoted text)\n"
            "- Category (code_snippet, concept, explanation, best_practice, etc.)\n"
            "- Level (techstack, component, architecture, project, concept)\n"
            "- Importance score (0-1)\n\n"
            "Format as JSON:\n"
            '[{{"content": "...", "metadata": {{"category": "...", "level": "...", "importance": 0.X}}}}]\n\n'
            "User message: {user_message}\n"
            "Assistant response: {assistant_response}\n\n"
            "Extract knowledge items (return JSON array only):"
        ),
        description="Prompt for knowledge extraction"
    )
    
    chat_model: ChatModel = Field(
        ...,
        description="Chat model to use for knowledge extraction"
    )
    
    settings: KnowledgeCaptureSetting = Field(
        default_factory=KnowledgeCaptureSetting,
        description="Settings for knowledge extraction"
    )
    
    async def extract(
        self, 
        user_message: str, 
        assistant_response: str
    ) -> List[KnowledgeEntry]:
        """Extract knowledge from user and assistant messages.
        
        Args:
            user_message: User message
            assistant_response: Assistant response
            
        Returns:
            List of extracted knowledge entries
        """
        if not self.settings.enabled:
            logger.debug("Knowledge extraction is disabled")
            return []
        
        if not user_message or not assistant_response:
            logger.warning("Empty message, skipping knowledge extraction")
            return []
        
        # Skip if assistant response is too short
        if len(assistant_response.split()) < 20:
            logger.debug("Assistant response too short, skipping knowledge extraction")
            return []
        
        # Prepare extraction prompt
        extraction_input = self.extraction_prompt.format(
            user_message=user_message,
            assistant_response=assistant_response
        )
        
        try:
            # Get extraction from model
            response = await self.chat_model.generate(
                messages=[SystemMessage(content=extraction_input)],
                temperature=self.settings.extraction_temperature,
                max_tokens=1000
            )
            
            if not response:
                logger.warning("Empty extraction response")
                return []
            
            # Parse extraction response
            extraction_text = response[0].content if isinstance(response, list) else response.content
            
            # Try to extract JSON
            json_match = re.search(r'\[.*\]', extraction_text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in extraction response")
                return []
            
            json_str = json_match.group(0)
            extracted_items = json.loads(json_str)
            
            # Filter and convert to KnowledgeEntry objects
            knowledge_entries = []
            for item in extracted_items:
                content = item.get("content", "").strip()
                metadata = item.get("metadata", {})
                
                # Skip if content is too short or too long
                if (
                    not content or 
                    len(content.split()) < self.settings.min_token_length or
                    len(content.split()) > self.settings.max_token_length
                ):
                    continue
                
                # Skip if importance is below threshold
                importance = float(metadata.get("importance", 0))
                if importance < self.settings.importance_threshold:
                    continue
                
                # Skip if content type is not in allowed types
                category = metadata.get("category", "")
                if self.settings.content_types and category not in self.settings.content_types:
                    continue
                
                # Add source to metadata
                if "source" not in metadata:
                    metadata["source"] = "chat_conversation"
                
                # Create knowledge entry
                knowledge_entries.append(
                    KnowledgeEntry(content=content, metadata=metadata)
                )
            
            logger.info(f"Extracted {len(knowledge_entries)} knowledge entries")
            return knowledge_entries
            
        except Exception as e:
            logger.error(f"Error in knowledge extraction: {str(e)}")
            return []


class KnowledgeCaptureProcessor:
    """Processes and captures knowledge from conversations."""
    
    def __init__(
        self,
        vector_provider: VectorMemoryProvider,
        chat_model: ChatModel,
        settings: Optional[KnowledgeCaptureSetting] = None
    ):
        """Initialize the knowledge capture processor.
        
        Args:
            vector_provider: Vector memory provider for storing knowledge
            chat_model: Chat model to use for knowledge extraction
            settings: Settings for knowledge capture
        """
        self.vector_provider = vector_provider
        self.settings = settings or KnowledgeCaptureSetting()
        self.extractor = KnowledgeExtractor(
            chat_model=chat_model,
            settings=self.settings
        )
    
    async def process_message_pair(
        self, 
        user_message: str, 
        assistant_response: str,
        context_metadata: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeEntry]:
        """Process a message pair to extract and store knowledge.
        
        Args:
            user_message: User message
            assistant_response: Assistant response
            context_metadata: Additional metadata for context
            
        Returns:
            List of extracted and stored knowledge entries
        """
        if not self.settings.enabled:
            return []
        
        # Extract knowledge from messages
        knowledge_entries = await self.extractor.extract(
            user_message=user_message,
            assistant_response=assistant_response
        )
        
        if not knowledge_entries:
            return []
        
        # Add context metadata if provided
        if context_metadata:
            for entry in knowledge_entries:
                # Merge metadata, preserving entry-specific values
                entry.metadata.update({
                    k: v for k, v in context_metadata.items() 
                    if k not in entry.metadata
                })
        
        # Store knowledge in vector memory
        stored_entries = []
        for entry in knowledge_entries:
            try:
                metadata = ContextMetadata(**entry.metadata)
                await self.vector_provider.store_context(
                    content=entry.content,
                    metadata=metadata
                )
                stored_entries.append(entry)
            except Exception as e:
                logger.error(f"Error storing knowledge entry: {str(e)}")
        
        logger.info(f"Stored {len(stored_entries)} knowledge entries")
        return stored_entries
    
    async def process_conversation(
        self, 
        messages: List[Message],
        context_metadata: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeEntry]:
        """Process an entire conversation to extract and store knowledge.
        
        Args:
            messages: List of messages in the conversation
            context_metadata: Additional metadata for context
            
        Returns:
            List of extracted and stored knowledge entries
        """
        if not self.settings.enabled or not messages:
            return []
        
        # Pair user messages with assistant responses
        all_stored_entries = []
        
        last_user_message = None
        for message in messages:
            if isinstance(message, UserMessage):
                last_user_message = message.content
            elif isinstance(message, AssistantMessage) and last_user_message:
                # Process the pair
                stored_entries = await self.process_message_pair(
                    user_message=last_user_message,
                    assistant_response=message.content,
                    context_metadata=context_metadata
                )
                all_stored_entries.extend(stored_entries)
                last_user_message = None
        
        return all_stored_entries
    
    def extract_knowledge_from_content(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> KnowledgeEntry:
        """Create a knowledge entry directly from content and metadata.
        
        Args:
            content: Content to store
            metadata: Metadata for the content
            
        Returns:
            Knowledge entry
        """
        # Create knowledge entry with required metadata
        entry_metadata = {
            "source": metadata.get("source", "manual_entry"),
            "category": metadata.get("category", "general"),
            "level": metadata.get("level", "concept"),
            "importance": metadata.get("importance", 0.8)
        }
        
        # Add any additional metadata
        for key, value in metadata.items():
            if key not in entry_metadata:
                entry_metadata[key] = value
        
        return KnowledgeEntry(
            content=content,
            metadata=entry_metadata
        )
    
    async def store_knowledge_from_content(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Store knowledge directly from content and metadata.
        
        Args:
            content: Content to store
            metadata: Metadata for the content
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            entry = self.extract_knowledge_from_content(content, metadata)
            vector_metadata = ContextMetadata(**entry.metadata)
            
            await self.vector_provider.store_context(
                content=entry.content,
                metadata=vector_metadata
            )
            
            logger.info("Stored knowledge from content successfully")
            return True
        except Exception as e:
            logger.error(f"Error storing knowledge from content: {str(e)}")
            return False


class KnowledgeCaptureMiddlewareConfig(MiddlewareConfig):
    """Configuration for the Knowledge Capture Middleware."""
    
    enabled: bool = Field(default=True, description="Whether knowledge capture is enabled")
    importance_threshold: float = Field(
        default=0.6,
        description="Minimum importance score to capture knowledge (0-1)"
    )
    content_types: List[str] = Field(
        default=["code_snippet", "concept", "explanation", "best_practice"],
        description="Types of content to capture"
    )
    skip_short_responses: bool = Field(
        default=True,
        description="Skip processing short responses"
    )
    min_response_length: int = Field(
        default=50,
        description="Minimum token length for responses to process"
    )
    batch_processing: bool = Field(
        default=False,
        description="Whether to use batch processing for captured knowledge"
    )
    batch_size: int = Field(
        default=10,
        description="Batch size for knowledge storage"
    )
    track_request_metadata: bool = Field(
        default=True,
        description="Track request metadata for attribution"
    )
    capture_conversation_context: bool = Field(
        default=True,
        description="Capture context of the conversation, not just the response"
    )


class KnowledgeCaptureMiddleware(Middleware):
    """
    Middleware that automatically extracts and stores knowledge from LLM interactions.
    
    This middleware:
    1. Intercepts LLM responses
    2. Extracts valuable knowledge using the KnowledgeCaptureProcessor
    3. Scores and categorizes the knowledge
    4. Stores it in the vector memory system for future contextual retrieval
    
    Knowledge capture happens asynchronously to avoid impacting user experience.
    """
    
    def __init__(
        self,
        vector_provider: VectorMemoryProvider,
        chat_model: ChatModel,
        config: Optional[KnowledgeCaptureMiddlewareConfig] = None,
        knowledge_processor: Optional[KnowledgeCaptureProcessor] = None,
    ):
        """
        Initialize the knowledge capture middleware.
        
        Args:
            vector_provider: Vector memory provider for storing knowledge
            chat_model: Chat model for knowledge extraction
            config: Configuration for the middleware
            knowledge_processor: Optional pre-configured knowledge processor
        """
        super().__init__(config or KnowledgeCaptureMiddlewareConfig())
        self.config = config or KnowledgeCaptureMiddlewareConfig()
        
        # Use provided processor or create a new one
        self.knowledge_processor = knowledge_processor or KnowledgeCaptureProcessor(
            vector_provider=vector_provider,
            chat_model=chat_model,
            settings=KnowledgeCaptureSetting(
                enabled=self.config.enabled,
                importance_threshold=self.config.importance_threshold,
                content_types=self.config.content_types,
                min_token_length=self.config.min_response_length,
            )
        )
        
        # Track batched knowledge entries if batch processing is enabled
        self.pending_entries: List[KnowledgeEntry] = []
        self.last_batch_time = time.time()
        self.capture_tasks = []
        
        logger.info("Initialized KnowledgeCaptureMiddleware")
    
    async def process(self, context: MiddlewareContext) -> MiddlewareContext:
        """
        Process the request context through this middleware.
        
        This method runs before the LLM generates a response.
        It will set up metadata tracking for knowledge capture.
        
        Args:
            context: The request context to process
            
        Returns:
            The (potentially modified) context
        """
        if not self.config.enabled:
            logger.debug("Knowledge capture is disabled, skipping")
            return context
        
        # Extract request metadata only if tracking is enabled
        if self.config.track_request_metadata:
            request_metadata = self._extract_request_metadata(context)
            
            # Store metadata in context for post-processing
            if request_metadata:
                context.metadata["knowledge_capture"] = request_metadata
                logger.debug("Added request metadata for knowledge capture")
        
        return context
    
    def _extract_request_metadata(self, context: MiddlewareContext) -> Dict[str, Any]:
        """Extract metadata from the request for knowledge attribution."""
        metadata = {}
        
        # Extract user query if present
        user_query = self._extract_user_query(context.request)
        if user_query:
            metadata["user_query"] = user_query
            
        # Add timestamp
        metadata["timestamp"] = datetime.now().isoformat()
        
        # Add additional context metadata if available
        if "metadata" in context.metadata and isinstance(context.metadata["metadata"], dict):
            metadata.update(context.metadata["metadata"])
            
        return metadata
    
    def _extract_user_query(self, request: Any) -> Optional[str]:
        """Extract the user query from the request object."""
        # Handle different request types
        if hasattr(request, "messages") and isinstance(request.messages, list):
            # Extract from message list (assuming last user message)
            for msg in reversed(request.messages):
                if isinstance(msg, UserMessage) or (hasattr(msg, "role") and msg.role == "user"):
                    return msg.content
        
        # Try to access content or query attributes
        if hasattr(request, "content"):
            return request.content
        if hasattr(request, "query"):
            return request.query
            
        return None
    
    async def post_process(self, context: MiddlewareContext) -> MiddlewareContext:
        """
        Process the response context after LLM has generated a response.
        
        This method extracts knowledge from the response and stores it.
        It runs asynchronously to avoid impacting user experience.
        
        Args:
            context: The response context to process
            
        Returns:
            The unmodified context
        """
        if not self.config.enabled or not context.response_generated:
            return context
        
        # Schedule knowledge capture as a background task
        # This avoids impacting the response time for the user
        asyncio.create_task(self._capture_knowledge_async(context))
        
        return context
    
    async def _capture_knowledge_async(self, context: MiddlewareContext) -> None:
        """
        Capture knowledge asynchronously after response is sent.
        
        Args:
            context: The completed context with request and response
        """
        try:
            # Extract user query and assistant response
            user_query = self._extract_user_query(context.request)
            assistant_response = self._extract_assistant_response(context.response)
            
            if not user_query or not assistant_response:
                logger.debug("Missing user query or assistant response, skipping knowledge capture")
                return
                
            # Skip short responses if configured
            if self.config.skip_short_responses and len(assistant_response.split()) < self.config.min_response_length:
                logger.debug(f"Response too short ({len(assistant_response.split())} words), skipping knowledge capture")
                return
                
            # Get request metadata
            metadata = context.metadata.get("knowledge_capture", {})
            
            # Extract and store knowledge
            logger.debug("Extracting knowledge from interaction")
            knowledge_entries = await self.knowledge_processor.process_message_pair(
                user_message=user_query,
                assistant_response=assistant_response,
                context_metadata=metadata
            )
            
            if not knowledge_entries:
                logger.debug("No knowledge entries extracted")
                return
                
            logger.info(f"Extracted {len(knowledge_entries)} knowledge entries from interaction")
            
            # Process entries (either batch or immediate)
            if self.config.batch_processing:
                self._add_to_batch(knowledge_entries)
            else:
                # Process each entry individually
                for entry in knowledge_entries:
                    await self.knowledge_processor.vector_provider.add_context(
                        content=entry.content,
                        metadata=entry.metadata
                    )
                    logger.debug(f"Stored knowledge entry: {entry.content[:50]}...")
        
        except Exception as e:
            logger.error(f"Error in knowledge capture: {str(e)}")
    
    def _extract_assistant_response(self, response: Any) -> Optional[str]:
        """Extract the assistant response from the response object."""
        # Handle different response types
        if isinstance(response, str):
            return response
            
        if isinstance(response, AssistantMessage) or (hasattr(response, "role") and response.role == "assistant"):
            return response.content
            
        if hasattr(response, "content"):
            return response.content
            
        if isinstance(response, list) and len(response) > 0:
            # Try to get content from first item if it's a list
            if hasattr(response[0], "content"):
                return response[0].content
                
        return None
    
    def _add_to_batch(self, entries: List[KnowledgeEntry]) -> None:
        """Add knowledge entries to the batch for later processing."""
        self.pending_entries.extend(entries)
        
        # Process batch if it's full or if enough time has passed
        if (len(self.pending_entries) >= self.config.batch_size or 
                time.time() - self.last_batch_time > 60):  # 60 seconds max batch delay
            # Create a task to process the batch
            batch_task = asyncio.create_task(self._process_batch())
            self.capture_tasks.append(batch_task)
            
            # Clean up completed tasks
            self.capture_tasks = [t for t in self.capture_tasks if not t.done()]
    
    async def _process_batch(self) -> None:
        """Process a batch of knowledge entries."""
        if not self.pending_entries:
            return
            
        # Get current batch and reset
        batch = self.pending_entries
        self.pending_entries = []
        self.last_batch_time = time.time()
        
        try:
            # Prepare batch for vector store
            contexts = [
                {
                    "content": entry.content,
                    "metadata": entry.metadata
                }
                for entry in batch
            ]
            
            # Store batch
            await self.knowledge_processor.vector_provider.add_contexts_batch(contexts)
            logger.info(f"Stored batch of {len(batch)} knowledge entries")
            
        except Exception as e:
            logger.error(f"Error processing knowledge batch: {str(e)}")
    
    async def shutdown(self) -> None:
        """Process any remaining knowledge entries before shutdown."""
        if self.config.batch_processing and self.pending_entries:
            logger.info(f"Processing {len(self.pending_entries)} remaining knowledge entries before shutdown")
            await self._process_batch()
            
        # Wait for all pending capture tasks to complete
        if self.capture_tasks:
            logger.info(f"Waiting for {len(self.capture_tasks)} knowledge capture tasks to complete")
            await asyncio.gather(*self.capture_tasks, return_exceptions=True) 