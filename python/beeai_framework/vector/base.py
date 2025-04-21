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
Vector Memory Provider Base Module

This module defines the abstract base classes for vector memory integration
including the core VectorMemoryProvider interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


class VectorMemoryProviderConfig(BaseModel):
    """Base configuration for vector memory providers."""
    
    dimension: int = Field(default=1536, description="Vector embedding dimension")
    enabled: bool = Field(default=True, description="Whether this provider is enabled")
    similarity_threshold: float = Field(
        default=0.7, description="Minimum similarity score for context retrieval (0-1)"
    )
    max_results: int = Field(default=10, description="Maximum number of results to return from queries")


class ContextMetadata(BaseModel):
    """Metadata model for contextual information."""
    
    source: str = Field(description="Source of the context (e.g., documentation, code, user)")
    category: Optional[str] = Field(default=None, description="Category of context (e.g., project_info, error)")
    level: str = Field(
        default="project", 
        description="Hierarchical level (domain, techstack, project)"
    )
    importance: Optional[float] = Field(default=None, description="Importance score (0-1)")
    timestamp: Optional[str] = Field(default=None, description="Timestamp when context was added")
    
    # Additional metadata fields can be added as needed
    custom_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional custom metadata"
    )


class ContextResult(BaseModel):
    """A single context result from a vector query."""
    
    content: str = Field(description="The context content")
    metadata: ContextMetadata = Field(description="Metadata about this context")
    score: float = Field(description="Similarity score for this result (0-1)")


class VectorMemoryProvider(ABC):
    """
    Abstract base class for vector memory providers.
    
    This defines the interface that all vector database integrations must implement
    to support the Selective Contextual Retrieval system.
    """
    
    def __init__(self, config: VectorMemoryProviderConfig):
        """
        Initialize the vector memory provider.
        
        Args:
            config: Configuration for this provider
        """
        self.config = config
        self.emitter = Emitter.root().child(
            namespace=["vector", "provider", self.__class__.__name__.lower()],
            creator=self,
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the vector memory provider.
        
        This method should be called before any other methods to ensure
        the provider is properly connected and configured.
        """
        pass
    
    @abstractmethod
    async def get_context(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[ContextResult]:
        """
        Retrieve context based on a query and optional filters.
        
        Args:
            query: The query text to find similar contexts
            metadata_filter: Optional metadata-based filters
            limit: Optional limit on number of results (defaults to config.max_results)
            
        Returns:
            List of context results ordered by relevance
        """
        pass
    
    @abstractmethod
    async def add_context(
        self, 
        content: str, 
        metadata: Union[Dict[str, Any], ContextMetadata],
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Add context to the vector memory.
        
        Args:
            content: The context content to add
            metadata: Metadata about this context
            embedding: Optional pre-computed embedding vector
            
        Returns:
            ID of the added context
        """
        pass
    
    @abstractmethod
    async def add_contexts_batch(
        self,
        contexts: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Add multiple contexts in a batch operation.
        
        Args:
            contexts: List of context objects with 'content', 'metadata', and optional 'embedding'
            
        Returns:
            List of IDs for the added contexts
        """
        pass
        
    @abstractmethod
    async def clear_context(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Clear context from the vector memory.
        
        Args:
            filter_metadata: Optional metadata to filter which contexts to clear
            
        Returns:
            Number of contexts removed
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector memory.
        
        Returns:
            Dictionary of statistics including counts by level, category, etc.
        """
        pass
    
    async def shutdown(self) -> None:
        """
        Properly shut down the vector memory provider.
        
        Override this method if the provider requires special shutdown logic.
        """
        pass 