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
Embedding Service Module

This module provides functionality for generating, caching, and managing
vector embeddings for text content.
"""

import os
import json
import hashlib
import asyncio
from typing import Any, Callable, Dict, List, Optional, Union, Literal
from datetime import datetime, timedelta
from pathlib import Path
import logging
from pydantic import BaseModel, Field
from functools import wraps

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError

logger = logging.getLogger(__name__)


class EmbeddingCacheItem(BaseModel):
    """A cached embedding along with metadata."""
    
    embedding: List[float]
    created_at: datetime
    expires_at: Optional[datetime] = None
    model_id: str


class EmbeddingServiceConfig(BaseModel):
    """Configuration for the embedding service."""
    
    cache_dir: str = Field(default="./cache/embeddings", description="Directory for caching embeddings")
    cache_ttl: int = Field(default=86400, description="Time-to-live for cached embeddings in seconds")
    enabled: bool = Field(default=True, description="Whether embedding caching is enabled")
    primary_model_id: str = Field(default="default", description="ID of the primary embedding model")
    fallback_model_ids: List[str] = Field(default_factory=list, description="IDs of fallback embedding models")
    max_input_length: int = Field(default=8000, description="Maximum text length for embedding")
    batch_size: int = Field(default=16, description="Batch size for batch embedding operations")
    
    
class EmbeddingService:
    """
    Service for generating and caching text embeddings.
    
    This service handles:
    - Text embedding generation with configurable models
    - Local embedding caching for improved performance
    - Batch processing for efficiency
    - Fallback mechanisms for reliability
    """
    
    def __init__(
        self,
        embedding_fn: Callable[[str, str], Union[List[float], List[List[float]]]],
        config: Optional[EmbeddingServiceConfig] = None,
        embedding_models: Optional[Dict[str, Callable[[str], List[float]]]] = None,
    ):
        """
        Initialize the embedding service.
        
        Args:
            embedding_fn: Function that takes text and model_id and returns embedding vector(s)
            config: Configuration for the embedding service
            embedding_models: Optional dictionary mapping model IDs to embedding functions
        """
        self.config = config or EmbeddingServiceConfig()
        self._embedding_fn = embedding_fn
        self._embedding_models = embedding_models or {}
        
        # Set up cache directory
        if self.config.enabled:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.emitter = Emitter.root().child(
            namespace=["vector", "embedding_service"],
            creator=self,
        )
    
    def _get_cache_key(self, text: str, model_id: str) -> str:
        """Generate a unique cache key for the text and model."""
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{model_id}_{text_hash}"
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return Path(self.config.cache_dir) / f"{key}.json"
    
    async def _load_from_cache(self, text: str, model_id: str) -> Optional[List[float]]:
        """Load an embedding from cache if available and not expired."""
        if not self.config.enabled:
            return None
            
        key = self._get_cache_key(text, model_id)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        try:
            cache_data = json.loads(cache_path.read_text())
            item = EmbeddingCacheItem(**cache_data)
            
            # Check if cache is expired
            if item.expires_at and datetime.fromisoformat(item.expires_at) < datetime.now():
                logger.debug(f"Expired cache for key {key}")
                return None
                
            logger.debug(f"Cache hit for key {key}")
            return item.embedding
        except Exception as e:
            logger.warning(f"Error loading from cache: {str(e)}")
            return None
    
    async def _save_to_cache(self, text: str, embedding: List[float], model_id: str) -> None:
        """Save an embedding to the cache."""
        if not self.config.enabled:
            return
            
        key = self._get_cache_key(text, model_id)
        cache_path = self._get_cache_path(key)
        
        try:
            now = datetime.now()
            expires_at = now + timedelta(seconds=self.config.cache_ttl) if self.config.cache_ttl > 0 else None
            
            item = EmbeddingCacheItem(
                embedding=embedding,
                created_at=now,
                expires_at=expires_at,
                model_id=model_id,
            )
            
            cache_path.write_text(item.model_dump_json())
            logger.debug(f"Saved to cache for key {key}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
    
    async def get_embedding(
        self, 
        text: str, 
        model_id: Optional[str] = None,
        use_cache: bool = True
    ) -> List[float]:
        """
        Get an embedding for the given text.
        
        Args:
            text: The text to embed
            model_id: Optional model ID to use (defaults to primary_model_id)
            use_cache: Whether to use the cache
            
        Returns:
            The embedding vector
        """
        model_id = model_id or self.config.primary_model_id
        
        # Truncate text if needed
        if len(text) > self.config.max_input_length:
            logger.warning(f"Truncating text from {len(text)} to {self.config.max_input_length} characters")
            text = text[:self.config.max_input_length]
        
        # Try to load from cache
        if use_cache and self.config.enabled:
            cached_embedding = await self._load_from_cache(text, model_id)
            if cached_embedding:
                return cached_embedding
        
        # Generate embedding
        try:
            embedding = await self._generate_embedding(text, model_id)
            
            # Save to cache if enabled
            if use_cache and self.config.enabled:
                await self._save_to_cache(text, embedding, model_id)
                
            return embedding
        except Exception as e:
            # Try fallback models if available
            if model_id != self.config.primary_model_id:
                # We're already using a fallback, don't cascade
                raise FrameworkError(f"Failed to generate embedding with model {model_id}: {str(e)}")
                
            for fallback_id in self.config.fallback_model_ids:
                try:
                    logger.warning(f"Falling back to embedding model {fallback_id}")
                    return await self.get_embedding(text, fallback_id, use_cache)
                except Exception:
                    continue
                    
            # If we got here, all fallbacks failed
            raise FrameworkError(f"Failed to generate embedding with all models: {str(e)}")
    
    async def _generate_embedding(self, text: str, model_id: str) -> List[float]:
        """Generate an embedding for the given text using the specified model."""
        if model_id in self._embedding_models:
            # Use the specific model function if available
            embedding_fn = self._embedding_models[model_id]
            result = embedding_fn(text)
            # Handle both synchronous and async functions
            if asyncio.iscoroutine(result):
                result = await result
            return result
        
        # Use the default embedding function
        result = self._embedding_fn(text, model_id)
        # Handle both synchronous and async functions
        if asyncio.iscoroutine(result):
            result = await result
        return result
    
    async def get_embeddings_batch(
        self, 
        texts: List[str], 
        model_id: Optional[str] = None,
        use_cache: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            model_id: Optional model ID to use
            use_cache: Whether to use the cache
            batch_size: Optional batch size (defaults to config.batch_size)
            
        Returns:
            List of embedding vectors
        """
        model_id = model_id or self.config.primary_model_id
        batch_size = batch_size or self.config.batch_size
        
        # Check cache for all texts first if enabled
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        if use_cache and self.config.enabled:
            for i, text in enumerate(texts):
                cached_embedding = await self._load_from_cache(text, model_id)
                if cached_embedding:
                    cached_embeddings[i] = cached_embedding
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts in batches
        new_embeddings = {}
        
        for i in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[i:i + batch_size]
            batch_indices = uncached_indices[i:i + batch_size]
            
            # Truncate texts if needed
            truncated_batch = [
                text[:self.config.max_input_length] if len(text) > self.config.max_input_length else text
                for text in batch_texts
            ]
            
            try:
                # Use batch embedding if the function supports it
                embeddings = await self._generate_embedding_batch(truncated_batch, model_id)
                
                # Save to cache and add to results
                for j, (text, embedding) in enumerate(zip(batch_texts, embeddings)):
                    idx = batch_indices[j]
                    new_embeddings[idx] = embedding
                    
                    if use_cache and self.config.enabled:
                        await self._save_to_cache(text, embedding, model_id)
            except Exception as e:
                # Fall back to individual embedding if batch fails
                logger.warning(f"Batch embedding failed, falling back to individual: {str(e)}")
                for j, text in enumerate(batch_texts):
                    idx = batch_indices[j]
                    try:
                        embedding = await self.get_embedding(text, model_id, use_cache)
                        new_embeddings[idx] = embedding
                    except Exception as inner_e:
                        logger.error(f"Failed to embed text at index {idx}: {str(inner_e)}")
                        # Use a zero vector as fallback to avoid breaking the entire batch
                        # The dimension should match the model's output dimension
                        # This is a last resort to avoid breaking the entire process
                        new_embeddings[idx] = [0.0] * self.config.dimension
        
        # Combine cached and new embeddings
        results = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                results.append(cached_embeddings[i])
            else:
                results.append(new_embeddings[i])
                
        return results
    
    async def _generate_embedding_batch(self, texts: List[str], model_id: str) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        # First, try calling the function with a list of texts
        try:
            result = self._embedding_fn(texts, model_id)
            # Handle both synchronous and async functions
            if asyncio.iscoroutine(result):
                result = await result
            # If result is a nested list (batch result), return it
            if isinstance(result, list) and all(isinstance(emb, list) for emb in result):
                return result
        except Exception as e:
            logger.warning(f"Batch embedding with model {model_id} failed: {str(e)}")
        
        # If batch embedding failed, fall back to individual embedding
        embeddings = []
        for text in texts:
            embedding = await self._generate_embedding(text, model_id)
            embeddings.append(embedding)
            
        return embeddings
    
    async def clear_cache(self) -> int:
        """
        Clear the embedding cache.
        
        Returns:
            Number of cache files removed
        """
        if not self.config.enabled:
            return 0
            
        cache_dir = Path(self.config.cache_dir)
        count = 0
        
        for cache_file in cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {str(e)}")
                
        return count
    
    @property
    def cache_size(self) -> int:
        """Get the number of cached embeddings."""
        if not self.config.enabled:
            return 0
            
        cache_dir = Path(self.config.cache_dir)
        return len(list(cache_dir.glob("*.json")))
        
    @property
    def dimension(self) -> Optional[int]:
        """
        Get the dimension of the embedding vectors.
        
        This is a helper to access the default dimension from config,
        but actual dimensions depend on the embedding model.
        """
        return getattr(self.config, "dimension", None) 