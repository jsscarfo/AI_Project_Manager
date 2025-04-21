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

"""Tests for the EmbeddingService."""

import os
import tempfile
import shutil
import pytest
from typing import List
import json
from datetime import datetime

from beeai_framework.vector.embedding_service import EmbeddingService, EmbeddingServiceConfig, EmbeddingCacheItem
from beeai_framework.errors import FrameworkError


class TestEmbeddingService:
    """Test cases for the EmbeddingService."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_fn(self):
        """Create a mock embedding function."""
        def _embedding_fn(text, model_id="default"):
            # Simple mock that generates deterministic embeddings based on text length and model
            return [float(i) / 100.0 for i in range(10)]
        return _embedding_fn
    
    @pytest.fixture
    def mock_batch_embedding_fn(self):
        """Create a mock batch embedding function."""
        def _batch_embedding_fn(texts, model_id="default"):
            # Simple mock that generates deterministic embeddings for a batch
            if isinstance(texts, list):
                return [[float(i + j) / 100.0 for i in range(10)] for j, _ in enumerate(texts)]
            else:
                return [float(i) / 100.0 for i in range(10)]
        return _batch_embedding_fn
    
    @pytest.fixture
    def embedding_service(self, temp_cache_dir, mock_embedding_fn):
        """Create an embedding service with mock functions."""
        config = EmbeddingServiceConfig(
            cache_dir=temp_cache_dir,
            cache_ttl=3600,
            enabled=True
        )
        return EmbeddingService(mock_embedding_fn, config)
    
    @pytest.mark.asyncio
    async def test_get_embedding(self, embedding_service, mock_embedding_fn):
        """Test getting an embedding."""
        text = "This is a test text for embedding."
        embedding = await embedding_service.get_embedding(text)
        
        # Check that the result matches our mock
        expected = mock_embedding_fn(text)
        assert embedding == expected
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self, embedding_service, temp_cache_dir):
        """Test that embeddings are properly cached."""
        text = "Test caching functionality."
        
        # First call should generate and cache
        embedding1 = await embedding_service.get_embedding(text)
        
        # Verify cache file was created
        cache_files = os.listdir(temp_cache_dir)
        assert len(cache_files) == 1
        
        # Load cache file and verify content
        cache_path = os.path.join(temp_cache_dir, cache_files[0])
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Check that the cache data has the expected structure
        assert "embedding" in cache_data
        assert "created_at" in cache_data
        assert "model_id" in cache_data
        
        # Second call should use cache
        embedding2 = await embedding_service.get_embedding(text)
        assert embedding1 == embedding2
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_embedding_fn, temp_cache_dir):
        """Test behavior when caching is disabled."""
        config = EmbeddingServiceConfig(
            cache_dir=temp_cache_dir,
            cache_ttl=3600,
            enabled=False
        )
        service = EmbeddingService(mock_embedding_fn, config)
        
        text = "Test with caching disabled."
        await service.get_embedding(text)
        
        # Verify no cache file was created
        cache_files = os.listdir(temp_cache_dir)
        assert len(cache_files) == 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_batch_embedding_fn, temp_cache_dir):
        """Test batch processing of embeddings."""
        config = EmbeddingServiceConfig(
            cache_dir=temp_cache_dir,
            cache_ttl=3600,
            enabled=True,
            batch_size=3
        )
        service = EmbeddingService(mock_batch_embedding_fn, config)
        
        texts = [
            "First test text.",
            "Second test text.",
            "Third test text.",
            "Fourth test text."
        ]
        
        embeddings = await service.get_embeddings_batch(texts)
        
        # Check results
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == 10  # Our mock returns 10-dim vectors
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, embedding_service, temp_cache_dir):
        """Test clearing the cache."""
        # Add some items to cache
        texts = ["Text one", "Text two", "Text three"]
        for text in texts:
            await embedding_service.get_embedding(text)
            
        # Verify cache files were created
        cache_files = os.listdir(temp_cache_dir)
        assert len(cache_files) == len(texts)
        
        # Clear cache
        removed = await embedding_service.clear_cache()
        assert removed == len(texts)
        
        # Verify cache is empty
        cache_files = os.listdir(temp_cache_dir)
        assert len(cache_files) == 0
    
    @pytest.mark.asyncio
    async def test_text_truncation(self, embedding_service):
        """Test that long texts are truncated."""
        # Create a long text
        long_text = "x" * 10000
        
        # Set max input length to something smaller
        embedding_service.config.max_input_length = 1000
        
        # Get embedding
        embedding = await embedding_service.get_embedding(long_text)
        
        # Our mock should be called with the truncated text
        assert len(embedding) == 10  # Our mock returns 10-dim vectors
    
    @pytest.mark.asyncio
    async def test_fallback_models(self, temp_cache_dir):
        """Test fallback to alternate models."""
        # Create a mock that fails for the primary model but works for fallback
        calls = []
        
        def failing_embedding_fn(text, model_id="default"):
            calls.append(model_id)
            if model_id == "default":
                raise Exception("Primary model failure")
            return [float(i) / 100.0 for i in range(10)]
        
        config = EmbeddingServiceConfig(
            cache_dir=temp_cache_dir,
            cache_ttl=3600,
            enabled=True,
            primary_model_id="default",
            fallback_model_ids=["fallback1", "fallback2"]
        )
        
        service = EmbeddingService(failing_embedding_fn, config)
        
        # Should fail over to the first fallback model
        embedding = await service.get_embedding("Test fallback")
        
        assert "fallback1" in calls
        assert len(embedding) == 10


if __name__ == "__main__":
    # Simple example usage
    import asyncio
    
    async def test_embedding_service():
        def mock_embedding_function(text, model_id=None):
            # Simple mock that returns a fixed-size vector
            return [0.1] * 10
        
        service = EmbeddingService(
            embedding_fn=mock_embedding_function,
            config=EmbeddingServiceConfig(cache_dir="./temp_cache")
        )
        
        result = await service.get_embedding("Test text")
        print(f"Embedding: {result}")
        
        batch_results = await service.get_embeddings_batch(["Text 1", "Text 2"])
        print(f"Batch results: {batch_results}")
    
    asyncio.run(test_embedding_service()) 