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

"""Tests for the WeaviateProvider."""

import os
import pytest
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock, AsyncMock
import uuid

from beeai_framework.vector.weaviate_provider import WeaviateProvider, WeaviateProviderConfig
from beeai_framework.vector.embedding_service import EmbeddingService
from beeai_framework.vector.base import ContextMetadata, ContextResult
from beeai_framework.errors import FrameworkError


class MockWeaviateClient:
    """Mock Weaviate client for testing."""
    
    def __init__(self):
        self.schema = MagicMock()
        self.data_object = MagicMock()
        self.batch = MagicMock()
        self.query = MagicMock()
        
        # Setup mocked methods
        self.is_ready = MagicMock(return_value=True)
        self.schema.get = MagicMock(return_value={"classes": []})
        self.schema.create_class = MagicMock()
        
        self.query.get = MagicMock()
        self.query.get.return_value.with_near_vector = MagicMock()
        self.query.get.return_value.with_near_vector.return_value.with_limit = MagicMock()
        self.query.get.return_value.with_near_vector.return_value.with_limit.return_value.do = MagicMock()
        
        self.data_object.create = MagicMock()
        
        # Setup batch
        self.batch.delete_objects = MagicMock(return_value={"results": {"successful": 5}})
        
        # Make batch context manager work
        self.batch.__enter__ = MagicMock(return_value=self.batch)
        self.batch.__exit__ = MagicMock(return_value=None)
        self.batch.add_data_object = MagicMock()
        
        # Setup raw query
        self.query.raw = MagicMock()
        self.query.raw.return_value = {
            "data": {
                "Aggregate": {
                    "ContextMemory": [
                        {
                            "meta": {"count": 5},
                            "level": [{"value": "project", "count": 3}, {"value": "domain", "count": 2}],
                            "category": [{"value": "code_snippet", "count": 2}, {"value": "documentation", "count": 3}],
                            "source": [{"value": "user", "count": 3}, {"value": "system", "count": 2}]
                        }
                    ]
                }
            }
        }


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock = MagicMock(spec=EmbeddingService)
    mock.get_embedding = AsyncMock(return_value=[0.1] * 10)
    return mock


@pytest.fixture
def mock_weaviate():
    """Patch the Weaviate client import and return a mock."""
    with patch('beeai_framework.vector.weaviate_provider.weaviate') as mock_weaviate:
        mock_weaviate.Client.return_value = MockWeaviateClient()
        yield mock_weaviate


@pytest.fixture
def weaviate_provider(mock_weaviate, mock_embedding_service):
    """Create a WeaviateProvider with mocks."""
    config = WeaviateProviderConfig(
        host="localhost",
        port=8080,
        class_name="ContextMemory",
        dimension=10
    )
    provider = WeaviateProvider(config, mock_embedding_service)
    return provider


class TestWeaviateProvider:
    """Test cases for the WeaviateProvider."""

    @pytest.mark.asyncio
    async def test_initialization(self, weaviate_provider, mock_weaviate):
        """Test initialization and schema creation."""
        await weaviate_provider.initialize()
        
        # Check if client was created
        mock_weaviate.Client.assert_called_once()
        
        # Check if schema was created
        weaviate_provider.client.schema.create_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_context(self, weaviate_provider, mock_embedding_service):
        """Test adding context."""
        await weaviate_provider.initialize()
        
        # Define test data
        content = "Test content for vector storage"
        metadata = ContextMetadata(
            source="test",
            category="unittest",
            level="project"
        )
        
        # Add context
        result_id = await weaviate_provider.add_context(content, metadata)
        
        # Verify embedding was generated
        mock_embedding_service.get_embedding.assert_called_once_with(content)
        
        # Verify object was created
        weaviate_provider.client.data_object.create.assert_called_once()
        
        # Check returned ID format
        assert isinstance(result_id, str)
        # Should be a valid UUID
        uuid.UUID(result_id)
    
    @pytest.mark.asyncio
    async def test_get_context(self, weaviate_provider, mock_embedding_service):
        """Test retrieving context."""
        await weaviate_provider.initialize()
        
        # Set up mock response
        query_result = {
            "data": {
                "Get": {
                    "ContextMemory": [
                        {
                            "content": "Test content",
                            "source": "test",
                            "category": "unittest",
                            "level": "project",
                            "importance": 0.8,
                            "timestamp": "2023-01-01T00:00:00",
                            "_additional": {"certainty": 0.95}
                        }
                    ]
                }
            }
        }
        weaviate_provider.client.query.get.return_value.with_near_vector.return_value.with_limit.return_value.do.return_value = query_result
        
        # Test query
        query_text = "Find test content"
        result = await weaviate_provider.get_context(query_text)
        
        # Verify embedding was generated for query
        mock_embedding_service.get_embedding.assert_called_with(query_text)
        
        # Verify query was made
        weaviate_provider.client.query.get.assert_called_once()
        
        # Check result
        assert len(result) == 1
        assert isinstance(result[0], ContextResult)
        assert result[0].content == "Test content"
        assert result[0].metadata.source == "test"
        assert result[0].metadata.level == "project"
        assert result[0].score == 0.95
    
    @pytest.mark.asyncio
    async def test_add_contexts_batch(self, weaviate_provider, mock_embedding_service):
        """Test batch adding of contexts."""
        await weaviate_provider.initialize()
        
        # Define test data
        contexts = [
            {
                "content": "Batch item 1",
                "metadata": {
                    "source": "test",
                    "category": "unittest",
                    "level": "project"
                }
            },
            {
                "content": "Batch item 2",
                "metadata": {
                    "source": "test",
                    "category": "documentation",
                    "level": "domain"
                }
            }
        ]
        
        # Add batch
        result_ids = await weaviate_provider.add_contexts_batch(contexts)
        
        # Verify batch was used
        assert weaviate_provider.client.batch.add_data_object.call_count == 2
        
        # Check results
        assert len(result_ids) == 2
        for id in result_ids:
            # Should be valid UUIDs
            uuid.UUID(id)
    
    @pytest.mark.asyncio
    async def test_clear_context(self, weaviate_provider):
        """Test clearing context."""
        await weaviate_provider.initialize()
        
        # Clear with no filters
        count = await weaviate_provider.clear_context()
        
        # Verify delete was called
        weaviate_provider.client.batch.delete_objects.assert_called_once()
        
        # Check count
        assert count == 5  # From our mock
        
        # Test with filter
        filter_metadata = {"level": "project"}
        await weaviate_provider.clear_context(filter_metadata)
        
        # Should be called again with filter
        assert weaviate_provider.client.batch.delete_objects.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_stats(self, weaviate_provider):
        """Test getting stats."""
        await weaviate_provider.initialize()
        
        stats = await weaviate_provider.get_stats()
        
        # Verify raw query was used
        assert weaviate_provider.client.query.raw.call_count >= 1
        
        # Check stats structure
        assert "total_count" in stats
        assert stats["total_count"] == 5
        assert "by_level" in stats
        assert "project" in stats["by_level"]
        assert stats["by_level"]["project"] == 3
        assert "by_category" in stats
        assert "by_source" in stats
    
    @pytest.mark.asyncio
    async def test_metadata_filters(self, weaviate_provider):
        """Test conversion of metadata filters."""
        await weaviate_provider.initialize()
        
        # Test simple filter
        simple_filter = {"level": "project"}
        simple_result = weaviate_provider._prepare_metadata_filters(simple_filter)
        assert simple_result["operator"] == "Equal"
        assert simple_result["path"] == ["level"]
        assert simple_result["valueText"] == "project"
        
        # Test complex filter
        complex_filter = {"level": {"$in": ["domain", "techstack"]}}
        complex_result = weaviate_provider._prepare_metadata_filters(complex_filter)
        assert complex_result["operator"] == "Or"
        assert "operands" in complex_result
        assert len(complex_result["operands"]) == 2


if __name__ == "__main__":
    # Simple manual test
    import asyncio
    
    async def run_test():
        # This would require a real Weaviate instance
        print("This test requires a running Weaviate instance")
        print("Please run the unit tests with pytest instead")
    
    asyncio.run(run_test()) 