"""Tests for the Weaviate Provider."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import os
import time
import numpy as np

from v5.extensions.middleware.vector.weaviate_provider import (
    WeaviateProvider,
    WeaviateProviderConfig
)


class MockWeaviateClient:
    """Mock Weaviate client for testing."""
    
    def __init__(self):
        self.schema = MagicMock()
        self.schema.exists.return_value = False
        self.schema.create_class = MagicMock()
        
        self.data_object = MagicMock()
        self.data_object.create.return_value = "test-uuid"
        
        self.query = MagicMock()
        self.get_result = MagicMock()
        self.get_result.do.return_value = {
            "data": {
                "Get": {
                    "Memory": [
                        {
                            "content": "Test content 1",
                            "source": "test",
                            "category": "general",
                            "created": time.time(),
                            "metadata": {"key": "value"},
                            "_additional": {
                                "id": "test-id-1",
                                "certainty": 0.95
                            }
                        },
                        {
                            "content": "Test content 2",
                            "source": "test",
                            "category": "general",
                            "created": time.time(),
                            "metadata": {"key": "value2"},
                            "_additional": {
                                "id": "test-id-2",
                                "certainty": 0.85
                            }
                        }
                    ]
                }
            }
        }
        
        # Setup query builder chain
        self.query.get.return_value = self.get_result
        self.get_result.with_near_vector.return_value = self.get_result
        self.get_result.with_limit.return_value = self.get_result
        self.get_result.with_additional.return_value = self.get_result
        self.get_result.with_where.return_value = self.get_result
        
        # Cluster status
        self.cluster = MagicMock()
        self.cluster.get_nodes_status.return_value = {"nodes": [{"status": "HEALTHY"}]}


@pytest.fixture
def mock_weaviate():
    """Create a mock weaviate module."""
    mock_module = MagicMock()
    mock_module.Client.return_value = MockWeaviateClient()
    mock_module.auth = MagicMock()
    mock_module.auth.AuthApiKey = MagicMock(return_value="mock-auth")
    return mock_module


@pytest.fixture
def weaviate_provider(mock_weaviate):
    """Create a weaviate provider with mocked client."""
    with patch.dict('sys.modules', {'weaviate': mock_weaviate}):
        # Force weaviate availability to be true
        provider = WeaviateProvider()
        provider._weaviate_available = True
        return provider


@pytest.mark.asyncio
async def test_initialization(weaviate_provider, mock_weaviate):
    """Test initialization of weaviate provider."""
    # Initialize the provider
    await weaviate_provider.initialize()
    
    # Check that the client was created
    assert weaviate_provider._client is not None
    
    # Check that schema creation was attempted
    mock_client = weaviate_provider._client
    mock_client.schema.exists.assert_called_once_with(weaviate_provider.config.default_collection)
    mock_client.schema.create_class.assert_called_once()


@pytest.mark.asyncio
async def test_get_context(weaviate_provider):
    """Test retrieving context from weaviate."""
    # Initialize the provider
    await weaviate_provider.initialize()
    
    # Mock embedding generation
    with patch.object(weaviate_provider, '_generate_embedding', 
                     return_value=np.random.randn(384).tolist()):
        # Get context
        results = await weaviate_provider.get_context("test query")
        
        # Check results
        assert len(results) == 2
        assert results[0]["content"] == "Test content 1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"]["id"] == "test-id-1"
        assert results[1]["content"] == "Test content 2"


@pytest.mark.asyncio
async def test_add_context(weaviate_provider):
    """Test adding context to weaviate."""
    # Initialize the provider
    await weaviate_provider.initialize()
    
    # Mock embedding generation
    with patch.object(weaviate_provider, '_generate_embedding', 
                     return_value=np.random.randn(384).tolist()):
        # Add context
        result_id = await weaviate_provider.add_context(
            "Test content", 
            {"source": "test", "category": "general", "key": "value"}
        )
        
        # Check result
        assert result_id == "test-uuid"
        
        # Check that create was called correctly
        mock_client = weaviate_provider._client
        mock_client.data_object.create.assert_called_once()
        
        # Check call arguments
        args, kwargs = mock_client.data_object.create.call_args
        assert kwargs["class_name"] == weaviate_provider.config.default_collection
        assert kwargs["data_object"]["content"] == "Test content"
        assert kwargs["data_object"]["source"] == "test"
        assert kwargs["data_object"]["category"] == "general"
        assert "vector" in kwargs


@pytest.mark.asyncio
async def test_filter_building(weaviate_provider):
    """Test building filters for weaviate queries."""
    # Initialize the provider
    await weaviate_provider.initialize()
    
    # Test simple filter
    metadata = {"source": "test", "category": "general"}
    filter_expr = weaviate_provider._build_filter(metadata)
    
    # Check filter structure
    assert filter_expr["operator"] == "And"
    assert len(filter_expr["operands"]) == 2
    
    # Check source filter
    source_filter = filter_expr["operands"][0]
    assert source_filter["path"] == ["source"]
    assert source_filter["operator"] == "Equal"
    assert source_filter["valueString"] == "test"
    
    # Test complex filter
    metadata = {
        "filters": [
            {"field": "created", "operator": "gt", "value": 1000},
            {"field": "key", "operator": "eq", "value": "value"},
            {"field": "flag", "operator": "eq", "value": True}
        ]
    }
    filter_expr = weaviate_provider._build_filter(metadata)
    
    # Check complex filter structure
    assert filter_expr["operator"] == "And"
    assert len(filter_expr["operands"]) == 3
    
    # Check numeric filter
    num_filter = filter_expr["operands"][0]
    assert num_filter["path"] == ["created"]
    assert num_filter["operator"] == "GreaterThan"
    assert num_filter["valueNumber"] == 1000
    
    # Check string filter
    str_filter = filter_expr["operands"][1]
    assert str_filter["path"] == ["metadata", "key"]
    assert str_filter["operator"] == "Equal"
    assert str_filter["valueString"] == "value"
    
    # Check boolean filter
    bool_filter = filter_expr["operands"][2]
    assert bool_filter["path"] == ["metadata", "flag"]
    assert bool_filter["operator"] == "Equal"
    assert bool_filter["valueBoolean"] is True


@pytest.mark.asyncio
async def test_empty_query(weaviate_provider):
    """Test behavior with empty query."""
    # Initialize the provider
    await weaviate_provider.initialize()
    
    # Get context with empty query
    results = await weaviate_provider.get_context("")
    
    # Should return empty results without error
    assert results == []


@pytest.mark.asyncio
async def test_embed_fallback(weaviate_provider):
    """Test embedding fallback behavior."""
    # Initialize the provider
    await weaviate_provider.initialize()
    
    # Test actual embedding generation
    embedding = await weaviate_provider._generate_embedding("test")
    
    # Should return a vector of the correct dimension
    assert len(embedding) == weaviate_provider.config.vector_dimension
    
    # Values should be normalized (for cosine similarity)
    embedding_norm = np.linalg.norm(embedding)
    assert abs(embedding_norm - 1.0) < 0.0001


@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client"""
    mock_client = MagicMock()
    
    # Mock schema methods
    mock_client.schema.get.return_value = {"classes": []}
    
    # Mock batch context manager
    mock_batch = MagicMock()
    mock_client.batch.__enter__.return_value = mock_batch
    
    # Mock query builder
    mock_query = MagicMock()
    mock_query.get.return_value = mock_query
    mock_query.with_additional.return_value = mock_query
    mock_query.with_near_vector.return_value = mock_query
    mock_query.with_where.return_value = mock_query
    mock_query.with_limit.return_value = mock_query
    
    mock_client.query = mock_query
    
    return mock_client


@pytest.fixture
def weaviate_provider(mock_weaviate_client):
    """Create a WeaviateProvider with mocked client"""
    with patch('weaviate.Client', return_value=mock_weaviate_client):
        config = WeaviateProviderConfig(
            host="test_host",
            port=8080,
            class_name="TestClass"
        )
        provider = WeaviateProvider(config)
        # Ensure we use the mock
        provider.client = mock_weaviate_client
        provider.schema_created = True
        return provider


@pytest.mark.asyncio
async def test_add_memories(weaviate_provider, mock_weaviate_client):
    """Test adding memories to WeaviateProvider"""
    # Test data
    contents = ["Test memory 1", "Test memory 2"]
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    metadata_list = [
        {"source": "test1", "category": "A"},
        {"source": "test2", "category": "B"}
    ]
    
    # Configure mock
    mock_batch = mock_weaviate_client.batch.__enter__.return_value
    
    # Call method
    ids = await weaviate_provider.add_memories(
        contents=contents,
        embeddings=embeddings,
        metadata_list=metadata_list
    )
    
    # Assertions
    assert len(ids) == 2
    assert mock_batch.add_data_object.call_count == 2
    
    # Check first call args
    args1 = mock_batch.add_data_object.call_args_list[0][1]
    assert args1["data_object"]["content"] == "Test memory 1"
    assert args1["data_object"]["metadata"] == {"source": "test1", "category": "A"}
    assert args1["vector"] == [0.1, 0.2, 0.3]
    assert args1["class_name"] == "TestClass"


@pytest.mark.asyncio
async def test_get_memories(weaviate_provider, mock_weaviate_client):
    """Test retrieving memories by IDs"""
    # Test data
    test_ids = ["id1", "id2"]
    
    # Configure mock
    mock_client = weaviate_provider.client
    mock_client.data_object.get_by_id.side_effect = [
        {
            "properties": {
                "content": "Test content 1",
                "metadata": {"source": "test1"},
                "timestamp": 1625097600
            },
            "vector": [0.1, 0.2, 0.3]
        },
        {
            "properties": {
                "content": "Test content 2",
                "metadata": {"source": "test2"},
                "timestamp": 1625097700
            },
            "vector": [0.4, 0.5, 0.6]
        }
    ]
    
    # Call method
    results = await weaviate_provider.get_memories(test_ids)
    
    # Assertions
    assert len(results) == 2
    assert mock_client.data_object.get_by_id.call_count == 2
    
    assert results[0]["id"] == "id1"
    assert results[0]["content"] == "Test content 1"
    assert results[0]["metadata"] == {"source": "test1"}
    assert results[0]["embedding"] == [0.1, 0.2, 0.3]
    
    assert results[1]["id"] == "id2"
    assert results[1]["content"] == "Test content 2"
    assert results[1]["metadata"] == {"source": "test2"}
    assert results[1]["embedding"] == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_search_memories(weaviate_provider, mock_weaviate_client):
    """Test searching memories by vector similarity"""
    # Test data
    query_embedding = [0.1, 0.2, 0.3]
    
    # Configure mock
    mock_query = weaviate_provider.client.query
    mock_query.do.return_value = {
        "data": {
            "Get": {
                "TestClass": [
                    {
                        "content": "Result 1",
                        "metadata": {"source": "test1"},
                        "timestamp": 1625097600,
                        "_additional": {
                            "id": "result1",
                            "vector": [0.11, 0.21, 0.31],
                            "distance": 0.05
                        }
                    },
                    {
                        "content": "Result 2",
                        "metadata": {"source": "test2"},
                        "timestamp": 1625097700,
                        "_additional": {
                            "id": "result2",
                            "vector": [0.12, 0.22, 0.32],
                            "distance": 0.1
                        }
                    }
                ]
            }
        }
    }
    
    # Call method
    results = await weaviate_provider.search_memories(
        embedding=query_embedding,
        limit=5,
        metadata_filter={"source": "test1"}
    )
    
    # Assertions
    assert len(results) == 2
    assert results[0]["id"] == "result1"
    assert results[0]["content"] == "Result 1"
    assert results[0]["score"] == pytest.approx(0.95)  # 1.0 - 0.05
    
    # Check query construction
    mock_query.get.assert_called_once_with(
        class_name="TestClass",
        properties=["content", "metadata", "timestamp"]
    )
    mock_query.with_near_vector.assert_called_once()
    mock_query.with_limit.assert_called_once_with(5)


@pytest.mark.asyncio
async def test_delete_memories(weaviate_provider, mock_weaviate_client):
    """Test deleting memories by IDs"""
    # Test data
    test_ids = ["id1", "id2"]
    
    # Call method
    deleted_ids = await weaviate_provider.delete_memories(test_ids)
    
    # Assertions
    assert len(deleted_ids) == 2
    assert deleted_ids == test_ids
    assert weaviate_provider.client.data_object.delete.call_count == 2
    
    # Check call args
    weaviate_provider.client.data_object.delete.assert_any_call(
        "id1", class_name="TestClass"
    )
    weaviate_provider.client.data_object.delete.assert_any_call(
        "id2", class_name="TestClass"
    )


@pytest.mark.asyncio
async def test_clear_all(weaviate_provider, mock_weaviate_client):
    """Test clearing all memories"""
    # Call method
    result = await weaviate_provider.clear_all()
    
    # Assertions
    assert result is True
    weaviate_provider.client.schema.delete_class.assert_called_once_with("TestClass")


def test_config_validation():
    """Test configuration validation"""
    # Valid config
    valid_config = WeaviateProviderConfig(
        host="localhost",
        port=8080,
        class_name="TestClass"
    )
    assert valid_config.class_name == "TestClass"
    
    # Invalid class name (should start with uppercase)
    with pytest.raises(ValueError):
        WeaviateProviderConfig(
            host="localhost",
            port=8080,
            class_name="testClass"  # Lowercase first letter
        )


def test_build_where_filter(weaviate_provider):
    """Test building Weaviate where filters from metadata"""
    # Test with string value
    filter1 = weaviate_provider._build_where_filter({"source": "test"})
    assert filter1["path"] == ["metadata", "source"]
    assert filter1["operator"] == "Equal"
    assert filter1["valueString"] == "test"
    
    # Test with numeric value
    filter2 = weaviate_provider._build_where_filter({"priority": 5})
    assert filter2["path"] == ["metadata", "priority"]
    assert filter2["operator"] == "Equal"
    assert filter2["valueNumber"] == 5
    
    # Test with multiple values (AND)
    filter3 = weaviate_provider._build_where_filter({
        "source": "test",
        "priority": 5
    })
    assert filter3["operator"] == "And"
    assert len(filter3["operands"]) == 2


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__]) 