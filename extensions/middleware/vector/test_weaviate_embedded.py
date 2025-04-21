"""
Test script for the Embedded Weaviate provider.

This test ensures that the WeaviateProviderFactory properly instantiates
an embedded Weaviate instance and operations work correctly without
requiring Docker.
"""
import os
import pytest
import tempfile
import numpy as np
from typing import List, Dict, Any

from weaviate_provider_factory import WeaviateProviderFactory, WeaviateDeploymentType

# Test data
TEST_CONTENTS = [
    "This is a test memory for embedded Weaviate.",
    "Another test memory with different content.",
    "A third memory to retrieve and test with."
]

# Create sample embeddings (normalized random vectors)
def create_test_embeddings(count: int, dim: int = 384) -> List[List[float]]:
    """Create normalized random test embeddings"""
    embeddings = []
    for _ in range(count):
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec.tolist())
    return embeddings

TEST_EMBEDDINGS = create_test_embeddings(len(TEST_CONTENTS))

TEST_METADATA = [
    {"source": "test", "category": "general", "priority": "high"},
    {"source": "test", "category": "specific", "priority": "medium"},
    {"source": "test", "category": "general", "priority": "low"}
]


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for Weaviate data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def embedded_provider(temp_data_dir):
    """Create an embedded Weaviate provider with temporary data directory"""
    provider = WeaviateProviderFactory.create(
        deployment_type=WeaviateDeploymentType.EMBEDDED,
        config={
            "class_name": "TestMemory",
            "persistence_data_path": temp_data_dir,
            "vector_dimensions": 384
        }
    )
    
    # Ensure the schema exists
    provider.ensure_schema_exists()
    
    yield provider
    
    # Clean up after tests
    provider.clear_collection()


def test_embedded_provider_initialization(embedded_provider):
    """Test that the embedded provider initializes correctly"""
    assert embedded_provider is not None
    assert embedded_provider.class_name == "TestMemory"
    assert embedded_provider.client is not None


def test_add_and_retrieve_memories(embedded_provider):
    """Test adding memories and retrieving them"""
    # Add memories
    memory_ids = embedded_provider.add_memories(
        contents=TEST_CONTENTS,
        embeddings=TEST_EMBEDDINGS,
        metadatas=TEST_METADATA
    )
    
    # Verify IDs were returned
    assert len(memory_ids) == len(TEST_CONTENTS)
    
    # Retrieve and verify each memory
    for i, memory_id in enumerate(memory_ids):
        memory = embedded_provider.get_memory(memory_id)
        
        # Check content and metadata
        assert memory["content"] == TEST_CONTENTS[i]
        for key, value in TEST_METADATA[i].items():
            assert memory["metadata"][key] == value


def test_vector_search(embedded_provider):
    """Test vector search functionality"""
    # Add memories
    memory_ids = embedded_provider.add_memories(
        contents=TEST_CONTENTS,
        embeddings=TEST_EMBEDDINGS,
        metadatas=TEST_METADATA
    )
    
    # Use the first embedding as a query vector
    query_vector = TEST_EMBEDDINGS[0]
    
    # Perform search
    results = embedded_provider.search_by_vector(
        query_vector=query_vector,
        limit=3
    )
    
    # Verify results
    assert len(results) > 0
    # The most similar result should be the first one
    assert results[0]["content"] == TEST_CONTENTS[0]


def test_filtered_search(embedded_provider):
    """Test search with metadata filters"""
    # Add memories
    memory_ids = embedded_provider.add_memories(
        contents=TEST_CONTENTS,
        embeddings=TEST_EMBEDDINGS,
        metadatas=TEST_METADATA
    )
    
    # Search with filter for "general" category
    results = embedded_provider.search_by_vector(
        query_vector=TEST_EMBEDDINGS[0],  # Use first embedding as query
        limit=3,
        filters={
            "metadata_filter": {
                "path": ["metadata", "category"],
                "operator": "Equal",
                "valueText": "general"
            }
        }
    )
    
    # Should return only general category memories (2 items)
    assert len(results) == 2
    for result in results:
        assert result["metadata"]["category"] == "general"


def test_memory_deletion(embedded_provider):
    """Test deleting memories"""
    # Add memories
    memory_ids = embedded_provider.add_memories(
        contents=TEST_CONTENTS,
        embeddings=TEST_EMBEDDINGS,
        metadatas=TEST_METADATA
    )
    
    # Delete the first memory
    embedded_provider.delete_memory(memory_ids[0])
    
    # Verify it's gone
    with pytest.raises(Exception):
        embedded_provider.get_memory(memory_ids[0])
    
    # But others should still be there
    assert embedded_provider.get_memory(memory_ids[1]) is not None


def test_collection_clearing(embedded_provider):
    """Test clearing the entire collection"""
    # Add memories
    memory_ids = embedded_provider.add_memories(
        contents=TEST_CONTENTS,
        embeddings=TEST_EMBEDDINGS,
        metadatas=TEST_METADATA
    )
    
    # Clear collection
    embedded_provider.clear_collection()
    
    # Perform search - should return empty results
    results = embedded_provider.search_by_vector(
        query_vector=TEST_EMBEDDINGS[0],
        limit=3
    )
    
    assert len(results) == 0


if __name__ == "__main__":
    # Can be run directly for manual testing
    print("Testing embedded Weaviate provider...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        provider = WeaviateProviderFactory.create(
            deployment_type=WeaviateDeploymentType.EMBEDDED,
            config={
                "class_name": "TestMemory",
                "persistence_data_path": temp_dir,
                "vector_dimensions": 384
            }
        )
        
        provider.ensure_schema_exists()
        
        print("Adding test memories...")
        memory_ids = provider.add_memories(
            contents=TEST_CONTENTS,
            embeddings=TEST_EMBEDDINGS,
            metadatas=TEST_METADATA
        )
        
        print(f"Added {len(memory_ids)} memories")
        
        print("Retrieving memory...")
        memory = provider.get_memory(memory_ids[0])
        print(f"Retrieved: {memory}")
        
        print("Performing vector search...")
        results = provider.search_by_vector(
            query_vector=TEST_EMBEDDINGS[0],
            limit=2
        )
        print(f"Search results: {results}")
        
        print("Cleaning up...")
        provider.clear_collection()
        
        print("Test completed successfully!") 