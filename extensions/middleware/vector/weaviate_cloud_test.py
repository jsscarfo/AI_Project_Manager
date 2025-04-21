"""
Standalone test script for the Weaviate Cloud provider.

This script tests connectivity and operations against a Weaviate Cloud instance.
"""
import os
import sys
import json
import logging
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import modules
current_dir = str(Path(__file__).parent.absolute())
sys.path.insert(0, current_dir)
print(f"Added {current_dir} to sys.path")

# Import the factory from the standalone test file
from weaviate_standalone_test import WeaviateProviderFactory, WeaviateDeploymentType

# Cloud instance credentials
CLOUD_CONFIG = {
    "url": "jingwenlong-l1sjqf4g.weaviate.network",
    "grpc_url": "jingwenlong-l1sjqf4g.weaviate.network:443",
    "api_key": "rY78JgCBfH3bFLsVcTalwJtm6QbO6lwgLhJc",
    "class_name": "CloudMemory",
    "vector_dimensions": 384
}

# Test data
TEST_CONTENTS = [
    "This is a test memory for Weaviate Cloud.",
    "Cloud instances offer scalability and managed infrastructure.",
    "Testing vector search capabilities in a production environment."
]

# Create sample embeddings (simplified for the example)
TEST_EMBEDDINGS = [
    [0.1, 0.2, 0.3] + [0.0] * 381,  # Simplified 384-dim vector
    [0.2, 0.3, 0.4] + [0.0] * 381,  # Simplified 384-dim vector
    [0.3, 0.4, 0.5] + [0.0] * 381,  # Simplified 384-dim vector
]

TEST_METADATA = [
    {"source": "cloud_test", "category": "general", "priority": "high"},
    {"source": "cloud_test", "category": "technology", "priority": "medium"},
    {"source": "cloud_test", "category": "development", "priority": "low"}
]


def run_cloud_test():
    """Run a demonstration of the Weaviate Cloud provider"""
    logger.info("Starting Weaviate Cloud test...")
    
    try:
        # Create a cloud provider
        provider = WeaviateProviderFactory.create(
            deployment_type=WeaviateDeploymentType.CLOUD,
            config=CLOUD_CONFIG
        )
        
        # Ensure the schema exists
        logger.info("Initializing schema...")
        provider.ensure_schema_exists()
        
        # Add test data
        logger.info("Adding test memories...")
        memory_ids = provider.add_memories(
            contents=TEST_CONTENTS,
            embeddings=TEST_EMBEDDINGS,
            metadatas=TEST_METADATA
        )
        
        logger.info(f"Added {len(memory_ids)} memories with IDs: {memory_ids}")
        
        # Retrieve a memory
        logger.info("Retrieving a memory...")
        memory = provider.get_memory(memory_ids[0])
        logger.info(f"Retrieved memory content: {memory['content']}")
        
        # Perform a vector search
        logger.info("Performing vector search...")
        query_embedding = [0.15, 0.25, 0.35] + [0.0] * 381  # Simplified query vector
        search_results = provider.search_by_vector(
            query_vector=query_embedding,
            limit=3
        )
        
        logger.info(f"Found {len(search_results)} results")
        for i, result in enumerate(search_results):
            logger.info(f"Result {i+1}: {result['content']}")
        
        # Search with a filter
        logger.info("Performing filtered search for 'technology' category...")
        filtered_results = provider.search_by_vector(
            query_vector=query_embedding,
            limit=3,
            filters={
                "metadata_filter": {
                    "path": ["metadata", "category"],
                    "operator": "Equal",
                    "valueText": "technology"
                }
            }
        )
        
        logger.info(f"Found {len(filtered_results)} filtered results")
        for i, result in enumerate(filtered_results):
            logger.info(f"Filtered Result {i+1}: {result['content']}")
        
        # Test deletion
        logger.info("Testing memory deletion...")
        provider.delete_memory(memory_ids[0])
        
        try:
            provider.get_memory(memory_ids[0])
            logger.error("Memory still exists after deletion!")
        except Exception as e:
            logger.info(f"Successfully deleted memory: {e}")
        
        # Clean up
        logger.info("Cleaning up...")
        provider.clear_collection()
        
        logger.info("Cloud test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during cloud test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_cloud_test()
    sys.exit(0 if success else 1) 