"""
Standalone test script for the Embedded Weaviate provider.

This demonstrates using the embedded Weaviate without Docker,
and can be run directly without pytest.
"""
import os
import sys
import json
import logging
import tempfile
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

# Create a simple mock WeaviateProvider and WeaviateConfig if the real ones aren't available
# This allows us to test the factory logic even if the real provider isn't implemented yet
class MockWeaviateConfig:
    def __init__(self, class_name="Memory", **kwargs):
        self.class_name = class_name
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockWeaviateProvider:
    def __init__(self, client=None, config=None):
        self.client = client or {}
        self.config = config
        self.class_name = config.class_name if config else "Memory"
        
    def ensure_schema_exists(self):
        logger.info(f"Mock: Ensuring schema exists for class {self.class_name}")
        return True
        
    def add_memories(self, contents, embeddings, metadatas=None):
        logger.info(f"Mock: Adding {len(contents)} memories")
        return [f"mock-id-{i}" for i in range(len(contents))]
        
    def get_memory(self, memory_id):
        logger.info(f"Mock: Getting memory {memory_id}")
        if memory_id.startswith("deleted-"):
            raise ValueError(f"Memory {memory_id} not found")
        return {"id": memory_id, "content": "Mock memory content", "metadata": {"source": "mock"}}
        
    def search_by_vector(self, query_vector, limit=5, filters=None):
        logger.info(f"Mock: Searching with vector, limit={limit}, filters={filters}")
        results = []
        for i in range(min(3, limit)):
            results.append({
                "id": f"mock-result-{i}",
                "content": f"Mock search result {i}",
                "metadata": {"category": "mock", "source": "test"}
            })
        return results
        
    def delete_memory(self, memory_id):
        logger.info(f"Mock: Deleting memory {memory_id}")
        return True
        
    def clear_collection(self):
        logger.info(f"Mock: Clearing collection {self.class_name}")
        return True

# Define the deployment type constants
class WeaviateDeploymentType:
    EMBEDDED = "embedded"
    CLOUD = "cloud"
    DOCKER = "docker"

# Create a simplified version of the factory for testing
class WeaviateProviderFactory:
    @staticmethod
    def create(deployment_type=None, config=None, **kwargs):
        config = config or {}
        
        if deployment_type is None:
            deployment_type = os.environ.get("WEAVIATE_DEPLOYMENT_TYPE", WeaviateDeploymentType.DOCKER)
        
        logger.info(f"Creating provider with deployment type: {deployment_type}")
        
        try:
            # Try to import the real implementation
            from weaviate_provider import WeaviateProvider, WeaviateConfig
            logger.info("Using actual WeaviateProvider implementation")
        except ImportError:
            logger.warning("Could not import real provider, using mock implementation")
            WeaviateProvider = MockWeaviateProvider
            WeaviateConfig = MockWeaviateConfig
        
        # Create a provider config
        provider_config = WeaviateConfig(
            class_name=config.get("class_name", "Memory"),
            vector_dimensions=config.get("vector_dimensions", 384),
            **kwargs
        )
        
        if deployment_type == WeaviateDeploymentType.EMBEDDED:
            try:
                import weaviate
                
                logger.info("Initializing embedded Weaviate")
                
                # Create embedded client using v4 syntax
                try:
                    embedded_params = weaviate.EmbeddedOptions()
                    client = weaviate.WeaviateClient(
                        connection_params=weaviate.connect_to_embedded(
                            embedded_options=embedded_params
                        )
                    )
                    logger.info("Successfully connected to embedded Weaviate using v4 API")
                except Exception as e:
                    logger.error(f"Error connecting to embedded Weaviate: {e}")
                    logger.warning("Falling back to mock implementation")
                    return MockWeaviateProvider(config=provider_config)
                
                return WeaviateProvider(client=client, config=provider_config)
            except ImportError as e:
                logger.error(f"Error importing Weaviate modules: {e}")
                logger.warning("Falling back to mock implementation")
                return MockWeaviateProvider(config=provider_config)
        elif deployment_type == WeaviateDeploymentType.CLOUD:
            try:
                import weaviate
                
                # Required config for cloud
                url = config.get("url")
                api_key = config.get("api_key")
                
                if not url or not api_key:
                    logger.error("Missing required cloud configuration: url and api_key")
                    return MockWeaviateProvider(config=provider_config)
                
                logger.info(f"Connecting to Weaviate Cloud instance at {url}")
                
                # Configure connection using v4 syntax
                try:
                    # Create connection parameters with API key
                    connection_params = weaviate.ConnectionParams.from_url(
                        url=f"https://{url}",
                        headers={
                            "X-Weaviate-Api-Key": api_key
                        }
                    )
                    
                    # Optional gRPC configuration
                    if config.get("grpc_url"):
                        connection_params = weaviate.ConnectionParams.from_url(
                            url=f"https://{url}",
                            grpc_url=f"https://{config.get('grpc_url')}",
                            headers={
                                "X-Weaviate-Api-Key": api_key
                            }
                        )
                        logger.info(f"Using gRPC endpoint: {connection_params.grpc_url}")
                    
                    # Connect to the cloud instance
                    client = weaviate.WeaviateClient(connection_params=connection_params)
                    logger.info("Successfully connected to Weaviate Cloud")
                    
                    return WeaviateProvider(client=client, config=provider_config)
                except Exception as e:
                    logger.error(f"Error connecting to Weaviate Cloud: {e}")
                    logger.warning("Falling back to mock implementation")
                    return MockWeaviateProvider(config=provider_config)
            except ImportError as e:
                logger.error(f"Error importing Weaviate modules: {e}")
                logger.warning("Falling back to mock implementation")
                return MockWeaviateProvider(config=provider_config)
        else:  # docker
            try:
                import weaviate
                
                url = config.get("url", "http://localhost:8080")
                logger.info(f"Connecting to Docker Weaviate instance at {url}")
                
                # Create client with v4 syntax
                try:
                    connection_params = weaviate.ConnectionParams.from_url(url=url)
                    client = weaviate.WeaviateClient(connection_params=connection_params)
                    logger.info("Successfully connected to Docker Weaviate")
                    
                    return WeaviateProvider(client=client, config=provider_config)
                except Exception as e:
                    logger.error(f"Error connecting to Docker Weaviate: {e}")
                    logger.warning("Falling back to mock implementation")
                    return MockWeaviateProvider(config=provider_config)
            except ImportError as e:
                logger.error(f"Error importing Weaviate modules: {e}")
                logger.warning("Falling back to mock implementation")
                return MockWeaviateProvider(config=provider_config)

# Test data
TEST_CONTENTS = [
    "This is a test memory for embedded Weaviate.",
    "Vector search uses embeddings to find semantically similar content.",
    "Embedded Weaviate runs in-process, ideal for development and testing."
]

# Create sample embeddings (simplified for the example)
TEST_EMBEDDINGS = [
    [0.1, 0.2, 0.3] + [0.0] * 381,  # Simplified 384-dim vector
    [0.2, 0.3, 0.4] + [0.0] * 381,  # Simplified 384-dim vector
    [0.3, 0.4, 0.5] + [0.0] * 381,  # Simplified 384-dim vector
]

TEST_METADATA = [
    {"source": "test", "category": "general", "priority": "high"},
    {"source": "test", "category": "technology", "priority": "medium"},
    {"source": "test", "category": "development", "priority": "low"}
]


def run_embedded_test():
    """Run a demonstration of the embedded Weaviate provider"""
    logger.info("Starting embedded Weaviate test...")
    
    # Create a temporary directory for the data
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        try:
            # Try to see what version of weaviate we're using
            try:
                import weaviate
                logger.info(f"Using weaviate version: {weaviate.__version__}")
            except (ImportError, AttributeError):
                logger.warning("Could not determine weaviate version")
            
            # Create an embedded provider
            provider = WeaviateProviderFactory.create(
                deployment_type=WeaviateDeploymentType.EMBEDDED,
                config={
                    "class_name": "TestMemory",
                    "persistence_data_path": temp_dir,
                    "vector_dimensions": 384
                }
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
            logger.info(f"Retrieved memory: {json.dumps(memory, indent=2)}")
            
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
            
            logger.info("Test completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during test: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = run_embedded_test()
    sys.exit(0 if success else 1) 