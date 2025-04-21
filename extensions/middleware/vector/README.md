# Weaviate Vector Integration for BeeAI

This directory contains the implementation of Weaviate vector database integration for the BeeAI framework, providing high-performance vector search capabilities for context retrieval and memory storage.

## Features

- Support for multiple deployment options:
  - Embedded Weaviate (development/testing)
  - Docker-based Weaviate (local deployment)
  - Cloud-hosted Weaviate (production)
- Unified API through abstraction layer
- Automatic schema creation
- Batch operations for efficient ingestion
- Vector and hybrid search capabilities
- Metadata filtering

## Setup Options

### Option 1: Embedded Weaviate (Recommended for Development)

Embedded Weaviate runs directly in your Python process without requiring Docker, ideal for development and testing:

```python
from weaviate_provider_factory import WeaviateProviderFactory, WeaviateDeploymentType

# Create embedded provider
provider = WeaviateProviderFactory.create(
    deployment_type=WeaviateDeploymentType.EMBEDDED,
    config={
        "class_name": "Memory",
        "persistence_data_path": "./weaviate-data"  # Local path for data storage
    }
)
```

### Option 2: Docker-based Weaviate (Local Deployment)

For a more production-like setup while still running locally:

1. Start the Weaviate Docker container:
   ```bash
   # Using the provided scripts
   .\start_weaviate.bat  # Windows
   ./start_weaviate.sh   # Linux/Mac
   ```

2. In your code:
   ```python
   from weaviate_provider_factory import WeaviateProviderFactory, WeaviateDeploymentType

   # Create Docker-based provider (default)
   provider = WeaviateProviderFactory.create(
       deployment_type=WeaviateDeploymentType.DOCKER,
       config={
           "host": "localhost",
           "port": "8080",
           "class_name": "Memory"
       }
   )
   ```

### Option 3: Cloud-hosted Weaviate (Production)

For production deployments, use Weaviate Cloud or your own hosted Weaviate instance:

```python
from weaviate_provider_factory import WeaviateProviderFactory, WeaviateDeploymentType

# Create cloud provider
provider = WeaviateProviderFactory.create(
    deployment_type=WeaviateDeploymentType.CLOUD,
    config={
        "url": "https://your-cluster.weaviate.cloud",  # Your Weaviate Cloud URL
        "api_key": "your-api-key",                     # Your API key
        "class_name": "Memory"
    }
)
```

## Usage Example

```python
# Import the factory
from weaviate_provider_factory import WeaviateProviderFactory, WeaviateDeploymentType

# Create provider (defaults to Docker if not specified)
provider = WeaviateProviderFactory.create(
    config={"class_name": "Memory"}
)

# Ensure schema exists
provider.ensure_schema_exists()

# Add memories
memory_ids = provider.add_memories(
    contents=["This is a test memory", "Another test memory"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],  # Your embeddings here
    metadatas=[{"source": "test"}, {"source": "test"}]
)

# Search by vector
results = provider.search_by_vector(
    query_vector=[0.2, 0.3, ...],  # Your query embedding
    limit=5
)

# Search with metadata filter
filtered_results = provider.search_by_vector(
    query_vector=[0.2, 0.3, ...],
    limit=5,
    filters={"metadata_filter": {"path": ["metadata", "source"], "operator": "Equal", "valueText": "test"}}
)
```

## Docker Scripts

Several utility scripts are provided to manage the Docker-based Weaviate instance:

- `start_weaviate.bat` / `start_weaviate.sh`: Start the Weaviate container
- `stop_weaviate.bat` / `stop_weaviate.sh`: Stop the Weaviate container
- `restart_weaviate.bat`: Restart the Weaviate container
- `setup_weaviate.bat`: Install dependencies and set up Weaviate

For systems without Docker Compose, alternative versions are provided:
- `start_weaviate_alternative.bat`
- `stop_weaviate_alternative.bat`
- `restart_weaviate_alternative.bat`
- `setup_weaviate_alternative.bat`

## Environment Variables

The following environment variables can be used to configure the Weaviate provider:

- `WEAVIATE_DEPLOYMENT_TYPE`: Set to "embedded", "docker", or "cloud"
- `WEAVIATE_HOST`: Hostname for Docker deployment (default: "localhost")
- `WEAVIATE_PORT`: Port for Docker deployment (default: "8080") 
- `WEAVIATE_CLOUD_URL`: URL for cloud deployment
- `WEAVIATE_API_KEY`: API key for cloud deployment
- `WEAVIATE_PERSISTENCE_PATH`: Data storage path for embedded deployment

## Running the Demo

A demonstration script is provided to showcase the abstraction layer:

```bash
# Run with embedded Weaviate (default)
python weaviate_example.py embedded

# Run with Docker-based Weaviate
python weaviate_example.py docker

# Run with cloud-hosted Weaviate
python weaviate_example.py cloud
```

## Dependencies

Required:
- weaviate-client>=3.26.0
- numpy>=1.24.0

Optional:
- sentence-transformers>=2.2.0 (for generating embeddings)
- openai>=1.3.0 (for OpenAI embeddings)
- cohere>=4.32 (for Cohere embeddings)

## Testing

Run the test suite:

```bash
pytest -xvs test_weaviate.py
``` 