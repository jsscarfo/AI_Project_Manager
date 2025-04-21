# Weaviate Integration Tools

This directory contains tools for working with Weaviate vector database (v4) in the BeeAI Framework.

## Validation and Testing Tools

### Weaviate Control Center

The control center provides a unified interface for all Weaviate operations:

```
Weaviate Control Center
======================

1. Validate Local Weaviate Instance
2. Validate Remote Weaviate Instance
3. Test Embedded Weaviate (no Docker)
4. Test Weaviate Cloud Connection
5. Start Local Docker Instance
6. Stop Local Docker Instance
7. Restart Local Docker Instance
8. Run Batch Import Process
9. Update Weaviate Schema
```

To use it, simply run one of the following depending on your OS:

**Windows**:
```
weaviate_control.bat
```

**Linux/Mac**:
```
chmod +x weaviate_control.sh  # First time only
./weaviate_control.sh
```

And select the option you need.

## Configuration Files

- `docker-compose.weaviate.yml` - Docker Compose configuration for running Weaviate locally
- `requirements.txt` - Python dependencies for the Weaviate integration

## Core Implementation Files

- `weaviate_provider.py` - The main Weaviate provider implementation (using v4 API)
- `vector_provider.py` - The abstract vector provider interface

## Test Scripts

- `weaviate_standalone_test.py` - Tests embedded Weaviate functionality (no Docker needed)
- `weaviate_cloud_test.py` - Tests Weaviate Cloud functionality

## Validation Script

- `validate_weaviate.py` - Validates connections to Weaviate (local, remote, or cloud)

## Batch Scripts

- `batch_import.py` - Import data in batches from files or database
- `update_schema.py` - Update or create Weaviate schema

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for local deployment)
- Required Python packages:
  ```
  pip install -r requirements.txt
  ```

## Usage

1. Start the local Weaviate instance (option 5 in the control center)
2. Validate the connection (option 1)
3. Run your application using the Weaviate provider

## Cloud Integration

To connect to a Weaviate Cloud instance:

1. Sign up for Weaviate Cloud at https://console.weaviate.cloud/
2. Create a new cluster and note the URL and API key
3. Use the "Validate Remote Weaviate Instance" or "Test Weaviate Cloud Connection" options in the tool

## Weaviate v4 Notes

This implementation uses Weaviate Client v4, which has some key differences from v3:

- Client initialization: We use `weaviate.WeaviateClient()` instead of `weaviate.Client()`
- Authentication: API keys are passed as headers rather than using AuthApiKey
- Collection operations: We use `collections.get().data` instead of `data_object`
- Schema operations: We use `schema.create_classes([cls])` instead of `schema.create_class(cls)`
- Search operations: We use `collections.get().query.near_vector()` instead of `query.get().with_near_vector()`

For more details, see the [Weaviate v4 documentation](https://weaviate.io/developers/weaviate/client-libraries/python).

## Troubleshooting

If you encounter issues with the Docker container:
- Check Docker is running
- Try restarting the container using option 7
- Check for port conflicts if port 8080 is already in use 