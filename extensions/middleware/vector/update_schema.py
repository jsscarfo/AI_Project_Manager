#!/usr/bin/env python
"""
Weaviate Schema Update Script.

This script updates or creates the Weaviate schema for the BeeAI Framework.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import modules
current_dir = str(Path(__file__).parent.absolute())
sys.path.insert(0, current_dir)

try:
    # Try to import the WeaviateProviderFactory
    from weaviate_standalone_test import WeaviateProviderFactory, WeaviateDeploymentType
    logger.info("Using WeaviateProviderFactory for provider creation")
except ImportError:
    logger.error("Could not import WeaviateProviderFactory, please check your installation")
    sys.exit(1)

# Default schema definition
DEFAULT_SCHEMA = {
    "classes": [
        {
            "class": "Memory",
            "description": "A memory chunk with vector embedding for semantic search",
            "vectorizer": "none",  # We provide our own vectors
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "skip": False,
                "ef": 100,
                "efConstruction": 128,
                "maxConnections": 64,
                "dynamicEfMin": 100,
                "dynamicEfMax": 500,
                "dynamicEfFactor": 8,
                "vectorCacheMaxObjects": 1000000
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The text content of the memory"
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata about the memory",
                    "nestedProperties": [
                        {
                            "name": "source",
                            "dataType": ["text"],
                            "description": "Source of the memory"
                        },
                        {
                            "name": "category",
                            "dataType": ["text"],
                            "description": "Category of the memory",
                            "indexFilterable": True,
                            "indexSearchable": True
                        },
                        {
                            "name": "timestamp",
                            "dataType": ["date"],
                            "description": "When the memory was created",
                            "indexFilterable": True
                        },
                        {
                            "name": "priority",
                            "dataType": ["text"],
                            "description": "Priority level of the memory",
                            "indexFilterable": True
                        },
                        {
                            "name": "tags",
                            "dataType": ["text[]"],
                            "description": "Tags associated with the memory",
                            "indexFilterable": True,
                            "indexSearchable": True
                        }
                    ]
                }
            ]
        }
    ]
}

def get_schema_from_file(schema_file):
    """
    Read schema definition from a file.
    
    Args:
        schema_file: Path to the schema JSON file
        
    Returns:
        Schema definition as a dictionary or None if file doesn't exist
    """
    import json
    
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading schema file: {e}")
        return None

def update_schema(provider, schema):
    """
    Update the Weaviate schema using the provided definition.
    
    Args:
        provider: Weaviate provider
        schema: Schema definition
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the client from the provider
        client = getattr(provider, "client", None)
        
        if not client:
            logger.error("Provider does not have a client attribute")
            return False
            
        # First check if we have a schema with classes that exist
        existing_schema = client.schema.get()
        existing_classes = [c['class'] for c in existing_schema['classes']] if existing_schema.get('classes') else []
        
        # Process each class in the schema
        for class_def in schema.get('classes', []):
            class_name = class_def.get('class')
            
            if class_name in existing_classes:
                logger.info(f"Class {class_name} already exists. Updating...")
                
                # Delete existing class if the force flag is set
                if args.force:
                    logger.warning(f"Force flag set. Deleting class {class_name}")
                    client.schema.delete_class(class_name)
                    
                    # Create the class again with v4 syntax
                    client.schema.create_classes([class_def])
                else:
                    # Try to update properties if possible
                    # Note: Weaviate has limited schema update capabilities
                    logger.info(f"Skipping full update for {class_name}. Use --force to recreate")
            else:
                logger.info(f"Creating new class {class_name}")
                # Create class with v4 syntax
                client.schema.create_classes([class_def])
        
        logger.info("Schema update completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error updating schema: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Update Weaviate schema")
    
    # Schema options
    parser.add_argument("--schema-file", type=str, help="Path to schema definition JSON file")
    parser.add_argument("--force", action="store_true", help="Force recreate existing classes")
    
    # Weaviate connection options
    parser.add_argument("--deployment", type=str, choices=["local", "embedded", "cloud"], default="local",
                      help="Weaviate deployment type")
    parser.add_argument("--url", type=str, help="Weaviate URL for cloud deployment")
    parser.add_argument("--api-key", type=str, help="API key for cloud deployment")
    
    global args
    args = parser.parse_args()
    
    # Validate cloud arguments
    if args.deployment == "cloud" and (not args.url or not args.api_key):
        logger.error("--url and --api-key are required for cloud deployment")
        sys.exit(1)
    
    # Get schema definition
    if args.schema_file:
        schema = get_schema_from_file(args.schema_file)
        if not schema:
            logger.error(f"Could not read schema from {args.schema_file}")
            sys.exit(1)
    else:
        logger.info("Using default schema definition")
        schema = DEFAULT_SCHEMA
    
    # Create provider based on deployment type
    deployment_type_map = {
        "local": WeaviateDeploymentType.DOCKER,
        "embedded": WeaviateDeploymentType.EMBEDDED,
        "cloud": WeaviateDeploymentType.CLOUD
    }
    
    deployment_type = deployment_type_map[args.deployment]
    
    # Prepare config
    config = {
        "class_name": "Memory",  # Default class name
        "vector_dimensions": 384  # Default dimension
    }
    
    if args.deployment == "cloud":
        config.update({
            "url": args.url,
            "api_key": args.api_key
        })
    
    # Create provider
    provider = WeaviateProviderFactory.create(
        deployment_type=deployment_type,
        config=config
    )
    
    # Update schema
    success = update_schema(provider, schema)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 