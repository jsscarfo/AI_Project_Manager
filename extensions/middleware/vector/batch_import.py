#!/usr/bin/env python
"""
Batch Import Script for Weaviate.

This script provides efficient batch import functionality for Weaviate,
supporting import from JSON files or database sources.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

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

# Constants
BATCH_SIZE = 100  # How many objects to import at once
DEFAULT_DIMENSION = 384  # Default embedding dimension

def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        else:
            logger.error("JSON file must contain a list of objects")
            return []
    except Exception as e:
        logger.error(f"Error reading JSON file: {e}")
        return []

def read_from_database() -> List[Dict[str, Any]]:
    """
    Read data from a database. Implement your database connection here.
    
    Returns:
        List of dictionaries containing the data
    """
    # TODO: Implement your database connection logic
    logger.warning("Database import not implemented yet")
    return []

def create_embeddings(texts: List[str], embedding_model: str = None) -> List[List[float]]:
    """
    Create embeddings for the given texts.
    
    Args:
        texts: List of texts to create embeddings for
        embedding_model: Optional model to use for embeddings
        
    Returns:
        List of embeddings as float vectors
    """
    if embedding_model:
        logger.info(f"Creating embeddings using {embedding_model}")
        # TODO: Implement your embedding logic with the specified model
        
        # For now, return random embeddings as a placeholder
        return [list(np.random.rand(DEFAULT_DIMENSION).astype(float)) for _ in texts]
    else:
        logger.info("Using random embeddings for testing")
        return [list(np.random.rand(DEFAULT_DIMENSION).astype(float)) for _ in texts]

def batch_import(provider, items: List[Dict[str, Any]], batch_size: int = BATCH_SIZE) -> int:
    """
    Import data in batches.
    
    Args:
        provider: The Weaviate provider
        items: List of items to import
        batch_size: Number of items to import at once
        
    Returns:
        Number of successfully imported items
    """
    logger.info(f"Starting batch import of {len(items)} items with batch size {batch_size}")
    
    total_imported = 0
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_contents = []
        batch_embeddings = []
        batch_metadata = []
        
        for item in batch:
            # Extract content, embedding, and metadata from the item
            content = item.get("content", "")
            embedding = item.get("embedding")
            metadata = item.get("metadata", {})
            
            # If no embedding is provided, create one
            if embedding is None:
                embedding = create_embeddings([content])[0]
            
            batch_contents.append(content)
            batch_embeddings.append(embedding)
            batch_metadata.append(metadata)
        
        try:
            # Add batch to Weaviate
            memory_ids = provider.add_memories(
                contents=batch_contents,
                embeddings=batch_embeddings,
                metadatas=batch_metadata
            )
            
            if memory_ids:
                total_imported += len(memory_ids)
                logger.info(f"Imported batch {i//batch_size + 1}: {len(memory_ids)} items")
            else:
                logger.warning(f"Batch {i//batch_size + 1} returned no memory IDs")
                
        except Exception as e:
            logger.error(f"Error importing batch {i//batch_size + 1}: {e}")
    
    logger.info(f"Import completed. Total imported: {total_imported}/{len(items)}")
    return total_imported

def main():
    parser = argparse.ArgumentParser(description="Batch import data into Weaviate")
    
    # Source options
    parser.add_argument("--source", type=str, choices=["file", "database"], required=True,
                      help="Source of data (file or database)")
    parser.add_argument("--file", type=str, help="Path to JSON file (required if source is file)")
    
    # Weaviate connection options
    parser.add_argument("--deployment", type=str, choices=["local", "embedded", "cloud"], default="local",
                      help="Weaviate deployment type")
    parser.add_argument("--url", type=str, help="Weaviate URL for cloud deployment")
    parser.add_argument("--api-key", type=str, help="API key for cloud deployment")
    parser.add_argument("--class-name", type=str, default="Memory", 
                      help="Weaviate class name to import into")
    
    # Import options
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                      help="Batch size for import")
    parser.add_argument("--embedding-model", type=str,
                      help="Model to use for generating embeddings")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source == "file" and not args.file:
        logger.error("--file is required when source is file")
        sys.exit(1)
    
    if args.deployment == "cloud" and (not args.url or not args.api_key):
        logger.error("--url and --api-key are required for cloud deployment")
        sys.exit(1)
    
    # Load data
    if args.source == "file":
        logger.info(f"Reading data from file: {args.file}")
        items = read_json_file(args.file)
    else:
        logger.info("Reading data from database")
        items = read_from_database()
    
    if not items:
        logger.error("No items found to import")
        sys.exit(1)
    
    logger.info(f"Loaded {len(items)} items for import")
    
    # Create provider based on deployment type
    deployment_type_map = {
        "local": WeaviateDeploymentType.DOCKER,
        "embedded": WeaviateDeploymentType.EMBEDDED,
        "cloud": WeaviateDeploymentType.CLOUD
    }
    
    deployment_type = deployment_type_map[args.deployment]
    
    # Prepare config
    config = {
        "class_name": args.class_name,
        "vector_dimensions": DEFAULT_DIMENSION
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
    
    # Ensure schema exists
    provider.ensure_schema_exists()
    
    # Run batch import
    imported_count = batch_import(
        provider=provider,
        items=items,
        batch_size=args.batch_size
    )
    
    logger.info(f"Import complete. {imported_count}/{len(items)} items imported successfully.")
    
    return 0 if imported_count == len(items) else 1

if __name__ == "__main__":
    sys.exit(main()) 