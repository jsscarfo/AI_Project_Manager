# Batch Data Import with Weaviate

This guide explains how to efficiently import large datasets into Weaviate for use with the BeeAI Framework.

## Batch Import Script

The following script demonstrates how to batch import data into Weaviate:

```python
import os
import weaviate
import numpy as np
import time
from tqdm import tqdm
import json
from typing import List, Dict, Any

# Load environment variables
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Memory")

# Initialize Weaviate client
client = weaviate.Client(
    url=WEAVIATE_URL
)

# Check if schema exists, create if not
if not client.schema.exists(COLLECTION_NAME):
    class_obj = {
        "class": COLLECTION_NAME,
        "vectorizer": "none",  # We'll provide our own vectors
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The main content text"
            },
            {
                "name": "metadata",
                "dataType": ["object"],
                "description": "Additional metadata for the content"
            },
            {
                "name": "source",
                "dataType": ["string"],
                "description": "Source of the content"
            },
            {
                "name": "category",
                "dataType": ["string"],
                "description": "Category of the content"
            },
            {
                "name": "created",
                "dataType": ["number"],
                "description": "Creation timestamp"
            }
        ]
    }
    client.schema.create_class(class_obj)
    print(f"Created schema for {COLLECTION_NAME}")

# Configure the batch
client.batch.configure(
    batch_size=BATCH_SIZE,  # Number of objects per batch
    timeout_retries=3,      # Number of retries if batch fails
    callback=None,          # Optional callback function
)

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_embedding(text: str, dimension: int = 384) -> List[float]:
    """
    Generate embedding for text - replace with your embedding generation logic.
    
    This is just a placeholder that generates random vectors.
    In a real scenario, you would use:
    - A language model like sentence-transformers, OpenAI, etc.
    - A local embedding model
    - An API-based embedding service
    """
    # Use text hash for deterministic "embedding" for demo purposes
    np.random.seed(hash(text) % 2**32)
    embedding = np.random.randn(dimension).astype(np.float32)
    # Normalize for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

def batch_import(data: List[Dict[str, Any]]) -> None:
    """Import data in batches."""
    print(f"Starting batch import of {len(data)} records...")
    
    with client.batch as batch:
        for i, item in enumerate(tqdm(data)):
            # Extract fields
            content = item.get("content", "")
            if not content:
                continue
                
            # Generate embedding for the content
            vector = generate_embedding(content)
            
            # Extract metadata
            source = item.get("source", "import")
            category = item.get("category", "general")
            created = item.get("timestamp", time.time())
            metadata = item.get("metadata", {})
            
            # Prepare object properties
            properties = {
                "content": content,
                "source": source,
                "category": category,
                "created": created,
                "metadata": metadata
            }
            
            # Add object to batch
            batch.add_data_object(
                data_object=properties,
                class_name=COLLECTION_NAME,
                vector=vector
            )
    
    print("Batch import completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Import data into Weaviate")
    parser.add_argument("--file", type=str, required=True, help="JSON file containing data to import")
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.file)
    
    # Import data
    batch_import(data)
```

## Data Format

Prepare your data in JSON format:

```json
[
  {
    "content": "This is the content to vectorize and store",
    "source": "documentation",
    "category": "api",
    "timestamp": 1646870400,
    "metadata": {
      "author": "John Doe",
      "version": "1.0",
      "tags": ["python", "weaviate", "vector-db"]
    }
  },
  {
    "content": "More content to import in batch",
    "source": "knowledge-base",
    "category": "tutorial",
    "timestamp": 1646956800,
    "metadata": {
      "author": "Jane Smith",
      "version": "2.0",
      "tags": ["batch-import", "guide"]
    }
  }
]
```

## Running the Import

To run the import script:

```bash
# Set environment variables
export WEAVIATE_URL=http://localhost:8080
export BATCH_SIZE=100
export COLLECTION_NAME=Memory

# Run the import script
python batch_import.py --file data.json
```

## Performance Tips

1. **Optimal Batch Size**: The ideal batch size depends on your data and hardware. Start with 100-200 objects per batch and adjust based on performance.

2. **Memory Considerations**: Monitor memory usage during import. If you encounter memory issues:
   - Reduce batch size
   - Process data in chunks from disk rather than loading everything into memory

3. **Parallel Embedding Generation**: For large datasets, generate embeddings in parallel before import:

   ```python
   from concurrent.futures import ProcessPoolExecutor
   
   def parallel_embed(texts, max_workers=4):
       results = []
       with ProcessPoolExecutor(max_workers=max_workers) as executor:
           results = list(executor.map(generate_embedding, texts))
       return results
   ```

4. **Progress Monitoring**: For large imports, implement proper logging and progress tracking.

5. **Schema Optimization**: Consider adding indexes on properties you frequently filter on:

   ```python
   # Add an index on the 'source' property
   client.schema.property.create(
       COLLECTION_NAME,
       {
           "name": "source",
           "dataType": ["string"],
           "indexInverted": True
       }
   )
   ```

## Validation After Import

Verify your import with a simple search:

```python
# Perform a test query
query_vector = generate_embedding("test query")
result = (
    client.query
    .get(COLLECTION_NAME, ["content", "source", "category"])
    .with_near_vector({"vector": query_vector})
    .with_limit(5)
    .do()
)

# Print results
if "data" in result and "Get" in result["data"]:
    items = result["data"]["Get"][COLLECTION_NAME]
    for i, item in enumerate(items):
        print(f"{i+1}. {item['content'][:100]}...")
else:
    print("No results found")
```

## Troubleshooting

1. **Connection Issues**: Ensure Weaviate is running and accessible at the specified URL

2. **Memory Errors**: If you encounter memory errors:
   - Reduce batch size
   - Process data in smaller chunks
   - Use streaming import for very large datasets

3. **Slow Performance**: 
   - Check network latency between your script and Weaviate
   - Optimize embedding generation
   - Consider running the script on the same machine as Weaviate

4. **Data Validation Errors**:
   - Ensure all required fields are present
   - Check for malformed data (e.g., null values, wrong data types)
   - Look for invalid characters in content 