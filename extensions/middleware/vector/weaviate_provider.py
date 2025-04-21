"""
Weaviate Vector Memory Provider implementation.

This implements the vector memory provider interface using Weaviate as a backend.
"""
import os
import sys
import uuid
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Add parent directory to path so we can import the abstract provider
current_dir = str(Path(__file__).parent.absolute())
parent_dir = str(Path(current_dir).parent.absolute())
sys.path.insert(0, parent_dir)

try:
    # Try to import any base classes or interfaces
    from vector.vector_provider import VectorMemoryProvider, VectorMemoryProviderConfig
    HAS_BASE = True
except ImportError:
    logger.warning("Base vector provider not found, using standalone implementation")
    HAS_BASE = False

# Define a standalone config if base not available
if not HAS_BASE:
    class VectorMemoryProviderConfig:
        def __init__(self, class_name="Memory", **kwargs):
            self.class_name = class_name
            for key, value in kwargs.items():
                setattr(self, key, value)


class WeaviateConfig(VectorMemoryProviderConfig):
    """Configuration for the Weaviate vector memory provider."""
    
    def __init__(
        self,
        class_name: str = "Memory",
        host: str = "localhost",
        port: str = "8080",
        grpc_port: Optional[str] = None,
        scheme: str = "http",
        vector_dimensions: int = 384,
        **kwargs
    ):
        """
        Initialize the Weaviate configuration.
        
        Args:
            class_name: The Weaviate class name to use for memory objects
            host: Hostname of the Weaviate instance
            port: Port of the Weaviate instance
            grpc_port: Optional gRPC port for more efficient data operations
            scheme: HTTP scheme to use (http or https)
            vector_dimensions: Dimensions of the vector embeddings
            **kwargs: Additional configuration options
        """
        super().__init__(class_name=class_name, **kwargs)
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.scheme = scheme
        self.vector_dimensions = vector_dimensions
        self.api_key = kwargs.get("api_key")


class WeaviateProvider:
    """Vector memory provider implementation using Weaviate."""
    
    def __init__(self, client=None, config: WeaviateConfig = None):
        """
        Initialize the Weaviate vector memory provider.
        
        Args:
            client: Optional pre-configured Weaviate client
            config: Configuration for the provider
        """
        self.config = config or WeaviateConfig()
        self.client = client
        
        # If no client provided, create one
        if self.client is None:
            self._connect()
            
        self.collection_name = self.config.class_name
            
    def _connect(self):
        """Connect to Weaviate instance using configuration parameters."""
        try:
            import weaviate
            
            url = f"{self.config.scheme}://{self.config.host}:{self.config.port}"
            logger.info(f"Connecting to Weaviate at {url}")
            
            # Configure connection parameters using v4 syntax
            connection_params = None
            
            if self.config.api_key:
                # Connection with API key for cloud instances
                connection_params = weaviate.ConnectionParams.from_url(
                    url=url,
                    headers={"X-Weaviate-Api-Key": self.config.api_key}
                )
                
                if self.config.grpc_port:
                    grpc_url = f"{self.config.scheme}://{self.config.host}:{self.config.grpc_port}"
                    connection_params = weaviate.ConnectionParams.from_url(
                        url=url,
                        grpc_url=grpc_url,
                        headers={"X-Weaviate-Api-Key": self.config.api_key}
                    )
            else:
                # Standard connection
                connection_params = weaviate.ConnectionParams.from_url(url=url)
                
                if self.config.grpc_port:
                    grpc_url = f"{self.config.scheme}://{self.config.host}:{self.config.grpc_port}"
                    connection_params = weaviate.ConnectionParams.from_url(
                        url=url,
                        grpc_url=grpc_url
                    )
            
            # Create the client
            self.client = weaviate.WeaviateClient(connection_params=connection_params)
            logger.info(f"Successfully connected to Weaviate at {url}")
            
        except ImportError:
            raise ImportError(
                "The 'weaviate' package is required to use the WeaviateProvider. "
                "Please install it with 'pip install weaviate-client'."
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {str(e)}")
    
    def ensure_schema_exists(self):
        """Ensure the collection with the appropriate schema exists in Weaviate."""
        try:
            # Check if collection exists
            collections = self.client.schema.get()
            
            collection_exists = False
            if collections.get("classes"):
                for cls in collections.get("classes", []):
                    if cls.get("class") == self.collection_name:
                        collection_exists = True
                        break
            
            if not collection_exists:
                logger.info(f"Creating collection '{self.collection_name}'")
                
                # Define the class schema with v4 syntax
                class_definition = {
                    "class": self.collection_name,
                    "vectorizer": "none",  # We'll provide our own vectors
                    "vectorIndexConfig": {
                        "skip": False,
                        "dimension": self.config.vector_dimensions,
                        "distance": "cosine"
                    },
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The textual content of the memory"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                            "description": "Additional metadata for the memory"
                        }
                    ]
                }
                
                # Create the class
                self.client.schema.create_classes([class_definition])
                logger.info(f"Created collection '{self.collection_name}'")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
            return True
        except Exception as e:
            logger.error(f"Error creating schema: {str(e)}")
            return False
    
    def add_memories(self, contents: List[str], embeddings: List[List[float]], 
                  metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add multiple memory chunks with their embeddings and metadata.
        
        Args:
            contents: List of text content for each memory
            embeddings: List of vector embeddings for each memory
            metadatas: Optional list of metadata for each memory
            
        Returns:
            List of memory IDs for the added memories
        """
        if not self.client:
            raise ValueError("Weaviate client not initialized")
        
        if len(contents) != len(embeddings):
            raise ValueError("Number of contents must match number of embeddings")
        
        if metadatas and len(metadatas) != len(contents):
            raise ValueError("Number of metadatas must match number of contents")
        
        metadatas = metadatas or [{} for _ in contents]
        memory_ids = []
        
        # Create objects with v4 syntax
        for i, (content, embedding, metadata) in enumerate(zip(contents, embeddings, metadatas)):
            try:
                # Generate a UUID for the object
                memory_id = str(uuid.uuid4())
                
                # Add object with the v4 client
                collection = self.client.collections.get(self.collection_name)
                result = collection.data.insert(
                    properties={
                        "content": content,
                        "metadata": metadata
                    },
                    uuid=memory_id,
                    vector=embedding
                )
                
                if result:
                    memory_ids.append(memory_id)
                else:
                    logger.warning(f"Failed to add memory {i}, no error but no ID returned")
                    
            except Exception as e:
                logger.error(f"Error adding memory {i}: {str(e)}")
        
        return memory_ids
    
    def get_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Retrieve a memory by its ID.
        
        Args:
            memory_id: The ID of the memory to retrieve
            
        Returns:
            Dict containing the memory content and metadata
        """
        if not self.client:
            raise ValueError("Weaviate client not initialized")
        
        try:
            # Get object with v4 client
            collection = self.client.collections.get(self.collection_name)
            result = collection.data.get_by_id(
                uuid=memory_id,
                include_vector=False
            )
            
            if not result:
                raise ValueError(f"Memory with ID {memory_id} not found")
            
            # Extract content and metadata
            memory = {
                "id": memory_id,
                "content": result.properties.get("content", ""),
                "metadata": result.properties.get("metadata", {})
            }
            
            return memory
        except Exception as e:
            raise ValueError(f"Error retrieving memory: {str(e)}")
    
    def get_memories(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple memories by their IDs.
        
        Args:
            memory_ids: List of memory IDs to retrieve
            
        Returns:
            List of dictionaries containing the memories
        """
        if not self.client:
            raise ValueError("Weaviate client not initialized")
        
        memories = []
        for memory_id in memory_ids:
            try:
                memory = self.get_memory(memory_id)
                memories.append(memory)
            except Exception as e:
                logger.error(f"Error retrieving memory {memory_id}: {str(e)}")
        
        return memories
    
    def search_by_vector(self, query_vector: List[float], limit: int = 5,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for memories by vector similarity.
        
        Args:
            query_vector: The query vector to search with
            limit: Maximum number of results to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of similar memories with their content and metadata
        """
        if not self.client:
            raise ValueError("Weaviate client not initialized")
        
        try:
            # Prepare search options with v4 syntax
            collection = self.client.collections.get(self.collection_name)
            search_options = {
                "near_vector": query_vector,
                "limit": limit,
                "include_vector": False
            }
            
            # Add filter if provided
            if filters and "metadata_filter" in filters:
                filter_obj = filters["metadata_filter"]
                
                # Create Where filter with v4 syntax
                path = filter_obj.get("path", [])
                operator = filter_obj.get("operator", "Equal")
                
                # Different value types
                value = None
                if "valueText" in filter_obj:
                    value = filter_obj["valueText"]
                elif "valueNumber" in filter_obj:
                    value = filter_obj["valueNumber"]
                elif "valueBoolean" in filter_obj:
                    value = filter_obj["valueBoolean"]
                
                # Create where filter using v4 syntax
                where_filter = {
                    "path": path,
                    "operator": operator,
                    "valueText": value if isinstance(value, str) else None,
                    "valueNumber": value if isinstance(value, (int, float)) else None,
                    "valueBoolean": value if isinstance(value, bool) else None
                }
                
                # Remove None values
                where_filter = {k: v for k, v in where_filter.items() if v is not None}
                
                # Add filter to search options
                search_options["where_filter"] = where_filter
            
            # Execute search with v4 syntax
            results = collection.query.near_vector(**search_options)
            
            # Process results
            memories = []
            for obj in results.objects:
                memory = {
                    "id": obj.uuid,
                    "content": obj.properties.get("content", ""),
                    "metadata": obj.properties.get("metadata", {}),
                }
                memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by its ID.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            raise ValueError("Weaviate client not initialized")
        
        try:
            # Delete object with v4 syntax
            collection = self.client.collections.get(self.collection_name)
            result = collection.data.delete_by_id(uuid=memory_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Remove all objects from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            raise ValueError("Weaviate client not initialized")
        
        try:
            # Delete all objects with v4 syntax
            collection = self.client.collections.get(self.collection_name)
            result = collection.data.delete_all()
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

# Simple test to verify functionality when run directly
if __name__ == "__main__":
    # Test configuration
    config = WeaviateConfig(
        host="localhost",
        port="8080",
        scheme="http",
        class_name="BeeAIMemory"
    )
    
    # Initialize provider
    provider = WeaviateProvider(config)
    
    # Output success message
    print("WeaviateProvider initialized successfully.") 