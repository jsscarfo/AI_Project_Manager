"""
Vector Memory Context Provider

Adapts the VectorMemory class to the ContextProvider interface for 
use with the Contextual Enhancement Middleware.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np
from pydantic import BaseModel, Field

# Import our middleware core
from ..core.context_middleware import ContextProvider, ContextProviderConfig

logger = logging.getLogger(__name__)

class VectorMemoryProviderConfig(ContextProviderConfig):
    """Configuration for the Vector Memory Provider."""
    dimension: int = 384
    similarity: str = "cosine"
    storage_dir: str = "data/vector-memory"
    default_collection: str = "default"
    collections: List[str] = ["default"]
    create_if_missing: bool = True
    
    # Search parameters
    limit: int = 10
    threshold: float = 0.65
    hybrid_search: bool = True
    
    # Load from environment variables if available
    def __init__(self, **data: Any):
        # Override defaults with environment variables if they exist
        if "VECTOR_DIMENSION" in os.environ:
            data["dimension"] = int(os.environ["VECTOR_DIMENSION"])
        if "VECTOR_STORAGE_DIR" in os.environ:
            data["storage_dir"] = os.environ["VECTOR_STORAGE_DIR"]
        if "VECTOR_DEFAULT_COLLECTION" in os.environ:
            data["default_collection"] = os.environ["VECTOR_DEFAULT_COLLECTION"]
        if "VECTOR_COLLECTIONS" in os.environ:
            data["collections"] = os.environ["VECTOR_COLLECTIONS"].split(",")
        
        super().__init__(**data)

class VectorMemoryProvider(ContextProvider):
    """
    Context provider that uses vector memory for retrieval.
    
    This provider adapts our existing VectorMemory system to work
    with the contextual enhancement middleware.
    """
    
    def __init__(self, config: Optional[VectorMemoryProviderConfig] = None):
        """
        Initialize the Vector Memory Provider.
        
        Args:
            config: Configuration for the vector memory provider
        """
        self.config = config or VectorMemoryProviderConfig()
        super().__init__(self.config)
        
        # We'll import the VectorMemory class lazily to avoid circular imports
        # and allow for custom Vector Memory implementations
        try:
            # First try importing from our extensions
            from ...vector.vector_memory import VectorMemory
        except ImportError:
            # Fall back to the original implementation
            from src.memory.VectorMemory import VectorMemory, createChunksWithContext
            self.create_chunks = createChunksWithContext
        
        # Memory instance cache
        self._memory_instances = {}
        logger.info(f"Initialized VectorMemoryProvider with collections: {self.config.collections}")
    
    async def initialize(self) -> None:
        """Initialize the vector memory instances."""
        # Initialize all collections in parallel
        tasks = []
        for collection_name in self.config.collections:
            tasks.append(self._get_memory_instance(collection_name))
        
        await asyncio.gather(*tasks)
        logger.info(f"Initialized {len(self._memory_instances)} vector memory collections")
    
    async def _get_memory_instance(self, collection_name: str):
        """
        Get or create a vector memory instance for the specified collection.
        
        Args:
            collection_name: Name of the memory collection/index
            
        Returns:
            VectorMemory instance
        """
        if collection_name in self._memory_instances:
            return self._memory_instances[collection_name]
        
        try:
            # First try importing from our extensions
            from ...vector.vector_memory import VectorMemory
        except ImportError:
            # Fall back to the original implementation
            from src.memory.VectorMemory import VectorMemory
        
        # Create a new instance
        memory = VectorMemory({
            "dimension": self.config.dimension,
            "similarity": self.config.similarity,
            "storageDir": self.config.storage_dir,
            "indexName": collection_name,
            "createIfMissing": self.config.create_if_missing
        })
        
        # Initialize the memory
        await memory.initialize()
        
        # Cache the instance
        self._memory_instances[collection_name] = memory
        
        return memory
    
    async def get_context(self, query: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve context relevant to the query from vector memory.
        
        Args:
            query: The query to retrieve context for
            metadata: Additional metadata for the query
            
        Returns:
            List of context items
        """
        if not query:
            return []
        
        metadata = metadata or {}
        collections_to_search = metadata.get("collections", [self.config.default_collection])
        
        # Ensure we're using collections that exist in our config
        collections_to_search = [
            c for c in collections_to_search 
            if c in self.config.collections
        ]
        
        if not collections_to_search:
            collections_to_search = [self.config.default_collection]
        
        # Search all specified collections in parallel
        search_tasks = []
        for collection_name in collections_to_search:
            memory = await self._get_memory_instance(collection_name)
            
            if self.config.hybrid_search:
                search_tasks.append(self._search_with_hybrid(memory, query, metadata))
            else:
                search_tasks.append(self._search_standard(memory, query, metadata))
        
        # Gather all search results
        all_results = await asyncio.gather(*search_tasks)
        
        # Flatten and format results
        flattened_results = []
        for collection_idx, results in enumerate(all_results):
            collection_name = collections_to_search[collection_idx]
            for result in results:
                flattened_results.append({
                    "content": result["content"],
                    "metadata": {
                        **result["metadata"],
                        "collection": collection_name
                    },
                    "score": result["score"]
                })
        
        # Sort by relevance
        sorted_results = sorted(flattened_results, key=lambda x: x["score"], reverse=True)
        
        return sorted_results
    
    async def _search_with_hybrid(self, memory, query: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform a hybrid search (vector + keyword)."""
        try:
            # Use the enhancedSearch method if available (preferred)
            results = await memory.enhancedSearch(
                query,
                {"limit": self.config.limit, "threshold": self.config.threshold}
            )
        except AttributeError:
            # Fall back to standard vector search
            results = await memory.searchByText(
                query, 
                {"limit": self.config.limit, "threshold": self.config.threshold}
            )
        
        return results
    
    async def _search_standard(self, memory, query: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform a standard vector search."""
        return await memory.searchByText(
            query, 
            {"limit": self.config.limit, "threshold": self.config.threshold}
        )
    
    async def add_context(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add content to vector memory.
        
        Args:
            content: The content to store
            metadata: Metadata for the content
            
        Returns:
            ID of the stored memory
        """
        if not content:
            raise ValueError("Content cannot be empty")
        
        metadata = metadata or {}
        collection_name = metadata.get("collection", self.config.default_collection)
        
        # Make sure the collection exists in our config
        if collection_name not in self.config.collections:
            logger.warning(f"Collection {collection_name} not in configured collections, using default")
            collection_name = self.config.default_collection
        
        # Get the memory instance
        memory = await self._get_memory_instance(collection_name)
        
        # Check if we should chunk the content
        chunk_size = metadata.get("chunk_size", None)
        if chunk_size:
            overlap_size = metadata.get("overlap_size", chunk_size // 5)
            
            # Import chunking function if needed
            if not hasattr(self, "create_chunks"):
                try:
                    from ...vector.vector_memory import create_chunks_with_context
                    self.create_chunks = create_chunks_with_context
                except ImportError:
                    from src.memory.VectorMemory import createChunksWithContext
                    self.create_chunks = createChunksWithContext
            
            # Create chunks and add them
            chunks = self.create_chunks(content, chunk_size, overlap_size)
            
            # Add all chunks with the same metadata
            ids = []
            for chunk in chunks:
                chunk_id = await memory.addMemory(chunk, metadata)
                ids.append(chunk_id)
            
            return ",".join(ids)
        else:
            # Add as a single entry
            return await memory.addMemory(content, metadata) 