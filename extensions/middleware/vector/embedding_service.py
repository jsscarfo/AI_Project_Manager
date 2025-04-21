"""
Embedding Service for generating and managing embeddings
"""
import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Union, Any, Callable
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings from different providers."""
    
    def __init__(self, 
                embedding_fn: Optional[Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]] = None,
                cache_dir: Optional[str] = None,
                cache_ttl: int = 86400):  # Default TTL: 1 day
        """Initialize the embedding service.
        
        Args:
            embedding_fn: Function that takes text and returns embeddings
            cache_dir: Directory to store embedding cache (None for no caching)
            cache_ttl: Time-to-live for cached embeddings in seconds
        """
        self.embedding_fn = embedding_fn
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.logger.info(f"Embedding cache directory: {self.cache_dir}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        # Check cache first
        cache_hit = False
        if self.cache_dir:
            cache_key = self._get_cache_key(text)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
            
            if os.path.exists(cache_path):
                cache_age = time.time() - os.path.getmtime(cache_path)
                if cache_age < self.cache_ttl:
                    try:
                        embedding = np.load(cache_path).tolist()
                        cache_hit = True
                        self.logger.debug(f"Cache hit for text: {text[:30]}...")
                        return embedding
                    except Exception as e:
                        self.logger.warning(f"Failed to load cached embedding: {str(e)}")
        
        # If we don't have a cache hit, generate the embedding
        if not cache_hit:
            if not self.embedding_fn:
                raise ValueError("No embedding function provided")
                
            try:
                result = await self._call_embedding_fn(text)
                
                # Handle different return types from embedding functions
                if isinstance(result, list) and all(isinstance(x, (int, float)) for x in result):
                    embedding = result
                elif isinstance(result, list) and len(result) == 1:
                    embedding = result[0]
                else:
                    embedding = result
                    
                # Cache the result if caching is enabled
                if self.cache_dir:
                    cache_key = self._get_cache_key(text)
                    cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
                    np.save(cache_path, np.array(embedding))
                    
                return embedding
            except Exception as e:
                self.logger.error(f"Error generating embedding: {str(e)}")
                raise
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector embeddings
        """
        # For small batches, just process sequentially using get_embedding
        if len(texts) <= 5:
            return [await self.get_embedding(text) for text in texts]
            
        # For larger batches, optimize by batching the API call and using cache efficiently
        cache_hits = {}
        texts_to_embed = []
        text_indices = []
        
        # Check cache first for all texts
        if self.cache_dir:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
                
                if os.path.exists(cache_path):
                    cache_age = time.time() - os.path.getmtime(cache_path)
                    if cache_age < self.cache_ttl:
                        try:
                            embedding = np.load(cache_path).tolist()
                            cache_hits[i] = embedding
                            continue
                        except Exception as e:
                            self.logger.warning(f"Failed to load cached embedding: {str(e)}")
                
                # If we're here, the cache missed
                texts_to_embed.append(text)
                text_indices.append(i)
        else:
            # No cache, embed all texts
            texts_to_embed = texts
            text_indices = list(range(len(texts)))
            
        # Generate embeddings for texts not in cache
        embeddings = []
        if texts_to_embed:
            if not self.embedding_fn:
                raise ValueError("No embedding function provided")
                
            try:
                batch_result = await self._call_embedding_fn(texts_to_embed)
                
                # Save to cache if enabled
                if self.cache_dir:
                    for i, text in enumerate(texts_to_embed):
                        cache_key = self._get_cache_key(text)
                        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
                        np.save(cache_path, np.array(batch_result[i]))
            except Exception as e:
                self.logger.error(f"Error generating batch embeddings: {str(e)}")
                raise
                
            embeddings = batch_result
        
        # Combine cache hits and newly generated embeddings
        result = [None] * len(texts)
        for i, embedding in cache_hits.items():
            result[i] = embedding
            
        for i, embedding in zip(text_indices, embeddings):
            result[i] = embedding
            
        return result
    
    async def _call_embedding_fn(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Call the embedding function, handling both sync and async functions.
        
        Args:
            text: Text or texts to embed
            
        Returns:
            Embedding or list of embeddings
        """
        try:
            if self.embedding_fn.__code__.co_flags & 0x80:  # Check if it's an async function
                result = await self.embedding_fn(text)
            else:
                # Run synchronous function in a way that doesn't block
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.embedding_fn, text)
                
            return result
        except Exception as e:
            self.logger.error(f"Error calling embedding function: {str(e)}")
            raise
            
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

if __name__ == "__main__":
    import asyncio
    
    async def test_embedding_service():
        service = EmbeddingService(embedding_fn=None, cache_dir="./embedding_cache")
        text = "This is a test text for embedding generation."
        
        embedding = await service.get_embedding(text)
        print(f"Generated embedding with length: {len(embedding)}")
        
        # Test batch embedding
        texts = ["First test text", "Second test text", "Third test text"]
        embeddings = await service.get_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
    asyncio.run(test_embedding_service()) 