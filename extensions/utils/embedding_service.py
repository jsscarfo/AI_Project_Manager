"""
Embedding Service for BeeAI Framework

Provides utilities for generating text embeddings using various
embedding models and services.
"""

import os
import logging
import asyncio
from typing import List, Optional, Dict, Any, Union, Callable
import numpy as np

logger = logging.getLogger(__name__)

# Default embedding dimension
DEFAULT_DIMENSION = 384

# Environment variables for embedding service configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "minilm-l6-v2")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "")


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        self.model_name = EMBEDDING_MODEL
        self.api_key = EMBEDDING_API_KEY
        self.api_url = EMBEDDING_API_URL
        self._model = None
        
        # Cache for embeddings to avoid regenerating
        self._cache = {}
        self._cache_size = 1000
        
        logger.info(f"Initialized EmbeddingService with model: {self.model_name}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a piece of text.
        
        Args:
            text: The text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        if text in self._cache:
            return self._cache[text]
        
        try:
            # Try different embedding approaches
            embedding = await self._get_embedding_impl(text)
            
            # Update cache (with LRU-like behavior)
            if len(self._cache) >= self._cache_size:
                # Remove a random item if cache is full
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Fallback to random embedding
            return self._get_fallback_embedding(text)
    
    async def _get_embedding_impl(self, text: str) -> List[float]:
        """Actual implementation of embedding generation."""
        # Try different approaches based on configuration
        
        # 1. Try local model if available
        if self._model is not None:
            return self._get_local_embedding(text)
        
        # 2. Try to load local model if not already loaded
        try:
            embedding = await self._get_local_embedding(text)
            return embedding
        except Exception:
            pass
        
        # 3. Try API-based approach if URL is set
        if self.api_url:
            try:
                embedding = await self._get_api_embedding(text)
                return embedding
            except Exception as e:
                logger.warning(f"API embedding failed: {str(e)}")
        
        # 4. Fall back to default approach
        logger.warning(f"Using fallback embedding for '{text[:20]}...'")
        return self._get_fallback_embedding(text)
    
    async def _get_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using a local model."""
        # Try to load the model if needed
        if self._model is None:
            await self._load_model()
        
        # Use the model to generate embedding
        if self.model_name.startswith("minilm"):
            # Handle sentence-transformers models
            embedding = self._model.encode(text)
            return embedding.tolist()
        
        # Default handling
        return self._model(text)
    
    async def _load_model(self):
        """Load the embedding model."""
        if self.model_name.startswith("minilm"):
            try:
                # Try to load sentence-transformers
                from sentence_transformers import SentenceTransformer
                
                # Load in a separate thread to avoid blocking
                def load_model():
                    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(None, load_model)
                logger.info(f"Loaded local model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available")
                raise
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    async def _get_api_embedding(self, text: str) -> List[float]:
        """Generate embedding using an API service."""
        import aiohttp
        
        if not self.api_url:
            raise ValueError("Embedding API URL not set")
        
        # Determine which API to use based on URL
        if "openai" in self.api_url:
            return await self._get_openai_embedding(text)
        elif "cohere" in self.api_url:
            return await self._get_cohere_embedding(text)
        
        # Generic API approach
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                json={"text": text},
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                # Extract embedding from result (structure depends on API)
                if "embedding" in result:
                    return result["embedding"]
                elif "data" in result and len(result["data"]) > 0:
                    return result["data"][0]["embedding"]
                else:
                    raise ValueError(f"Unexpected API response format: {result}")
    
    async def _get_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        import openai
        
        if not self.api_key:
            raise ValueError("OpenAI API key not set")
        
        openai.api_key = self.api_key
        
        # Use client to make API call
        try:
            result = await openai.Embedding.acreate(
                input=text,
                model="text-embedding-ada-002"
            )
            
            # Extract embedding from response
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            else:
                raise ValueError(f"Unexpected OpenAI response format: {result}")
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    async def _get_cohere_embedding(self, text: str) -> List[float]:
        """Generate embedding using Cohere API."""
        import cohere
        
        if not self.api_key:
            raise ValueError("Cohere API key not set")
        
        co = cohere.Client(self.api_key)
        
        # Convert to async with run_in_executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: co.embed(
                texts=[text],
                model="embed-english-v2.0"
            )
        )
        
        # Extract embedding from response
        embeddings = response.embeddings
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        else:
            raise ValueError(f"Unexpected Cohere response format")
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """Generate a fallback embedding when other methods fail."""
        logger.warning("Using fallback random embedding - NOT SUITABLE FOR PRODUCTION")
        
        # Use text hash for deterministic "embedding"
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(DEFAULT_DIMENSION).astype(np.float32)
        
        # Normalize to unit length for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()


# Singleton instance
_embedding_service = None

async def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a piece of text.
    
    Args:
        text: The text to generate embedding for
        
    Returns:
        List of floats representing the embedding vector
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    
    return await _embedding_service.get_embedding(text)


async def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for multiple texts in batch.
    
    Args:
        texts: List of texts to generate embeddings for
        
    Returns:
        List of embedding vectors
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    
    # Process in parallel
    tasks = [_embedding_service.get_embedding(text) for text in texts]
    return await asyncio.gather(*tasks) 