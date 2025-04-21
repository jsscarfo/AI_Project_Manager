#!/usr/bin/env python
"""
Tests for Knowledge Retrieval Middleware.

This module contains tests for the knowledge retrieval middleware
and its integration with sequential thinking.
"""

import os
import sys
import pytest
import json
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from beeai_framework.vector.base import VectorMemoryProvider
from beeai_framework.vector.knowledge_capture import KnowledgeEntry
from beeai_framework.middleware.base import MiddlewareRequest

from extensions.middleware.knowledge_retrieval import (
    KnowledgeRetrievalProcessor,
    KnowledgeRetrievalSettings,
    KnowledgeRetrievalMiddleware,
    KnowledgeRetrievalResult,
    ContextEnhancementProvider
)


class MockVectorProvider(VectorMemoryProvider):
    """Mock vector provider for testing."""
    
    def __init__(self, entries=None):
        """
        Initialize with mock entries.
        
        Args:
            entries: Optional list of mock entries
        """
        self.entries = entries or []
    
    async def store(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Store a new knowledge entry.
        
        Args:
            content: Content to store
            metadata: Metadata for the content
            
        Returns:
            True if successful
        """
        self.entries.append({"content": content, "metadata": metadata})
        return True
    
    async def search(self, query: str, metadata_filter: Dict[str, Any] = None, limit: int = 5, min_score: float = 0.0) -> List[Any]:
        """
        Search for knowledge entries.
        
        Args:
            query: Query string
            metadata_filter: Optional filter for metadata
            limit: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            List of search results
        """
        from collections import namedtuple
        SearchResult = namedtuple('SearchResult', ['content', 'metadata', 'score'])
        
        results = []
        for entry in self.entries:
            # Simple mock matching
            score = 0.7  # Default score for tests
            
            # Apply metadata filter if provided
            if metadata_filter and not self._matches_filter(entry["metadata"], metadata_filter):
                continue
                
            if score >= min_score:
                result = SearchResult(
                    content=entry["content"],
                    metadata=entry["metadata"],
                    score=score
                )
                results.append(result)
                
            if len(results) >= limit:
                break
                
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
        """
        Check if metadata matches the filter.
        
        Args:
            metadata: Entry metadata
            metadata_filter: Filter to apply
            
        Returns:
            True if metadata matches filter
        """
        for key, value in metadata_filter.items():
            if key not in metadata:
                return False
                
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
                
        return True


@pytest.fixture
def mock_entries():
    """Fixture for mock knowledge entries."""
    return [
        {
            "content": "Python decorators are a syntactic sugar for applying wrapper functions.",
            "metadata": {
                "source": "programming_guide",
                "category": "code_snippet",
                "level": "concept",
                "importance": 0.8
            }
        },
        {
            "content": "Sequential thinking helps break down complex problems into steps.",
            "metadata": {
                "source": "framework_documentation",
                "category": "concept",
                "level": "architecture",
                "importance": 0.9
            }
        },
        {
            "content": "Vector databases store embeddings for semantic search.",
            "metadata": {
                "source": "vector_db_guide",
                "category": "explanation",
                "level": "techstack",
                "importance": 0.75
            }
        }
    ]


@pytest.fixture
def vector_provider(mock_entries):
    """Fixture for vector provider."""
    return MockVectorProvider(entries=mock_entries)


@pytest.fixture
def knowledge_processor(vector_provider):
    """Fixture for knowledge retrieval processor."""
    settings = KnowledgeRetrievalSettings(
        enabled=True,
        max_results=3,
        similarity_threshold=0.5
    )
    return KnowledgeRetrievalProcessor(vector_provider=vector_provider, settings=settings)


@pytest.fixture
def middleware(vector_provider):
    """Fixture for knowledge retrieval middleware."""
    settings = KnowledgeRetrievalSettings(
        enabled=True,
        max_results=3,
        similarity_threshold=0.5
    )
    return KnowledgeRetrievalMiddleware(vector_provider=vector_provider, settings=settings)


@pytest.mark.asyncio
async def test_retrieve_knowledge(knowledge_processor):
    """Test retrieving knowledge with the processor."""
    # Basic retrieval
    result = await knowledge_processor.retrieve_knowledge(
        query="Python decorators"
    )
    
    # Check result
    assert isinstance(result, KnowledgeRetrievalResult)
    assert result.has_results
    assert len(result.entries) > 0
    assert "Python decorators" in result.entries[0].content
    assert "programming_guide" == result.entries[0].metadata.get("source")
    assert result.context_str != ""


@pytest.mark.asyncio
async def test_retrieve_knowledge_with_metadata_filter(knowledge_processor):
    """Test retrieving knowledge with metadata filter."""
    # Retrieval with filter
    result = await knowledge_processor.retrieve_knowledge(
        query="concept",
        metadata_filter={"category": "concept"}
    )
    
    # Check result
    assert result.has_results
    assert len(result.entries) > 0
    # Check that all entries have the correct category
    for entry in result.entries:
        assert entry.metadata.get("category") == "concept"


@pytest.mark.asyncio
async def test_retrieve_knowledge_with_previous_thoughts(knowledge_processor):
    """Test retrieving knowledge with previous thoughts context."""
    # Create previous thoughts
    previous_thoughts = [
        {"thought": "I need to understand how decorators work in Python."},
        {"thought": "Decorators seem to be related to modifying function behavior."}
    ]
    
    # Retrieval with previous thoughts
    result = await knowledge_processor.retrieve_knowledge(
        query="Python decorators",
        previous_thoughts=previous_thoughts,
        step_number=3
    )
    
    # Check result
    assert result.has_results
    assert "Python decorators" in result.entries[0].content


@pytest.mark.asyncio
async def test_disabled_processor(vector_provider):
    """Test behavior when retrieval is disabled."""
    # Create disabled processor
    settings = KnowledgeRetrievalSettings(enabled=False)
    processor = KnowledgeRetrievalProcessor(vector_provider=vector_provider, settings=settings)
    
    # Attempt retrieval
    result = await processor.retrieve_knowledge(query="Python")
    
    # Check that no results are returned
    assert not result.has_results
    assert len(result.entries) == 0
    assert result.metrics.get("enabled") is False


@pytest.mark.asyncio
async def test_context_enhancement_provider(knowledge_processor):
    """Test the context enhancement provider."""
    # Create provider
    provider = ContextEnhancementProvider(knowledge_processor=knowledge_processor)
    
    # Test enhancement
    enhanced = await provider.enhance_context(
        prompt="How do Python decorators work?",
        thought_number=1,
        previous_thoughts=[]
    )
    
    # Check result
    assert "enhanced_prompt" in enhanced
    assert enhanced["enhanced_prompt"] != "How do Python decorators work?"
    assert "context_metrics" in enhanced
    assert enhanced["context_metrics"]["entries_found"] > 0


@pytest.mark.asyncio
async def test_middleware_process(middleware):
    """Test processing a request with the middleware."""
    # Create request
    request = MiddlewareRequest(
        prompt="Explain Python decorators",
        context={"task_type": "explanation"},
        trace_id="test-trace"
    )
    
    # Process request
    response = await middleware.process(request)
    
    # Check response
    assert response.success
    assert response.prompt != request.prompt
    assert "knowledge_retrieval" in response.metadata
    assert response.metadata["knowledge_retrieval"]["entries_found"] > 0


@pytest.mark.asyncio
async def test_middleware_integration(middleware):
    """Test integration with sequential thinking middleware."""
    # Create mock sequential thinking middleware
    mock_sequential = MagicMock()
    mock_sequential.context_refinement_processor = None
    
    # Integrate
    await middleware.set_sequential_middleware(mock_sequential)
    
    # Check integration
    assert mock_sequential.context_refinement_processor is not None
    assert isinstance(mock_sequential.context_refinement_processor, ContextEnhancementProvider)


@pytest.mark.asyncio
async def test_enhance_query_with_previous_thoughts(knowledge_processor):
    """Test enhancing a query with previous thoughts."""
    # Create previous thoughts
    previous_thoughts = [
        {"thought": "I need to understand how decorators work in Python."},
        {"thought": "Decorators seem to be related to modifying function behavior."}
    ]
    
    # Get enhanced query
    enhanced_query = knowledge_processor._enhance_query(
        query="Python decorators",
        previous_thoughts=previous_thoughts,
        step_number=2
    )
    
    # Check that previous thoughts are incorporated
    assert len(enhanced_query) > len("Python decorators")
    assert "modifying function behavior" in enhanced_query


# Run a manual test if this file is executed directly
if __name__ == "__main__":
    async def main():
        # Create provider and processor
        entries = [
            {
                "content": "Python decorators are a syntactic sugar for applying wrapper functions.",
                "metadata": {
                    "source": "programming_guide",
                    "category": "code_snippet",
                    "level": "concept",
                    "importance": 0.8
                }
            },
            {
                "content": "Sequential thinking helps break down complex problems into steps.",
                "metadata": {
                    "source": "framework_documentation",
                    "category": "concept",
                    "level": "architecture",
                    "importance": 0.9
                }
            }
        ]
        
        provider = MockVectorProvider(entries=entries)
        processor = KnowledgeRetrievalProcessor(
            vector_provider=provider,
            settings=KnowledgeRetrievalSettings(enabled=True)
        )
        
        # Test knowledge retrieval
        result = await processor.retrieve_knowledge(query="Python decorators")
        
        print(f"Retrieved {len(result.entries)} entries")
        print(f"Query: {result.query}")
        print(f"Formatted context:\n{result.context_str}")
        
        # Test context enhancement
        context_provider = ContextEnhancementProvider(knowledge_processor=processor)
        enhanced = await context_provider.enhance_context(
            prompt="How can Python decorators be used in a framework?",
            thought_number=1,
            previous_thoughts=[]
        )
        
        print("\nEnhanced prompt:")
        print(enhanced["enhanced_prompt"])
        
        print("\nContext metrics:")
        print(json.dumps(enhanced["context_metrics"], indent=2))
    
    asyncio.run(main()) 