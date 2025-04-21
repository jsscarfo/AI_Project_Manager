#!/usr/bin/env python
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Knowledge Retrieval module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime
from typing import Dict, List, Any

from beeai_framework.vector.knowledge_retrieval import (
    SequentialThinkingKnowledgeRetriever,
    KnowledgeRetrievalConfig,
    StepContextManager,
    RetrievedKnowledge,
    KnowledgeRetrievalResult
)
from beeai_framework.vector.base import VectorMemoryProvider, ContextResult, ContextMetadata


@pytest.fixture
def mock_vector_provider():
    """Create a mock vector memory provider."""
    provider = AsyncMock(spec=VectorMemoryProvider)
    
    # Mock get_context to return different results based on level
    async def mock_get_context(query, metadata_filter=None, limit=None):
        level = metadata_filter.get("level") if metadata_filter else None
        
        if level == "domain":
            return [
                ContextResult(
                    content="Python is a high-level programming language.",
                    metadata=ContextMetadata(
                        source="documentation",
                        category="explanation",
                        level="domain",
                        importance=0.8
                    ),
                    score=0.9
                )
            ]
        elif level == "techstack":
            return [
                ContextResult(
                    content="Flask is a micro web framework for Python.",
                    metadata=ContextMetadata(
                        source="documentation",
                        category="explanation",
                        level="techstack",
                        importance=0.8
                    ),
                    score=0.85
                )
            ]
        elif level == "project":
            return [
                ContextResult(
                    content="Our application uses Flask for the API endpoints.",
                    metadata=ContextMetadata(
                        source="project_docs",
                        category="implementation",
                        level="project",
                        importance=0.9
                    ),
                    score=0.95
                )
            ]
        else:
            return []
    
    provider.get_context = mock_get_context
    return provider


@pytest.fixture
def retriever(mock_vector_provider):
    """Create a SequentialThinkingKnowledgeRetriever instance."""
    config = KnowledgeRetrievalConfig(
        max_results=5,
        similarity_threshold=0.6,
        boost_previous_concepts=True
    )
    return SequentialThinkingKnowledgeRetriever(
        vector_provider=mock_vector_provider,
        config=config
    )


@pytest.fixture
def context_manager(retriever):
    """Create a StepContextManager instance."""
    return StepContextManager(
        knowledge_retriever=retriever,
        max_context_items_per_step=3,
        enable_context_carryover=True
    )


@pytest.mark.asyncio
async def test_retriever_retrieve_for_step(retriever):
    """Test retrieving knowledge for a specific step."""
    # Arrange
    query = "How to use Flask in our project?"
    step_type = "implementation"
    step_number = 3
    
    # Act
    result = await retriever.retrieve_for_step(
        query=query,
        step_type=step_type,
        step_number=step_number
    )
    
    # Assert
    assert isinstance(result, KnowledgeRetrievalResult)
    assert len(result.items) == 3  # One for each level
    assert result.query == query
    assert result.step_type == step_type
    assert "retrieval_time_ms" in result.metrics
    
    # Check items content
    content_texts = [item.content for item in result.items]
    assert any("Python" in text for text in content_texts)
    assert any("Flask" in text for text in content_texts)
    assert any("application" in text for text in content_texts)


@pytest.mark.asyncio
async def test_retriever_level_weighting_by_step(retriever):
    """Test that level weights are appropriate for different step types."""
    # Test problem_definition step
    result1 = await retriever.retrieve_for_step(
        query="What is Flask?",
        step_type="problem_definition",
        step_number=1
    )
    
    # Test implementation step
    result2 = await retriever.retrieve_for_step(
        query="How to implement Flask routes?",
        step_type="implementation",
        step_number=3
    )
    
    # Assert: problem_definition should prioritize domain knowledge
    domain_items1 = [i for i in result1.items if i.metadata.get("level") == "domain"]
    domain_items2 = [i for i in result2.items if i.metadata.get("level") == "domain"]
    
    # Check if domain items are weighted higher in problem_definition
    assert domain_items1[0].score > domain_items2[0].score


@pytest.mark.asyncio
async def test_enhance_query_for_step(retriever):
    """Test query enhancement for different step types."""
    # Test with problem_definition step
    enhanced1 = await retriever._enhance_query_for_step(
        query="What is Flask?",
        step_type="problem_definition",
        step_number=1,
        previous_concepts=set()
    )
    
    # Test with implementation step
    enhanced2 = await retriever._enhance_query_for_step(
        query="How to implement Flask routes?",
        step_type="implementation",
        step_number=3,
        previous_concepts={"Python", "Web"}
    )
    
    # Assert: Enhancement should include step-specific terms
    assert "definition" in enhanced1 or "requirement" in enhanced1
    assert "implementation" in enhanced2 or "code" in enhanced2
    
    # Check if previous concepts are included in the enhanced query
    assert "Python" in enhanced2
    assert "Web" in enhanced2


@pytest.mark.asyncio
async def test_context_manager_get_context_for_step(context_manager):
    """Test getting context for a step through the context manager."""
    # Arrange
    query = "How to implement Flask routes?"
    step_type = "implementation"
    step_number = 3
    
    # Act
    result = await context_manager.get_context_for_step(
        query=query,
        step_type=step_type,
        step_number=step_number
    )
    
    # Assert
    assert "formatted_context" in result
    assert "context_items" in result
    assert "retrieval_metrics" in result
    
    # Check context content
    assert "Relevant Context for Implementation" in result["formatted_context"]
    assert len(result["context_items"]) <= context_manager.max_context_items_per_step
    
    # Check metrics
    assert "retrieval_time_ms" in result["retrieval_metrics"]
    assert result["retrieval_metrics"]["result_count"] > 0


@pytest.mark.asyncio
async def test_context_carryover(context_manager):
    """Test context carryover between steps."""
    # Step 1
    result1 = await context_manager.get_context_for_step(
        query="What is Flask?",
        step_type="problem_definition",
        step_number=1
    )
    
    # Step 2 (should carry over relevant context)
    result2 = await context_manager.get_context_for_step(
        query="How to use Flask in our project?",
        step_type="implementation",
        step_number=2,
        explicit_concepts=["Flask"]
    )
    
    # Assert: Check that context is carried over
    assert len(context_manager.context_usage) == 2
    assert context_manager.step_history[0]["step_number"] == 1
    assert context_manager.step_history[1]["step_number"] == 2
    
    # Verify that context items exist and are properly tracked
    assert len(context_manager.current_context_items) > 0


@pytest.mark.asyncio
async def test_context_usage_stats(context_manager):
    """Test getting context usage statistics."""
    # Run a few steps to generate stats
    await context_manager.get_context_for_step(
        query="What is Flask?",
        step_type="problem_definition",
        step_number=1
    )
    
    await context_manager.get_context_for_step(
        query="How to implement Flask routes?",
        step_type="implementation",
        step_number=2
    )
    
    # Get stats
    stats = context_manager.get_context_usage_stats()
    
    # Assert
    assert stats["steps"] == 2
    assert "avg_items_per_step" in stats
    assert "avg_retrieval_time_ms" in stats
    assert "step_distribution" in stats
    assert "problem_definition" in stats["step_distribution"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 