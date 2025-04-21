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

"""Tests for the Knowledge Retrieval Middleware integration with Sequential Thinking."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from beeai_framework.middleware.base import MiddlewareContext
from beeai_framework.vector.middleware import KnowledgeRetrievalMiddleware
from beeai_framework.vector.sequential_thinking_integration import (
    SequentialKnowledgeIntegration,
    IntegrationConfig
)
from beeai_framework.vector.knowledge_retrieval import (
    SequentialThinkingKnowledgeRetriever,
    KnowledgeRetrievalConfig,
    RetrievedKnowledge
)
from beeai_framework.vector.base import VectorMemoryProvider
from beeai_framework.vector.knowledge_capture import KnowledgeCaptureProcessor


@pytest.fixture
def mock_vector_provider():
    """Create a mock vector memory provider."""
    provider = MagicMock(spec=VectorMemoryProvider)
    
    # Set up semantic_search to return different results
    async def mock_semantic_search(query, limit=5, metadata_filter=None):
        # Return different results based on query content
        results = []
        
        if "vector operations" in query or "Python" in query:
            results = [
                (
                    "Efficient vector operations in Python can be implemented using NumPy, which provides optimized array operations.",
                    {"source": "documentation", "importance": 0.8},
                    0.92
                ),
                (
                    "For large-scale vector operations, consider using libraries like PyTorch or TensorFlow which provide GPU acceleration.",
                    {"source": "techstack", "importance": 0.7},
                    0.85
                ),
                (
                    "When working with vector operations, memory management is crucial for performance.",
                    {"source": "best_practices", "importance": 0.6},
                    0.78
                )
            ]
        elif "context window" in query or "optimization" in query:
            results = [
                (
                    "Context window optimization involves selecting the most relevant information to fit within token constraints.",
                    {"source": "documentation", "importance": 0.9},
                    0.94
                ),
                (
                    "Token budget management is essential for effective context window usage.",
                    {"source": "best_practices", "importance": 0.75},
                    0.88
                )
            ]
        else:
            results = [
                (
                    "Sequential thinking involves breaking down complex problems into steps.",
                    {"source": "methodology", "importance": 0.85},
                    0.90
                ),
                (
                    "Knowledge retrieval enhances reasoning by providing contextual information.",
                    {"source": "documentation", "importance": 0.8},
                    0.85
                )
            ]
        
        # Limit to requested number
        return results[:min(limit, len(results))]
    
    provider.semantic_search = AsyncMock(side_effect=mock_semantic_search)
    return provider


@pytest.fixture
def mock_knowledge_capture():
    """Create a mock knowledge capture processor."""
    processor = MagicMock(spec=KnowledgeCaptureProcessor)
    processor.store_knowledge_from_content = AsyncMock()
    return processor


@pytest.fixture
def middleware(mock_vector_provider, mock_knowledge_capture):
    """Create a KnowledgeRetrievalMiddleware instance for testing."""
    integration_config = IntegrationConfig(
        enable_knowledge_capture=True,
        enable_context_enhancement=True,
        enable_feedback_collection=True,
        enable_reasoning_path_optimization=True
    )
    
    retrieval_config = KnowledgeRetrievalConfig(
        default_max_results=3,
        default_similarity_threshold=0.7
    )
    
    # Token estimator that counts spaces to simulate tokens
    def simple_token_estimator(text):
        if not text:
            return 0
        return text.count(" ") + 1
    
    return KnowledgeRetrievalMiddleware(
        vector_provider=mock_vector_provider,
        knowledge_capture_processor=mock_knowledge_capture,
        integration_config=integration_config,
        retrieval_config=retrieval_config,
        token_estimator=simple_token_estimator
    )


class MockMiddlewareContext:
    """Mock middleware context for testing."""
    
    def __init__(self, request=None, response=None):
        self.request = request or {}
        self.response = response
        self.metadata = {}
        self.response_generated = response is not None
    
    def enhance_metadata(self, metadata):
        """Add metadata to the context."""
        self.metadata.update(metadata)


@pytest.mark.asyncio
async def test_process_sequential_thinking_request(middleware):
    """Test processing a sequential thinking request."""
    # Create a sequential thinking request
    request = {
        "thought_number": 1,
        "total_thoughts": 5,
        "system_prompt": "You are a helpful assistant.",
        "prompt": "How can I implement efficient vector operations in Python?",
        "trace_id": "test-trace-123"
    }
    
    # Create mock context
    context = MockMiddlewareContext(request=request)
    
    # Process the request
    result = await middleware.process(context)
    
    # Verify that the request was enhanced with context
    assert "knowledge_retrieval" in result.metadata
    assert result.metadata["knowledge_retrieval"]["applied"] is True
    assert result.metadata["knowledge_retrieval"]["context_items_count"] > 0
    
    # Verify that the system_prompt was updated with context
    assert len(request["system_prompt"]) > len("You are a helpful assistant.")
    assert "NumPy" in request["system_prompt"] or "vector operations" in request["system_prompt"]


@pytest.mark.asyncio
async def test_post_process_sequential_thinking_response(middleware):
    """Test post-processing a sequential thinking response."""
    # Create a sequential thinking request and response
    request = {
        "thought_number": 1,
        "total_thoughts": 5,
        "system_prompt": "You are a helpful assistant.",
        "prompt": "How can I implement efficient vector operations in Python?",
        "trace_id": "test-trace-123"
    }
    
    response = {
        "thought": "To implement efficient vector operations in Python, I would use NumPy which provides optimized array operations. As mentioned in context [1], NumPy offers significant performance benefits for vector calculations.",
        "thought_number": 1,
        "next_thought_needed": True,
        "total_thoughts": 5
    }
    
    # Create mock context with metadata from the process step
    context = MockMiddlewareContext(request=request, response=response)
    context.metadata["knowledge_retrieval"] = {
        "applied": True,
        "context_items_count": 2,
        "step_type": "problem_definition",
        "step_number": 1,
        "trace_id": "test-trace-123",
        "context_items": [
            {
                "content": "Efficient vector operations in Python can be implemented using NumPy.",
                "metadata": {"source": "documentation"},
                "similarity": 0.92
            },
            {
                "content": "For large-scale vector operations, consider using libraries like PyTorch.",
                "metadata": {"source": "techstack"},
                "similarity": 0.85
            }
        ]
    }
    
    # Post-process the response
    result = await middleware.post_process(context)
    
    # Verify that knowledge was captured
    mock_knowledge_capture = middleware.integration.knowledge_capture_processor
    mock_knowledge_capture.store_knowledge_from_content.assert_called_once()
    
    # Verify that feedback was collected
    assert "knowledge_retrieval_result" in result.metadata
    assert result.metadata["knowledge_retrieval_result"]["knowledge_captured"] is True
    assert result.metadata["knowledge_retrieval_result"]["feedback_collected"] is True
    
    # Check that the context quality was scored
    assert result.metadata["knowledge_retrieval_result"]["context_quality_score"] > 0
    
    # Verify that step history was updated
    trace_id = "test-trace-123"
    assert trace_id in middleware.step_history
    assert len(middleware.step_history[trace_id]) == 1
    assert middleware.step_history[trace_id][0]["step_number"] == 1


@pytest.mark.asyncio
async def test_context_window_optimization(middleware):
    """Test context window optimization for large context."""
    # Create a request with a large system prompt
    large_system_prompt = "You are a helpful assistant. " + "Token padding. " * 100
    
    request = {
        "thought_number": 2,
        "total_thoughts": 5,
        "system_prompt": large_system_prompt,
        "prompt": "How can I implement context window optimization?",
        "trace_id": "test-trace-123"
    }
    
    # Create mock context
    context = MockMiddlewareContext(request=request)
    
    # Override the token budget to force optimization
    original_optimize = middleware._optimize_context_window
    
    def mock_optimize(*args, **kwargs):
        result = original_optimize(*args, **kwargs)
        # Force optimization by setting a very low token budget
        result["token_usage"]["total_tokens"] = 1000
        return result
    
    middleware._optimize_context_window = mock_optimize
    
    # Process the request
    result = await middleware.process(context)
    
    # Restore original method
    middleware._optimize_context_window = original_optimize
    
    # Verify that context window optimization was applied
    assert result.metadata["knowledge_retrieval"]["optimized"] is True
    
    # Verify that token usage was tracked
    assert "token_usage" in result.metadata["knowledge_retrieval"]
    token_usage = result.metadata["knowledge_retrieval"]["token_usage"]
    assert "system_tokens" in token_usage
    assert "context_tokens" in token_usage
    assert "total_tokens" in token_usage


@pytest.mark.asyncio
async def test_reasoning_path_analysis(middleware):
    """Test reasoning path analysis in response processing."""
    # Create a sequential thinking request and response that triggers path analysis
    request = {
        "thought_number": 2,
        "total_thoughts": 5,
        "system_prompt": "You are a helpful assistant.",
        "prompt": "How should I approach vector operations in Python?",
        "trace_id": "test-trace-123"
    }
    
    # Response indicating need for more context
    response = {
        "thought": "I need more information about the specific vector operations required. According to context [1], NumPy provides optimized array operations, but I need to understand the scale and nature of your operations.",
        "thought_number": 2,
        "next_thought_needed": True,
        "total_thoughts": 5
    }
    
    # Create mock context with metadata from the process step
    context = MockMiddlewareContext(request=request, response=response)
    context.metadata["knowledge_retrieval"] = {
        "applied": True,
        "context_items_count": 2,
        "step_type": "information_gathering",
        "step_number": 2,
        "trace_id": "test-trace-123",
        "context_items": [
            {
                "content": "Efficient vector operations in Python can be implemented using NumPy.",
                "metadata": {"source": "documentation"},
                "similarity": 0.92
            }
        ]
    }
    
    # Add a previous step to the history
    middleware.step_history["test-trace-123"] = [
        {
            "step_number": 1,
            "thought": "I'll explore vector operations in Python.",
            "timestamp": datetime.now().timestamp()
        }
    ]
    
    # Post-process the response
    result = await middleware.post_process(context)
    
    # Verify reasoning path analysis was performed
    assert "reasoning_path_analysis" in result.metadata["knowledge_retrieval_result"]
    path_analysis = result.metadata["knowledge_retrieval_result"]["reasoning_path_analysis"]
    
    # Since the response mentions "need more information", it should be detected
    assert path_analysis["needs_more_context"] is True
    assert path_analysis["adjustment_suggested"] is True


@pytest.mark.asyncio
async def test_feedback_loop_relevance(middleware):
    """Test the feedback loop's impact on relevance scoring."""
    # First request/response pair with explicit reference to context
    request1 = {
        "thought_number": 1,
        "total_thoughts": 5,
        "system_prompt": "You are a helpful assistant.",
        "prompt": "How can I implement efficient vector operations in Python?",
        "trace_id": "test-trace-loop"
    }
    
    response1 = {
        "thought": "According to context [1], efficient vector operations in Python can be implemented using NumPy, which provides optimized array operations.",
        "thought_number": 1,
        "next_thought_needed": True,
        "total_thoughts": 5
    }
    
    # Process first request
    context1 = MockMiddlewareContext(request=request1)
    await middleware.process(context1)
    
    # Add context metadata manually for testing
    context1.metadata["knowledge_retrieval"] = {
        "applied": True,
        "context_items_count": 2,
        "step_type": "problem_definition",
        "step_number": 1,
        "trace_id": "test-trace-loop",
        "context_items": [
            {
                "content": "Efficient vector operations in Python can be implemented using NumPy.",
                "metadata": {"source": "documentation"},
                "vector_id": "item1",
                "similarity": 0.92
            },
            {
                "content": "For large-scale vector operations, consider using libraries like PyTorch.",
                "metadata": {"source": "techstack"},
                "vector_id": "item2",
                "similarity": 0.85
            }
        ]
    }
    
    # Post-process first response
    context1.response = response1
    context1.response_generated = True
    await middleware.post_process(context1)
    
    # Second request/response pair that should benefit from feedback
    request2 = {
        "thought_number": 2,
        "total_thoughts": 5,
        "system_prompt": "You are a helpful assistant.",
        "prompt": "What are the advantages of NumPy for vector operations?",
        "trace_id": "test-trace-loop"
    }
    
    # Process second request
    context2 = MockMiddlewareContext(request=request2)
    await middleware.process(context2)
    
    # Verify that the feedback from the first step influenced the second step
    # Check that item1 (which was explicitly referenced) has been tracked as relevant
    assert "item1" in middleware.integration.feedback["relevant_contexts"]
    
    # Check quality metrics tracking
    assert len(middleware.integration.feedback["step_quality_scores"]) > 0
    
    # Verify the step history is properly maintained
    assert "test-trace-loop" in middleware.step_history
    assert len(middleware.step_history["test-trace-loop"]) >= 1


@pytest.mark.asyncio
async def test_non_sequential_thinking_request(middleware):
    """Test that non-sequential thinking requests are passed through unchanged."""
    # Create a regular request (not sequential thinking)
    request = {
        "system_prompt": "You are a helpful assistant.",
        "prompt": "What is Python?",
    }
    
    # Create mock context
    context = MockMiddlewareContext(request=request)
    
    # Process the request
    result = await middleware.process(context)
    
    # Verify that the request was not modified
    assert "knowledge_retrieval" not in result.metadata
    assert request["system_prompt"] == "You are a helpful assistant."
    assert request["prompt"] == "What is Python?"


@pytest.mark.asyncio
async def test_extract_key_concepts(middleware):
    """Test extraction of key concepts from step content."""
    # Test directly with the integration component
    integration = middleware.integration
    
    # Test text with clear concepts
    text = "NumPy provides efficient vector operations in Python. Memory management is crucial for performance."
    
    concepts = integration._extract_key_concepts(text)
    
    # Verify concepts were extracted
    assert len(concepts) > 0
    assert any("NumPy" in concept["concept"] for concept in concepts)
    assert all("importance" in concept for concept in concepts)
    assert all(0 <= concept["importance"] <= 1 for concept in concepts)


def test_token_estimation(middleware):
    """Test token estimation for context window optimization."""
    # Test the token estimator
    test_text = "This is a test sentence with multiple words."
    
    # Count tokens
    tokens = middleware.token_estimator(test_text)
    
    # Simple word count should match our test estimator (words + 1)
    assert tokens == 9  # 8 words + 1
    
    # Test with empty text
    assert middleware.token_estimator("") == 0
    
    # Test optimization logic with a mock context
    context_items = [
        {"content": "Short content", "similarity": 0.9},
        {"content": "Medium length content with more details", "similarity": 0.8},
        {"content": "Long content " + "with many words " * 10, "similarity": 0.7}
    ]
    
    optimized = middleware._optimize_context_window(
        enhanced_data={
            "context_applied": True,
            "context_items": context_items,
            "system_prompt": "System prompt",
            "user_prompt": "User prompt",
            "apply_to_system_prompt": True
        },
        system_prompt="System prompt",
        user_prompt="User prompt",
        previous_steps=[]
    )
    
    # Verify token usage calculation
    assert "token_usage" in optimized
    assert "system_tokens" in optimized["token_usage"]
    assert "user_tokens" in optimized["token_usage"]
    assert "context_tokens" in optimized["token_usage"]


def test_context_quality_tracking(middleware):
    """Test context quality tracking across steps."""
    # Set up test quality scores
    integration = middleware.integration
    integration.feedback["step_quality_scores"] = [0.5, 0.6, 0.7, 0.8]
    
    # Get quality metrics
    metrics = integration.track_context_quality("test-trace")
    
    # Verify metrics
    assert metrics["average_quality"] == 0.65
    assert metrics["quality_trend"] == "improving"
    assert metrics["step_count"] == 4
    assert metrics["high_quality_steps"] == 2  # Scores >= 0.7
    assert metrics["low_quality_steps"] == 0   # Scores <= 0.3


if __name__ == "__main__":
    # Manual test
    import asyncio
    
    async def run_test():
        """Run a manual test of the middleware."""
        provider = MagicMock(spec=VectorMemoryProvider)
        provider.semantic_search = AsyncMock(return_value=[
            (
                "NumPy provides efficient vector operations in Python.",
                {"source": "documentation"},
                0.9
            )
        ])
        
        processor = MagicMock(spec=KnowledgeCaptureProcessor)
        processor.store_knowledge_from_content = AsyncMock()
        
        middleware = KnowledgeRetrievalMiddleware(
            vector_provider=provider,
            knowledge_capture_processor=processor
        )
        
        request = {
            "thought_number": 1,
            "system_prompt": "You are a helpful assistant.",
            "prompt": "How can I implement vector operations in Python?",
            "trace_id": "manual-test"
        }
        
        context = MockMiddlewareContext(request=request)
        
        result = await middleware.process(context)
        print(f"Process result metadata: {json.dumps(result.metadata, indent=2)}")
        print(f"Enhanced system prompt: {request['system_prompt'][:100]}...")
        
        # Test post-processing
        response = {
            "thought": "NumPy is excellent for vector operations as mentioned in context [1].",
            "thought_number": 1,
            "next_thought_needed": True
        }
        
        context.response = response
        context.response_generated = True
        
        post_result = await middleware.post_process(context)
        print(f"Post-process result metadata: {json.dumps(post_result.metadata, indent=2)}")
        
    asyncio.run(run_test()) 