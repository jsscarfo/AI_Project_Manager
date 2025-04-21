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

"""Tests for the Sequential Thinking Knowledge Integration module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime

from beeai_framework.vector.sequential_thinking_integration import (
    SequentialKnowledgeIntegration,
    IntegrationConfig,
    ReasoningFeedbackCollector
)
from beeai_framework.vector.knowledge_retrieval import (
    KnowledgeRetrievalConfig,
    RetrievedKnowledge
)
from beeai_framework.vector.base import VectorMemoryProvider
from beeai_framework.vector.knowledge_capture import KnowledgeCaptureProcessor


@pytest.fixture
def mock_vector_provider():
    """Create a mock vector memory provider."""
    provider = MagicMock(spec=VectorMemoryProvider)
    
    # Set up get_context to return different results based on level
    async def mock_get_context(query, count=5, level=None, **kwargs):
        # Simulate different results based on level
        results = []
        
        if level == "domain":
            results = [
                {
                    "content": "Domain knowledge about software architecture principles.",
                    "metadata": {
                        "id": "domain1",
                        "level": "domain",
                        "category": "architecture",
                        "importance": 0.85
                    }
                },
                {
                    "content": "Domain knowledge about design patterns in software engineering.",
                    "metadata": {
                        "id": "domain2",
                        "level": "domain",
                        "category": "patterns",
                        "importance": 0.8
                    }
                }
            ]
        elif level == "techstack":
            results = [
                {
                    "content": "Technical documentation for Python asyncio module.",
                    "metadata": {
                        "id": "tech1",
                        "level": "techstack",
                        "category": "python",
                        "importance": 0.75
                    }
                },
                {
                    "content": "Best practices for Python vector operations and memory management.",
                    "metadata": {
                        "id": "tech2",
                        "level": "techstack",
                        "category": "python",
                        "importance": 0.7
                    }
                }
            ]
        elif level == "project":
            results = [
                {
                    "content": "Knowledge about sequential thinking implementation details in this project.",
                    "metadata": {
                        "id": "project1",
                        "level": "project", 
                        "category": "implementation",
                        "importance": 0.9
                    }
                },
                {
                    "content": "Project-specific guidelines for knowledge retrieval in reasoning.",
                    "metadata": {
                        "id": "project2",
                        "level": "project",
                        "category": "guidelines",
                        "importance": 0.85
                    }
                }
            ]
        else:
            # Mixed results when no specific level
            results = [
                {
                    "content": "Knowledge about sequential thinking implementation details in this project.",
                    "metadata": {
                        "id": "project1",
                        "level": "project",
                        "importance": 0.9
                    }
                },
                {
                    "content": "Technical documentation for Python asyncio module.",
                    "metadata": {
                        "id": "tech1",
                        "level": "techstack",
                        "importance": 0.75
                    }
                }
            ]
        
        # Limit to requested count
        return results[:min(count, len(results))]
    
    # Set up the mock
    provider.get_context = AsyncMock(side_effect=mock_get_context)
    return provider


@pytest.fixture
def mock_knowledge_capture():
    """Create a mock knowledge capture processor."""
    processor = MagicMock(spec=KnowledgeCaptureProcessor)
    processor.store_knowledge_from_content = AsyncMock()
    processor.extract_knowledge_from_content = AsyncMock(return_value={
        "content": "Extracted knowledge",
        "metadata": {"importance": 0.8}
    })
    return processor


@pytest.fixture
def integration(mock_vector_provider, mock_knowledge_capture):
    """Create an instance of the integration class for testing."""
    config = IntegrationConfig(
        enable_knowledge_capture=True,
        enable_context_enhancement=True,
        enable_knowledge_retrieval=True
    )
    
    retrieval_config = KnowledgeRetrievalConfig()
    
    return SequentialKnowledgeIntegration(
        vector_provider=mock_vector_provider,
        knowledge_capture_processor=mock_knowledge_capture,
        config=config,
        retrieval_config=retrieval_config
    )


@pytest.mark.asyncio
async def test_enhance_step(integration):
    """Test enhancing a sequential thinking step with knowledge."""
    # Set up test data
    step_info = {
        "thought_number": 2,
        "description": "Information gathering step"
    }
    system_prompt = "You are a helpful assistant."
    user_prompt = "How can I implement vector operations in Python efficiently?"
    
    # Call the method
    result = await integration.enhance_step(step_info, system_prompt, user_prompt)
    
    # Verify result structure
    assert "system_prompt" in result
    assert "user_prompt" in result
    assert "context_applied" in result
    assert "context_items" in result
    assert "step_type" in result
    assert result["context_applied"] is True
    
    # Verify context was applied to system prompt (based on config)
    assert len(result["system_prompt"]) > len(system_prompt)
    assert "implementation details" in result["system_prompt"] or "Technical documentation" in result["system_prompt"]


@pytest.mark.asyncio
async def test_process_step_result(integration):
    """Test processing a step result."""
    # Set up test data
    step_result = {
        "thought_number": 2,
        "thought": "To implement vector operations efficiently in Python, we should use NumPy which provides optimized vector operations. Technical documentation for Python asyncio module can also be useful for managing concurrent operations."
    }
    
    original_request = {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "How can I implement vector operations in Python efficiently?"
    }
    
    enhanced_request = {
        "system_prompt": "You are a helpful assistant.\n\nTechnical documentation for Python asyncio module.",
        "user_prompt": "How can I implement vector operations in Python efficiently?",
        "context_applied": True,
        "context_items": [
            {
                "content": "Technical documentation for Python asyncio module.",
                "metadata": {"id": "tech1", "level": "techstack"}
            }
        ],
        "step_type": "information_gathering"
    }
    
    # Call the method
    result = await integration.process_step_result(step_result, original_request, enhanced_request)
    
    # Verify result
    assert "step_result" in result
    assert "knowledge_captured" in result
    assert "feedback_collected" in result
    assert "next_step_suggestions" in result
    
    # Verify knowledge capture was called
    integration.knowledge_capture_processor.store_knowledge_from_content.assert_called_once()
    
    # Verify next step suggestions
    assert len(result["next_step_suggestions"]) > 0


@pytest.mark.asyncio
async def test_disabled_knowledge_retrieval(integration):
    """Test step enhancement with disabled knowledge retrieval."""
    # Disable knowledge retrieval
    integration.config.enable_knowledge_retrieval = False
    
    # Set up test data
    step_info = {"thought_number": 1}
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is sequential thinking?"
    
    # Call the method
    result = await integration.enhance_step(step_info, system_prompt, user_prompt)
    
    # Verify result
    assert result["system_prompt"] == system_prompt
    assert result["user_prompt"] == user_prompt
    assert result["context_applied"] is False


@pytest.mark.asyncio
async def test_disabled_knowledge_capture(integration):
    """Test step result processing with disabled knowledge capture."""
    # Disable knowledge capture
    integration.config.enable_knowledge_capture = False
    
    # Set up test data
    step_result = {
        "thought_number": 1,
        "thought": "Sequential thinking involves breaking down problems into steps."
    }
    original_request = {}
    enhanced_request = {"step_type": "problem_definition"}
    
    # Call the method
    result = await integration.process_step_result(step_result, original_request, enhanced_request)
    
    # Verify knowledge capture was not called
    integration.knowledge_capture_processor.store_knowledge_from_content.assert_not_called()
    assert result["knowledge_captured"] is False


def test_determine_step_type(integration):
    """Test determination of step type based on step number."""
    # Test different step numbers
    assert integration._determine_step_type(1, {}) == "problem_definition"
    assert integration._determine_step_type(2, {}) == "information_gathering"
    assert integration._determine_step_type(3, {}) == "analysis"
    assert integration._determine_step_type(4, {}) == "solution_formulation"
    assert integration._determine_step_type(5, {}) == "implementation"
    assert integration._determine_step_type(7, {}) == "verification"
    
    # Test with explicitly provided step type
    assert integration._determine_step_type(1, {"step_type": "custom_type"}) == "custom_type"


def test_apply_context_to_prompts(integration):
    """Test applying context to prompts."""
    system_prompt = "You are a helpful assistant."
    user_prompt = "Tell me about sequential thinking."
    context = "Sequential thinking is a problem-solving approach."
    step_type = "problem_definition"
    
    # Test with context in system prompt
    integration.config.apply_context_to_system_prompt = True
    result = integration._apply_context_to_prompts(system_prompt, user_prompt, context, step_type)
    assert result["system_prompt"] == f"{system_prompt}\n\n{context}"
    assert result["user_prompt"] == user_prompt
    
    # Test with context in user prompt
    integration.config.apply_context_to_system_prompt = False
    result = integration._apply_context_to_prompts(system_prompt, user_prompt, context, step_type)
    assert result["system_prompt"] == system_prompt
    assert result["user_prompt"] == f"{context}\n\n{user_prompt}"
    
    # Test with empty context
    result = integration._apply_context_to_prompts(system_prompt, user_prompt, "", step_type)
    assert result["system_prompt"] == system_prompt
    assert result["user_prompt"] == user_prompt


@pytest.mark.asyncio
async def test_capture_step_knowledge(integration):
    """Test capturing knowledge from a step."""
    step_content = "Sequential thinking involves breaking down problems into steps for better analysis."
    step_number = 1
    step_type = "problem_definition"
    
    # Call the method
    await integration._capture_step_knowledge(step_content, step_number, step_type)
    
    # Verify knowledge capture was called with correct parameters
    integration.knowledge_capture_processor.store_knowledge_from_content.assert_called_once()
    call_args = integration.knowledge_capture_processor.store_knowledge_from_content.call_args[1]
    assert call_args["content"] == step_content
    assert call_args["metadata"]["source"] == "sequential_reasoning"
    assert call_args["metadata"]["step_number"] == step_number
    assert call_args["metadata"]["step_type"] == step_type


def test_collect_feedback_from_step(integration):
    """Test collecting feedback from a step."""
    step_content = "Based on the technical documentation for Python asyncio, we can implement concurrent operations efficiently."
    context_items = [
        {
            "content": "Technical documentation for Python asyncio module.",
            "metadata": {"id": "tech1"}
        },
        {
            "content": "Knowledge about sequential thinking implementation.",
            "metadata": {"id": "project1"}
        }
    ]
    
    # Call the method
    integration._collect_feedback_from_step(step_content, context_items)
    
    # Verify feedback was collected
    assert "tech1" in integration.feedback["relevant_contexts"]
    # The second context item might be in relevant or irrelevant based on the implementation
    assert ("project1" in integration.feedback["relevant_contexts"] or 
            "project1" in integration.feedback["irrelevant_contexts"])


def test_extract_key_phrases(integration):
    """Test extraction of key phrases from text."""
    text = "Python asyncio is a library that provides tools for writing concurrent code using the async/await syntax."
    
    # Call the method
    phrases = integration._extract_key_phrases(text, 3)
    
    # Verify results
    assert len(phrases) <= 3
    assert all(isinstance(phrase, str) for phrase in phrases)
    assert all(len(phrase) > 0 for phrase in phrases)


def test_generate_next_step_suggestions(integration):
    """Test generation of next step suggestions."""
    step_content = "Sequential thinking involves breaking down problems into steps."
    
    # Test for different step types
    for step_type in ["problem_definition", "information_gathering", "analysis"]:
        suggestions = integration._generate_next_step_suggestions(step_content, step_type, 1)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)


def test_get_integration_stats(integration):
    """Test getting integration statistics."""
    # Set up some feedback data
    integration.feedback["relevant_contexts"].add("item1")
    integration.feedback["relevant_contexts"].add("item2")
    integration.feedback["irrelevant_contexts"].add("item3")
    
    # Call the method
    stats = integration.get_integration_stats()
    
    # Verify stats structure
    assert "context_usage" in stats
    assert "feedback" in stats
    assert "configuration" in stats
    
    # Verify feedback stats
    assert stats["feedback"]["relevant_contexts"] == 2
    assert stats["feedback"]["irrelevant_contexts"] == 1
    assert stats["feedback"]["relevance_ratio"] == 2/3


# Tests for ReasoningFeedbackCollector

@pytest.fixture
def feedback_collector(mock_vector_provider):
    """Create a feedback collector instance for testing."""
    return ReasoningFeedbackCollector(
        vector_provider=mock_vector_provider,
        enable_explicit_feedback=True,
        enable_implicit_feedback=True
    )


def test_collect_feedback_from_step_explicit(feedback_collector):
    """Test collecting explicit feedback from a step."""
    step_content = "Based on context [1], we should use Python's asyncio library. As mentioned in item 2, this provides tools for concurrent programming."
    step_number = 2
    context_items = [
        {
            "content": "Technical documentation for Python asyncio module.",
            "metadata": {"id": "tech1"}
        },
        {
            "content": "Project-specific guidelines for knowledge retrieval in reasoning.",
            "metadata": {"id": "project2"}
        }
    ]
    
    # Call the method
    metrics = feedback_collector.collect_feedback_from_step(step_content, step_number, context_items)
    
    # Verify explicit feedback was collected
    assert metrics["explicit_references"] > 0
    assert "tech1" in feedback_collector.explicit_feedback
    assert "project2" in feedback_collector.explicit_feedback


def test_collect_feedback_from_step_implicit(feedback_collector):
    """Test collecting implicit feedback from a step."""
    step_content = "Technical documentation for Python asyncio module shows how to implement concurrent operations efficiently."
    step_number = 2
    context_items = [
        {
            "content": "Technical documentation for Python asyncio module.",
            "metadata": {"id": "tech1"}
        },
        {
            "content": "Project-specific guidelines for knowledge retrieval.",
            "metadata": {"id": "project2"}
        }
    ]
    
    # Call the method
    metrics = feedback_collector.collect_feedback_from_step(step_content, step_number, context_items)
    
    # Verify implicit feedback was collected
    assert metrics["implicit_references"] > 0
    assert "tech1" in feedback_collector.implicit_feedback


def test_get_feedback_stats(feedback_collector):
    """Test getting feedback statistics."""
    # Set up some feedback data
    feedback_collector.explicit_feedback = {"item1": 2, "item2": 1}
    feedback_collector.implicit_feedback = {"item1": 1, "item3": 1}
    
    # Call the method
    stats = feedback_collector.get_feedback_stats()
    
    # Verify stats structure
    assert "items_with_feedback" in stats
    assert "explicit_feedback_count" in stats
    assert "implicit_feedback_count" in stats
    assert "top_items" in stats
    assert "feedback_distribution" in stats
    
    # Verify stats values
    assert stats["items_with_feedback"] == 3
    assert stats["explicit_feedback_count"] == 2
    assert stats["implicit_feedback_count"] == 2
    assert len(stats["top_items"]) > 0
    # item1 should be at the top with highest score
    assert stats["top_items"][0][0] == "item1"


@pytest.mark.asyncio
async def test_apply_feedback_adjustments(feedback_collector):
    """Test applying feedback adjustments."""
    # Set up some feedback data
    feedback_collector.explicit_feedback = {"item1": 2, "item2": 1}
    
    # Call the method
    result = await feedback_collector.apply_feedback_adjustments()
    
    # Verify result
    assert "adjustments_applied" in result
    assert "status" in result
    assert result["adjustments_applied"] == 2
    assert result["status"] == "success"


# Integration tests

@pytest.mark.asyncio
async def test_full_integration_flow(integration):
    """Test a full integration flow from step enhancement to processing."""
    # Step 1: Enhance a step
    step_info = {"thought_number": 1}
    system_prompt = "You are a helpful assistant."
    user_prompt = "How can I implement sequential thinking in my application?"
    
    enhanced = await integration.enhance_step(step_info, system_prompt, user_prompt)
    
    # Step 2: Process the step result
    step_result = {
        "thought_number": 1,
        "thought": "Sequential thinking involves breaking down complex problems into steps. Knowledge about sequential thinking implementation details in this project shows that we can structure this as a middleware component."
    }
    
    processed = await integration.process_step_result(step_result, 
                                                     {"system_prompt": system_prompt, "user_prompt": user_prompt}, 
                                                     enhanced)
    
    # Step 3: Get integration stats
    stats = integration.get_integration_stats()
    
    # Verify the complete flow
    assert enhanced["context_applied"] is True
    assert processed["knowledge_captured"] is True
    assert processed["feedback_collected"] is True
    assert len(processed["next_step_suggestions"]) > 0
    assert "feedback" in stats
    assert "context_usage" in stats


if __name__ == "__main__":
    # Manual test
    import asyncio
    
    async def run_test():
        vector_provider = MagicMock(spec=VectorMemoryProvider)
        vector_provider.get_context = AsyncMock(return_value=[
            {
                "content": "Technical documentation for Python asyncio module.",
                "metadata": {"id": "tech1", "level": "techstack"}
            }
        ])
        
        knowledge_capture = MagicMock(spec=KnowledgeCaptureProcessor)
        knowledge_capture.store_knowledge_from_content = AsyncMock()
        
        integration = SequentialKnowledgeIntegration(
            vector_provider=vector_provider,
            knowledge_capture_processor=knowledge_capture
        )
        
        step_info = {"thought_number": 1}
        system_prompt = "You are a helpful assistant."
        user_prompt = "How can I implement sequential thinking?"
        
        enhanced = await integration.enhance_step(step_info, system_prompt, user_prompt)
        print(f"Enhanced step: {json.dumps(enhanced, indent=2)}")
        
        step_result = {
            "thought_number": 1,
            "thought": "Sequential thinking involves breaking down complex problems."
        }
        
        processed = await integration.process_step_result(step_result, 
                                                         {"system_prompt": system_prompt, "user_prompt": user_prompt}, 
                                                         enhanced)
        print(f"Processed result: {json.dumps(processed, indent=2)}")
        
        stats = integration.get_integration_stats()
        print(f"Integration stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(run_test()) 