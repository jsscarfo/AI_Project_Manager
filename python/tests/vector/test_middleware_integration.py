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

"""Integration tests for the contextual enhancement middleware and knowledge capture system."""

import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock, AsyncMock

from beeai_framework.vector.middleware import ContextualEnhancementMiddleware, ContextualEnhancementConfig
from beeai_framework.vector.knowledge_capture import KnowledgeCaptureProcessor, KnowledgeCaptureConfig, ExtractedKnowledgeItem
from beeai_framework.vector.base import ContextResult, ContextMetadata, VectorMemoryProvider
from beeai_framework.middleware.base import MiddlewareContext
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage, AssistantMessage, SystemMessage


class MockVectorProvider(AsyncMock):
    """Mock vector provider for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_contexts = []
        
        # Setup get_context method
        self.get_context = AsyncMock(return_value=[
            ContextResult(
                content="This is a test context about Python programming.",
                metadata=ContextMetadata(
                    source="documentation",
                    category="code_snippet",
                    level="techstack",
                    importance=0.8
                ),
                score=0.95
            ),
            ContextResult(
                content="Project XYZ requires Python 3.10+ and follows PEP8.",
                metadata=ContextMetadata(
                    source="requirements",
                    category="project_info",
                    level="project",
                    importance=0.9
                ),
                score=0.85
            )
        ])
        
        # Setup add_context method
        self.add_context = AsyncMock(side_effect=self._mock_add_context)
        
        # Setup other required methods
        self.add_contexts_batch = AsyncMock(return_value=["id1", "id2"])
        self.clear_context = AsyncMock(return_value=2)
        self.get_stats = AsyncMock(return_value={"total_count": 2})
    
    async def _mock_add_context(self, content, metadata, embedding=None):
        """Mock implementation to store added contexts."""
        self.stored_contexts.append({"content": content, "metadata": metadata})
        return f"id_{len(self.stored_contexts)}"


class MockChatModel(AsyncMock):
    """Mock chat model for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Setup create_json method for knowledge extraction
        self.create_json = AsyncMock(return_value=[
            {
                "content": "Python 3.10 introduces match statements for pattern matching.",
                "category": "best_practice",
                "level": "techstack",
                "confidence": 0.9,
                "importance": 0.8,
                "related_entities": ["Python"]
            }
        ])


@pytest.fixture
def mock_vector_provider():
    """Create a mock vector provider."""
    return MockVectorProvider(spec=VectorMemoryProvider)


@pytest.fixture
def mock_chat_model():
    """Create a mock chat model."""
    return MockChatModel(spec=ChatModel)


@pytest.fixture
def contextual_middleware(mock_vector_provider):
    """Create a contextual enhancement middleware with mock provider."""
    config = ContextualEnhancementConfig(
        max_context_items=3,
        min_similarity_threshold=0.6
    )
    return ContextualEnhancementMiddleware(mock_vector_provider, config)


@pytest.fixture
def knowledge_capture(mock_vector_provider, mock_chat_model):
    """Create a knowledge capture processor with mocks."""
    config = KnowledgeCaptureConfig(
        min_confidence_threshold=0.7,
        min_importance_threshold=0.6
    )
    return KnowledgeCaptureProcessor(mock_vector_provider, mock_chat_model, config)


class TestContextualEnhancementMiddleware:
    """Test cases for the contextual enhancement middleware."""
    
    @pytest.mark.asyncio
    async def test_process_chat_request(self, contextual_middleware):
        """Test processing a chat request."""
        # Create a mock request with messages
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            UserMessage(content="Help me with Python programming")
        ]
        
        # Create a middleware context
        middleware_context = MiddlewareContext(request={"messages": messages})
        
        # Process the context
        result = await contextual_middleware.process(middleware_context)
        
        # Check that context was added
        assert "messages" in result.request
        assert len(result.request["messages"]) >= len(messages)
        
        # Verify task analysis
        assert "contextual_enhancement" in result.metadata
        assert result.metadata["contextual_enhancement"]["task_type"] == "code_implementation"
        
        # Verify vector provider was called
        contextual_middleware.vector_provider.get_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_request(self, contextual_middleware):
        """Test request analysis for different task types."""
        # Test planning query
        planning_query = "Design a high-level architecture for a new system"
        task_type, reasoning_step = await contextual_middleware._analyze_request(planning_query)
        assert task_type == "high_level_planning"
        
        # Test coding query
        coding_query = "Write a Python function to sort a list"
        task_type, reasoning_step = await contextual_middleware._analyze_request(coding_query)
        assert task_type == "code_implementation"
        
        # Test debugging query
        debug_query = "Fix this error in my code: TypeError is not callable"
        task_type, reasoning_step = await contextual_middleware._analyze_request(debug_query)
        assert task_type == "debugging"
    
    @pytest.mark.asyncio
    async def test_sequential_thinking_integration(self, contextual_middleware):
        """Test integration with sequential thinking."""
        # Test getting context for a specific step
        task = "Implement a sorting algorithm"
        step = "problem_definition"
        
        context = await contextual_middleware.get_context_for_step(task, step)
        
        # Verify vector provider was called
        contextual_middleware.vector_provider.get_context.assert_called_once()
        
        # Check result
        assert "retrieved_context" in context
        assert "step" in context
        assert context["step"] == step


class TestKnowledgeCaptureProcessor:
    """Test cases for the knowledge capture processor."""
    
    @pytest.mark.asyncio
    async def test_process_message(self, knowledge_capture):
        """Test processing a message for knowledge extraction."""
        # Test with simple text
        message = "Python 3.10 introduces match statements for pattern matching."
        
        result = await knowledge_capture.process_message(message, "user_input")
        
        # Verify LLM was called
        knowledge_capture.llm.create_json.assert_called_once()
        
        # Check results
        assert len(result) > 0
        assert isinstance(result[0], ExtractedKnowledgeItem)
        assert result[0].category == "best_practice"
        assert result[0].level == "techstack"
        
        # Verify storage
        knowledge_capture.vector_provider.add_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pattern_extraction(self, knowledge_capture):
        """Test pattern-based knowledge extraction."""
        # Message with code block
        message = """
Here's how to use a Python match statement:

```python
def process_value(value):
    match value:
        case [x, y]:
            return x + y
        case {"key": k}:
            return k
        case _:
            return None
```

This is a powerful pattern matching feature.
"""
        
        # Disable LLM extraction to focus on patterns
        knowledge_capture.config.use_llm_extraction = False
        
        result = await knowledge_capture.process_message(message)
        
        # Should extract the code block
        assert len(result) > 0
        assert "```python" not in result[0].content  # Should remove the markers
        assert "match value:" in result[0].content
        assert result[0].category == "code_snippet"
    
    @pytest.mark.asyncio
    async def test_process_conversation(self, knowledge_capture):
        """Test processing a full conversation."""
        # Create a simple conversation
        conversation = [
            UserMessage(content="How do I use Python match statements?"),
            AssistantMessage(content="""
Match statements are new in Python 3.10. Here's an example:

```python
def process_value(value):
    match value:
        case [x, y]:
            return x + y
        case {"key": k}:
            return k
        case _:
            return None
```

This allows pattern matching similar to switch statements in other languages.
"""),
            SystemMessage(content="System message should be ignored"),
            UserMessage(content="Thanks, that's helpful!")
        ]
        
        result = await knowledge_capture.process_conversation(conversation)
        
        # Should have extracted knowledge from both user and assistant messages
        # but not from system message
        assert len(result) > 0
        
        # Check storage
        assert knowledge_capture.vector_provider.add_context.call_count > 0


class TestIntegratedSystem:
    """Test the integration between middleware and knowledge capture."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, contextual_middleware, knowledge_capture):
        """Test the complete flow from context enhancement to knowledge capture."""
        # Create a request
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            UserMessage(content="Explain Python decorators")
        ]
        
        middleware_context = MiddlewareContext(request={"messages": messages})
        
        # Process through middleware to enhance with context
        enhanced_context = await contextual_middleware.process(middleware_context)
        
        # Simulate LLM response
        llm_response = """
Decorators in Python are a powerful way to modify functions.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def hello():
    print("Hello world!")
```

This is a common pattern used in frameworks like Flask.
"""
        
        assistant_message = AssistantMessage(content=llm_response)
        
        # Extract knowledge from the response
        extracted_knowledge = await knowledge_capture.process_message(assistant_message)
        
        # Verify the full cycle worked
        assert len(extracted_knowledge) > 0
        assert knowledge_capture.vector_provider.add_context.call_count > 0
        
        # This knowledge should be available for future context enhancement
        contextual_middleware.vector_provider.get_context.reset_mock()
        
        # Test a follow-up question that should benefit from previously captured knowledge
        follow_up_messages = [
            SystemMessage(content="You are a helpful assistant"),
            UserMessage(content="Show me how to use decorators with arguments")
        ]
        
        follow_up_context = MiddlewareContext(request={"messages": follow_up_messages})
        await contextual_middleware.process(follow_up_context)
        
        # Verify context retrieval was called again
        contextual_middleware.vector_provider.get_context.assert_called_once()


if __name__ == "__main__":
    # Simple manual test for demonstrating the flow
    import asyncio
    
    async def run_demo():
        # This is just a demonstration - use pytest for actual testing
        print("Running demonstration of contextual enhancement and knowledge capture")
        
        # Create mock components
        vector_provider = MockVectorProvider(spec=VectorMemoryProvider)
        chat_model = MockChatModel(spec=ChatModel)
        
        # Create middleware and knowledge capture
        middleware = ContextualEnhancementMiddleware(vector_provider)
        knowledge_processor = KnowledgeCaptureProcessor(vector_provider, chat_model)
        
        # Demonstrate workflow
        print("1. User asks a question")
        user_question = "How do I implement a Python decorator?"
        
        print("2. Middleware enhances with relevant context")
        # (In a real system, this would go through the middleware chain)
        
        print("3. LLM generates a response")
        llm_response = """
Decorators in Python are a powerful way to modify functions.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def hello():
    print("Hello world!")
```

This pattern is used in many frameworks like Flask and Django.
"""
        
        print("4. Knowledge capture extracts and stores knowledge")
        knowledge = await knowledge_processor.process_message(llm_response)
        
        print(f"Extracted {len(knowledge)} knowledge items")
        for item in knowledge:
            print(f"  - {item.content[:50]}... ({item.category}, {item.level}, confidence: {item.confidence})")
        
        print("Demo complete")
    
    asyncio.run(run_demo()) 