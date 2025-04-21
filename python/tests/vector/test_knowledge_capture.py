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

"""Tests for the KnowledgeCaptureProcessor."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from beeai_framework.vector.knowledge_capture import (
    KnowledgeCaptureProcessor,
    KnowledgeCaptureSetting,
    KnowledgeEntry,
    KnowledgeExtractor,
    KnowledgeCaptureMiddleware,
    KnowledgeCaptureMiddlewareConfig
)
from beeai_framework.vector.base import ContextMetadata, VectorMemoryProvider
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage, AssistantMessage, SystemMessage
from beeai_framework.middleware.base import MiddlewareContext


@pytest.fixture
def mock_vector_provider():
    """Create a mock vector memory provider."""
    provider = AsyncMock(spec=VectorMemoryProvider)
    provider.store_context = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_chat_model():
    """Create a mock chat model."""
    model = AsyncMock(spec=ChatModel)
    # Mock the generate method to return sample extraction
    model.generate = AsyncMock(return_value=[
        MagicMock(content=json.dumps([
            {
                "content": "Python decorators are syntactic sugar for applying wrapper functions.",
                "metadata": {
                    "category": "explanation",
                    "level": "concept",
                    "importance": 0.8
                }
            }
        ]))
    ])
    return model


@pytest.fixture
def processor(mock_vector_provider, mock_chat_model):
    """Create a KnowledgeCaptureProcessor instance."""
    settings = KnowledgeCaptureSetting(
        enabled=True,
        importance_threshold=0.5,
        min_token_length=5
    )
    return KnowledgeCaptureProcessor(
        vector_provider=mock_vector_provider,
        chat_model=mock_chat_model,
        settings=settings
    )


@pytest.fixture
def middleware(mock_vector_provider, mock_chat_model, processor):
    """Create a KnowledgeCaptureMiddleware instance."""
    config = KnowledgeCaptureMiddlewareConfig(
        enabled=True,
        importance_threshold=0.5,
        min_response_length=5,
        batch_processing=False
    )
    return KnowledgeCaptureMiddleware(
        vector_provider=mock_vector_provider,
        chat_model=mock_chat_model,
        config=config,
        knowledge_processor=processor
    )


@pytest.fixture
def middleware_context():
    """Create a middleware context for testing."""
    # Mock a request with messages
    request = MagicMock()
    request.messages = [
        UserMessage(content="What are Python decorators?")
    ]
    
    # Create context with request
    context = MiddlewareContext(request=request)
    return context


@pytest.mark.asyncio
async def test_process_message_pair(processor, mock_vector_provider):
    """Test processing a message pair."""
    # Arrange
    user_message = "What are Python decorators?"
    assistant_response = "Python decorators are syntactic sugar for applying wrapper functions."
    
    # Act
    result = await processor.process_message_pair(
        user_message=user_message,
        assistant_response=assistant_response
    )
    
    # Assert
    assert len(result) == 1
    assert isinstance(result[0], KnowledgeEntry)
    assert "Python decorators" in result[0].content
    assert result[0].metadata["category"] == "explanation"
    assert result[0].metadata["importance"] >= 0.5
    mock_vector_provider.store_context.assert_called_once()


@pytest.mark.asyncio
async def test_process_conversation(processor, mock_vector_provider):
    """Test processing a conversation with multiple messages."""
    # Arrange
    messages = [
        UserMessage(content="What are Python decorators?"),
        AssistantMessage(content="Python decorators are syntactic sugar for applying wrapper functions."),
        UserMessage(content="Can you show an example?"),
        AssistantMessage(content="Here's an example of a decorator.")
    ]
    
    # Act
    result = await processor.process_conversation(messages)
    
    # Assert
    assert len(result) == 2  # Two pairs of messages
    assert mock_vector_provider.store_context.call_count == 2


@pytest.mark.asyncio
async def test_disabled_processor(mock_vector_provider, mock_chat_model):
    """Test that disabled processor doesn't extract knowledge."""
    # Arrange
    settings = KnowledgeCaptureSetting(enabled=False)
    processor = KnowledgeCaptureProcessor(
        vector_provider=mock_vector_provider,
        chat_model=mock_chat_model,
        settings=settings
    )
    
    # Act
    result = await processor.process_message_pair(
        user_message="What are Python decorators?",
        assistant_response="Python decorators are syntactic sugar for applying wrapper functions."
    )
    
    # Assert
    assert result == []
    mock_vector_provider.store_context.assert_not_called()


@pytest.mark.asyncio
async def test_store_knowledge_from_content(processor, mock_vector_provider):
    """Test storing knowledge directly from content."""
    # Arrange
    content = "When testing async code, use pytest's asyncio plugin."
    metadata = {
        "source": "documentation",
        "category": "best_practice",
        "importance": 0.9
    }
    
    # Act
    result = await processor.store_knowledge_from_content(content, metadata)
    
    # Assert
    assert result is True
    mock_vector_provider.store_context.assert_called_once()
    
    # Verify metadata was passed correctly
    args, kwargs = mock_vector_provider.store_context.call_args
    assert isinstance(kwargs["metadata"], ContextMetadata)
    assert kwargs["metadata"].source == "documentation"
    assert kwargs["metadata"].category == "best_practice"
    assert kwargs["metadata"].importance == 0.9
    assert kwargs["content"] == content


@pytest.mark.asyncio
async def test_middleware_process(middleware, middleware_context):
    """Test the process method of the middleware."""
    # Act
    result = await middleware.process(middleware_context)
    
    # Assert
    assert result is middleware_context
    assert "knowledge_capture" in result.metadata
    assert "timestamp" in result.metadata["knowledge_capture"]
    assert result.metadata["knowledge_capture"]["user_query"] == "What are Python decorators?"


@pytest.mark.asyncio
async def test_middleware_post_process(middleware, middleware_context, mock_vector_provider):
    """Test the post_process method of the middleware."""
    # Arrange
    middleware_context.response = AssistantMessage(content="Python decorators are syntactic sugar for applying wrapper functions.")
    middleware_context.response_generated = True
    
    # Patch asyncio.create_task to run the coroutine immediately instead of scheduling it
    with patch('asyncio.create_task', side_effect=lambda coro: coro):
        # Act
        result = await middleware.post_process(middleware_context)
        
        # Assert
        assert result is middleware_context
        # Verify that vector_provider.add_context was called with appropriate content
        mock_vector_provider.add_context.assert_called_once()
        args, kwargs = mock_vector_provider.add_context.call_args
        assert "Python decorators" in kwargs["content"]


@pytest.mark.asyncio
async def test_middleware_disabled(mock_vector_provider, mock_chat_model, middleware_context):
    """Test that disabled middleware doesn't process."""
    # Arrange
    config = KnowledgeCaptureMiddlewareConfig(enabled=False)
    middleware = KnowledgeCaptureMiddleware(
        vector_provider=mock_vector_provider,
        chat_model=mock_chat_model,
        config=config
    )
    
    # Act
    result = await middleware.process(middleware_context)
    
    # Assert
    assert result is middleware_context
    assert "knowledge_capture" not in result.metadata
    
    # Test post_process with disabled middleware
    middleware_context.response = AssistantMessage(content="Test response")
    middleware_context.response_generated = True
    result = await middleware.post_process(middleware_context)
    assert result is middleware_context
    mock_vector_provider.add_context.assert_not_called()


@pytest.mark.asyncio
async def test_middleware_batch_processing(mock_vector_provider, mock_chat_model, processor, middleware_context):
    """Test batch processing in the middleware."""
    # Arrange
    config = KnowledgeCaptureMiddlewareConfig(
        enabled=True,
        batch_processing=True,
        batch_size=2
    )
    middleware = KnowledgeCaptureMiddleware(
        vector_provider=mock_vector_provider,
        chat_model=mock_chat_model,
        config=config,
        knowledge_processor=processor
    )
    
    # Setup multiple contexts
    contexts = []
    for i in range(3):
        ctx = MiddlewareContext(request=MagicMock())
        ctx.request.messages = [UserMessage(content=f"Question {i}")]
        ctx.response = AssistantMessage(content=f"Answer {i}")
        ctx.response_generated = True
        contexts.append(ctx)
    
    # Patch asyncio.create_task and time.time
    with patch('asyncio.create_task', side_effect=lambda coro: coro), \
         patch('time.time', side_effect=[100, 101, 102, 103, 104, 105, 106, 170]):
        
        # Act - process first context
        await middleware.process(contexts[0])
        result1 = await middleware.post_process(contexts[0])
        assert len(middleware.pending_entries) == 1
        
        # Process second context
        await middleware.process(contexts[1])
        result2 = await middleware.post_process(contexts[1])
        
        # At this point, a batch of 2 should be processed
        assert len(middleware.pending_entries) == 0
        mock_vector_provider.add_contexts_batch.assert_called_once()
        
        # Reset mock and process third context
        mock_vector_provider.add_contexts_batch.reset_mock()
        await middleware.process(contexts[2])
        result3 = await middleware.post_process(contexts[2])
        
        # This time it shouldn't batch yet (only 1 entry)
        assert len(middleware.pending_entries) == 1
        mock_vector_provider.add_contexts_batch.assert_not_called()
        
        # Force batch processing due to timeout (time.time patch returns 170)
        await middleware._capture_knowledge_async(contexts[0])
        
        # Now it should have processed the batch
        assert len(middleware.pending_entries) == 0
        mock_vector_provider.add_contexts_batch.assert_called_once()


@pytest.mark.asyncio
async def test_middleware_shutdown(mock_vector_provider, mock_chat_model, processor):
    """Test shutdown processing of pending entries."""
    # Arrange
    config = KnowledgeCaptureMiddlewareConfig(
        enabled=True,
        batch_processing=True,
        batch_size=10  # Large batch size to prevent auto-processing
    )
    middleware = KnowledgeCaptureMiddleware(
        vector_provider=mock_vector_provider,
        chat_model=mock_chat_model,
        config=config,
        knowledge_processor=processor
    )
    
    # Manually add some pending entries
    middleware.pending_entries = [
        KnowledgeEntry(
            content="Test entry 1",
            metadata={"source": "test", "category": "test", "importance": 0.8}
        ),
        KnowledgeEntry(
            content="Test entry 2",
            metadata={"source": "test", "category": "test", "importance": 0.8}
        )
    ]
    
    # Act
    await middleware.shutdown()
    
    # Assert
    assert len(middleware.pending_entries) == 0
    mock_vector_provider.add_contexts_batch.assert_called_once()
    args, kwargs = mock_vector_provider.add_contexts_batch.call_args
    assert len(args[0]) == 2  # Two contexts in batch


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 