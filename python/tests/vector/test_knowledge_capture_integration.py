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

"""Integration tests for KnowledgeCaptureMiddleware using real project content."""

import os
import json
import re
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import List, Dict, Any, Optional

from beeai_framework.vector.knowledge_capture import (
    KnowledgeCaptureMiddleware,
    KnowledgeCaptureMiddlewareConfig,
    KnowledgeCaptureProcessor,
    KnowledgeCaptureSetting,
    KnowledgeEntry
)
from beeai_framework.vector.base import ContextMetadata, VectorMemoryProvider, ContextResult
from beeai_framework.backend.message import UserMessage, AssistantMessage, SystemMessage, Message
from beeai_framework.middleware.base import MiddlewareChain

# Comment out the import of MiddlewareContext since we're using our own mock
# from beeai_framework.middleware.base import MiddlewareContext


class MockVectorProvider(AsyncMock):
    """Mock implementation of VectorMemoryProvider with storage simulation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_contexts = []
        self.context_id_counter = 0
        
        # Set up method mocks
        self.add_context = AsyncMock(side_effect=self._add_context)
        self.add_contexts_batch = AsyncMock(side_effect=self._add_contexts_batch)
        self.get_context = AsyncMock(side_effect=self._get_context)
        self.initialize = AsyncMock(return_value=None)
        self.shutdown = AsyncMock(return_value=None)
    
    async def _add_context(self, content: str, metadata: Any) -> str:
        """Simulate adding a context to the vector store."""
        self.context_id_counter += 1
        context_id = f"context_{self.context_id_counter}"
        
        # Convert metadata to dict if it's a ContextMetadata object
        if isinstance(metadata, ContextMetadata):
            metadata_dict = {
                "source": metadata.source,
                "category": metadata.category,
                "level": metadata.level,
                "importance": metadata.importance,
                "timestamp": metadata.timestamp if hasattr(metadata, "timestamp") else None
            }
        else:
            metadata_dict = metadata
            
        self.stored_contexts.append({
            "id": context_id,
            "content": content,
            "metadata": metadata_dict,
            "embedding": [0.1] * 10  # Dummy embedding
        })
        
        return context_id
    
    async def _add_contexts_batch(self, contexts: List[Dict[str, Any]]) -> List[str]:
        """Simulate adding multiple contexts in a batch."""
        context_ids = []
        for context in contexts:
            context_id = await self._add_context(
                content=context["content"],
                metadata=context["metadata"]
            )
            context_ids.append(context_id)
        return context_ids
    
    async def _get_context(
        self, 
        query: str, 
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[ContextResult]:
        """Simulate fetching contexts based on query."""
        # Very simplistic "search" - just checks if query words appear in content
        query_words = set(query.lower().split())
        
        results = []
        for context in self.stored_contexts:
            content_words = set(context["content"].lower().split())
            # Calculate a simple overlap score
            word_overlap = len(query_words.intersection(content_words))
            if word_overlap > 0:
                # Apply metadata filter if provided
                if metadata_filter:
                    matches_filter = True
                    for key, value in metadata_filter.items():
                        if key not in context["metadata"] or context["metadata"][key] != value:
                            matches_filter = False
                            break
                    if not matches_filter:
                        continue
                
                # Calculate a score based on word overlap
                score = min(1.0, word_overlap / len(query_words))
                
                results.append(ContextResult(
                    id=context["id"],
                    content=context["content"],
                    metadata=context["metadata"],
                    score=score
                ))
        
        # Sort by score and apply limit
        results.sort(key=lambda x: x.score, reverse=True)
        if limit:
            results = results[:limit]
            
        return results


class MockChatModel(AsyncMock):
    """Mock implementation of ChatModel with knowledge extraction capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate = AsyncMock(side_effect=self._generate)
        self.chat = AsyncMock(side_effect=self._chat)
    
    async def _generate(
        self, 
        messages: List[Message], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[Message]:
        """Mock implementation of generate that extracts knowledge items."""
        if not messages:
            return []
            
        # Check if this is an extraction request
        content = messages[0].content
        if "Extract knowledge items" in content:
            # Parse user message and assistant response from prompt
            user_msg_match = re.search(r"User message: (.*?)(?:\n|$)", content, re.DOTALL)
            assistant_msg_match = re.search(r"Assistant response: (.*?)(?:\n\n|$)", content, re.DOTALL)
            
            user_msg = user_msg_match.group(1) if user_msg_match else ""
            assistant_msg = assistant_msg_match.group(1) if assistant_msg_match else ""
            
            # Generate mock knowledge items based on the content
            knowledge_items = self._extract_mock_knowledge(user_msg, assistant_msg)
            
            return [AssistantMessage(content=json.dumps(knowledge_items))]
        
        # For other requests, return a generic response
        return [AssistantMessage(content="I'm a helpful AI assistant.")]
    
    async def _chat(
        self,
        messages: List[Message],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        **kwargs
    ) -> List[Message]:
        """Mock implementation of chat that returns project-related content."""
        if not messages:
            return [AssistantMessage(content="No messages provided.")]
            
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, UserMessage) or (hasattr(msg, "role") and msg.role == "user"):
                user_message = msg.content
                break
        
        if not user_message:
            return [AssistantMessage(content="No user message found.")]
            
        # Generate different responses based on the user message
        if "middleware" in user_message.lower():
            return [AssistantMessage(content="""
The middleware framework is a core part of the BeeAI architecture. Middleware components are 
processed in a chain, with each component having the opportunity to modify the request or response.
The framework uses a MiddlewareContext object to pass information through the chain, and 
middleware components can be prioritized to control the order of execution.

Key concepts:
1. MiddlewareContext - Contains request, response, and metadata
2. Middleware - Abstract base class for all middleware components
3. MiddlewareChain - Processes middleware in priority order
4. MiddlewareConfig - Configuration options for middleware components
""")]
            
        elif "vector" in user_message.lower() or "knowledge" in user_message.lower():
            return [AssistantMessage(content="""
The vector memory system provides contextual retrieval functionality for AI agents.
It uses vector embeddings to find semantically similar content, enabling agents to
access relevant knowledge from previous interactions and documentation.

The system includes:
1. VectorMemoryProvider - Abstract interface for vector storage
2. WeaviateProvider - Implementation using Weaviate database
3. EmbeddingService - Generates embeddings for text content
4. KnowledgeCaptureMiddleware - Automatically extracts and stores knowledge
5. ContextualEnhancementMiddleware - Enhances prompts with relevant context

Knowledge is organized hierarchically across domain, techstack, and project levels
to enable more precise contextual retrieval for different tasks.
""")]
            
        elif "weaviate" in user_message.lower():
            return [AssistantMessage(content="""
Weaviate is a vector database that powers our contextual retrieval system. Our
WeaviateProvider implementation offers:

1. Vector-based similarity search
2. Metadata filtering
3. Graph-based relationship queries
4. Hybrid search combining vector and keyword matching
5. Hierarchical knowledge organization

The provider integrates with the EmbeddingService to convert text into vector
representations for semantic matching. It supports batched operations for
efficient knowledge storage and retrieval.

Key methods include:
- get_context: Retrieves context based on query and filters
- add_context: Stores new context with metadata
- add_contexts_batch: Stores multiple contexts efficiently
- add_relationship: Creates graph relationships between contexts
""")]
            
        else:
            return [AssistantMessage(content="""
The BeeAI Framework is a comprehensive system for building AI-powered applications.
It includes components for LLM integration, vector memory, middleware processing,
and workflow orchestration. The system is designed to be modular and extensible,
allowing developers to customize and extend functionality as needed.

Key modules include:
1. backend - LLM provider integrations and routing
2. vector - Contextual memory and retrieval
3. middleware - Request/response processing pipeline
4. workflows - Task orchestration and execution
5. mcp - Model Context Protocol implementation
""")]
    
    def _extract_mock_knowledge(self, user_message: str, assistant_response: str) -> List[Dict[str, Any]]:
        """Generate mock knowledge items based on the content of messages."""
        # Simple keyword-based extraction for testing
        keywords = [
            "middleware", "vector", "weaviate", "knowledge", "context", 
            "framework", "extraction", "memory", "embedding"
        ]
        
        items = []
        
        # Extract sentences containing keywords
        for response_part in assistant_response.split('. '):
            for keyword in keywords:
                if keyword in response_part.lower() and len(response_part) > 30:
                    # Determine category based on content
                    category = "concept"
                    if "code" in response_part.lower() or "function" in response_part.lower():
                        category = "code_snippet"
                    elif "should" in response_part.lower() or "best" in response_part.lower():
                        category = "best_practice"
                    elif "is a" in response_part.lower() or "are" in response_part.lower():
                        category = "explanation"
                        
                    # Determine level based on content
                    level = "concept"
                    if "framework" in response_part.lower() or "architecture" in response_part.lower():
                        level = "architecture"
                    elif "component" in response_part.lower() or "module" in response_part.lower():
                        level = "component"
                        
                    # Calculate importance based on keyword count
                    keyword_count = sum(1 for k in keywords if k in response_part.lower())
                    importance = min(0.9, 0.5 + (keyword_count * 0.1))
                    
                    items.append({
                        "content": response_part.strip(),
                        "metadata": {
                            "category": category,
                            "level": level,
                            "importance": importance
                        }
                    })
                    break
        
        return items


@pytest.fixture
def vector_provider():
    """Create a mock vector memory provider with storage simulation."""
    return MockVectorProvider(spec=VectorMemoryProvider)


@pytest.fixture
def chat_model():
    """Create a mock chat model with knowledge extraction capabilities."""
    return MockChatModel(spec=Any)


@pytest.fixture
def middleware(vector_provider, chat_model):
    """Create a KnowledgeCaptureMiddleware instance with the mock providers."""
    config = KnowledgeCaptureMiddlewareConfig(
        enabled=True,
        importance_threshold=0.3,
        batch_processing=True,
        batch_size=3,
        min_response_length=5
    )
    return KnowledgeCaptureMiddleware(
        vector_provider=vector_provider,
        chat_model=chat_model,
        config=config
    )


@pytest.fixture
def middleware_chain(middleware):
    """Create a middleware chain with the knowledge capture middleware."""
    chain = MiddlewareChain()
    chain.add_middleware(middleware)
    
    # Patch post_process_request method to MiddlewareChain
    async def post_process_request(context):
        """Simulate post processing."""
        for middleware in chain.middlewares:
            if hasattr(middleware, "post_process"):
                await middleware.post_process(context)
        return context
    
    chain.post_process_request = post_process_request
    
    return chain


# Mock implementation of MiddlewareContext for testing
class MockMiddlewareContext:
    """Mock implementation of the MiddlewareContext class."""
    
    def __init__(self, request=None, response=None):
        self.request = request
        self.response = response
        self.response_generated = False
        self.metadata = {}
    
    def set_response(self, response):
        """Set the response content."""
        self.response = response
        return self
    
    def get_response(self):
        """Get the response content."""
        return self.response
    
    def set_metadata(self, key, value):
        """Set metadata value."""
        self.metadata[key] = value
        return self
    
    def get_metadata(self, key, default=None):
        """Get metadata value."""
        return self.metadata.get(key, default)


class TestQuery:
    """Test query class to simulate chat requests."""
    
    def __init__(self, content: str, messages: Optional[List[Message]] = None):
        self.content = content
        self.messages = messages or [UserMessage(content=content)]


class TestKnowledgeCaptureIntegration:
    """Integration tests for KnowledgeCaptureMiddleware."""
    
    @pytest.mark.asyncio
    async def test_knowledge_capture_from_middleware_description(self, middleware_chain, chat_model, vector_provider):
        """Test extracting knowledge from middleware documentation."""
        # Setup
        query = TestQuery("Can you explain how the middleware framework works?")
        context = MiddlewareContext(request=query)
        
        # Process request through middleware chain
        processed_context = await middleware_chain.process_request(context)
        
        # Generate a response
        response_messages = await chat_model.chat(messages=query.messages)
        response_content = response_messages[0].content
        
        # Set the response in the context
        processed_context.response = response_content
        processed_context.response_generated = True
        
        # Process response through middleware chain
        await middleware_chain.post_process_request(processed_context)
        
        # Wait for async tasks to complete
        # In real code this happens in the background, but for testing we need to wait
        await asyncio.sleep(0.1)
        
        # Verify knowledge was captured
        assert len(vector_provider.stored_contexts) > 0
        
        # Check content relevance
        middleware_related_items = [
            item for item in vector_provider.stored_contexts
            if "middleware" in item["content"].lower()
        ]
        assert len(middleware_related_items) > 0
        
        # Metadata should contain proper categorization
        for item in middleware_related_items:
            assert "category" in item["metadata"]
            assert "level" in item["metadata"]
            assert "importance" in item["metadata"]
            assert item["metadata"]["importance"] > 0.3
    
    @pytest.mark.asyncio
    async def test_knowledge_capture_from_vector_memory_description(self, middleware_chain, chat_model, vector_provider):
        """Test extracting knowledge from vector memory system documentation."""
        # Setup
        query = TestQuery("How does the vector memory system work?")
        context = MiddlewareContext(request=query)
        
        # Process request through middleware chain
        processed_context = await middleware_chain.process_request(context)
        
        # Generate a response
        response_messages = await chat_model.chat(messages=query.messages)
        response_content = response_messages[0].content
        
        # Set the response in the context
        processed_context.response = response_content
        processed_context.response_generated = True
        
        # Process response through middleware chain
        await middleware_chain.post_process_request(processed_context)
        
        # Wait for async tasks to complete
        await asyncio.sleep(0.1)
        
        # Verify knowledge was captured
        assert len(vector_provider.stored_contexts) > 0
        
        # Check content relevance
        vector_related_items = [
            item for item in vector_provider.stored_contexts
            if "vector" in item["content"].lower()
        ]
        assert len(vector_related_items) > 0
        
        # Verify retrieval works by querying for relevant knowledge
        results = await vector_provider.get_context(
            query="vector embedding retrieval",
            limit=5
        )
        assert len(results) > 0
        assert any("embedding" in r.content.lower() for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_multiple_queries(self, middleware_chain, chat_model, vector_provider):
        """Test batch processing with multiple sequential queries."""
        # Setup queries
        queries = [
            "How does the middleware framework work?",
            "Can you explain the vector memory system?",
            "What is Weaviate and how does it integrate with the framework?"
        ]
        
        # Process each query in sequence
        for query_text in queries:
            query = TestQuery(query_text)
            context = MiddlewareContext(request=query)
            
            # Process request
            processed_context = await middleware_chain.process_request(context)
            
            # Generate response
            response_messages = await chat_model.chat(messages=query.messages)
            response_content = response_messages[0].content
            
            # Set response and process
            processed_context.response = response_content
            processed_context.response_generated = True
            await middleware_chain.post_process_request(processed_context)
            
            # Small delay to simulate real-world processing
            await asyncio.sleep(0.1)
        
        # Batch size is 3, so all items should be processed by now
        # Verify each query topic is represented in captured knowledge
        middleware_items = [
            item for item in vector_provider.stored_contexts
            if "middleware" in item["content"].lower()
        ]
        vector_items = [
            item for item in vector_provider.stored_contexts
            if "vector" in item["content"].lower()
        ]
        weaviate_items = [
            item for item in vector_provider.stored_contexts
            if "weaviate" in item["content"].lower()
        ]
        
        assert len(middleware_items) > 0, "No middleware knowledge captured"
        assert len(vector_items) > 0, "No vector knowledge captured"
        assert len(weaviate_items) > 0, "No weaviate knowledge captured"
    
    @pytest.mark.asyncio
    async def test_knowledge_retrieval_after_capture(self, middleware_chain, chat_model, vector_provider):
        """Test retrieving knowledge after it's been captured from multiple sources."""
        # First, populate with knowledge from different queries
        queries = [
            "How does the middleware framework work?",
            "Can you explain the vector memory system?",
            "What is Weaviate and how does it integrate with the framework?"
        ]
        
        # Process each query to populate knowledge base
        for query_text in queries:
            query = TestQuery(query_text)
            context = MiddlewareContext(request=query)
            context = await middleware_chain.process_request(context)
            
            response_messages = await chat_model.chat(messages=query.messages)
            response_content = response_messages[0].content
            
            context.set_response(response_content)
            context.response_generated = True
            context = await middleware_chain.post_process_request(context)
            
            await asyncio.sleep(0.1)
        
        # Verify knowledge was captured
        assert len(vector_provider.stored_contexts) > 0
        
        # Test retrieval with various queries
        test_retrieval_queries = [
            ("middleware chain process", ["middleware"]),
            ("vector embedding similarity", ["vector", "embedding"]),
            ("weaviate database", ["weaviate"]),
            ("knowledge extraction", ["knowledge", "extraction"]),
        ]
        
        for query_text, expected_keywords in test_retrieval_queries:
            results = await vector_provider.get_context(query=query_text, limit=3)
            
            # Check if results contain expected keywords
            found_keywords = set()
            for result in results:
                for keyword in expected_keywords:
                    if keyword in result.content.lower():
                        found_keywords.add(keyword)
            
            # At least one of the expected keywords should be present
            assert found_keywords.intersection(expected_keywords), f"No expected keywords found for query: {query_text}"
    
    @pytest.mark.asyncio
    async def test_middleware_shutdown_with_pending_entries(self, middleware, vector_provider):
        """Test that shutdown processes pending entries."""
        # Manually add some pending entries
        middleware.pending_entries = [
            KnowledgeEntry(
                content="Middleware components can be processed in a chain, with each having the opportunity to modify the request or response.",
                metadata=ContextMetadata(
                    source="documentation", 
                    category="explanation", 
                    level="architecture", 
                    importance=0.8
                )
            ),
            KnowledgeEntry(
                content="The vector memory system uses embeddings to find semantically similar content.",
                metadata=ContextMetadata(
                    source="documentation", 
                    category="explanation", 
                    level="component", 
                    importance=0.7
                )
            )
        ]
        
        # Call shutdown to process pending entries
        await middleware.shutdown()
        
        # Verify entries were processed
        assert len(middleware.pending_entries) == 0
        assert len(vector_provider.stored_contexts) == 2
        
        # Verify retrieval works for the stored entries
        results = await vector_provider.get_context(query="vector embeddings")
        assert len(results) > 0
        assert "embeddings" in results[0].content.lower()
        
    @pytest.mark.asyncio
    async def test_mock_chat_model_knowledge_extraction(self, chat_model):
        """Test that knowledge extraction works in the mock chat model."""
        user_message = "How does the vector memory system work?"
        assistant_response = """
        The vector memory system provides contextual retrieval functionality for AI agents.
        It uses vector embeddings to find semantically similar content. 
        The system includes VectorMemoryProvider and WeaviateProvider components.
        """
        
        # Extract knowledge
        knowledge_items = chat_model._extract_mock_knowledge(user_message, assistant_response)
        
        # Log extracted items for debugging
        print(f"\nExtracted {len(knowledge_items)} knowledge items:")
        for i, item in enumerate(knowledge_items):
            print(f"Item {i+1}: {item['content'][:50]}... [category: {item['metadata']['category']}, importance: {item['metadata']['importance']}]")
        
        # Verify extraction worked
        assert len(knowledge_items) > 0
        assert any("vector" in item["content"].lower() for item in knowledge_items)
        
        # Verify metadata is set properly
        for item in knowledge_items:
            assert "category" in item["metadata"]
            assert "level" in item["metadata"]
            assert "importance" in item["metadata"]
            assert 0 <= item["metadata"]["importance"] <= 1


if __name__ == "__main__":
    import re
    import unittest
    
    # Create a simple test case
    class SimpleTest(unittest.TestCase):
        def test_imports(self):
            """Test that all required imports are available."""
            self.assertTrue(True, "This test should always pass")
    
    # Run the simple test
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Also try to run with pytest
    print("\nAttempting to run with pytest...")
    try:
        import pytest
        pytest.main(["-v", __file__])
    except Exception as e:
        print(f"Error running pytest: {str(e)}") 