#!/usr/bin/env python
"""
Knowledge Retrieval with Sequential Thinking Example.

This script demonstrates how to use the knowledge retrieval middleware
with sequential thinking for enhanced contextual reasoning.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any

from beeai_framework.vector.base import VectorMemoryProvider
from beeai_framework.vector.knowledge_capture import KnowledgeCaptureProcessor, KnowledgeCaptureSetting
from beeai_framework.backend.chat import ChatModel

# Import from extensions
from extensions.middleware.sequential.middleware import SequentialThinkingMiddleware, SequentialThinkingRequest
from extensions.middleware.knowledge_retrieval import (
    KnowledgeRetrievalMiddleware,
    KnowledgeRetrievalSettings,
    KnowledgeRetrievalRequest
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockVectorProvider(VectorMemoryProvider):
    """Mock vector provider for testing."""
    
    def __init__(self):
        """Initialize with some mock knowledge entries."""
        self.entries = [
            {
                "content": "Python decorators are a syntactic sugar for applying wrapper functions. They are used to modify the behavior of a function without changing its code.",
                "metadata": {
                    "source": "programming_guide",
                    "category": "code_snippet",
                    "level": "concept",
                    "importance": 0.8
                }
            },
            {
                "content": "The Sequential Thinking framework helps language models break down complex problems into step-by-step reasoning processes.",
                "metadata": {
                    "source": "framework_documentation",
                    "category": "concept",
                    "level": "architecture",
                    "importance": 0.9
                }
            },
            {
                "content": "Vector databases store embeddings which enable semantic search based on meaning rather than exact keyword matches.",
                "metadata": {
                    "source": "vector_db_guide",
                    "category": "explanation",
                    "level": "techstack",
                    "importance": 0.75
                }
            }
        ]
    
    async def store(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Store a new knowledge entry."""
        self.entries.append({"content": content, "metadata": metadata})
        return True
    
    async def search(self, query: str, metadata_filter: Dict[str, Any] = None, limit: int = 5, min_score: float = 0.0) -> List[Any]:
        """Search for knowledge entries."""
        from collections import namedtuple
        SearchResult = namedtuple('SearchResult', ['content', 'metadata', 'score'])
        
        # Simple mock implementation - in real use, this would use embeddings and similarity
        results = []
        for entry in self.entries:
            # Very simple matching logic for the mock
            score = 0.0
            if any(term in entry["content"].lower() for term in query.lower().split()):
                score = 0.7
            
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
        """Check if metadata matches the filter."""
        for key, value in metadata_filter.items():
            if key not in metadata:
                return False
                
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
                
        return True


class MockChatModel(ChatModel):
    """Mock chat model for testing."""
    
    async def generate(self, messages, **kwargs):
        """Generate a response."""
        from beeai_framework.backend.message import Message, AssistantMessage
        
        prompt = messages[-1].content if isinstance(messages, list) else messages.content
        
        # For sequential thinking responses
        if "Task:" in prompt and ("first thought" in prompt or "next thought" in prompt):
            if "first thought" in prompt:
                thought_number = 1
                thought = "This is my first thought about the problem. I need to understand what we're trying to accomplish."
            else:
                # Extract the previous thought number
                import re
                match = re.search(r'thought_number": (\d+)', prompt)
                thought_number = int(match.group(1)) + 1 if match else 2
                thought = f"This is thought #{thought_number}. Building on my previous thinking, I'm getting closer to a solution."
            
            # For the last thought, set next_thought_needed to false
            next_thought_needed = thought_number < 3
            
            response = json.dumps({
                "thought": thought,
                "next_thought_needed": next_thought_needed,
                "thought_number": thought_number,
                "total_thoughts": 3
            })
        else:
            response = "This is a mock response from the chat model."
            
        return AssistantMessage(content=response)


async def demo_knowledge_with_sequential_thinking():
    """Demonstrate knowledge retrieval with sequential thinking."""
    try:
        # Initialize components
        vector_provider = MockVectorProvider()
        chat_model = MockChatModel()
        
        # Initialize knowledge capture processor
        knowledge_capture = KnowledgeCaptureProcessor(
            vector_provider=vector_provider,
            chat_model=chat_model,
            settings=KnowledgeCaptureSetting(enabled=True)
        )
        
        # Capture some knowledge
        user_message = "Can you explain how Python decorators work?"
        assistant_response = """
        Python decorators are a powerful feature that allows you to modify the behavior of functions and methods. 
        They use the @symbol syntax and are applied above the function definition.
        
        Here's a simple example:
        
        ```python
        def my_decorator(func):
            def wrapper():
                print("Something is happening before the function is called.")
                func()
                print("Something is happening after the function is called.")
            return wrapper
        
        @my_decorator
        def say_hello():
            print("Hello!")
        
        say_hello()
        ```
        
        When you run this code, the output will be:
        
        ```
        Something is happening before the function is called.
        Hello!
        Something is happening after the function is called.
        ```
        
        Decorators are commonly used for:
        1. Logging
        2. Timing functions
        3. Authentication and authorization
        4. Rate limiting
        5. Caching
        
        You can also create decorators that take arguments to make them more flexible.
        """
        
        await knowledge_capture.process_message_pair(user_message, assistant_response)
        
        # Initialize knowledge retrieval middleware
        knowledge_retrieval = KnowledgeRetrievalMiddleware(
            vector_provider=vector_provider,
            settings=KnowledgeRetrievalSettings(
                enabled=True,
                max_results=3,
                similarity_threshold=0.5
            )
        )
        
        # Initialize sequential thinking middleware
        sequential_thinking = SequentialThinkingMiddleware(
            llm_client=chat_model,
            context_refinement_processor=None  # Will be set by integration
        )
        
        # Integrate knowledge retrieval with sequential thinking
        await knowledge_retrieval.set_sequential_middleware(sequential_thinking)
        
        # Define progress callback
        def progress_callback(step_data):
            logger.info(f"Step {step_data.step_number}: {step_data.step_content[:50]}...")
        
        # Process a task with sequential thinking
        task = "Explain how Python decorators could be used with a Sequential Thinking framework."
        
        request = SequentialThinkingRequest(
            prompt=task,
            task_type="explanation",
            context={
                "metadata_filter": {
                    "category": ["code_snippet", "concept"]
                }
            },
            streaming=True
        )
        
        logger.info("Processing task with sequential thinking and knowledge retrieval...")
        response = await sequential_thinking.process_request(request, progress_callback=progress_callback)
        
        logger.info(f"Final result: {response.result}")
        
        # Show steps
        logger.info("\nReasoning steps:")
        for i, step in enumerate(response.steps, 1):
            logger.info(f"Step {i}: {step['content'][:100]}...")
        
        # Show metrics
        logger.info("\nMetrics:")
        for key, value in response.metrics.items():
            logger.info(f"{key}: {value}")
        
        # Show context usage
        logger.info("\nContext usage:")
        for key, value in response.context_usage.items():
            logger.info(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(demo_knowledge_with_sequential_thinking()) 