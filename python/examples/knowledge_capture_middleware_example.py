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

"""
Knowledge Capture Middleware Example

This example demonstrates how to use the KnowledgeCaptureMiddleware to
automatically extract and store knowledge from LLM interactions.
"""

import os
import asyncio
import argparse
from typing import List, Dict, Any
import logging

from beeai_framework.backend.providers import OpenAIProvider
from beeai_framework.backend.router import ModelRouter
from beeai_framework.vector import (
    WeaviateProvider, 
    WeaviateProviderConfig,
    EmbeddingService,
    KnowledgeCaptureMiddleware,
    KnowledgeCaptureMiddlewareConfig
)
from beeai_framework.middleware.base import MiddlewareChain, MiddlewareContext
from beeai_framework.backend.message import UserMessage, SystemMessage, Message

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatRequest:
    """Simple chat request representation."""
    
    def __init__(self, messages: List[Message]):
        self.messages = messages


class ChatResponse:
    """Simple chat response representation."""
    
    def __init__(self, content: str):
        self.content = content


async def main():
    """Run the knowledge capture middleware example."""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    
    # Initialize Weaviate provider
    logger.info("Initializing Weaviate provider...")
    weaviate_config = WeaviateProviderConfig(
        host=weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
        port=int(weaviate_url.split(":")[-1]) if ":" in weaviate_url else 8080,
        protocol="http",
        class_name="KnowledgeCapture",
        create_schema_if_missing=True
    )
    
    # Initialize OpenAI provider for embeddings and chat
    openai_provider = OpenAIProvider(api_key=openai_api_key)
    
    # Create embedding service
    embedding_service = EmbeddingService(
        embedding_fn=lambda text, model_id: asyncio.run(openai_provider.get_embeddings(text))
    )
    
    # Initialize Weaviate provider with embedding service
    vector_provider = WeaviateProvider(
        config=weaviate_config,
        embedding_service=embedding_service
    )
    await vector_provider.initialize()
    
    # Initialize model router with OpenAI
    model_router = ModelRouter()
    model_router.register_provider("openai", openai_provider)
    
    # Configure the knowledge capture middleware
    logger.info("Setting up knowledge capture middleware...")
    knowledge_capture_config = KnowledgeCaptureMiddlewareConfig(
        enabled=True,
        importance_threshold=0.5,
        batch_processing=True,
        batch_size=3,
        min_response_length=10
    )
    
    knowledge_capture_middleware = KnowledgeCaptureMiddleware(
        vector_provider=vector_provider,
        chat_model=openai_provider,
        config=knowledge_capture_config
    )
    
    # Set up middleware chain
    middleware_chain = MiddlewareChain()
    middleware_chain.add_middleware(knowledge_capture_middleware)
    
    # In a production application, you might add other middleware components as well
    # For example:
    # middleware_chain.add_middleware(logging_middleware)
    # middleware_chain.add_middleware(error_handling_middleware)
    
    # The middleware chain processes components in order of priority
    # Lower priority values run first
    
    # Run example interactions
    await run_example_interactions(middleware_chain, openai_provider)
    
    # Perform a query to verify captured knowledge
    await verify_captured_knowledge(vector_provider)
    
    # Clean up resources
    await knowledge_capture_middleware.shutdown()
    await vector_provider.shutdown()


async def run_example_interactions(middleware_chain, chat_model):
    """Run a series of example interactions through the middleware chain."""
    example_queries = [
        (
            "What are Python generators and how do they work?",
            "System message: You are a helpful assistant."
        ),
        (
            "Explain the difference between shallow and deep copy in Python.",
            "System message: You are a technical expert."
        ),
        (
            "What are the best practices for error handling in Python?",
            "System message: You are an expert programmer."
        ),
        (
            "How does the Global Interpreter Lock (GIL) affect Python multithreading?",
            "System message: You are a concurrency expert."
        ),
    ]
    
    logger.info(f"Running {len(example_queries)} example interactions...")
    
    for query, system_prompt in example_queries:
        # Create a chat request
        request = ChatRequest(messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=query)
        ])
        
        # Create middleware context with the request
        context = MiddlewareContext(request=request)
        
        # Process through middleware (pre-response phase)
        context = await middleware_chain.process_request(context)
        
        # Generate a response
        logger.info(f"Processing query: {query}")
        response_messages = await chat_model.chat(
            messages=request.messages,
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        response_content = response_messages[0].content
        
        # Create a response and add it to the context
        context.set_response(ChatResponse(content=response_content))
        
        # Process through middleware (post-response phase)
        if hasattr(middleware_chain, "post_process_request"):
            context = await middleware_chain.post_process_request(context)
        
        # Print the response (truncated for brevity)
        logger.info(f"Response: {response_content[:100]}...")
        
        # Pause briefly between requests
        await asyncio.sleep(1)


async def verify_captured_knowledge(vector_provider):
    """Query the vector database to verify the captured knowledge."""
    logger.info("Verifying captured knowledge...")
    
    # Example query to check if knowledge was stored
    query = "Python generators"
    results = await vector_provider.get_context(
        query=query,
        limit=5
    )
    
    logger.info(f"Found {len(results)} knowledge entries for query: '{query}'")
    
    # Display a sample of the results
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: {result.content[:100]}...")
        logger.info(f"   Metadata: {result.metadata}")
        logger.info(f"   Score: {result.score}")
        logger.info("---")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        logger.info("Closing event loop")
        loop.close() 