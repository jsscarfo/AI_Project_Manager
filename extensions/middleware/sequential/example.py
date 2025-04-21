#!/usr/bin/env python
"""
Example usage of the Sequential Thinking Framework.

This script demonstrates how to use the sequential thinking framework
with context refinement and reasoning trace analysis.
"""

import asyncio
import logging
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleLLMClient:
    """Simple mock LLM client for demonstration purposes."""
    
    async def generate(self, prompt: str, system_prompt: str = None):
        """
        Generate a response from a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        # In a real implementation, this would call an actual LLM API
        # For demonstration, we'll return a mock response
        
        # Parse the prompt to determine which step we're on
        step_number = 1
        if "Previous thoughts:" in prompt:
            # Count the number of thought blocks to determine step number
            step_number = prompt.count("[Thought ") + 1
        
        # Create mock responses based on step
        if step_number == 1:
            return json.dumps({
                "thought": f"First, I need to understand the key aspects of this problem. The main points are: 1) [...] 2) [...]. Let me think about the best approach.",
                "next_thought_needed": True,
                "thought_number": 1,
                "total_thoughts": 4
            })
        elif step_number == 2:
            return json.dumps({
                "thought": f"Based on my initial analysis, I should gather additional information about [...]. This will help me develop a more comprehensive solution.",
                "next_thought_needed": True,
                "thought_number": 2,
                "total_thoughts": 4
            })
        elif step_number == 3:
            return json.dumps({
                "thought": f"Now I can analyze the information I've gathered. The key insights are: [...]. These suggest that the best approach would be [...].",
                "next_thought_needed": True,
                "thought_number": 3,
                "total_thoughts": 4
            })
        else:
            return json.dumps({
                "thought": f"Based on my analysis, the solution is [...]. This addresses all the requirements and constraints of the problem.",
                "next_thought_needed": False,
                "thought_number": 4,
                "total_thoughts": 4
            })


class SimpleVectorProvider:
    """Simple mock vector provider for demonstration purposes."""
    
    async def get_context(self, query: str, metadata_filter: Dict[str, Any] = None, limit: int = 5):
        """
        Get context for a query.
        
        Args:
            query: Query to search for
            metadata_filter: Filter for metadata
            limit: Maximum number of results
            
        Returns:
            List of context items
        """
        logger.info(f"Getting context for query: {query[:50]}...")
        
        # In a real implementation, this would query a vector database
        # For demonstration, we'll return mock results
        
        level = metadata_filter.get("level", "project") if metadata_filter else "project"
        
        results = []
        if level == "domain":
            results.append({
                "id": "dom1",
                "content": "Domain knowledge about the topic: [...]",
                "metadata": {
                    "source": "knowledge_base",
                    "level": "domain",
                    "timestamp": 1677721600  # March 2023
                }
            })
        elif level == "techstack":
            results.append({
                "id": "tech1",
                "content": "Technical documentation about the relevant technologies: [...]",
                "metadata": {
                    "source": "documentation",
                    "level": "techstack",
                    "timestamp": 1672531200  # January 2023
                }
            })
        else:  # project
            results.append({
                "id": "proj1",
                "content": "Project-specific information: [...]",
                "metadata": {
                    "source": "codebase",
                    "level": "project",
                    "timestamp": 1680393600  # April 2023
                }
            })
            
        return results[:limit]


class SimpleEmbeddingService:
    """Simple mock embedding service for demonstration purposes."""
    
    async def get_embedding(self, text: str):
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        # In a real implementation, this would use an embedding model
        # For demonstration, we'll return a mock embedding
        import hashlib
        import numpy as np
        
        # Generate a deterministic but unique embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16))
        
        # Create a mock embedding
        return np.random.randn(1536).tolist()


async def main():
    """Main function to demonstrate the sequential thinking framework."""
    # Import the necessary modules
    from core import SequentialThinkingProcessor
    from context_refinement import ContextRefinementProcessor
    from reasoning_trace import ReasoningTrace, ReasoningStep, ReasoningTraceStore, ReasoningTraceAnalyzer
    from context_templates import TemplateManager
    from middleware import SequentialThinkingMiddleware, SequentialThinkingRequest
    
    # Create mock components
    llm_client = SimpleLLMClient()
    vector_provider = SimpleVectorProvider()
    embedding_service = SimpleEmbeddingService()
    
    # Create context refinement processor
    context_processor = ContextRefinementProcessor(
        vector_provider=vector_provider,
        embedding_service=embedding_service,
        context_window_size=4000
    )
    
    # Create template manager
    template_manager = TemplateManager()
    
    # Create trace store
    trace_store = ReasoningTraceStore("./traces")
    
    # Create middleware
    middleware = SequentialThinkingMiddleware(
        llm_client=llm_client,
        context_refinement_processor=context_processor,
        template_manager=template_manager,
        trace_store=trace_store
    )
    
    # Define a progress callback
    def progress_callback(progress_data):
        logger.info(f"Step {progress_data.step_number} completed: {progress_data.step_content[:50]}...")
        if progress_data.is_final:
            logger.info("Final step reached!")
    
    # Create a request
    request = SequentialThinkingRequest(
        prompt="How can we implement a caching system for the API to improve performance?",
        task_type="planning",
        streaming=True
    )
    
    # Process the request
    logger.info("Processing request...")
    response = await middleware.process_request(request, progress_callback=progress_callback)
    
    # Print the result
    logger.info(f"Result: {response.result}")
    logger.info(f"Processing time: {response.timing['total']:.2f} seconds")
    
    # Analyze the trace
    logger.info("Analyzing trace...")
    analysis = await middleware.analyze_trace(response.trace_id)
    logger.info(f"Analysis complete: {json.dumps(analysis, indent=2)}")
    
    # Get visualization
    logger.info("Generating visualization...")
    visualization = await middleware.get_trace_visualization(response.trace_id)
    logger.info(f"Visualization generated: {json.dumps(visualization, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main()) 