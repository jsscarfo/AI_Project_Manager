#!/usr/bin/env python
"""
Tests for Context Refinement Processor.

This module contains tests for the ContextRefinementProcessor
class and related functionality for context optimization.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional

# Import component to test
from context_refinement import ContextRefinementProcessor


class TestContextRefinementProcessor:
    """Test cases for ContextRefinementProcessor."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create mock vector provider
        self.mock_vector_provider = MagicMock()
        self.mock_vector_provider.search = AsyncMock(return_value=[
            {"id": "doc1", "content": "This is document 1", "metadata": {"type": "code"}, "score": 0.95},
            {"id": "doc2", "content": "This is document 2", "metadata": {"type": "comment"}, "score": 0.85},
            {"id": "doc3", "content": "This is document 3", "metadata": {"type": "documentation"}, "score": 0.75},
        ])
        
        # Create mock embedding service
        self.mock_embedding_service = MagicMock()
        self.mock_embedding_service.get_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        
        # Create mock weaviate client
        self.mock_weaviate_client = MagicMock()
        mock_query_result = {
            "data": {
                "Get": {
                    "Context": [
                        {"content": "Weaviate document 1"},
                        {"content": "Weaviate document 2"}
                    ]
                }
            }
        }
        self.mock_weaviate_client.query.get.return_value.do.return_value = mock_query_result
        
        # Create processor instance for testing
        self.processor = ContextRefinementProcessor(
            vector_provider=self.mock_vector_provider,
            embedding_service=self.mock_embedding_service,
            context_window_size=4000,
            max_items_per_level=5,
            enable_progressive_refinement=True,
            enable_content_weighting=True,
            weaviate_client=self.mock_weaviate_client
        )

    @pytest.mark.asyncio
    async def test_get_enhanced_context_basic(self):
        """Test basic context enhancement without template."""
        # Setup
        task_text = "Write a function to calculate fibonacci numbers"
        step_number = 1
        
        # Execute
        result = await self.processor.get_enhanced_context(task_text, step_number)
        
        # Verify
        assert isinstance(result, dict)
        assert "context_items" in result
        assert len(result["context_items"]) > 0
        assert self.mock_vector_provider.search.called
        
        # Verify the vector provider was called with expected parameters
        call_args = self.mock_vector_provider.search.call_args[1]
        assert "query" in call_args
        assert call_args["limit"] == self.processor.max_items_per_level

    @pytest.mark.asyncio
    async def test_determine_step_type(self):
        """Test step type determination for different task types."""
        # Test coding task step types
        assert self.processor._determine_step_type("coding", 1) == "requirements_analysis"
        assert self.processor._determine_step_type("coding", 2) == "algorithm_design"
        assert self.processor._determine_step_type("coding", 3) == "implementation"
        assert self.processor._determine_step_type("coding", 4) == "testing"
        
        # Test research task step types
        assert self.processor._determine_step_type("research", 1) == "information_gathering"
        assert self.processor._determine_step_type("research", 2) == "analysis"
        assert self.processor._determine_step_type("research", 3) == "synthesis"
        
        # Test unknown task type or step
        assert self.processor._determine_step_type("unknown", 1) == "general"
        assert self.processor._determine_step_type("coding", 10) == "general"

    @pytest.mark.asyncio
    async def test_get_template_guided_context(self):
        """Test context retrieval with template guidance."""
        # Setup
        prompt = "Write a recursive function to calculate fibonacci numbers"
        step_type = "implementation"
        template_config = {
            "query_transformations": {
                "implementation": "examples of {topic} implementation in Python"
            },
            "content_filters": {
                "implementation": {"metadata.type": "code"}
            },
            "ranking": {
                "implementation": {
                    "relevance": 0.6,
                    "recency": 0.2,
                    "authority": 0.2
                }
            }
        }
        
        # Execute
        result = await self.processor._get_template_guided_context(
            prompt=prompt,
            step_type=step_type,
            template_config=template_config
        )
        
        # Verify
        assert isinstance(result, dict)
        assert "context_items" in result
        assert len(result["context_items"]) > 0
        
        # Check that template-specific transformations were applied
        call_args = self.mock_vector_provider.search.call_args[1]
        assert "examples of" in call_args["query"]
        assert "fibonacci" in call_args["query"]

    @pytest.mark.asyncio
    async def test_extract_key_terms(self):
        """Test extraction of key terms from text."""
        # Setup
        text = "Write a Python function to calculate the Fibonacci sequence recursively"
        
        # Execute
        key_terms = self.processor._extract_key_terms(text)
        
        # Verify
        assert isinstance(key_terms, list)
        assert len(key_terms) > 0
        assert "fibonacci" in [term.lower() for term in key_terms]
        assert "python" in [term.lower() for term in key_terms]
        assert "function" in [term.lower() for term in key_terms]
        assert "recursively" in [term.lower() for term in key_terms]

    @pytest.mark.asyncio
    async def test_generate_step_specific_query(self):
        """Test generation of step-specific queries."""
        # Setup
        task_text = "Write a Python function to calculate the Fibonacci sequence"
        
        # Test for implementation step
        query_implementation = await self.processor._generate_step_specific_query(
            task_text, "implementation"
        )
        assert "fibonacci" in query_implementation.lower()
        assert "python" in query_implementation.lower()
        assert "implementation" in query_implementation.lower()
        
        # Test for algorithm_design step
        query_design = await self.processor._generate_step_specific_query(
            task_text, "algorithm_design"
        )
        assert "fibonacci" in query_design.lower()
        assert "algorithm" in query_design.lower()
        
        # Test for testing step
        query_testing = await self.processor._generate_step_specific_query(
            task_text, "testing"
        )
        assert "fibonacci" in query_testing.lower()
        assert "test" in query_testing.lower()

    @pytest.mark.asyncio
    async def test_get_context_from_weaviate(self):
        """Test retrieval of context items from Weaviate."""
        # Setup - using the mock client from setup_method
        
        # Execute
        context_items = await self.processor._get_context_from_weaviate(
            query="fibonacci algorithm",
            limit=2
        )
        
        # Verify
        assert isinstance(context_items, list)
        assert len(context_items) == 2
        assert "Weaviate document 1" in context_items[0]["content"]
        assert "Weaviate document 2" in context_items[1]["content"]

    @pytest.mark.asyncio
    async def test_context_caching(self):
        """Test that context items are properly cached."""
        # Setup
        task_text = "Write a function to calculate fibonacci numbers"
        step_number = 1
        
        # Execute first call - should query the vector provider
        result1 = await self.processor.get_enhanced_context(task_text, step_number)
        
        # Reset the mock to verify it's not called again
        self.mock_vector_provider.search.reset_mock()
        
        # Execute second call with same parameters - should use cache
        result2 = await self.processor.get_enhanced_context(task_text, step_number)
        
        # Verify
        assert self.mock_vector_provider.search.called is False
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_progressive_refinement(self):
        """Test progressive refinement of context based on previous steps."""
        # Setup
        self.processor.enable_progressive_refinement = True
        task_text = "Write a function to calculate fibonacci numbers"
        
        # Mock previous steps
        previous_steps = [
            "First I'll analyze the requirements: we need a recursive function to calculate Fibonacci numbers",
            "The algorithm will use the formula F(n) = F(n-1) + F(n-2) with base cases F(0)=0 and F(1)=1"
        ]
        
        # Execute with previous steps
        with patch.object(self.processor, '_extract_key_terms_from_steps') as mock_extract:
            mock_extract.return_value = ["recursive", "formula", "base cases"]
            result = await self.processor.get_enhanced_context(
                task_text, 
                step_number=3,  # implementation step
                previous_steps=previous_steps
            )
        
        # Verify
        assert isinstance(result, dict)
        assert "context_items" in result
        assert len(result["context_items"]) > 0
        assert self.mock_vector_provider.search.called
        
        # Verify the extracted terms were used in the query formation
        args = self.mock_vector_provider.search.call_args[1]
        for term in ["recursive", "formula", "base cases"]:
            assert term.lower() in args["query"].lower()

    @pytest.mark.asyncio
    async def test_content_weighting(self):
        """Test content weighting based on content type and relevance."""
        # Setup - enable content weighting
        self.processor.enable_content_weighting = True
        
        # Create mock context items with different types
        context_items = [
            {"content": "Code example", "metadata": {"type": "code"}, "score": 0.8},
            {"content": "Documentation", "metadata": {"type": "documentation"}, "score": 0.9},
            {"content": "Comment", "metadata": {"type": "comment"}, "score": 0.7}
        ]
        
        # Mock the get_context_candidates method
        with patch.object(self.processor, '_get_context_candidates') as mock_get_context:
            mock_get_context.return_value = context_items
            
            # Execute for implementation step (should prefer code)
            result_impl = await self.processor.get_enhanced_context(
                "Write a function", 
                step_number=3  # implementation step
            )
            
            # Execute for requirements step (should prefer documentation)
            result_req = await self.processor.get_enhanced_context(
                "Write a function", 
                step_number=1  # requirements step
            )
        
        # Verify the context items were reordered based on the step type
        # For implementation step, code should be first
        assert "code" in result_impl["context_items"][0]["metadata"]["type"]
        
        # Verify that some weighting was applied


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 