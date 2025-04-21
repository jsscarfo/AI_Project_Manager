#!/usr/bin/env python
"""
Tests for Sequential Thinking Processor.

This module contains tests for the SequentialThinkingProcessor 
class and related functionality for step-by-step reasoning.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional

# Import components to test
from sequential_thinking import (
    SequentialThinkingProcessor,
    SequentialThought,
    SEQUENTIAL_THINKING_SYSTEM_PROMPT
)


class TestSequentialThought:
    """Test cases for SequentialThought data class."""
    
    def test_initialization(self):
        """Test initialization of SequentialThought with required and optional fields."""
        # Basic initialization
        thought = SequentialThought(
            step_number=1,
            content="This is a test thought.",
            reasoning="The reasoning behind this thought."
        )
        
        assert thought.step_number == 1
        assert thought.content == "This is a test thought."
        assert thought.reasoning == "The reasoning behind this thought."
        assert thought.created_at is not None
        assert thought.metadata == {}
        
        # Initialization with metadata
        metadata = {"source": "test", "confidence": 0.9}
        thought_with_metadata = SequentialThought(
            step_number=2,
            content="Another test thought.",
            reasoning="More reasoning.",
            metadata=metadata
        )
        
        assert thought_with_metadata.step_number == 2
        assert thought_with_metadata.metadata == metadata
    
    def test_to_dict(self):
        """Test conversion of SequentialThought to dictionary."""
        thought = SequentialThought(
            step_number=1,
            content="Test content",
            reasoning="Test reasoning",
            metadata={"key": "value"}
        )
        
        thought_dict = thought.to_dict()
        
        assert isinstance(thought_dict, dict)
        assert thought_dict["step_number"] == 1
        assert thought_dict["content"] == "Test content"
        assert thought_dict["reasoning"] == "Test reasoning"
        assert thought_dict["metadata"] == {"key": "value"}
        assert "created_at" in thought_dict
    
    def test_from_dict(self):
        """Test creation of SequentialThought from dictionary."""
        thought_dict = {
            "step_number": 3,
            "content": "Content from dict",
            "reasoning": "Reasoning from dict",
            "metadata": {"source": "dict"},
            "created_at": "2023-01-01T00:00:00"
        }
        
        thought = SequentialThought.from_dict(thought_dict)
        
        assert thought.step_number == 3
        assert thought.content == "Content from dict"
        assert thought.reasoning == "Reasoning from dict"
        assert thought.metadata == {"source": "dict"}
        assert thought.created_at == "2023-01-01T00:00:00"


class TestSequentialThinkingProcessor:
    """Test cases for SequentialThinkingProcessor."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create mock LLM provider
        self.mock_llm_provider = MagicMock()
        self.mock_llm_provider.generate = AsyncMock(return_value="Step 1: First, I'll analyze the requirements.\nReasoning: It's important to understand what is needed before starting to code.")
        self.mock_llm_provider.generate_with_details = AsyncMock(return_value={
            "content": "Step 1: First, I'll analyze the requirements.\nReasoning: It's important to understand what is needed before starting to code.",
            "usage": {"total_tokens": 100}
        })
        
        # Create mock context provider
        self.mock_context_provider = MagicMock()
        self.mock_context_provider.get_enhanced_context = AsyncMock(return_value={
            "context_items": [
                {"content": "Example code", "metadata": {"type": "code"}}
            ],
            "query": "test query"
        })
        
        # Create processor instance for testing
        self.processor = SequentialThinkingProcessor(
            llm_provider=self.mock_llm_provider,
            context_provider=self.mock_context_provider,
            max_steps=5,
            persist_thoughts=True,
            system_prompt=SEQUENTIAL_THINKING_SYSTEM_PROMPT
        )
    
    @pytest.mark.asyncio
    async def test_process_single_step(self):
        """Test processing a single thinking step."""
        # Setup
        task_text = "Write a function to calculate fibonacci numbers"
        step_number = 1
        
        # Execute
        thought = await self.processor.process_step(task_text, step_number)
        
        # Verify
        assert isinstance(thought, SequentialThought)
        assert thought.step_number == 1
        assert "analyze the requirements" in thought.content.lower()
        assert "reasoning" in thought.reasoning.lower()
        
        # Verify context provider was called
        self.mock_context_provider.get_enhanced_context.assert_called_once_with(
            task_text, 
            step_number=1,
            previous_steps=None
        )
        
        # Verify LLM provider was called with appropriate prompt
        call_args = self.mock_llm_provider.generate.call_args[0]
        assert "Step 1" in call_args[0]
        assert task_text in call_args[0]
    
    @pytest.mark.asyncio
    async def test_process_multiple_steps(self):
        """Test processing multiple sequential steps."""
        # Setup
        task_text = "Write a function to calculate fibonacci numbers"
        
        # Modify the mock to return different responses for different steps
        self.mock_llm_provider.generate.side_effect = [
            "Step 1: First, I'll analyze the requirements.\nReasoning: It's important to understand what is needed before starting to code.",
            "Step 2: Now I'll design the algorithm.\nReasoning: A recursive approach works well for Fibonacci.",
            "Step 3: Implementation time.\nReasoning: Python has a clean syntax for recursive functions."
        ]
        
        # Execute
        thoughts = await self.processor.process_multiple_steps(task_text, num_steps=3)
        
        # Verify
        assert isinstance(thoughts, list)
        assert len(thoughts) == 3
        assert all(isinstance(t, SequentialThought) for t in thoughts)
        
        # Verify step sequence
        assert thoughts[0].step_number == 1
        assert thoughts[1].step_number == 2
        assert thoughts[2].step_number == 3
        
        # Verify content of each step
        assert "analyze the requirements" in thoughts[0].content.lower()
        assert "design the algorithm" in thoughts[1].content.lower()
        assert "implementation" in thoughts[2].content.lower()
        
        # Verify the LLM provider was called multiple times
        assert self.mock_llm_provider.generate.call_count == 3
    
    @pytest.mark.asyncio
    async def test_format_thinking_step_prompt(self):
        """Test formatting of the thinking step prompt."""
        # Setup
        task_text = "Write a function to calculate fibonacci numbers"
        step_number = 2
        previous_steps = [
            SequentialThought(
                step_number=1,
                content="First, I'll analyze the requirements",
                reasoning="Understanding the requirements is important"
            )
        ]
        
        # Execute
        prompt = self.processor._format_thinking_step_prompt(
            task_text, 
            step_number, 
            previous_steps
        )
        
        # Verify
        assert isinstance(prompt, str)
        assert task_text in prompt
        assert f"Step {step_number}" in prompt
        assert "Step 1" in prompt
        assert "analyze the requirements" in prompt.lower()
        assert "understanding the requirements" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_parse_thinking_step_response(self):
        """Test parsing of LLM response into a thinking step."""
        # Setup
        response = """
        Step 2: I will design the algorithm using recursion.
        
        Reasoning: Recursion is a natural fit for the Fibonacci sequence since each number 
        is defined in terms of the previous two numbers in the sequence.
        """
        
        # Execute
        thought = self.processor._parse_thinking_step_response(response, 2)
        
        # Verify
        assert isinstance(thought, SequentialThought)
        assert thought.step_number == 2
        assert "design the algorithm" in thought.content.lower()
        assert "recursion is a natural fit" in thought.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_save_and_load_thoughts(self):
        """Test saving and loading thoughts."""
        # Setup - create some thoughts
        thoughts = [
            SequentialThought(
                step_number=1,
                content="Step 1 content",
                reasoning="Step 1 reasoning"
            ),
            SequentialThought(
                step_number=2,
                content="Step 2 content",
                reasoning="Step 2 reasoning"
            )
        ]
        
        # Mock file operations
        with patch("builtins.open", MagicMock()), \
             patch("json.dump") as mock_dump, \
             patch("json.load") as mock_load, \
             patch("os.path.exists") as mock_exists:
            
            # Setup for loading
            mock_exists.return_value = True
            mock_load.return_value = [t.to_dict() for t in thoughts]
            
            # Execute save
            self.processor._save_thoughts(thoughts, "test_task")
            
            # Verify save was called with the right data
            save_args = mock_dump.call_args[0]
            saved_data = save_args[0]
            assert isinstance(saved_data, list)
            assert len(saved_data) == 2
            assert saved_data[0]["step_number"] == 1
            assert saved_data[1]["step_number"] == 2
            
            # Execute load
            loaded_thoughts = self.processor._load_thoughts("test_task")
            
            # Verify loaded thoughts
            assert isinstance(loaded_thoughts, list)
            assert len(loaded_thoughts) == 2
            assert loaded_thoughts[0].step_number == 1
            assert loaded_thoughts[1].step_number == 2
    
    @pytest.mark.asyncio
    async def test_dynamic_system_prompt(self):
        """Test customization of system prompt."""
        # Setup - custom system prompt
        custom_prompt = "You are an AI assistant that thinks step by step. {task}"
        
        # Create processor with custom prompt
        processor = SequentialThinkingProcessor(
            llm_provider=self.mock_llm_provider,
            context_provider=self.mock_context_provider,
            max_steps=5,
            system_prompt=custom_prompt
        )
        
        # Execute
        task_text = "Write a function"
        await processor.process_step(task_text, 1)
        
        # Verify system prompt was used
        call_args = self.mock_llm_provider.generate.call_args[1]
        assert custom_prompt.format(task=task_text) in call_args.get("system_prompt", "")
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during step processing."""
        # Setup - make the LLM provider raise an exception
        self.mock_llm_provider.generate.side_effect = Exception("LLM error")
        
        # Execute and verify exception is handled
        with pytest.raises(Exception) as exc_info:
            await self.processor.process_step("Write a function", 1)
        
        assert "LLM error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_step_with_details(self):
        """Test processing a step with detailed response."""
        # Setup
        task_text = "Write a function to calculate fibonacci numbers"
        step_number = 1
        
        # Execute
        result = await self.processor.process_step_with_details(task_text, step_number)
        
        # Verify
        assert isinstance(result, dict)
        assert "thought" in result
        assert isinstance(result["thought"], SequentialThought)
        assert "metadata" in result
        assert "usage" in result["metadata"]
        assert result["metadata"]["usage"]["total_tokens"] == 100


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 