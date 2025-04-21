#!/usr/bin/env python
"""
Sequential Thinking Core implementation.

This module implements the core logic for sequential thinking,
ported from the TypeScript implementation with enhancements
for integration with the contextual retrieval system.
"""

import logging
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# System prompt ported from TypeScript implementation
SEQUENTIAL_THINKING_SYSTEM_PROMPT = """
You are a dynamic and reflective problem-solver that analyzes problems through a flexible thinking process that can adapt and evolve.
Each thought can build on, question, or revise previous insights as understanding deepens.

Follow these guidelines:
1. Start with an initial estimate of needed thoughts, but be ready to adjust.
2. Feel free to question or revise previous thoughts within the 'thought' text itself.
3. Don't hesitate to add more thoughts if needed, even if it exceeds the initial 'total_thoughts' estimate.
4. Express uncertainty when present.
5. Ignore information that is irrelevant to the current step.
6. Generate a solution hypothesis when appropriate.
7. Verify the hypothesis based on the Chain of Thought steps.
8. Repeat the process until satisfied with the solution.
9. Provide a single, correct answer or the final generated content within the 'thought' field of the last step.
10. Only set next_thought_needed to false when truly done and a satisfactory answer is reached.

Your response MUST be a valid JSON object with ONLY these fields:
- thought: (string) Your current thinking step, analysis, or generated content for this step.
- next_thought_needed: (boolean) True if you need more thinking steps to complete the task, False otherwise.
- thought_number: (integer) Current step number in the sequence (must be positive).
- total_thoughts: (integer) Current estimate of the total thoughts needed (must be positive, can be adjusted).
"""

class SequentialThought(BaseModel):
    """Data model for a sequential thought."""
    thought: str = Field(..., description="The content of the thought")
    next_thought_needed: bool = Field(..., description="Whether another thought is needed")
    thought_number: int = Field(..., description="Current step number in the sequence", gt=0)
    total_thoughts: int = Field(..., description="Estimated total thoughts needed", gt=0)
    
    @validator('thought_number', 'total_thoughts')
    def must_be_positive(cls, v, values, **kwargs):
        if v <= 0:
            param_name = kwargs.get('field').name
            raise ValueError(f"'{param_name}' must be a positive number")
        return v


class SequentialThinkingProcessor:
    """
    Implements sequential thinking logic with context awareness.
    
    This class processes tasks through a step-by-step thinking approach,
    with each step potentially enhanced by contextual information.
    """
    
    def __init__(self, 
                 llm_client: Any, 
                 context_provider: Optional[Any] = None,
                 max_sequential_thoughts: int = 10):
        """
        Initialize the sequential thinking processor.
        
        Args:
            llm_client: The LLM client to use for generating thoughts
            context_provider: Optional provider for contextual information
            max_sequential_thoughts: Maximum number of thoughts to generate
        """
        self.llm_client = llm_client
        self.context_provider = context_provider
        self.max_sequential_thoughts = max_sequential_thoughts
        self._steps = []
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """
        Get all captured steps from the thinking process.
        
        Returns:
            List of step data dictionaries
        """
        return self._steps
    
    async def process_task(self,
                          task: str,
                          system_prompt: Optional[str] = None,
                          initial_context: Optional[Dict[str, Any]] = None,
                          step_callback: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Dict[str, Any]:
        """
        Process a complete task using sequential thinking with enhanced control flow.
        
        This method provides more detailed control and output compared to
        process_with_sequential_thinking, including full step data and metrics.
        
        Args:
            task: The task description to process
            system_prompt: Optional system prompt to guide the thinking
            initial_context: Optional initial context to use
            step_callback: Optional callback function called after each step
            
        Returns:
            Dictionary containing result and metadata about the process
        """
        start_time = time.time()
        self._steps = []
        
        # Process the steps
        result = await self.process_with_sequential_thinking(
            user_prompt=task,
            system_prompt=system_prompt,
            initial_context=initial_context,
            step_callback=step_callback
        )
        
        # Calculate metrics
        completion_time = time.time()
        
        return {
            'result': result,
            'steps': self._steps,
            'metrics': {
                'step_count': len(self._steps),
                'total_time': completion_time - start_time,
                'avg_time_per_step': (completion_time - start_time) / max(1, len(self._steps)),
                'estimated_vs_actual_steps': self._get_estimate_accuracy()
            }
        }
        
    def _get_estimate_accuracy(self) -> Dict[str, Any]:
        """
        Calculate the accuracy of step estimations.
        
        Returns:
            Dictionary with estimation accuracy metrics
        """
        if not self._steps:
            return {'initial_estimate': 0, 'actual_count': 0, 'accuracy': 0.0}
            
        initial_estimate = self._steps[0].get('total_thoughts', 0)
        actual_count = len(self._steps)
        
        if initial_estimate == 0:
            accuracy = 0.0
        else:
            accuracy = min(1.0, actual_count / initial_estimate) if actual_count <= initial_estimate else initial_estimate / actual_count
            
        return {
            'initial_estimate': initial_estimate,
            'actual_count': actual_count,
            'accuracy': accuracy
        }
        
    async def process_with_sequential_thinking(self, 
                                              user_prompt: str, 
                                              system_prompt: Optional[str] = None, 
                                              initial_context: Optional[Dict[str, Any]] = None,
                                              step_callback: Optional[Callable[[Dict[str, Any]], Any]] = None) -> str:
        """
        Process a task using sequential thinking.
        
        Args:
            user_prompt: The prompt to send to the model
            system_prompt: Optional additional system prompt
            initial_context: Optional initial context to use
            step_callback: Optional callback function called after each step
            
        Returns:
            The final result of the sequential thinking process
        """
        thoughts: List[SequentialThought] = []
        current_thought = SequentialThought(
            thought="",
            next_thought_needed=True,
            thought_number=1,
            total_thoughts=5  # Initial estimate
        )
        
        # Combine sequential thinking system prompt with optional additional prompt
        full_system_prompt = f"{SEQUENTIAL_THINKING_SYSTEM_PROMPT}\n\n{system_prompt}" if system_prompt else SEQUENTIAL_THINKING_SYSTEM_PROMPT
        
        # Process thoughts sequentially until next_thought_needed is false or max thoughts reached
        while current_thought.next_thought_needed and len(thoughts) < self.max_sequential_thoughts:
            # Build context for this thought
            thought_context = self._get_thought_context(thoughts)
            
            # Prepare the prompt for this step
            if thought_context:
                initial_prompt = f"{thought_context}\n\nTask: {user_prompt}\n\nContinue with the next thought:"
            else:
                initial_prompt = f"Task: {user_prompt}\n\nProvide your first thought:"
            
            logger.debug(f"Processing thought {current_thought.thought_number} (total estimate: {current_thought.total_thoughts})...")
            
            # Track step start time for metrics
            step_start_time = time.time()
            
            # Get context-enhanced prompt if context provider is available
            enhanced_prompt = await self._enhance_with_context(
                initial_prompt, 
                current_thought.thought_number, 
                thoughts, 
                initial_context
            )
            
            # Calculate context enhancement time
            context_time = time.time() - step_start_time
            
            # Get the next thought with retry logic
            try:
                next_thought = await self._get_next_thought(
                    enhanced_prompt, 
                    full_system_prompt, 
                    current_thought.thought_number
                )
                
                # Add to thoughts history
                thoughts.append(next_thought)
                current_thought = next_thought
                
                # Calculate metrics and prepare step data
                step_end_time = time.time()
                step_data = {
                    **next_thought.dict(),
                    'step_duration': step_end_time - step_start_time,
                    'context_enhancement_time': context_time,
                    'llm_generation_time': step_end_time - step_start_time - context_time,
                }
                
                # Store step data
                self._steps.append(step_data)
                
                # Call step callback if provided
                if step_callback:
                    await self._safe_callback(step_callback, step_data)
                
            except Exception as e:
                logger.error(f"Error during thought generation: {str(e)}")
                raise
        
        # Check if the loop terminated due to max thoughts limit
        if len(thoughts) >= self.max_sequential_thoughts and current_thought.next_thought_needed:
            message = (f"Sequential thinking process terminated after reaching the maximum limit of "
                      f"{self.max_sequential_thoughts} thoughts. The final thought may be incomplete.")
            logger.error(message)
        
        # Extract the solution from the final thought
        return current_thought.thought
    
    async def _enhance_with_context(self, 
                                   prompt: str, 
                                   thought_number: int, 
                                   previous_thoughts: List[SequentialThought],
                                   initial_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance the prompt with contextual information if available.
        
        Args:
            prompt: The original prompt
            thought_number: Current thought number
            previous_thoughts: List of previous thoughts
            initial_context: Optional initial context
            
        Returns:
            Enhanced prompt with contextual information
        """
        if not self.context_provider:
            return prompt
        
        try:
            # Get the task stage from thought number
            if thought_number == 1:
                stage = "problem_definition"
            elif thought_number == 2:
                stage = "information_gathering"
            elif thought_number < len(previous_thoughts):
                stage = "analysis"
            else:
                stage = "conclusion"
            
            # Extract thought contents to provide as context
            thought_contents = [t.thought for t in previous_thoughts]
            
            # Get context from provider
            context = await self.context_provider.get_context_for_step(
                prompt=prompt,
                step=stage,
                previous_steps=thought_contents,
                current_context=initial_context or {}
            )
            
            # If context was found, add it to the prompt
            if context and context.get('relevant_context'):
                relevant_context = context.get('relevant_context')
                return f"Relevant context for this step:\n{relevant_context}\n\n{prompt}"
            
            return prompt
        except Exception as e:
            logger.warning(f"Error enhancing prompt with context: {str(e)}")
            return prompt
    
    def _get_thought_context(self, thoughts: List[SequentialThought]) -> str:
        """
        Build a context string that includes all previous thoughts.
        
        Args:
            thoughts: List of previous thoughts
            
        Returns:
            Formatted context string
        """
        if not thoughts:
            return ""
        
        return "Previous thoughts:\n" + "\n\n".join([
            f"[Thought {t.thought_number}/{t.total_thoughts}]: {t.thought}" 
            for t in thoughts
        ])
    
    async def _get_next_thought(self, 
                               prompt: str, 
                               system_prompt: str, 
                               current_thought_number: int) -> SequentialThought:
        """
        Get the next thought from the LLM, with retry logic for errors.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: The system prompt
            current_thought_number: The current thought number
            
        Returns:
            The next sequential thought
        """
        max_retries = 3
        current_prompt = prompt
        
        for attempt in range(1, max_retries + 1):
            try:
                # Call the LLM
                response = await self.llm_client.generate(
                    prompt=current_prompt,
                    system_prompt=system_prompt
                )
                
                # Parse the response
                try:
                    # First try to parse as JSON directly
                    thought_data = json.loads(response)
                    
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON from text
                    # This handles cases like ```json {...} ``` or other text wrapping
                    logger.debug("JSON parsing failed, attempting to extract JSON from text")
                    json_match = self._extract_json_from_text(response)
                    
                    if not json_match:
                        error_msg = f"LLM output was not valid JSON (attempt {attempt})"
                        logger.warning(error_msg)
                        
                        if attempt < max_retries:
                            # Prepare for retry
                            current_prompt = (f"{prompt}\n\nYour previous attempt (attempt {attempt}) failed. "
                                            f"Please provide a valid JSON object with the required fields.")
                            continue
                        else:
                            # If all retries failed, try to extract just the thought content as fallback
                            fallback_text = self._extract_fallback_thought_text(response)
                            return SequentialThought(
                                thought=fallback_text,
                                next_thought_needed=current_thought_number < 3,  # Simple heuristic
                                thought_number=current_thought_number,
                                total_thoughts=max(current_thought_number, 3)  # Simple heuristic
                            )
                    
                    thought_data = json.loads(json_match)
                
                # Validate the thought data
                thought = SequentialThought(**thought_data)
                return thought
                
            except Exception as e:
                logger.warning(f"Attempt {attempt} to get thought {current_thought_number} failed: {str(e)}")
                
                if attempt < max_retries:
                    # Prepare for retry
                    current_prompt = (f"{prompt}\n\nYour previous attempt (attempt {attempt}) failed with error: {str(e)}. "
                                    f"Please provide a valid JSON object with the required fields.")
                else:
                    # After all retries, rethrow the last error
                    raise
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON from text that might contain formatting or other content.
        
        Args:
            text: Text that might contain JSON
            
        Returns:
            Extracted JSON string or None if not found
        """
        # Try to extract from code blocks (```json ... ```)
        import re
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block_match:
            return code_block_match.group(1)
        
        # Try to find JSON object pattern
        json_pattern_match = re.search(r'(\{[\s\S]*\})', text)
        if json_pattern_match:
            return json_pattern_match.group(1)
        
        return None
    
    def _extract_fallback_thought_text(self, text: str) -> str:
        """
        Extract just the thought text as a fallback when JSON parsing fails.
        
        Args:
            text: Text to extract thought from
            
        Returns:
            Extracted thought text or original text if extraction fails
        """
        try:
            # Look for "thought": "..." pattern
            import re
            thought_match = re.search(r'[\'"]?thought[\'"]?\s*:\s*[\'"]?([\s\S]*?)(?:[\'"]?\s*(?:,|$|\n))', text, re.IGNORECASE)
            if thought_match and thought_match.group(1):
                extracted = thought_match.group(1).strip()
                
                # Remove trailing comma if present
                if extracted.endswith(','):
                    extracted = extracted[:-1].strip()
                
                # Remove surrounding quotes if present
                if ((extracted.startswith('"') and extracted.endswith('"')) or 
                    (extracted.startswith("'") and extracted.endswith("'"))):
                    extracted = extracted[1:-1]
                
                return extracted
            
            # If no match found, return the trimmed input
            return text.strip()
        except Exception as e:
            logger.error(f"Error during fallback text extraction: {str(e)}")
            return text.strip()
    
    async def _safe_callback(self, callback: Callable, data: Dict[str, Any]) -> None:
        """
        Safely execute a callback function, catching any exceptions.
        
        Args:
            callback: The callback function to call
            data: The data to pass to the callback
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"Error in step callback: {str(e)}") 