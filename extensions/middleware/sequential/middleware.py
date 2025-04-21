#!/usr/bin/env python
"""
Sequential Thinking Middleware Integration.

This module implements the middleware integration for sequential thinking,
providing hooks for pre-processing, post-processing, and monitoring.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union, Type
from pydantic import BaseModel, Field
import asyncio
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequentialThinkingRequest(BaseModel):
    """Model for sequential thinking middleware request."""
    prompt: str = Field(..., description="The prompt to process")
    task_type: str = Field("general", description="The type of task being performed")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    template_name: Optional[str] = Field(None, description="Name of template to use")
    max_steps: Optional[int] = Field(None, description="Maximum number of steps")
    trace_id: Optional[str] = Field(None, description="Trace ID for logging")
    streaming: bool = Field(False, description="Whether to stream results")


class SequentialThinkingResponse(BaseModel):
    """Model for sequential thinking middleware response."""
    result: str = Field(..., description="The final result")
    trace_id: str = Field(..., description="Trace ID for reference")
    task_type: str = Field(..., description="The type of task that was performed")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Steps in the reasoning process")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    context_usage: Dict[str, Any] = Field(default_factory=dict, description="Context usage statistics")
    timing: Dict[str, float] = Field(default_factory=dict, description="Timing information")


class StepProgressCallback(BaseModel):
    """Model for step progress callback data."""
    trace_id: str = Field(..., description="Trace ID for reference")
    step_number: int = Field(..., description="Current step number")
    step_content: str = Field(..., description="Content of the current step")
    step_type: str = Field(..., description="Type of the current step")
    total_steps: int = Field(..., description="Estimated total steps")
    is_final: bool = Field(False, description="Whether this is the final step")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Step-specific metrics")


class SequentialThinkingMiddleware:
    """
    Middleware for integrating sequential thinking with LLM providers.
    
    This middleware handles the orchestration of sequential thinking processes,
    including context enhancement, template selection, and progress tracking.
    """
    
    def __init__(self, 
                 llm_client: Any,
                 context_refinement_processor: Optional[Any] = None,
                 template_manager: Optional[Any] = None,
                 trace_store: Optional[Any] = None,
                 metrics_collector: Optional[Any] = None):
        """
        Initialize the sequential thinking middleware.
        
        Args:
            llm_client: Client for LLM API calls
            context_refinement_processor: Processor for context refinement
            template_manager: Manager for context templates
            trace_store: Store for reasoning traces
            metrics_collector: Collector for metrics
        """
        self.llm_client = llm_client
        self.context_refinement_processor = context_refinement_processor
        self.template_manager = template_manager
        self.trace_store = trace_store
        self.metrics_collector = metrics_collector
        
        # Import necessary modules
        try:
            from .core import SequentialThinkingProcessor
            from .reasoning_trace import ReasoningTrace, ReasoningStep
            self.SequentialThinkingProcessor = SequentialThinkingProcessor
            self.ReasoningTrace = ReasoningTrace
            self.ReasoningStep = ReasoningStep
        except ImportError as e:
            logger.error(f"Error importing required modules: {str(e)}")
            raise
    
    async def process_request(self, 
                             request: SequentialThinkingRequest, 
                             progress_callback: Optional[Callable[[StepProgressCallback], None]] = None) -> SequentialThinkingResponse:
        """
        Process a sequential thinking request.
        
        Args:
            request: The request to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Response with results and metadata
        """
        start_time = time.time()
        
        # Generate trace ID if not provided
        trace_id = request.trace_id or str(uuid.uuid4())
        
        # Create metrics dict
        metrics = {
            "trace_id": trace_id,
            "task_type": request.task_type,
            "start_time": start_time,
            "steps_completed": 0,
            "context_fetches": 0,
            "template_used": request.template_name
        }
        
        # Select template based on task type
        template = None
        if self.template_manager:
            if request.template_name:
                template = self.template_manager.get_template(request.template_name)
            if not template:
                template = self.template_manager.get_template_for_task(request.task_type)
                if template:
                    metrics["template_used"] = template.name
        
        # Create reasoning trace
        trace = self.ReasoningTrace(
            trace_id=trace_id,
            task=request.prompt,
            start_time=start_time,
            metadata={
                "task_type": request.task_type,
                "template": template.name if template else None
            }
        )
        
        # Initialize sequence processor
        processor = self.SequentialThinkingProcessor(
            llm_client=self.llm_client,
            context_provider=self.context_refinement_processor,
            max_sequential_thoughts=request.max_steps or 10
        )
        
        # Create context for steps
        initial_context = request.context or {}
        if template:
            initial_context["template"] = template.to_dict()
        
        # Define step callback to capture progress
        steps = []
        context_usage_data = {"fetches": 0, "sources": {}, "quality_scores": []}
        
        async def step_callback(step_data):
            # Create reasoning step
            step = self.ReasoningStep(
                step_id=str(uuid.uuid4()),
                thought_number=step_data["thought_number"],
                content=step_data["thought"],
                timestamp=time.time(),
                step_type=_determine_step_type(step_data["thought_number"]),
                requires_next_step=step_data["next_thought_needed"],
                context_references=step_data.get("context_references", []),
                metrics=step_data.get("metrics", {})
            )
            
            # Add step to trace
            trace.add_step(step)
            
            # Store step data for response
            steps.append({
                "number": step.thought_number,
                "content": step.content,
                "type": step.step_type,
                "requires_next_step": step.requires_next_step,
                "timestamp": step.timestamp
            })
            
            # Update metrics
            metrics["steps_completed"] = len(steps)
            
            # Update context usage data
            if "context_metrics" in step_data:
                context_usage_data["fetches"] += 1
                context_metrics = step_data["context_metrics"]
                context_usage_data["quality_scores"].append(context_metrics.get("quality_score", 0))
                
                # Track sources
                for source in context_metrics.get("sources", []):
                    if source not in context_usage_data["sources"]:
                        context_usage_data["sources"][source] = 0
                    context_usage_data["sources"][source] += 1
            
            # Call progress callback if provided
            if progress_callback:
                progress_data = StepProgressCallback(
                    trace_id=trace_id,
                    step_number=step.thought_number,
                    step_content=step.content,
                    step_type=step.step_type,
                    total_steps=step_data["total_thoughts"],
                    is_final=not step.requires_next_step,
                    metrics=step.metrics
                )
                
                # Call in a non-blocking way
                asyncio.create_task(
                    _call_callback_with_timeout(progress_callback, progress_data)
                )
        
        # Process with sequential thinking
        try:
            # Set up monitoring hooks
            pre_processing_time = time.time() - start_time
            
            if request.streaming:
                # Streaming mode with step callback
                result = await processor.process_with_sequential_thinking(
                    user_prompt=request.prompt,
                    system_prompt=request.system_prompt,
                    initial_context=initial_context,
                    step_callback=step_callback
                )
            else:
                # Non-streaming mode, process all at once
                result = await processor.process_with_sequential_thinking(
                    user_prompt=request.prompt,
                    system_prompt=request.system_prompt,
                    initial_context=initial_context
                )
                
                # Extract steps from processor for non-streaming case
                for step_data in processor.get_steps():
                    await step_callback(step_data)
            
            processing_time = time.time() - start_time - pre_processing_time
            
            # Store the reasoning trace if store is available
            if self.trace_store:
                await self.trace_store.add_trace(trace)
            
            # Calculate final metrics
            metrics["end_time"] = time.time()
            metrics["total_time"] = metrics["end_time"] - start_time
            metrics["processing_time"] = processing_time
            metrics["pre_processing_time"] = pre_processing_time
            metrics["success"] = True
            
            # Calculate average context quality
            if context_usage_data["quality_scores"]:
                context_usage_data["avg_quality"] = sum(context_usage_data["quality_scores"]) / len(context_usage_data["quality_scores"])
            else:
                context_usage_data["avg_quality"] = 0.0
            
            # Log metrics if collector is available
            if self.metrics_collector:
                await self.metrics_collector.record_metrics("sequential_thinking", metrics)
            
            # Create and return response
            return SequentialThinkingResponse(
                result=result,
                trace_id=trace_id,
                task_type=request.task_type,
                steps=steps,
                metrics=metrics,
                context_usage=context_usage_data,
                timing={
                    "total": metrics["total_time"],
                    "processing": processing_time,
                    "pre_processing": pre_processing_time
                }
            )
            
        except Exception as e:
            # Log error
            logger.error(f"Error processing sequential thinking request: {str(e)}")
            
            # Update metrics
            metrics["end_time"] = time.time()
            metrics["total_time"] = metrics["end_time"] - start_time
            metrics["success"] = False
            metrics["error"] = str(e)
            
            # Log metrics if collector is available
            if self.metrics_collector:
                await self.metrics_collector.record_metrics("sequential_thinking", metrics)
            
            # Re-raise the exception
            raise
    
    async def analyze_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Analyze a reasoning trace.
        
        Args:
            trace_id: ID of the trace to analyze
            
        Returns:
            Analysis results
        """
        if not self.trace_store:
            return {"error": "No trace store available"}
        
        # Get trace from store
        trace = await self.trace_store.get_trace(trace_id)
        if not trace:
            return {"error": f"Trace not found: {trace_id}"}
        
        # Import analyzer if needed
        try:
            from .reasoning_trace import ReasoningTraceAnalyzer
            analyzer = ReasoningTraceAnalyzer()
        except ImportError as e:
            logger.error(f"Error importing ReasoningTraceAnalyzer: {str(e)}")
            return {"error": f"Cannot analyze trace: {str(e)}"}
        
        # Analyze trace
        try:
            analysis = analyzer.analyze_trace(trace)
            
            # Add additional metrics
            if self.metrics_collector:
                additional_metrics = await self.metrics_collector.get_metrics_for_trace(trace_id)
                if additional_metrics:
                    analysis["additional_metrics"] = additional_metrics
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing trace: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def get_trace_visualization(self, trace_id: str, visualization_type: str = "summary") -> Dict[str, Any]:
        """
        Get visualization data for a reasoning trace.
        
        Args:
            trace_id: ID of the trace to visualize
            visualization_type: Type of visualization to generate
            
        Returns:
            Visualization data
        """
        if not self.trace_store:
            return {"error": "No trace store available"}
        
        # Get trace from store
        trace = await self.trace_store.get_trace(trace_id)
        if not trace:
            return {"error": f"Trace not found: {trace_id}"}
        
        # Import visualizer if needed
        try:
            from .reasoning_trace import ReasoningTraceVisualizer
            visualizer = ReasoningTraceVisualizer()
        except ImportError as e:
            logger.error(f"Error importing ReasoningTraceVisualizer: {str(e)}")
            return {"error": f"Cannot generate visualization: {str(e)}"}
        
        # Generate visualization
        try:
            if visualization_type == "summary":
                return visualizer.generate_trace_summary(trace)
            elif visualization_type == "context_usage":
                return visualizer.generate_context_usage_data(trace)
            elif visualization_type == "timeline":
                return visualizer.generate_timeline_data(trace)
            elif visualization_type == "transitions":
                return visualizer.generate_step_transition_data(trace)
            else:
                return {"error": f"Unknown visualization type: {visualization_type}"}
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return {"error": f"Visualization generation failed: {str(e)}"}


async def _call_callback_with_timeout(callback, data, timeout: float = 1.0):
    """
    Call a callback with a timeout to prevent blocking.
    
    Args:
        callback: The callback function
        data: Data to pass to the callback
        timeout: Timeout in seconds
    """
    try:
        # Create a task for the callback
        callback_task = asyncio.create_task(
            _ensure_async(callback)(data)
        )
        
        # Wait for the callback with timeout
        await asyncio.wait_for(callback_task, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Callback timed out after {timeout}s")
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")


async def _ensure_async(func):
    """
    Ensure a function is async-compatible.
    
    Args:
        func: Function to make async-compatible
        
    Returns:
        Async-compatible function
    """
    async def wrapper(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper


def _determine_step_type(step_number: int) -> str:
    """
    Determine the step type based on step number.
    
    Args:
        step_number: Number of the step
        
    Returns:
        Step type string
    """
    if step_number == 1:
        return "problem_definition"
    elif step_number == 2:
        return "information_gathering"
    elif step_number == 3:
        return "analysis"
    elif step_number < 6:
        return "solution_design"
    else:
        return "conclusion" 