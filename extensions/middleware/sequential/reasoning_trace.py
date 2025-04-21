#!/usr/bin/env python
"""
Reasoning Trace Framework for Sequential Thinking.

This module provides data structures and utilities for capturing,
analyzing, and visualizing reasoning traces from sequential thinking
processes.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContextReference(BaseModel):
    """Reference to a piece of context used in a reasoning step."""
    context_id: str = Field(..., description="Identifier of the referenced context")
    source: str = Field(..., description="Source of the context")
    usage_type: str = Field(..., description="How the context was used (e.g., 'supporting', 'contradicting')")
    relevance_score: float = Field(..., description="Relevance score (0-1)")


class ReasoningStep(BaseModel):
    """Data model for a single step in a reasoning trace."""
    step_id: str = Field(..., description="Unique identifier for this step")
    thought_number: int = Field(..., description="Position in the sequence", gt=0)
    content: str = Field(..., description="Content of the reasoning step")
    timestamp: float = Field(..., description="Unix timestamp when step was created")
    step_type: str = Field(..., description="Type of reasoning step")
    requires_next_step: bool = Field(..., description="Whether another step is needed")
    context_references: List[ContextReference] = Field(default_factory=list, 
                                                      description="References to context used in this step")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Step-specific metrics")
    
    @validator('timestamp', pre=True)
    def ensure_timestamp(cls, v):
        """Ensure timestamp is a float."""
        if v is None:
            return time.time()
        return float(v)
    
    @validator('step_id', pre=True)
    def ensure_step_id(cls, v):
        """Ensure step_id is a string."""
        if v is None:
            return str(uuid.uuid4())
        return str(v)


class ReasoningTrace(BaseModel):
    """Data model for a complete reasoning trace."""
    trace_id: str = Field(..., description="Unique identifier for this trace")
    task: str = Field(..., description="Description of the task being reasoned about")
    start_time: float = Field(..., description="Unix timestamp when trace was started")
    end_time: Optional[float] = Field(None, description="Unix timestamp when trace was completed")
    steps: List[ReasoningStep] = Field(default_factory=list, description="Steps in the reasoning process")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    final_result: Optional[str] = Field(None, description="Final result of the reasoning process")
    
    @validator('trace_id', pre=True)
    def ensure_trace_id(cls, v):
        """Ensure trace_id is a string."""
        if v is None:
            return str(uuid.uuid4())
        return str(v)
    
    @validator('start_time', pre=True)
    def ensure_start_time(cls, v):
        """Ensure start_time is a float."""
        if v is None:
            return time.time()
        return float(v)
    
    def add_step(self, step: ReasoningStep) -> None:
        """
        Add a step to the reasoning trace.
        
        Args:
            step: The reasoning step to add
        """
        self.steps.append(step)
        
        # If this step doesn't require a next step, mark the trace as complete
        if not step.requires_next_step:
            self.end_time = time.time()
            self.final_result = step.content
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the trace to a dictionary.
        
        Returns:
            Dictionary representation of the trace
        """
        return self.dict()
    
    def to_json(self, **kwargs) -> str:
        """
        Convert the trace to JSON.
        
        Args:
            **kwargs: Additional arguments to pass to json.dumps
            
        Returns:
            JSON representation of the trace
        """
        return json.dumps(self.dict(), **kwargs)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningTrace':
        """
        Create a trace from a dictionary.
        
        Args:
            data: Dictionary representation of the trace
            
        Returns:
            ReasoningTrace instance
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ReasoningTrace':
        """
        Create a trace from JSON.
        
        Args:
            json_str: JSON representation of the trace
            
        Returns:
            ReasoningTrace instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class ReasoningTraceAnalyzer:
    """
    Analyzer for reasoning traces.
    
    This class provides methods for analyzing reasoning traces to extract
    insights, identify patterns, and provide feedback for improvement.
    """
    
    def analyze_trace(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """
        Analyze a reasoning trace.
        
        Args:
            trace: The trace to analyze
            
        Returns:
            Dictionary of analysis results
        """
        results = {
            'trace_id': trace.trace_id,
            'task': trace.task,
            'duration': self._calculate_duration(trace),
            'step_count': len(trace.steps),
            'step_analysis': self._analyze_steps(trace.steps),
            'context_usage': self._analyze_context_usage(trace.steps),
            'patterns': self._identify_patterns(trace.steps),
            'quality_metrics': self._calculate_quality_metrics(trace)
        }
        
        return results
    
    def _calculate_duration(self, trace: ReasoningTrace) -> float:
        """
        Calculate the duration of the reasoning process.
        
        Args:
            trace: The trace to analyze
            
        Returns:
            Duration in seconds
        """
        if trace.end_time:
            return trace.end_time - trace.start_time
        
        # If trace is not complete, calculate based on last step
        if trace.steps:
            return trace.steps[-1].timestamp - trace.start_time
            
        return 0.0
    
    def _analyze_steps(self, steps: List[ReasoningStep]) -> List[Dict[str, Any]]:
        """
        Analyze each step in the reasoning process.
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            List of step analyses
        """
        analyses = []
        
        for i, step in enumerate(steps):
            # Calculate time delta from previous step
            time_delta = 0.0
            if i > 0:
                time_delta = step.timestamp - steps[i-1].timestamp
            
            # Analyze step content
            word_count = len(step.content.split())
            
            # Analyze context usage
            context_count = len(step.context_references)
            avg_context_relevance = 0.0
            if context_count > 0:
                avg_context_relevance = sum(ref.relevance_score for ref in step.context_references) / context_count
            
            analyses.append({
                'step_id': step.step_id,
                'step_number': i + 1,
                'time_delta': time_delta,
                'word_count': word_count,
                'context_count': context_count,
                'avg_context_relevance': avg_context_relevance,
                'step_type': step.step_type
            })
        
        return analyses
    
    def _analyze_context_usage(self, steps: List[ReasoningStep]) -> Dict[str, Any]:
        """
        Analyze how context was used across all steps.
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            Context usage analysis
        """
        # Collect all context references
        all_references = []
        for step in steps:
            all_references.extend(step.context_references)
        
        # Count unique context IDs
        context_ids = set(ref.context_id for ref in all_references)
        
        # Group references by source
        sources = {}
        for ref in all_references:
            if ref.source not in sources:
                sources[ref.source] = 0
            sources[ref.source] += 1
        
        # Group references by usage type
        usage_types = {}
        for ref in all_references:
            if ref.usage_type not in usage_types:
                usage_types[ref.usage_type] = 0
            usage_types[ref.usage_type] += 1
        
        # Calculate average relevance
        avg_relevance = 0.0
        if all_references:
            avg_relevance = sum(ref.relevance_score for ref in all_references) / len(all_references)
        
        return {
            'unique_contexts': len(context_ids),
            'total_references': len(all_references),
            'sources': sources,
            'usage_types': usage_types,
            'avg_relevance': avg_relevance
        }
    
    def _identify_patterns(self, steps: List[ReasoningStep]) -> Dict[str, Any]:
        """
        Identify patterns in the reasoning process.
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            Identified patterns
        """
        # Count step types
        step_types = {}
        for step in steps:
            if step.step_type not in step_types:
                step_types[step.step_type] = 0
            step_types[step.step_type] += 1
        
        # Identify linear vs. non-linear reasoning
        # (Non-linear reasoning revisits earlier topics)
        is_linear = True
        topics = set()
        for step in steps:
            # Simple topic extraction from step content
            words = step.content.lower().split()
            # Get first 5 non-stopwords as a simple proxy for topics
            new_topics = set(w for w in words[:20] if len(w) > 5)
            
            # If there's significant overlap with earlier topics, reasoning is non-linear
            if topics and len(topics.intersection(new_topics)) > 2:
                is_linear = False
                break
                
            topics.update(new_topics)
        
        return {
            'step_type_distribution': step_types,
            'reasoning_style': 'linear' if is_linear else 'non-linear',
            'topic_count': len(topics)
        }
    
    def _calculate_quality_metrics(self, trace: ReasoningTrace) -> Dict[str, float]:
        """
        Calculate overall quality metrics for the trace.
        
        Args:
            trace: The trace to analyze
            
        Returns:
            Quality metrics
        """
        if not trace.steps:
            return {
                'coherence': 0.0,
                'context_utilization': 0.0,
                'efficiency': 0.0,
                'depth': 0.0
            }
        
        # Coherence: measure how well steps connect to each other
        coherence = self._calculate_coherence(trace.steps)
        
        # Context utilization: measure how effectively context was used
        context_utilization = self._calculate_context_utilization(trace.steps)
        
        # Efficiency: inverse of number of steps relative to task complexity
        # (Simplified for now)
        efficiency = 1.0 / max(1, len(trace.steps) / 5)  # 5 steps is baseline for efficiency=1.0
        
        # Depth: measure complexity of reasoning
        depth = self._calculate_depth(trace.steps)
        
        return {
            'coherence': coherence,
            'context_utilization': context_utilization,
            'efficiency': efficiency,
            'depth': depth
        }
    
    def _calculate_coherence(self, steps: List[ReasoningStep]) -> float:
        """
        Calculate coherence between reasoning steps.
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            Coherence score (0-1)
        """
        # Simple heuristic based on word overlap between consecutive steps
        if len(steps) < 2:
            return 1.0
        
        overlap_scores = []
        
        for i in range(1, len(steps)):
            prev_words = set(steps[i-1].content.lower().split())
            curr_words = set(steps[i].content.lower().split())
            
            # Calculate Jaccard similarity
            if not prev_words or not curr_words:
                overlap_scores.append(0.0)
                continue
                
            intersection = len(prev_words.intersection(curr_words))
            union = len(prev_words.union(curr_words))
            
            overlap_scores.append(intersection / union)
        
        return sum(overlap_scores) / len(overlap_scores)
    
    def _calculate_context_utilization(self, steps: List[ReasoningStep]) -> float:
        """
        Calculate how effectively context was utilized.
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            Context utilization score (0-1)
        """
        # Calculate based on relevance scores and diversity of usage
        total_refs = sum(len(step.context_references) for step in steps)
        
        if total_refs == 0:
            return 0.0
        
        # Average relevance
        avg_relevance = sum(
            sum(ref.relevance_score for ref in step.context_references) 
            for step in steps if step.context_references
        ) / total_refs
        
        # Diversity of sources
        all_sources = set()
        for step in steps:
            for ref in step.context_references:
                all_sources.add(ref.source)
        
        source_diversity = min(1.0, len(all_sources) / 3)  # Normalize to max of 1.0
        
        # Usage across steps
        steps_with_context = sum(1 for step in steps if step.context_references)
        step_coverage = steps_with_context / len(steps)
        
        # Weighted score
        return (avg_relevance * 0.5) + (source_diversity * 0.2) + (step_coverage * 0.3)
    
    def _calculate_depth(self, steps: List[ReasoningStep]) -> float:
        """
        Calculate depth of reasoning.
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            Depth score (0-1)
        """
        # Simple heuristic based on word count, code presence, and context usage
        total_words = sum(len(step.content.split()) for step in steps)
        avg_words = total_words / len(steps)
        
        # Normalize word count to 0-1 range
        word_score = min(1.0, avg_words / 100)
        
        # Check for code snippets
        contains_code = any(
            '{' in step.content and '}' in step.content
            for step in steps
        )
        
        # Context usage per step
        context_per_step = sum(len(step.context_references) for step in steps) / len(steps)
        context_score = min(1.0, context_per_step / 2)  # Normalize to max of 1.0
        
        # Weighted score
        depth = (word_score * 0.4) + (float(contains_code) * 0.2) + (context_score * 0.4)
        
        return depth


class ReasoningTraceStore:
    """
    Storage for reasoning traces.
    
    This class provides methods for storing and retrieving reasoning traces,
    with support for optional persistence to a database.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the trace store.
        
        Args:
            storage_path: Optional path for persistent storage
        """
        self.traces = {}  # In-memory storage
        self.storage_path = storage_path
    
    def add_trace(self, trace: ReasoningTrace) -> str:
        """
        Add a trace to the store.
        
        Args:
            trace: The trace to store
            
        Returns:
            Trace ID
        """
        trace_id = trace.trace_id
        self.traces[trace_id] = trace
        
        # Persist if storage path is set
        if self.storage_path:
            self._persist_trace(trace)
        
        return trace_id
    
    def get_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """
        Get a trace by ID.
        
        Args:
            trace_id: ID of the trace to retrieve
            
        Returns:
            ReasoningTrace if found, None otherwise
        """
        # Try in-memory cache first
        if trace_id in self.traces:
            return self.traces[trace_id]
        
        # Try to load from storage if not in memory
        if self.storage_path:
            loaded_trace = self._load_trace(trace_id)
            if loaded_trace:
                self.traces[trace_id] = loaded_trace
                return loaded_trace
        
        return None
    
    def list_traces(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List traces with pagination.
        
        Args:
            limit: Maximum number of traces to return
            offset: Offset for pagination
            
        Returns:
            List of trace summaries
        """
        # Get trace IDs sorted by start time (newest first)
        sorted_ids = sorted(
            self.traces.keys(),
            key=lambda tid: self.traces[tid].start_time,
            reverse=True
        )
        
        # Apply pagination
        page_ids = sorted_ids[offset:offset+limit]
        
        # Create summaries
        return [
            {
                'trace_id': tid,
                'task': self.traces[tid].task,
                'start_time': self.traces[tid].start_time,
                'end_time': self.traces[tid].end_time,
                'step_count': len(self.traces[tid].steps),
                'is_complete': self.traces[tid].end_time is not None
            }
            for tid in page_ids
        ]
    
    def delete_trace(self, trace_id: str) -> bool:
        """
        Delete a trace by ID.
        
        Args:
            trace_id: ID of the trace to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if trace_id in self.traces:
            del self.traces[trace_id]
            
            # Delete from storage if path is set
            if self.storage_path:
                self._delete_persisted_trace(trace_id)
            
            return True
        
        return False
    
    def _persist_trace(self, trace: ReasoningTrace) -> None:
        """
        Persist a trace to storage.
        
        Args:
            trace: The trace to persist
        """
        if not self.storage_path:
            return
        
        try:
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Write to file
            file_path = os.path.join(self.storage_path, f"{trace.trace_id}.json")
            with open(file_path, 'w') as f:
                f.write(trace.to_json(indent=2))
                
        except Exception as e:
            logger.error(f"Error persisting trace {trace.trace_id}: {str(e)}")
    
    def _load_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """
        Load a trace from storage.
        
        Args:
            trace_id: ID of the trace to load
            
        Returns:
            ReasoningTrace if found and loaded, None otherwise
        """
        if not self.storage_path:
            return None
        
        try:
            import os
            
            # Check if file exists
            file_path = os.path.join(self.storage_path, f"{trace_id}.json")
            if not os.path.exists(file_path):
                return None
            
            # Read from file
            with open(file_path, 'r') as f:
                trace_json = f.read()
            
            # Parse JSON
            return ReasoningTrace.from_json(trace_json)
            
        except Exception as e:
            logger.error(f"Error loading trace {trace_id}: {str(e)}")
            return None
    
    def _delete_persisted_trace(self, trace_id: str) -> None:
        """
        Delete a persisted trace.
        
        Args:
            trace_id: ID of the trace to delete
        """
        if not self.storage_path:
            return
        
        try:
            import os
            
            # Check if file exists
            file_path = os.path.join(self.storage_path, f"{trace_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                
        except Exception as e:
            logger.error(f"Error deleting trace {trace_id}: {str(e)}")


class ReasoningTraceVisualizer:
    """
    Visualizer for reasoning traces.
    
    This class provides methods for visualizing reasoning traces to aid
    in understanding the reasoning process.
    """
    
    def generate_trace_summary(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """
        Generate a summary of a reasoning trace.
        
        Args:
            trace: The trace to summarize
            
        Returns:
            Summary data for visualization
        """
        # Format timestamps for display
        formatted_timestamps = []
        for step in trace.steps:
            dt = datetime.fromtimestamp(step.timestamp)
            formatted_timestamps.append(dt.strftime("%H:%M:%S"))
        
        # Extract step contents
        step_contents = [step.content for step in trace.steps]
        
        # Calculate time deltas between steps
        time_deltas = [0]
        for i in range(1, len(trace.steps)):
            delta = trace.steps[i].timestamp - trace.steps[i-1].timestamp
            time_deltas.append(round(delta, 2))
        
        # Get step types
        step_types = [step.step_type for step in trace.steps]
        
        # Count context references per step
        context_counts = [len(step.context_references) for step in trace.steps]
        
        # Duration
        duration = 0
        if trace.end_time:
            duration = trace.end_time - trace.start_time
        elif trace.steps:
            duration = trace.steps[-1].timestamp - trace.start_time
        
        return {
            'trace_id': trace.trace_id,
            'task': trace.task,
            'start_time': datetime.fromtimestamp(trace.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            'duration': round(duration, 2),
            'step_count': len(trace.steps),
            'is_complete': trace.end_time is not None,
            'steps': [
                {
                    'number': i + 1,
                    'timestamp': formatted_timestamps[i],
                    'time_delta': time_deltas[i],
                    'type': step_types[i],
                    'content': step_contents[i],
                    'context_count': context_counts[i]
                }
                for i in range(len(trace.steps))
            ],
            'final_result': trace.final_result
        }
    
    def generate_context_usage_data(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """
        Generate data about context usage in a reasoning trace.
        
        Args:
            trace: The trace to analyze
            
        Returns:
            Context usage data for visualization
        """
        # Collect all context references
        context_refs = {}
        for step in trace.steps:
            for ref in step.context_references:
                if ref.context_id not in context_refs:
                    context_refs[ref.context_id] = {
                        'id': ref.context_id,
                        'source': ref.source,
                        'usage_count': 0,
                        'usage_types': {},
                        'avg_relevance': 0,
                        'steps_used': []
                    }
                
                # Update usage count
                context_refs[ref.context_id]['usage_count'] += 1
                
                # Update steps used
                context_refs[ref.context_id]['steps_used'].append(step.thought_number)
                
                # Update usage types
                if ref.usage_type not in context_refs[ref.context_id]['usage_types']:
                    context_refs[ref.context_id]['usage_types'][ref.usage_type] = 0
                context_refs[ref.context_id]['usage_types'][ref.usage_type] += 1
                
                # Update average relevance
                current_avg = context_refs[ref.context_id]['avg_relevance']
                current_count = context_refs[ref.context_id]['usage_count']
                new_avg = (current_avg * (current_count - 1) + ref.relevance_score) / current_count
                context_refs[ref.context_id]['avg_relevance'] = new_avg
        
        # Group by source
        sources = {}
        for ref_id, ref_data in context_refs.items():
            source = ref_data['source']
            if source not in sources:
                sources[source] = {
                    'name': source,
                    'count': 0,
                    'contexts': []
                }
            
            sources[source]['count'] += 1
            sources[source]['contexts'].append(ref_id)
        
        return {
            'trace_id': trace.trace_id,
            'context_count': len(context_refs),
            'contexts': list(context_refs.values()),
            'sources': list(sources.values()),
            'total_references': sum(ref['usage_count'] for ref in context_refs.values())
        }
    
    def generate_timeline_data(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """
        Generate timeline data for a reasoning trace.
        
        Args:
            trace: The trace to visualize
            
        Returns:
            Timeline data for visualization
        """
        # Extract step timestamps and durations
        events = []
        
        for i, step in enumerate(trace.steps):
            # Calculate end time (either next step timestamp or current + 1.0s as fallback)
            end_time = step.timestamp + 1.0
            if i < len(trace.steps) - 1:
                end_time = trace.steps[i+1].timestamp
            
            events.append({
                'step_number': step.thought_number,
                'start_time': step.timestamp,
                'end_time': end_time,
                'duration': end_time - step.timestamp,
                'type': step.step_type,
                'content_preview': step.content[:50] + ('...' if len(step.content) > 50 else '')
            })
        
        return {
            'trace_id': trace.trace_id,
            'start_time': trace.start_time,
            'end_time': trace.end_time or (events[-1]['end_time'] if events else trace.start_time),
            'events': events
        }
    
    def generate_step_transition_data(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """
        Generate data about transitions between steps.
        
        Args:
            trace: The trace to analyze
            
        Returns:
            Step transition data for visualization
        """
        # Count transitions between step types
        transitions = {}
        
        for i in range(1, len(trace.steps)):
            from_type = trace.steps[i-1].step_type
            to_type = trace.steps[i].step_type
            
            key = f"{from_type}→{to_type}"
            if key not in transitions:
                transitions[key] = {
                    'from': from_type,
                    'to': to_type,
                    'count': 0,
                    'examples': []
                }
            
            transitions[key]['count'] += 1
            if len(transitions[key]['examples']) < 3:  # Limit to 3 examples
                transitions[key]['examples'].append(i)
        
        return {
            'trace_id': trace.trace_id,
            'transitions': list(transitions.values()),
            'step_types': list(set(step.step_type for step in trace.steps))
        } 