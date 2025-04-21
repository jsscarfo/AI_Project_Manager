#!/usr/bin/env python
"""
Steps Visualizer Component.

This module implements visualization components for reasoning steps,
including step sequences, transitions, and detailed step explorations.
"""

import json
from typing import Dict, List, Optional, Any, Union, Callable
import uuid
from datetime import datetime
from pydantic import BaseModel, Field

from ..core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData
)


class StepTransitionVisualization:
    """
    Visualizer for transitions between reasoning steps.
    
    This component generates data for visualizing how steps
    transition from one to another, including timing and type changes.
    """
    
    def generate_transition_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate transition data between steps.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Transition visualization data
        """
        transitions = []
        
        for i in range(len(data.steps) - 1):
            current_step = data.steps[i]
            next_step = data.steps[i + 1]
            
            transition = {
                "transition_id": str(uuid.uuid4()),
                "from_step": current_step.step_number,
                "to_step": next_step.step_number,
                "from_type": current_step.step_type,
                "to_type": next_step.step_type,
                "duration": next_step.timestamp - current_step.timestamp,
                "context_change": len(next_step.context_references) - len(current_step.context_references)
            }
            
            # Add transition annotations
            if current_step.step_type != next_step.step_type:
                transition["annotation"] = f"Transition from {current_step.step_type} to {next_step.step_type}"
                transition["type_change"] = True
            else:
                transition["type_change"] = False
            
            transitions.append(transition)
        
        return {
            "trace_id": data.trace_id,
            "total_transitions": len(transitions),
            "transitions": transitions,
            "type_changes": sum(1 for t in transitions if t.get("type_change", False)),
            "avg_transition_time": (sum(t["duration"] for t in transitions) / len(transitions)) if transitions else 0
        }
    
    def generate_step_flow_chart(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate data for a step flow chart visualization.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Flow chart visualization data
        """
        nodes = []
        edges = []
        
        # Create nodes for each step
        for step in data.steps:
            nodes.append({
                "id": f"step_{step.step_number}",
                "label": f"Step {step.step_number}",
                "type": step.step_type,
                "data": {
                    "step_number": step.step_number,
                    "content_preview": step.content_preview,
                    "timestamp": step.timestamp,
                    "requires_next_step": step.requires_next_step
                }
            })
        
        # Create edges between steps
        for i in range(len(data.steps) - 1):
            current_step = data.steps[i]
            next_step = data.steps[i + 1]
            
            edges.append({
                "id": f"edge_{current_step.step_number}_{next_step.step_number}",
                "source": f"step_{current_step.step_number}",
                "target": f"step_{next_step.step_number}",
                "data": {
                    "duration": next_step.timestamp - current_step.timestamp,
                    "type_change": current_step.step_type != next_step.step_type,
                    "label": f"{round(next_step.timestamp - current_step.timestamp, 2)}s"
                }
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "direction": "LR",  # Left to right
            "trace_id": data.trace_id,
            "step_types": list(set(step.step_type for step in data.steps))
        }


class StepDetailsVisualization:
    """
    Visualizer for detailed step information.
    
    This component provides detailed visualizations of individual
    reasoning steps, including context used and metrics.
    """
    
    def generate_step_details(self, step: StepVisualizationData, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate detailed visualization data for a specific step.
        
        Args:
            step: Step to visualize
            trace_data: Complete trace data for context
            
        Returns:
            Step details visualization data
        """
        # Find relevant context sources for this step
        context_sources = []
        for source in trace_data.context_sources:
            if step.step_number in source.steps_referenced:
                context_sources.append({
                    "source_id": source.source_id,
                    "name": source.name,
                    "source_type": source.source_type,
                    "relevance": next((ref["relevance_score"] for ref in step.context_references 
                                     if ref["source"] == source.source_id), 0.0)
                })
        
        # Get step position metrics
        position_metrics = {
            "is_first": step.step_number == 1,
            "is_last": step.step_number == len(trace_data.steps),
            "position_percent": (step.step_number / len(trace_data.steps)) * 100 if trace_data.steps else 0
        }
        
        # Create the step details
        return {
            "step_id": step.step_id,
            "step_number": step.step_number,
            "title": step.title,
            "content": step.content,
            "step_type": step.step_type,
            "timestamp": step.timestamp,
            "formatted_time": step.formatted_timestamp,
            "duration": step.duration,
            "context_sources": context_sources,
            "context_count": len(step.context_references),
            "metrics": step.metrics,
            "position": position_metrics,
            "annotations": step.annotations,
            "requires_next_step": step.requires_next_step
        }
    
    def generate_step_comparison(self, steps: List[StepVisualizationData]) -> Dict[str, Any]:
        """
        Generate comparison visualization data for multiple steps.
        
        Args:
            steps: Steps to compare
            
        Returns:
            Step comparison visualization data
        """
        if not steps:
            return {"error": "No steps provided for comparison"}
        
        comparison = {
            "step_numbers": [step.step_number for step in steps],
            "step_types": [step.step_type for step in steps],
            "timestamps": [step.timestamp for step in steps],
            "durations": [step.duration for step in steps],
            "context_counts": [len(step.context_references) for step in steps],
            "metrics_comparison": {}
        }
        
        # Compare metrics across steps
        all_metric_keys = set()
        for step in steps:
            all_metric_keys.update(step.metrics.keys())
        
        for key in all_metric_keys:
            comparison["metrics_comparison"][key] = [step.metrics.get(key, None) for step in steps]
        
        return comparison


class StepProgressVisualization:
    """
    Visualizer for reasoning progress tracking.
    
    This component generates visualization data for progress
    indicators and summaries of the reasoning process.
    """
    
    def generate_progress_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate progress visualization data.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Progress visualization data
        """
        # Calculate overall progress
        is_complete = data.final_result is not None
        last_step = data.steps[-1] if data.steps else None
        
        # Calculate estimated completion if not complete
        estimated_remaining_steps = 0
        if not is_complete and last_step and last_step.requires_next_step:
            # Estimate based on current progress rate
            avg_step_duration = sum(step.duration for step in data.steps) / len(data.steps) if data.steps else 0
            estimated_remaining_steps = 3  # Simple estimation for demonstration
        
        # Group steps by type
        steps_by_type = {}
        for step in data.steps:
            if step.step_type not in steps_by_type:
                steps_by_type[step.step_type] = []
            steps_by_type[step.step_type].append(step.step_number)
        
        # Generate step timeline
        timeline = []
        current_time = data.steps[0].timestamp if data.steps else 0
        for step in data.steps:
            timeline.append({
                "step_number": step.step_number,
                "time_offset": step.timestamp - current_time,
                "timestamp": step.timestamp,
                "step_type": step.step_type,
                "content_preview": step.content_preview
            })
        
        return {
            "trace_id": data.trace_id,
            "is_complete": is_complete,
            "steps_completed": len(data.steps),
            "estimated_remaining_steps": estimated_remaining_steps,
            "current_step": last_step.step_number if last_step else 0,
            "current_step_type": last_step.step_type if last_step else "",
            "steps_by_type": steps_by_type,
            "timeline": timeline,
            "total_reasoning_time": sum(step.duration for step in data.steps)
        }
    
    def generate_step_type_distribution(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate visualization data for step type distribution.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Step type distribution visualization data
        """
        type_counts = {}
        type_durations = {}
        
        for step in data.steps:
            # Count steps by type
            if step.step_type not in type_counts:
                type_counts[step.step_type] = 0
                type_durations[step.step_type] = 0
            
            type_counts[step.step_type] += 1
            type_durations[step.step_type] += step.duration
        
        # Create data for chart visualization
        chart_data = [
            {"type": step_type, "count": count, "total_duration": type_durations[step_type]}
            for step_type, count in type_counts.items()
        ]
        
        return {
            "trace_id": data.trace_id,
            "step_types": list(type_counts.keys()),
            "counts": list(type_counts.values()),
            "durations": list(type_durations.values()),
            "chart_data": chart_data
        }


class StepsVisualizer:
    """
    Main visualizer for reasoning steps.
    
    This class integrates various step visualization components
    to provide a comprehensive step visualization system.
    """
    
    def __init__(self):
        """Initialize the steps visualizer."""
        self.transition_visualizer = StepTransitionVisualization()
        self.details_visualizer = StepDetailsVisualization()
        self.progress_visualizer = StepProgressVisualization()
    
    def generate_visualization_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate complete visualization data for steps.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Complete step visualization data
        """
        result = {
            "trace_id": data.trace_id,
            "steps": [
                self.details_visualizer.generate_step_details(step, data)
                for step in data.steps
            ],
            "transitions": self.transition_visualizer.generate_transition_data(data),
            "flow_chart": self.transition_visualizer.generate_step_flow_chart(data),
            "progress": self.progress_visualizer.generate_progress_data(data),
            "type_distribution": self.progress_visualizer.generate_step_type_distribution(data)
        }
        
        return result
    
    def to_json(self, data: TraceVisualizationData, **kwargs) -> str:
        """
        Convert visualization data to JSON.
        
        Args:
            data: Trace visualization data
            **kwargs: Additional arguments for json.dumps
            
        Returns:
            JSON representation of visualization data
        """
        visualization_data = self.generate_visualization_data(data)
        return json.dumps(visualization_data, **kwargs)
    
    def visualize_step(self, step: StepVisualizationData, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate visualization for a specific step.
        
        Args:
            step: Step to visualize
            trace_data: Complete trace data for context
            
        Returns:
            Step visualization data
        """
        return self.details_visualizer.generate_step_details(step, trace_data)
    
    def compare_steps(self, steps: List[StepVisualizationData]) -> Dict[str, Any]:
        """
        Generate comparison visualization for multiple steps.
        
        Args:
            steps: Steps to compare
            
        Returns:
            Step comparison visualization data
        """
        return self.details_visualizer.generate_step_comparison(steps) 