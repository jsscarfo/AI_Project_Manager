#!/usr/bin/env python
"""
Simple test for Steps Visualizer functionality.

This script tests basic StepsVisualizer functionality with mocked data objects.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class StepVisualizationData:
    """Simple mock of StepVisualizationData."""
    step_id: str
    step_number: int
    title: str
    content: str
    step_type: str
    timestamp: float
    duration: float = 0.0
    requires_next_step: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)
    context_references: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    content_preview: Optional[str] = None
    formatted_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.content_preview is None:
            max_length = 100
            self.content_preview = (
                self.content[:max_length] + "..." 
                if len(self.content) > max_length else self.content
            )
        if self.formatted_timestamp is None:
            self.formatted_timestamp = datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ContextSourceData:
    """Simple mock of ContextSourceVisualizationData."""
    source_id: str
    name: str
    content: str
    source_type: str
    steps_referenced: List[int] = field(default_factory=list)


@dataclass
class TraceVisualizationData:
    """Simple mock of TraceVisualizationData."""
    trace_id: str
    task: str
    steps: List[StepVisualizationData]
    context_sources: List[ContextSourceData]
    final_result: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StepTransitionVisualization:
    """Mock of StepTransitionVisualization."""
    
    def generate_transition_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate mock transition data."""
        transitions = []
        
        for i in range(len(data.steps) - 1):
            current_step = data.steps[i]
            next_step = data.steps[i + 1]
            
            type_change = current_step.step_type != next_step.step_type
            
            transitions.append({
                "from_step": current_step.step_number,
                "to_step": next_step.step_number,
                "from_type": current_step.step_type,
                "to_type": next_step.step_type,
                "duration": next_step.timestamp - current_step.timestamp,
                "type_change": type_change,
                "annotation": f"Transition from {current_step.step_type} to {next_step.step_type}" if type_change else ""
            })
        
        return {
            "trace_id": data.trace_id,
            "total_transitions": len(transitions),
            "transitions": transitions,
            "type_changes": sum(1 for t in transitions if t["type_change"])
        }
    
    def generate_step_flow_chart(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate mock flow chart data."""
        nodes = []
        edges = []
        
        # Create nodes
        for step in data.steps:
            nodes.append({
                "id": f"step_{step.step_number}",
                "label": f"Step {step.step_number}",
                "type": step.step_type,
                "data": {
                    "step_number": step.step_number,
                    "content_preview": step.content_preview,
                    "requires_next_step": step.requires_next_step
                }
            })
        
        # Create edges
        for i in range(len(data.steps) - 1):
            current_step = data.steps[i]
            next_step = data.steps[i + 1]
            
            edges.append({
                "id": f"edge_{current_step.step_number}_{next_step.step_number}",
                "source": f"step_{current_step.step_number}",
                "target": f"step_{next_step.step_number}",
                "type": "transition",
                "data": {
                    "duration": next_step.timestamp - current_step.timestamp,
                    "type_change": current_step.step_type != next_step.step_type
                }
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "trace_id": data.trace_id,
            "step_types": list(set(step.step_type for step in data.steps))
        }


class StepDetailsVisualization:
    """Mock of StepDetailsVisualization."""
    
    def generate_step_details(self, step: StepVisualizationData, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate mock step details."""
        # Find relevant context sources
        context_sources = []
        for source in trace_data.context_sources:
            if step.step_number in source.steps_referenced:
                context_sources.append({
                    "source_id": source.source_id,
                    "name": source.name,
                    "source_type": source.source_type
                })
        
        # Get position metrics
        position_metrics = {
            "is_first": step.step_number == 1,
            "is_last": step.step_number == len(trace_data.steps),
            "position_percent": (step.step_number / len(trace_data.steps)) * 100
        }
        
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
        """Generate mock step comparison."""
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
    """Mock of StepProgressVisualization."""
    
    def generate_progress_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate mock progress data."""
        # Calculate overall progress
        is_complete = data.final_result is not None
        last_step = data.steps[-1] if data.steps else None
        
        # Group steps by type
        steps_by_type = {}
        for step in data.steps:
            if step.step_type not in steps_by_type:
                steps_by_type[step.step_type] = []
            steps_by_type[step.step_type].append(step.step_number)
        
        # Generate step timeline
        timeline = []
        for step in data.steps:
            timeline.append({
                "step_number": step.step_number,
                "timestamp": step.timestamp,
                "step_type": step.step_type,
                "content_preview": step.content_preview
            })
        
        return {
            "trace_id": data.trace_id,
            "is_complete": is_complete,
            "steps_completed": len(data.steps),
            "current_step": last_step.step_number if last_step else 0,
            "current_step_type": last_step.step_type if last_step else "",
            "steps_by_type": steps_by_type,
            "timeline": timeline,
            "total_reasoning_time": sum(step.duration for step in data.steps)
        }
    
    def generate_step_type_distribution(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate mock step type distribution data."""
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
    """Mock of StepsVisualizer."""
    
    def __init__(self):
        """Initialize with component instances."""
        self.transition_visualizer = StepTransitionVisualization()
        self.details_visualizer = StepDetailsVisualization()
        self.progress_visualizer = StepProgressVisualization()
    
    def generate_visualization_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate mock visualization data."""
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
        """Convert mock visualization data to JSON."""
        visualization_data = self.generate_visualization_data(data)
        return json.dumps(visualization_data, **kwargs)
    
    def visualize_step(self, step: StepVisualizationData, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate mock visualization for a specific step."""
        return self.details_visualizer.generate_step_details(step, trace_data)
    
    def compare_steps(self, steps: List[StepVisualizationData]) -> Dict[str, Any]:
        """Generate mock comparison visualization for multiple steps."""
        return self.details_visualizer.generate_step_comparison(steps)


def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data...")
    
    # Create steps
    steps = []
    
    steps.append(StepVisualizationData(
        step_id="step_1",
        step_number=1,
        title="Planning",
        content="In this step, we analyze the requirements for the project management integration.",
        step_type="planning",
        timestamp=1609459200.0,  # 2021-01-01
        duration=2.5,
        requires_next_step=True,
        metrics={"complexity": 0.3, "importance": 0.8},
        context_references=[
            {"source": "source_1", "relevance_score": 0.9},
            {"source": "source_2", "relevance_score": 0.7}
        ],
        annotations=["Initial planning", "Requirements analysis"]
    ))
    
    steps.append(StepVisualizationData(
        step_id="step_2",
        step_number=2,
        title="Design",
        content="Based on the requirements, we design the project management data models and operations.",
        step_type="design",
        timestamp=1609459205.0,  # 5 seconds later
        duration=3.5,
        requires_next_step=True,
        metrics={"complexity": 0.6, "quality": 0.75},
        context_references=[
            {"source": "source_1", "relevance_score": 0.6},
            {"source": "source_3", "relevance_score": 0.8}
        ],
        annotations=["Architecture design", "Data model creation"]
    ))
    
    steps.append(StepVisualizationData(
        step_id="step_3",
        step_number=3,
        title="Implementation",
        content="Now we implement the project management integration system with all required components.",
        step_type="implementation",
        timestamp=1609459210.0,  # 5 seconds later
        duration=5.0,
        requires_next_step=True,
        metrics={"complexity": 0.8, "completeness": 0.7},
        context_references=[
            {"source": "source_2", "relevance_score": 0.85},
            {"source": "source_3", "relevance_score": 0.9}
        ],
        annotations=["Core implementation", "Integration points"]
    ))
    
    steps.append(StepVisualizationData(
        step_id="step_4",
        step_number=4,
        title="Testing",
        content="Finally, we test the project management integration to ensure it meets all requirements.",
        step_type="testing",
        timestamp=1609459220.0,  # 10 seconds later
        duration=4.0,
        requires_next_step=False,
        metrics={"coverage": 0.9, "quality": 0.85},
        context_references=[
            {"source": "source_1", "relevance_score": 0.7},
            {"source": "source_4", "relevance_score": 0.95}
        ],
        annotations=["Test execution", "Validation of requirements"]
    ))
    
    # Create context sources
    sources = []
    
    sources.append(ContextSourceData(
        source_id="source_1",
        name="Requirements Document",
        content="Project requirements specification...",
        source_type="document",
        steps_referenced=[1, 2, 4]
    ))
    
    sources.append(ContextSourceData(
        source_id="source_2",
        name="Architecture Guide",
        content="System architecture guidelines...",
        source_type="guide",
        steps_referenced=[1, 3]
    ))
    
    sources.append(ContextSourceData(
        source_id="source_3",
        name="API Documentation",
        content="API reference and examples...",
        source_type="documentation",
        steps_referenced=[2, 3]
    ))
    
    sources.append(ContextSourceData(
        source_id="source_4",
        name="Test Plan",
        content="Testing procedures and validation criteria...",
        source_type="document",
        steps_referenced=[4]
    ))
    
    # Create trace data
    trace_data = TraceVisualizationData(
        trace_id="pm_integration_001",
        task="Implement Project Management Integration",
        steps=steps,
        context_sources=sources,
        final_result="Project Management Integration system successfully implemented and tested.",
        metrics={
            "overall_complexity": 0.65,
            "overall_quality": 0.82,
            "completion_time": 15.0
        },
        metadata={
            "project": "BeeAI Project Management",
            "author": "Test Runner",
            "created_at": datetime.now().isoformat()
        }
    )
    
    return trace_data


def test_visualizer():
    """Test the visualizer components with sample data."""
    # Create sample data
    trace_data = create_sample_data()
    print(f"Created trace with {len(trace_data.steps)} steps and {len(trace_data.context_sources)} context sources")
    
    # Create visualizer
    visualizer = StepsVisualizer()
    
    # Generate visualization data
    print("\nGenerating visualization data...")
    visualization_data = visualizer.generate_visualization_data(trace_data)
    
    # Print summary info
    print(f"Trace ID: {visualization_data['trace_id']}")
    print(f"Steps: {len(visualization_data['steps'])}")
    print(f"Transitions: {visualization_data['transitions']['total_transitions']}")
    print(f"Type changes: {visualization_data['transitions']['type_changes']}")
    print(f"Step types: {visualization_data['flow_chart']['step_types']}")
    print(f"Completion status: {'Complete' if visualization_data['progress']['is_complete'] else 'Incomplete'}")
    
    # Test step comparison
    print("\nComparing first and last steps:")
    comparison = visualizer.compare_steps([trace_data.steps[0], trace_data.steps[-1]])
    print(f"Step numbers compared: {comparison['step_numbers']}")
    print(f"Step types compared: {comparison['step_types']}")
    
    # Save output to file
    output_file = "python/beeai_framework/visualization/examples/simple_steps_visualization.json"
    with open(output_file, "w") as f:
        json.dump(visualization_data, f, indent=2)
    
    print(f"\nVisualization data saved to {output_file}")
    return visualization_data


if __name__ == "__main__":
    print("===== Simple Steps Visualizer Test =====")
    print("Running visualization test with mock data...")
    try:
        test_visualizer()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc() 