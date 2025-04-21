#!/usr/bin/env python
"""
Test runner for Steps Visualizer component.

This script tests the StepsVisualizer component functionality
by creating sample data and running the main visualization methods.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add the parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import directly for test runner to avoid circular dependencies
from beeai_framework.visualization.components.steps_visualizer import (
    StepTransitionVisualization,
    StepDetailsVisualization,
    StepProgressVisualization,
    StepsVisualizer
)

from beeai_framework.visualization.core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData,
    ContextSourceVisualizationData,
    VisualizationMetadata
)


def create_sample_data():
    """Create sample trace data for testing."""
    print("Creating sample trace data...")
    
    # Create multiple steps
    step1 = StepVisualizationData(
        step_id="step_1",
        step_number=1,
        title="First Step",
        content="This is the first step of the reasoning process. We need to analyze the requirements.",
        content_preview="This is the first step...",
        step_type="planning",
        timestamp=1609459200.0,  # 2021-01-01
        formatted_timestamp="2021-01-01 00:00:00",
        duration=2.5,
        context_references=[
            {"source": "source_1", "relevance_score": 0.85, "content_slice": "relevant text 1"},
            {"source": "source_2", "relevance_score": 0.65, "content_slice": "relevant text 2"}
        ],
        metrics={
            "coherence": 0.75,
            "relevance": 0.82
        },
        annotations=["Critical planning step", "Sets project scope"],
        requires_next_step=True
    )
    
    step2 = StepVisualizationData(
        step_id="step_2",
        step_number=2,
        title="Second Step",
        content="This is the second step of the reasoning process. We need to design the solution architecture.",
        content_preview="This is the second step...",
        step_type="design",
        timestamp=1609459203.0,  # 3 seconds after step 1
        formatted_timestamp="2021-01-01 00:00:03",
        duration=4.2,
        context_references=[
            {"source": "source_1", "relevance_score": 0.72, "content_slice": "relevant text 3"},
            {"source": "source_3", "relevance_score": 0.91, "content_slice": "relevant text 4"}
        ],
        metrics={
            "coherence": 0.81,
            "relevance": 0.79,
            "completeness": 0.65
        },
        annotations=["Architecture design", "Component identification"],
        requires_next_step=True
    )
    
    step3 = StepVisualizationData(
        step_id="step_3",
        step_number=3,
        title="Third Step",
        content="This is the third step of the reasoning process. We need to implement the core functionality.",
        content_preview="This is the third step...",
        step_type="implementation",
        timestamp=1609459209.0,  # 6 seconds after step 2
        formatted_timestamp="2021-01-01 00:00:09",
        duration=8.5,
        context_references=[
            {"source": "source_2", "relevance_score": 0.88, "content_slice": "relevant text 5"},
            {"source": "source_3", "relevance_score": 0.79, "content_slice": "relevant text 6"}
        ],
        metrics={
            "coherence": 0.92,
            "relevance": 0.85,
            "completeness": 0.78,
            "complexity": 0.72
        },
        annotations=["Core implementation", "Integration with existing systems"],
        requires_next_step=True
    )
    
    step4 = StepVisualizationData(
        step_id="step_4",
        step_number=4,
        title="Final Step",
        content="This is the final step of the reasoning process. We need to test and validate the solution.",
        content_preview="This is the final step...",
        step_type="validation",
        timestamp=1609459220.0,  # 11 seconds after step 3
        formatted_timestamp="2021-01-01 00:00:20",
        duration=3.0,
        context_references=[
            {"source": "source_2", "relevance_score": 0.75, "content_slice": "relevant text 7"},
            {"source": "source_4", "relevance_score": 0.95, "content_slice": "relevant text 8"}
        ],
        metrics={
            "coherence": 0.88,
            "relevance": 0.90,
            "completeness": 0.95,
            "accuracy": 0.87
        },
        annotations=["Final validation", "Acceptance criteria check"],
        requires_next_step=False
    )
    
    # Create context sources
    source1 = ContextSourceVisualizationData(
        source_id="source_1",
        name="Project Requirements",
        content="This document contains the project requirements and scope.",
        source_type="document",
        steps_referenced=[1, 2]
    )
    
    source2 = ContextSourceVisualizationData(
        source_id="source_2",
        name="Technical Documentation",
        content="This document contains technical specifications and API references.",
        source_type="document",
        steps_referenced=[1, 3, 4]
    )
    
    source3 = ContextSourceVisualizationData(
        source_id="source_3",
        name="Architecture Guide",
        content="This document provides architecture patterns and best practices.",
        source_type="guide",
        steps_referenced=[2, 3]
    )
    
    source4 = ContextSourceVisualizationData(
        source_id="source_4",
        name="Testing Framework",
        content="This document describes the testing methodology and criteria.",
        source_type="guide",
        steps_referenced=[4]
    )
    
    # Create metadata
    metadata = VisualizationMetadata(
        title="Project Management Integration Trace",
        description="Visualization of the project management integration implementation",
        creator="BeeAI Test System",
        tags=["project-management", "integration", "test"]
    )
    
    # Create the trace data
    trace_data = TraceVisualizationData(
        trace_id="test_trace_001",
        task="Implement Project Management Integration",
        steps=[step1, step2, step3, step4],
        context_sources=[source1, source2, source3, source4],
        final_result="The Project Management Integration has been designed, implemented, and validated.",
        metrics={
            "overall_coherence": 0.86,
            "overall_relevance": 0.84,
            "overall_completeness": 0.82,
            "success_rate": 0.95
        },
        metadata=metadata
    )
    
    return trace_data


def test_step_transition_visualization(trace_data):
    """Test the StepTransitionVisualization component."""
    print("\n===== Testing StepTransitionVisualization =====")
    
    # Create the component
    transition_visualizer = StepTransitionVisualization()
    
    # Test transition data generation
    print("Generating transition data...")
    transition_data = transition_visualizer.generate_transition_data(trace_data)
    print(f"Generated {transition_data['total_transitions']} transitions")
    print(f"Type changes detected: {transition_data['type_changes']}")
    
    # Test flow chart generation
    print("\nGenerating step flow chart...")
    flow_chart = transition_visualizer.generate_step_flow_chart(trace_data)
    print(f"Generated flow chart with {len(flow_chart['nodes'])} nodes and {len(flow_chart['edges'])} edges")
    print(f"Step types in flow chart: {flow_chart['step_types']}")
    
    return transition_data, flow_chart


def test_step_details_visualization(trace_data):
    """Test the StepDetailsVisualization component."""
    print("\n===== Testing StepDetailsVisualization =====")
    
    # Create the component
    details_visualizer = StepDetailsVisualization()
    
    # Test step details generation for first step
    print("Generating details for first step...")
    first_step = trace_data.steps[0]
    first_step_details = details_visualizer.generate_step_details(first_step, trace_data)
    print(f"Step: {first_step_details['title']} (ID: {first_step_details['step_id']})")
    print(f"Type: {first_step_details['step_type']}")
    print(f"Position: {'First step' if first_step_details['position']['is_first'] else 'Not first'}, "
          f"{'Last step' if first_step_details['position']['is_last'] else 'Not last'}")
    print(f"Using {len(first_step_details['context_sources'])} context sources")
    
    # Test step comparison
    print("\nComparing all steps...")
    comparison = details_visualizer.generate_step_comparison(trace_data.steps)
    print(f"Compared {len(comparison['step_numbers'])} steps")
    print(f"Step types: {comparison['step_types']}")
    print(f"Metrics compared: {list(comparison['metrics_comparison'].keys())}")
    
    return first_step_details, comparison


def test_step_progress_visualization(trace_data):
    """Test the StepProgressVisualization component."""
    print("\n===== Testing StepProgressVisualization =====")
    
    # Create the component
    progress_visualizer = StepProgressVisualization()
    
    # Test progress data generation
    print("Generating progress data...")
    progress_data = progress_visualizer.generate_progress_data(trace_data)
    print(f"Completion status: {'Complete' if progress_data['is_complete'] else 'Incomplete'}")
    print(f"Steps completed: {progress_data['steps_completed']}")
    print(f"Current step: {progress_data['current_step']} ({progress_data['current_step_type']})")
    print(f"Total reasoning time: {progress_data['total_reasoning_time']:.2f} seconds")
    
    # Test step type distribution
    print("\nGenerating step type distribution...")
    distribution = progress_visualizer.generate_step_type_distribution(trace_data)
    print(f"Step types: {distribution['step_types']}")
    print(f"Counts: {dict(zip(distribution['step_types'], distribution['counts']))}")
    print(f"Total duration by type: {dict(zip(distribution['step_types'], distribution['durations']))}")
    
    return progress_data, distribution


def test_steps_visualizer(trace_data):
    """Test the main StepsVisualizer component."""
    print("\n===== Testing StepsVisualizer =====")
    
    # Create the component
    visualizer = StepsVisualizer()
    
    # Test generating complete visualization data
    print("Generating complete visualization data...")
    visualization_data = visualizer.generate_visualization_data(trace_data)
    print(f"Generated visualization with trace ID: {visualization_data['trace_id']}")
    print(f"Contains {len(visualization_data['steps'])} steps, "
          f"{len(visualization_data['transitions']['transitions'])} transitions")
    
    # Test JSON conversion
    print("\nConverting visualization to JSON...")
    json_data = visualizer.to_json(trace_data, indent=2)
    json_size = len(json_data)
    print(f"Generated JSON of size {json_size} bytes")
    
    # Test single step visualization
    print("\nVisualizing a single step...")
    step = trace_data.steps[2]  # Third step
    step_visualization = visualizer.visualize_step(step, trace_data)
    print(f"Visualized step: {step_visualization['title']} ({step_visualization['step_type']})")
    
    # Test comparing steps
    print("\nComparing specific steps...")
    # Compare first and last steps
    steps_to_compare = [trace_data.steps[0], trace_data.steps[-1]]
    comparison = visualizer.compare_steps(steps_to_compare)
    print(f"Compared step numbers: {comparison['step_numbers']}")
    print(f"Compared step types: {comparison['step_types']}")
    
    return visualization_data, json_size, step_visualization, comparison


def save_visualization_output(visualization_data, output_dir="output"):
    """Save visualization output to file."""
    print("\n===== Saving Visualization Output =====")
    
    # Ensure output directory exists
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    # Save visualization data as JSON
    output_file = output_path / "steps_visualization.json"
    with open(output_file, 'w') as f:
        json.dump(visualization_data, f, indent=2)
    
    print(f"Saved visualization data to: {output_file}")
    return output_file


def main():
    """Run the test script."""
    print("===== Steps Visualizer Test Runner =====")
    print(f"Running at: {datetime.now().isoformat()}")
    
    # Create sample data
    trace_data = create_sample_data()
    print(f"Created trace data with {len(trace_data.steps)} steps and {len(trace_data.context_sources)} context sources")
    
    # Test all components
    transition_data, flow_chart = test_step_transition_visualization(trace_data)
    step_details, step_comparison = test_step_details_visualization(trace_data)
    progress_data, distribution = test_step_progress_visualization(trace_data)
    visualization_data, json_size, single_step, steps_comparison = test_steps_visualizer(trace_data)
    
    # Save output
    output_file = save_visualization_output(visualization_data)
    
    # Print summary
    print("\n===== Test Summary =====")
    print(f"Successfully tested all StepsVisualizer components")
    print(f"Generated visualization data with {len(visualization_data['steps'])} steps")
    print(f"JSON output size: {json_size} bytes")
    print(f"Output saved to: {output_file}")
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main() 
    