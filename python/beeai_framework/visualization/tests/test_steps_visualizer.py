#!/usr/bin/env python
"""
Tests for Steps Visualizer component.

This module contains unit tests for the Steps Visualizer
component of the visualization framework.
"""

import pytest
import json
from datetime import datetime
import numpy as np
from typing import Dict, List, Any

from ..components.steps_visualizer import (
    StepTransitionVisualization,
    StepDetailsVisualization,
    StepProgressVisualization,
    StepsVisualizer
)
from ..core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData,
    ContextSourceVisualizationData,
    VisualizationMetadata
)


@pytest.fixture
def sample_step_data():
    """Create sample step data for testing."""
    return StepVisualizationData(
        step_id="step_1",
        step_number=1,
        title="Example Step",
        content="This is the first step of the reasoning process.",
        content_preview="This is the first step...",
        step_type="reasoning",
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
        annotations=["Important first step", "Sets direction"],
        requires_next_step=True
    )


@pytest.fixture
def sample_trace_data(sample_step_data):
    """Create sample trace data for testing."""
    # Create multiple steps
    step1 = sample_step_data
    
    step2 = StepVisualizationData(
        step_id="step_2",
        step_number=2,
        title="Second Step",
        content="This is the second step of the reasoning process.",
        content_preview="This is the second step...",
        step_type="reflection",
        timestamp=1609459203.0,  # 3 seconds after step 1
        formatted_timestamp="2021-01-01 00:00:03",
        duration=1.8,
        context_references=[
            {"source": "source_1", "relevance_score": 0.72, "content_slice": "relevant text 3"},
            {"source": "source_3", "relevance_score": 0.91, "content_slice": "relevant text 4"}
        ],
        metrics={
            "coherence": 0.81,
            "relevance": 0.79
        },
        annotations=["Reflection on previous step"],
        requires_next_step=True
    )
    
    step3 = StepVisualizationData(
        step_id="step_3",
        step_number=3,
        title="Final Step",
        content="This is the final step of the reasoning process.",
        content_preview="This is the final step...",
        step_type="conclusion",
        timestamp=1609459205.5,  # 2.5 seconds after step 2
        formatted_timestamp="2021-01-01 00:00:05",
        duration=3.0,
        context_references=[
            {"source": "source_2", "relevance_score": 0.88, "content_slice": "relevant text 5"},
            {"source": "source_3", "relevance_score": 0.79, "content_slice": "relevant text 6"}
        ],
        metrics={
            "coherence": 0.92,
            "relevance": 0.85,
            "completeness": 0.78
        },
        annotations=["Final conclusion"],
        requires_next_step=False
    )
    
    # Create context sources
    source1 = ContextSourceVisualizationData(
        source_id="source_1",
        name="Document A",
        content="Content from document A",
        source_type="document",
        steps_referenced=[1, 2]
    )
    
    source2 = ContextSourceVisualizationData(
        source_id="source_2",
        name="Document B",
        content="Content from document B",
        source_type="document",
        steps_referenced=[1, 3]
    )
    
    source3 = ContextSourceVisualizationData(
        source_id="source_3",
        name="Web Search",
        content="Content from web search",
        source_type="search",
        steps_referenced=[2, 3]
    )
    
    # Create metadata
    metadata = VisualizationMetadata(
        title="Test Trace",
        description="Test trace for steps visualizer testing",
        creator="test_suite",
        tags=["test", "reasoning"]
    )
    
    # Create the trace data
    return TraceVisualizationData(
        trace_id="trace_1",
        task="Sample reasoning task",
        steps=[step1, step2, step3],
        context_sources=[source1, source2, source3],
        final_result="The final answer is derived from the reasoning steps.",
        metrics={
            "overall_coherence": 0.83,
            "overall_relevance": 0.82,
            "overall_completeness": 0.78
        },
        metadata=metadata
    )


@pytest.fixture
def transition_visualizer():
    """Create a step transition visualizer instance for testing."""
    return StepTransitionVisualization()


@pytest.fixture
def details_visualizer():
    """Create a step details visualizer instance for testing."""
    return StepDetailsVisualization()


@pytest.fixture
def progress_visualizer():
    """Create a step progress visualizer instance for testing."""
    return StepProgressVisualization()


@pytest.fixture
def steps_visualizer():
    """Create a steps visualizer instance for testing."""
    return StepsVisualizer()


class TestStepTransitionVisualization:
    """Tests for the StepTransitionVisualization class."""
    
    def test_generate_transition_data(self, transition_visualizer, sample_trace_data):
        """Test generating transition data between steps."""
        transition_data = transition_visualizer.generate_transition_data(sample_trace_data)
        
        assert isinstance(transition_data, dict)
        assert "trace_id" in transition_data
        assert "total_transitions" in transition_data
        assert "transitions" in transition_data
        assert isinstance(transition_data["transitions"], list)
        assert len(transition_data["transitions"]) == len(sample_trace_data.steps) - 1
        
        # Check the first transition
        first_transition = transition_data["transitions"][0]
        assert first_transition["from_step"] == 1
        assert first_transition["to_step"] == 2
        assert first_transition["from_type"] == "reasoning"
        assert first_transition["to_type"] == "reflection"
        assert first_transition["type_change"] == True
        assert abs(first_transition["duration"] - 3.0) < 0.001  # 3 seconds between step 1 and 2
    
    def test_generate_step_flow_chart(self, transition_visualizer, sample_trace_data):
        """Test generating step flow chart visualization."""
        flow_chart = transition_visualizer.generate_step_flow_chart(sample_trace_data)
        
        assert isinstance(flow_chart, dict)
        assert "nodes" in flow_chart
        assert "edges" in flow_chart
        assert "trace_id" in flow_chart
        assert "step_types" in flow_chart
        
        assert len(flow_chart["nodes"]) == len(sample_trace_data.steps)
        assert len(flow_chart["edges"]) == len(sample_trace_data.steps) - 1
        
        # Check that all step types are represented
        step_types = set(step.step_type for step in sample_trace_data.steps)
        assert set(flow_chart["step_types"]) == step_types
        
        # Check the first node
        first_node = flow_chart["nodes"][0]
        assert first_node["id"] == "step_1"
        assert first_node["type"] == "reasoning"
        assert first_node["data"]["step_number"] == 1


class TestStepDetailsVisualization:
    """Tests for the StepDetailsVisualization class."""
    
    def test_generate_step_details(self, details_visualizer, sample_trace_data):
        """Test generating detailed visualization for a step."""
        # Test with first step
        step = sample_trace_data.steps[0]
        details = details_visualizer.generate_step_details(step, sample_trace_data)
        
        assert isinstance(details, dict)
        assert details["step_id"] == step.step_id
        assert details["step_number"] == step.step_number
        assert details["title"] == step.title
        assert details["content"] == step.content
        assert details["step_type"] == step.step_type
        assert details["timestamp"] == step.timestamp
        
        # Check context sources
        assert isinstance(details["context_sources"], list)
        assert len(details["context_sources"]) == 2  # Step 1 references 2 sources
        
        # Check position metrics
        assert details["position"]["is_first"] == True
        assert details["position"]["is_last"] == False
        
        # Test with last step
        last_step = sample_trace_data.steps[-1]
        last_details = details_visualizer.generate_step_details(last_step, sample_trace_data)
        assert last_details["position"]["is_first"] == False
        assert last_details["position"]["is_last"] == True
    
    def test_generate_step_comparison(self, details_visualizer, sample_trace_data):
        """Test generating step comparison visualization."""
        steps = sample_trace_data.steps
        comparison = details_visualizer.generate_step_comparison(steps)
        
        assert isinstance(comparison, dict)
        assert "step_numbers" in comparison
        assert "step_types" in comparison
        assert "timestamps" in comparison
        assert "durations" in comparison
        assert "context_counts" in comparison
        assert "metrics_comparison" in comparison
        
        # Check that all metrics are compared
        all_metrics = set()
        for step in steps:
            all_metrics.update(step.metrics.keys())
        
        for metric in all_metrics:
            assert metric in comparison["metrics_comparison"]
            assert len(comparison["metrics_comparison"][metric]) == len(steps)
        
        # Test empty steps list
        empty_comparison = details_visualizer.generate_step_comparison([])
        assert "error" in empty_comparison


class TestStepProgressVisualization:
    """Tests for the StepProgressVisualization class."""
    
    def test_generate_progress_data(self, progress_visualizer, sample_trace_data):
        """Test generating progress visualization data."""
        progress_data = progress_visualizer.generate_progress_data(sample_trace_data)
        
        assert isinstance(progress_data, dict)
        assert "trace_id" in progress_data
        assert "is_complete" in progress_data
        assert "steps_completed" in progress_data
        assert "current_step" in progress_data
        assert "current_step_type" in progress_data
        assert "steps_by_type" in progress_data
        assert "timeline" in progress_data
        assert "total_reasoning_time" in progress_data
        
        # Check completion status
        assert progress_data["is_complete"] == True  # Sample trace has final_result
        assert progress_data["steps_completed"] == len(sample_trace_data.steps)
        
        # Check timeline
        assert len(progress_data["timeline"]) == len(sample_trace_data.steps)
        
        # Check step type grouping
        steps_by_type = progress_data["steps_by_type"]
        assert "reasoning" in steps_by_type
        assert "reflection" in steps_by_type
        assert "conclusion" in steps_by_type
        assert 1 in steps_by_type["reasoning"]  # Step 1 is reasoning
        assert 2 in steps_by_type["reflection"]  # Step 2 is reflection
        assert 3 in steps_by_type["conclusion"]  # Step 3 is conclusion
    
    def test_generate_step_type_distribution(self, progress_visualizer, sample_trace_data):
        """Test generating step type distribution visualization."""
        distribution = progress_visualizer.generate_step_type_distribution(sample_trace_data)
        
        assert isinstance(distribution, dict)
        assert "trace_id" in distribution
        assert "step_types" in distribution
        assert "counts" in distribution
        assert "durations" in distribution
        assert "chart_data" in distribution
        
        # Check that all step types are included
        step_types = set(step.step_type for step in sample_trace_data.steps)
        assert set(distribution["step_types"]) == step_types
        
        # Check counts
        assert len(distribution["counts"]) == len(step_types)
        assert sum(distribution["counts"]) == len(sample_trace_data.steps)
        
        # Check chart data
        assert len(distribution["chart_data"]) == len(step_types)
        for item in distribution["chart_data"]:
            assert "type" in item
            assert "count" in item
            assert "total_duration" in item


class TestStepsVisualizer:
    """Tests for the StepsVisualizer class."""
    
    def test_generate_visualization_data(self, steps_visualizer, sample_trace_data):
        """Test generating complete visualization data."""
        visualization_data = steps_visualizer.generate_visualization_data(sample_trace_data)
        
        assert isinstance(visualization_data, dict)
        assert "trace_id" in visualization_data
        assert "steps" in visualization_data
        assert "transitions" in visualization_data
        assert "flow_chart" in visualization_data
        assert "progress" in visualization_data
        assert "type_distribution" in visualization_data
        
        # Check steps
        assert len(visualization_data["steps"]) == len(sample_trace_data.steps)
        
        # Check that all components are integrated
        assert visualization_data["trace_id"] == sample_trace_data.trace_id
        assert "transitions" in visualization_data["transitions"]
        assert "nodes" in visualization_data["flow_chart"]
        assert "is_complete" in visualization_data["progress"]
        assert "step_types" in visualization_data["type_distribution"]
    
    def test_to_json(self, steps_visualizer, sample_trace_data):
        """Test converting visualization data to JSON."""
        json_data = steps_visualizer.to_json(sample_trace_data)
        
        assert isinstance(json_data, str)
        
        # Parse JSON and check structure
        parsed = json.loads(json_data)
        assert isinstance(parsed, dict)
        assert "trace_id" in parsed
        assert "steps" in parsed
        assert "transitions" in parsed
        assert "flow_chart" in parsed
        assert "progress" in parsed
        assert "type_distribution" in parsed
    
    def test_visualize_step(self, steps_visualizer, sample_trace_data):
        """Test visualizing a specific step."""
        step = sample_trace_data.steps[1]  # Use the second step
        step_visualization = steps_visualizer.visualize_step(step, sample_trace_data)
        
        assert isinstance(step_visualization, dict)
        assert step_visualization["step_id"] == step.step_id
        assert step_visualization["step_number"] == step.step_number
        assert step_visualization["title"] == step.title
        assert step_visualization["content"] == step.content
        assert step_visualization["step_type"] == step.step_type
    
    def test_compare_steps(self, steps_visualizer, sample_trace_data):
        """Test comparing multiple steps."""
        steps = sample_trace_data.steps
        comparison = steps_visualizer.compare_steps(steps)
        
        assert isinstance(comparison, dict)
        assert "step_numbers" in comparison
        assert "step_types" in comparison
        assert "timestamps" in comparison
        assert "durations" in comparison
        assert "context_counts" in comparison
        assert "metrics_comparison" in comparison
        
        # Check that arrays have correct length
        assert len(comparison["step_numbers"]) == len(steps)
        assert len(comparison["step_types"]) == len(steps)
        
        # Check metrics comparison
        for metric_values in comparison["metrics_comparison"].values():
            assert len(metric_values) == len(steps)


# Test the whole component with dummy data
if __name__ == "__main__":
    # Create visualizer
    visualizer = StepsVisualizer()
    
    # Create dummy data
    step1 = StepVisualizationData(
        step_id="test_step_1",
        step_number=1,
        title="First Step",
        content="This is a test step content.",
        content_preview="This is a test...",
        step_type="reasoning",
        timestamp=1000.0,
        formatted_timestamp="2021-01-01 00:00:00",
        duration=1.0,
        context_references=[],
        metrics={},
        annotations=[],
        requires_next_step=True
    )
    
    trace_data = TraceVisualizationData(
        trace_id="test_trace",
        task="Test task",
        steps=[step1],
        context_sources=[],
        final_result=None,
        metrics={},
        metadata={}
    )
    
    # Generate visualization
    result = visualizer.generate_visualization_data(trace_data)
    print(f"Generated visualization with trace ID: {result['trace_id']}")
    print(f"Number of steps: {len(result['steps'])}") 