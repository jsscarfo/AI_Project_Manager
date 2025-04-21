#!/usr/bin/env python
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for the ReasoningTraceVisualizer component.
"""

import os
import json
import pytest
from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import plotly.graph_objects as go

from beeai_framework.visualization.components.reasoning_trace_visualizer import (
    ReasoningStep,
    ReasoningTrace,
    ReasoningTraceVisualizer
)


@pytest.fixture
def mock_reasoning_step():
    """Fixture that creates a mock reasoning step."""
    return ReasoningStep(
        step_number=1,
        step_type="information_gathering",
        content="This is a test step content.",
        timestamp=datetime.now(),
        context_items=[
            {
                "content": "Context item 1",
                "similarity": 0.85,
                "metadata": {
                    "source": "document_1",
                    "level": "domain",
                    "timestamp": datetime.now().isoformat()
                }
            },
            {
                "content": "Context item 2",
                "similarity": 0.75,
                "metadata": {
                    "source": "document_2",
                    "level": "techstack",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ],
        metrics={
            "tokens": 150,
            "confidence": 0.85,
            "relevance": 0.78,
            "novelty": 0.65
        },
        key_concepts=[
            {"concept": "algorithm", "importance": 0.9},
            {"concept": "performance", "importance": 0.7}
        ],
        next_step_suggestions=[
            "Evaluate algorithm performance",
            "Consider alternative approaches"
        ]
    )


@pytest.fixture
def mock_reasoning_trace(mock_reasoning_step):
    """Fixture that creates a mock reasoning trace with steps."""
    trace = ReasoningTrace(
        trace_id="test-trace-001",
        task="Optimize sorting algorithm performance",
        start_time=datetime.now(),
        overall_metrics={
            "completion_time_s": 5.2,
            "token_usage": 450,
            "total_context_items": 5
        }
    )
    
    # Add initial step
    trace.add_step(mock_reasoning_step)
    
    # Add second step
    step2 = ReasoningStep(
        step_number=2,
        step_type="analysis",
        content="Analyzing performance bottlenecks in the sorting algorithm.",
        timestamp=datetime.now(),
        context_items=[
            {
                "content": "Context item 3",
                "similarity": 0.92,
                "metadata": {
                    "source": "document_3",
                    "level": "project",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ],
        metrics={
            "tokens": 200,
            "confidence": 0.88,
            "relevance": 0.85,
            "novelty": 0.72
        },
        key_concepts=[
            {"concept": "time complexity", "importance": 0.95},
            {"concept": "space complexity", "importance": 0.85}
        ],
        next_step_suggestions=[
            "Implement optimization technique",
            "Benchmark against alternatives"
        ]
    )
    trace.add_step(step2)
    
    # Set end time
    trace.end_time = datetime.now()
    
    return trace


def test_reasoning_step_creation(mock_reasoning_step):
    """Test creation of a ReasoningStep object."""
    assert mock_reasoning_step.step_number == 1
    assert mock_reasoning_step.step_type == "information_gathering"
    assert "test step content" in mock_reasoning_step.content
    assert len(mock_reasoning_step.context_items) == 2
    assert mock_reasoning_step.metrics["confidence"] == 0.85
    assert len(mock_reasoning_step.key_concepts) == 2
    assert len(mock_reasoning_step.next_step_suggestions) == 2


def test_reasoning_step_to_dict(mock_reasoning_step):
    """Test converting a ReasoningStep to dictionary."""
    step_dict = mock_reasoning_step.to_dict()
    
    assert isinstance(step_dict, dict)
    assert step_dict["step_number"] == 1
    assert step_dict["step_type"] == "information_gathering"
    assert "test step content" in step_dict["content"]
    assert len(step_dict["context_items"]) == 2
    assert step_dict["metrics"]["confidence"] == 0.85
    assert len(step_dict["key_concepts"]) == 2


def test_reasoning_trace_creation(mock_reasoning_trace):
    """Test creation of a ReasoningTrace object."""
    assert mock_reasoning_trace.trace_id == "test-trace-001"
    assert mock_reasoning_trace.task == "Optimize sorting algorithm performance"
    assert mock_reasoning_trace.overall_metrics["completion_time_s"] == 5.2
    assert len(mock_reasoning_trace.steps) == 2


def test_reasoning_trace_to_dict(mock_reasoning_trace):
    """Test converting a ReasoningTrace to dictionary."""
    trace_dict = mock_reasoning_trace.to_dict()
    
    assert isinstance(trace_dict, dict)
    assert trace_dict["trace_id"] == "test-trace-001"
    assert trace_dict["task"] == "Optimize sorting algorithm performance"
    assert trace_dict["overall_metrics"]["completion_time_s"] == 5.2
    assert len(trace_dict["steps"]) == 2


def test_reasoning_trace_from_dict(mock_reasoning_trace):
    """Test creating a ReasoningTrace from dictionary."""
    trace_dict = mock_reasoning_trace.to_dict()
    new_trace = ReasoningTrace.from_dict(trace_dict)
    
    assert new_trace.trace_id == mock_reasoning_trace.trace_id
    assert new_trace.task == mock_reasoning_trace.task
    assert new_trace.overall_metrics["completion_time_s"] == mock_reasoning_trace.overall_metrics["completion_time_s"]
    assert len(new_trace.steps) == len(mock_reasoning_trace.steps)


def test_visualizer_initialization():
    """Test initialization of ReasoningTraceVisualizer."""
    visualizer = ReasoningTraceVisualizer()
    assert visualizer is not None
    assert hasattr(visualizer, 'create_step_visualization')
    assert hasattr(visualizer, 'create_trace_visualization')


def test_visualizer_load_trace(mock_reasoning_trace):
    """Test loading a trace into the visualizer."""
    visualizer = ReasoningTraceVisualizer()
    visualizer.load_trace(mock_reasoning_trace)
    
    assert visualizer.trace is not None
    assert visualizer.trace.trace_id == "test-trace-001"
    assert len(visualizer.trace.steps) == 2


def test_visualizer_create_step_visualization(mock_reasoning_trace):
    """Test creating a visualization for a specific step."""
    visualizer = ReasoningTraceVisualizer()
    visualizer.load_trace(mock_reasoning_trace)
    
    fig = visualizer.create_step_visualization(step_number=1)
    
    assert isinstance(fig, go.Figure)
    # Confirm that the figure has at least some traces or annotations
    assert (len(fig.data) > 0) or (len(fig.layout.annotations) > 0)


def test_visualizer_create_trace_visualization(mock_reasoning_trace):
    """Test creating a visualization for the entire trace."""
    visualizer = ReasoningTraceVisualizer()
    visualizer.load_trace(mock_reasoning_trace)
    
    fig = visualizer.create_trace_visualization()
    
    assert isinstance(fig, go.Figure)
    # Confirm that the figure has at least some traces or annotations
    assert (len(fig.data) > 0) or (len(fig.layout.annotations) > 0)


def test_visualizer_create_context_usage_visualization(mock_reasoning_trace):
    """Test creating a visualization for context usage."""
    visualizer = ReasoningTraceVisualizer()
    visualizer.load_trace(mock_reasoning_trace)
    
    fig = visualizer.create_context_usage_visualization()
    
    assert isinstance(fig, go.Figure)
    # Confirm that the figure has at least some traces or annotations
    assert (len(fig.data) > 0) or (len(fig.layout.annotations) > 0)


def test_visualizer_save_and_load_html(mock_reasoning_trace):
    """Test saving and loading HTML visualizations."""
    visualizer = ReasoningTraceVisualizer()
    visualizer.load_trace(mock_reasoning_trace)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save visualization
        output_path = os.path.join(tmpdir, "test_trace.html")
        visualizer.save_visualization(output_path)
        
        # Check that file exists and has content
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Check content has HTML structure
        with open(output_path, 'r') as f:
            content = f.read()
            assert '<html>' in content.lower()
            assert '<body>' in content.lower()
            assert 'plotly' in content.lower()


def test_visualizer_save_and_load_json(mock_reasoning_trace):
    """Test saving and loading trace as JSON."""
    visualizer = ReasoningTraceVisualizer()
    visualizer.load_trace(mock_reasoning_trace)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save trace as JSON
        output_path = os.path.join(tmpdir, "test_trace.json")
        visualizer.save_trace_json(output_path)
        
        # Check that file exists and has content
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Load the trace back and verify
        new_visualizer = ReasoningTraceVisualizer()
        new_visualizer.load_trace_json(output_path)
        
        assert new_visualizer.trace.trace_id == mock_reasoning_trace.trace_id
        assert new_visualizer.trace.task == mock_reasoning_trace.task
        assert len(new_visualizer.trace.steps) == len(mock_reasoning_trace.steps)


def test_edge_case_empty_trace():
    """Test visualizer with an empty trace (no steps)."""
    empty_trace = ReasoningTrace(
        trace_id="empty-trace",
        task="Empty task",
        start_time=datetime.now(),
        overall_metrics={}
    )
    
    visualizer = ReasoningTraceVisualizer()
    visualizer.load_trace(empty_trace)
    
    # Should handle empty trace without errors
    fig = visualizer.create_trace_visualization()
    assert isinstance(fig, go.Figure)


def test_edge_case_many_steps():
    """Test visualizer with a trace containing many steps."""
    many_steps_trace = ReasoningTrace(
        trace_id="many-steps-trace",
        task="Task with many steps",
        start_time=datetime.now(),
        overall_metrics={}
    )
    
    # Add 20 steps
    for i in range(1, 21):
        step = ReasoningStep(
            step_number=i,
            step_type="analysis",
            content=f"Step {i} content",
            timestamp=datetime.now(),
            context_items=[],
            metrics={},
            key_concepts=[],
            next_step_suggestions=[]
        )
        many_steps_trace.add_step(step)
    
    visualizer = ReasoningTraceVisualizer()
    visualizer.load_trace(many_steps_trace)
    
    # Should handle many steps without errors
    fig = visualizer.create_trace_visualization()
    assert isinstance(fig, go.Figure)


def test_failure_case_invalid_step_number(mock_reasoning_trace):
    """Test behavior when requesting a visualization for an invalid step number."""
    visualizer = ReasoningTraceVisualizer()
    visualizer.load_trace(mock_reasoning_trace)
    
    # Try to visualize a non-existent step
    with pytest.raises(ValueError):
        visualizer.create_step_visualization(step_number=99)


if __name__ == "__main__":
    # Run the tests manually if needed
    pytest.main(["-xvs", __file__]) 