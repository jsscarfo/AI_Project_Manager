#!/usr/bin/env python
"""
Tests for reasoning trace visualizer component.

This module contains unit tests for the reasoning trace visualizer
component of the visualization framework.
"""

import pytest
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any

import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ..core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData,
    ContextSourceVisualizationData,
    VisualizationMetadata
)

from ..components.reasoning_trace_visualizer import (
    ReasoningTraceVisualizer,
    ReasoningTrace,
    ReasoningStep
)


@pytest.fixture
def sample_reasoning_step():
    """Create a sample reasoning step for testing."""
    return ReasoningStep(
        step_number=1,
        step_type="analysis",
        content="This is a sample reasoning step with [context reference 1] and [context reference 2].",
        timestamp=datetime.now(),
        context_items=[
            {
                "source_id": "source_1",
                "text": "context reference 1",
                "relevance_score": 0.85,
                "source_type": "document"
            },
            {
                "source_id": "source_2",
                "text": "context reference 2",
                "relevance_score": 0.75,
                "source_type": "code"
            }
        ],
        metrics={
            "coherence": 0.92,
            "relevance": 0.87,
            "completeness": 0.78
        },
        key_concepts=[
            {"concept": "ML model", "importance": 0.9},
            {"concept": "data preprocessing", "importance": 0.8}
        ],
        next_step_suggestions=[
            "Analyze model performance",
            "Review data quality"
        ]
    )


@pytest.fixture
def sample_reasoning_trace(sample_reasoning_step):
    """Create a sample reasoning trace with multiple steps for testing."""
    # Create a trace with multiple steps
    trace = ReasoningTrace(
        trace_id="test-trace-001",
        task="Analyze ML model performance",
        start_time=datetime.now() - timedelta(minutes=30)
    )
    
    # Add the first step
    trace.add_step(sample_reasoning_step)
    
    # Add more steps with different types and data
    for i in range(2, 6):
        step = ReasoningStep(
            step_number=i,
            step_type="synthesis" if i % 2 == 0 else "analysis",
            content=f"This is step {i} of the reasoning process with [reference {i}].",
            timestamp=datetime.now() - timedelta(minutes=30-i*5),
            context_items=[
                {
                    "source_id": f"source_{i}",
                    "text": f"reference {i}",
                    "relevance_score": 0.7 + (i * 0.05),
                    "source_type": "document" if i % 2 == 0 else "code"
                }
            ],
            metrics={
                "coherence": 0.7 + (i * 0.05),
                "relevance": 0.75 + (i * 0.03),
                "completeness": 0.65 + (i * 0.07)
            },
            key_concepts=[
                {"concept": f"concept {i}.1", "importance": 0.8},
                {"concept": f"concept {i}.2", "importance": 0.7}
            ],
            next_step_suggestions=[
                f"Suggestion {i}.1",
                f"Suggestion {i}.2"
            ]
        )
        trace.add_step(step)
    
    # Set end time
    trace.end_time = datetime.now()
    
    # Set overall metrics
    trace.overall_metrics = {
        "total_steps": 5,
        "average_coherence": 0.85,
        "average_relevance": 0.83,
        "average_completeness": 0.78,
        "duration_seconds": 1800  # 30 minutes
    }
    
    return trace


@pytest.fixture
def visualizer():
    """Create a visualizer instance for testing."""
    return ReasoningTraceVisualizer(
        cache_traces=True,
        default_height=600,
        default_width=800
    )


class TestReasoningStep:
    """Tests for the ReasoningStep class."""
    
    def test_to_dict(self, sample_reasoning_step):
        """Test conversion to dictionary format."""
        # Convert to dict
        result = sample_reasoning_step.to_dict()
        
        # Check all expected fields
        assert isinstance(result, dict)
        assert result["step_number"] == 1
        assert result["step_type"] == "analysis"
        assert result["content"] == "This is a sample reasoning step with [context reference 1] and [context reference 2]."
        assert isinstance(result["timestamp"], str)
        assert len(result["context_items"]) == 2
        assert len(result["metrics"]) == 3
        assert len(result["key_concepts"]) == 2
        assert len(result["next_step_suggestions"]) == 2


class TestReasoningTrace:
    """Tests for the ReasoningTrace class."""
    
    def test_add_step(self, sample_reasoning_trace, sample_reasoning_step):
        """Test adding a step to a trace."""
        # Count initial steps
        initial_count = len(sample_reasoning_trace.steps)
        
        # Create a new step
        new_step = ReasoningStep(
            step_number=initial_count + 1,
            step_type="conclusion",
            content="This is the conclusion step",
            timestamp=datetime.now()
        )
        
        # Add the step
        sample_reasoning_trace.add_step(new_step)
        
        # Verify step was added
        assert len(sample_reasoning_trace.steps) == initial_count + 1
        assert sample_reasoning_trace.steps[-1].step_type == "conclusion"
    
    def test_to_dict(self, sample_reasoning_trace):
        """Test conversion to dictionary format."""
        # Convert to dict
        result = sample_reasoning_trace.to_dict()
        
        # Check all expected fields
        assert isinstance(result, dict)
        assert result["trace_id"] == "test-trace-001"
        assert result["task"] == "Analyze ML model performance"
        assert isinstance(result["start_time"], str)
        assert isinstance(result["end_time"], str)
        assert len(result["steps"]) == 5
        assert isinstance(result["overall_metrics"], dict)
        assert result["overall_metrics"]["total_steps"] == 5


class TestReasoningTraceVisualizer:
    """Tests for the ReasoningTraceVisualizer class."""
    
    def test_create_step_visualization(self, visualizer, sample_reasoning_trace):
        """Test creation of step visualization."""
        # Create visualization
        fig = visualizer.create_step_visualization(
            trace=sample_reasoning_trace,
            selected_step=1,
            highlight_context=True,
            show_metrics=True
        )
        
        # Check the result
        assert isinstance(fig, go.Figure)
        
        # Verify figure has expected elements (subplots)
        assert len(fig.data) > 0
        assert "Reasoning Step Content" in fig.layout.annotations[0].text
    
    def test_create_context_relevance_visualization(self, visualizer, sample_reasoning_trace):
        """Test creation of context relevance visualization."""
        # Create visualization
        fig = visualizer.create_context_relevance_visualization(
            trace=sample_reasoning_trace,
            selected_step=2
        )
        
        # Check the result
        assert isinstance(fig, go.Figure)
        
        # Verify figure has expected elements
        assert len(fig.data) > 0
    
    def test_create_knowledge_graph_visualization(self, visualizer, sample_reasoning_trace):
        """Test creation of knowledge graph visualization."""
        # Create visualization
        fig = visualizer.create_knowledge_graph_visualization(
            trace=sample_reasoning_trace
        )
        
        # Check the result
        assert isinstance(fig, go.Figure)
        
        # Verify figure has expected elements
        assert len(fig.data) > 0
    
    def test_create_context_evolution_timeline(self, visualizer, sample_reasoning_trace):
        """Test creation of context evolution timeline."""
        # Create visualization
        fig = visualizer.create_context_evolution_timeline(
            trace=sample_reasoning_trace
        )
        
        # Check the result
        assert isinstance(fig, go.Figure)
        
        # Verify figure has expected elements
        assert len(fig.data) > 0
    
    def test_export_and_import_trace(self, visualizer, sample_reasoning_trace, tmp_path):
        """Test exporting and importing a trace to and from JSON."""
        # Create a temporary file path
        temp_file = tmp_path / "test_trace.json"
        
        # Cache the trace
        visualizer.trace_cache[sample_reasoning_trace.trace_id] = sample_reasoning_trace
        
        # Export the trace
        success = visualizer.export_trace_to_json(
            trace_id=sample_reasoning_trace.trace_id,
            file_path=str(temp_file)
        )
        
        # Verify export succeeded
        assert success is True
        assert os.path.exists(temp_file)
        
        # Import the trace
        imported_trace = visualizer.import_trace_from_json(str(temp_file))
        
        # Verify import succeeded
        assert imported_trace is not None
        assert imported_trace.trace_id == sample_reasoning_trace.trace_id
        assert len(imported_trace.steps) == len(sample_reasoning_trace.steps)
    
    def test_dict_to_trace(self, visualizer, sample_reasoning_trace):
        """Test conversion from dictionary to trace object."""
        # Convert trace to dict
        trace_dict = sample_reasoning_trace.to_dict()
        
        # Convert back to trace
        result = visualizer._dict_to_trace(trace_dict)
        
        # Verify conversion
        assert isinstance(result, ReasoningTrace)
        assert result.trace_id == sample_reasoning_trace.trace_id
        assert len(result.steps) == len(sample_reasoning_trace.steps)
    
    def test_highlight_context_references(self, visualizer):
        """Test highlighting context references in text."""
        # Sample text and context items
        text = "This text contains reference to something important."
        context_items = [
            {
                "source_id": "src1",
                "text": "something important",
                "relevance_score": 0.85
            }
        ]
        
        # Highlight references
        result = visualizer._highlight_context_references(text, context_items)
        
        # Verify highlighting - should include HTML markup
        assert "something important" in result
        assert "<span" in result
        assert "data-source-id" in result 