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
Unit tests for the ReasoningQualityMetrics component.
"""

import os
import json
import pytest
from datetime import datetime
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from beeai_framework.visualization.components.reasoning_quality_metrics import (
    MetricLevel,
    QualityMetric,
    ReasoningQualityMetrics
)
from beeai_framework.visualization.components.reasoning_trace_visualizer import (
    ReasoningStep,
    ReasoningTrace
)


@pytest.fixture
def mock_reasoning_trace():
    """Create a mock reasoning trace for testing."""
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
    
    # Add first step
    step1 = ReasoningStep(
        step_number=1,
        step_type="information_gathering",
        content="This is a test step content for information gathering.",
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
    trace.add_step(step1)
    
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
    
    # Add third step with decision making
    step3 = ReasoningStep(
        step_number=3,
        step_type="decision_making",
        content="Deciding on the best optimization approach based on analysis.",
        timestamp=datetime.now(),
        context_items=[
            {
                "content": "Context item 4",
                "similarity": 0.88,
                "metadata": {
                    "source": "document_4",
                    "level": "project",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ],
        metrics={
            "tokens": 180,
            "confidence": 0.92,
            "relevance": 0.90,
            "novelty": 0.65
        },
        key_concepts=[
            {"concept": "optimization strategy", "importance": 0.95},
            {"concept": "benchmark results", "importance": 0.9}
        ],
        next_step_suggestions=[
            "Implement chosen optimization",
            "Document decision rationale"
        ]
    )
    trace.add_step(step3)
    
    # Set end time
    trace.end_time = datetime.now()
    
    return trace


def test_metric_level_enum():
    """Test the MetricLevel enum values."""
    assert MetricLevel.STEP.value == "step"
    assert MetricLevel.TRACE.value == "trace"
    assert MetricLevel.GLOBAL.value == "global"


def test_quality_metric_creation():
    """Test creation of a QualityMetric."""
    metric = QualityMetric(
        name="coherence",
        level=MetricLevel.STEP,
        value=0.85,
        description="Measures logical flow and consistency",
        metadata={"timestamp": datetime.now().isoformat()}
    )
    
    assert metric.name == "coherence"
    assert metric.level == MetricLevel.STEP
    assert metric.value == 0.85
    assert "logical flow" in metric.description
    assert "timestamp" in metric.metadata


def test_quality_metric_to_dict():
    """Test converting a QualityMetric to dictionary."""
    timestamp = datetime.now().isoformat()
    metric = QualityMetric(
        name="coherence",
        level=MetricLevel.STEP,
        value=0.85,
        description="Measures logical flow and consistency",
        metadata={"timestamp": timestamp}
    )
    
    metric_dict = metric.to_dict()
    
    assert isinstance(metric_dict, dict)
    assert metric_dict["name"] == "coherence"
    assert metric_dict["level"] == "step"
    assert metric_dict["value"] == 0.85
    assert metric_dict["metadata"]["timestamp"] == timestamp


def test_reasoning_quality_metrics_initialization():
    """Test initialization of ReasoningQualityMetrics."""
    metrics = ReasoningQualityMetrics()
    
    assert metrics is not None
    assert hasattr(metrics, "add_metric")
    assert hasattr(metrics, "evaluate_trace")
    assert hasattr(metrics, "get_metrics")
    assert len(metrics.metrics) == 0


def test_add_metric():
    """Test adding metrics to ReasoningQualityMetrics."""
    metrics = ReasoningQualityMetrics()
    
    # Add a step-level metric
    metrics.add_metric(
        name="coherence",
        level=MetricLevel.STEP,
        value=0.85,
        step_number=1,
        trace_id="trace-001",
        description="Measures logical flow"
    )
    
    # Add a trace-level metric
    metrics.add_metric(
        name="overall_quality",
        level=MetricLevel.TRACE,
        value=0.90,
        trace_id="trace-001",
        description="Overall reasoning quality"
    )
    
    # Add a global metric
    metrics.add_metric(
        name="system_reliability",
        level=MetricLevel.GLOBAL,
        value=0.95,
        description="System-wide reliability score"
    )
    
    assert len(metrics.metrics) == 3
    
    # Verify the step-level metric
    step_metrics = metrics.get_metrics(level=MetricLevel.STEP)
    assert len(step_metrics) == 1
    assert step_metrics[0].name == "coherence"
    assert step_metrics[0].value == 0.85
    assert step_metrics[0].metadata["step_number"] == 1
    assert step_metrics[0].metadata["trace_id"] == "trace-001"
    
    # Verify the trace-level metric
    trace_metrics = metrics.get_metrics(level=MetricLevel.TRACE)
    assert len(trace_metrics) == 1
    assert trace_metrics[0].name == "overall_quality"
    
    # Verify the global metric
    global_metrics = metrics.get_metrics(level=MetricLevel.GLOBAL)
    assert len(global_metrics) == 1
    assert global_metrics[0].name == "system_reliability"


def test_get_metrics_by_name():
    """Test getting metrics by name."""
    metrics = ReasoningQualityMetrics()
    
    # Add multiple metrics with the same name
    metrics.add_metric(name="coherence", level=MetricLevel.STEP, value=0.85, step_number=1, trace_id="trace-001")
    metrics.add_metric(name="coherence", level=MetricLevel.STEP, value=0.90, step_number=2, trace_id="trace-001")
    metrics.add_metric(name="relevance", level=MetricLevel.STEP, value=0.75, step_number=1, trace_id="trace-001")
    
    # Get metrics by name
    coherence_metrics = metrics.get_metrics(name="coherence")
    assert len(coherence_metrics) == 2
    assert all(m.name == "coherence" for m in coherence_metrics)
    
    relevance_metrics = metrics.get_metrics(name="relevance")
    assert len(relevance_metrics) == 1
    assert relevance_metrics[0].name == "relevance"


def test_get_metrics_by_trace():
    """Test getting metrics for a specific trace."""
    metrics = ReasoningQualityMetrics()
    
    # Add metrics for different traces
    metrics.add_metric(name="coherence", level=MetricLevel.STEP, value=0.85, trace_id="trace-001")
    metrics.add_metric(name="relevance", level=MetricLevel.STEP, value=0.90, trace_id="trace-002")
    metrics.add_metric(name="overall_quality", level=MetricLevel.TRACE, value=0.95, trace_id="trace-001")
    
    # Get metrics for trace-001
    trace_001_metrics = metrics.get_metrics(trace_id="trace-001")
    assert len(trace_001_metrics) == 2
    assert all(m.metadata.get("trace_id") == "trace-001" for m in trace_001_metrics)
    
    # Get metrics for trace-002
    trace_002_metrics = metrics.get_metrics(trace_id="trace-002")
    assert len(trace_002_metrics) == 1
    assert trace_002_metrics[0].metadata.get("trace_id") == "trace-002"


def test_evaluate_trace(mock_reasoning_trace):
    """Test evaluating a trace with different metrics."""
    metrics = ReasoningQualityMetrics()
    
    # Evaluate the trace
    results = metrics.evaluate_trace(mock_reasoning_trace)
    
    # Verify results
    assert isinstance(results, dict)
    assert "trace_metrics" in results
    assert "step_metrics" in results
    
    # Check if common metrics are calculated
    trace_metrics = results["trace_metrics"]
    assert "coherence" in trace_metrics
    assert "completeness" in trace_metrics
    assert "relevance" in trace_metrics
    
    # Check step metrics
    step_metrics = results["step_metrics"]
    assert len(step_metrics) == 3  # 3 steps in the mock trace
    assert all("reasoning_depth" in step for step in step_metrics)
    assert all("context_relevance" in step for step in step_metrics)


def test_create_metrics_visualization(mock_reasoning_trace):
    """Test creating metrics visualization."""
    metrics = ReasoningQualityMetrics()
    
    # First evaluate the trace to generate metrics
    metrics.evaluate_trace(mock_reasoning_trace)
    
    # Create visualization
    fig = metrics.create_metrics_visualization(trace_id=mock_reasoning_trace.trace_id)
    
    assert isinstance(fig, go.Figure)
    # Confirm that the figure has at least some traces or annotations
    assert (len(fig.data) > 0) or (len(fig.layout.annotations) > 0)


def test_create_step_metrics_comparison(mock_reasoning_trace):
    """Test comparing metrics across steps."""
    metrics = ReasoningQualityMetrics()
    
    # Evaluate the trace to generate metrics
    metrics.evaluate_trace(mock_reasoning_trace)
    
    # Create step comparison visualization
    fig = metrics.create_step_metrics_comparison(trace_id=mock_reasoning_trace.trace_id)
    
    assert isinstance(fig, go.Figure)
    # Should have data for each step
    assert len(fig.data) > 0


def test_create_trace_comparison_visualization():
    """Test comparing metrics across multiple traces."""
    metrics = ReasoningQualityMetrics()
    
    # Add trace-level metrics for different traces
    trace_ids = ["trace-001", "trace-002", "trace-003"]
    metric_names = ["coherence", "completeness", "relevance"]
    
    for trace_id in trace_ids:
        for name in metric_names:
            metrics.add_metric(
                name=name,
                level=MetricLevel.TRACE,
                value=np.random.uniform(0.7, 0.95),
                trace_id=trace_id
            )
    
    # Create comparison visualization
    fig = metrics.create_trace_comparison_visualization(trace_ids=trace_ids)
    
    assert isinstance(fig, go.Figure)
    # Should have data for each trace
    assert len(fig.data) > 0


def test_save_and_load_metrics():
    """Test saving and loading metrics to/from JSON."""
    metrics = ReasoningQualityMetrics()
    
    # Add some metrics
    metrics.add_metric(name="coherence", level=MetricLevel.STEP, value=0.85, step_number=1, trace_id="trace-001")
    metrics.add_metric(name="relevance", level=MetricLevel.TRACE, value=0.90, trace_id="trace-001")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save metrics to JSON
        output_path = os.path.join(tmpdir, "test_metrics.json")
        metrics.save_metrics(output_path)
        
        # Check that file exists and has content
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Load metrics into a new instance
        new_metrics = ReasoningQualityMetrics()
        new_metrics.load_metrics(output_path)
        
        # Verify metrics were loaded correctly
        assert len(new_metrics.metrics) == 2
        
        loaded_coherence = new_metrics.get_metrics(name="coherence")[0]
        assert loaded_coherence.value == 0.85
        assert loaded_coherence.metadata["step_number"] == 1
        
        loaded_relevance = new_metrics.get_metrics(name="relevance")[0]
        assert loaded_relevance.value == 0.90


def test_aggregate_metrics():
    """Test aggregating metrics."""
    metrics = ReasoningQualityMetrics()
    
    # Add step-level metrics for the same trace
    metrics.add_metric(name="coherence", level=MetricLevel.STEP, value=0.85, step_number=1, trace_id="trace-001")
    metrics.add_metric(name="coherence", level=MetricLevel.STEP, value=0.90, step_number=2, trace_id="trace-001")
    metrics.add_metric(name="coherence", level=MetricLevel.STEP, value=0.80, step_number=3, trace_id="trace-001")
    
    # Aggregate the metrics
    agg_result = metrics.aggregate_metrics(name="coherence", trace_id="trace-001")
    
    assert isinstance(agg_result, dict)
    assert "mean" in agg_result
    assert "median" in agg_result
    assert "min" in agg_result
    assert "max" in agg_result
    assert "std" in agg_result
    
    assert agg_result["mean"] == pytest.approx(0.85)
    assert agg_result["min"] == 0.80
    assert agg_result["max"] == 0.90


def test_edge_case_empty_metrics():
    """Test behavior with no metrics."""
    metrics = ReasoningQualityMetrics()
    
    # Get metrics should return empty list
    assert len(metrics.get_metrics()) == 0
    
    # Visualizations should handle empty metrics gracefully
    fig = metrics.create_metrics_visualization()
    assert isinstance(fig, go.Figure)
    
    # Aggregation should handle empty data
    agg_result = metrics.aggregate_metrics(name="nonexistent")
    assert agg_result is None or isinstance(agg_result, dict)


def test_edge_case_metric_thresholds():
    """Test handling of metrics with threshold values (0 and 1)."""
    metrics = ReasoningQualityMetrics()
    
    # Add metrics with extreme values
    metrics.add_metric(name="perfect", level=MetricLevel.STEP, value=1.0, step_number=1)
    metrics.add_metric(name="worst", level=MetricLevel.STEP, value=0.0, step_number=2)
    
    perfect_metrics = metrics.get_metrics(name="perfect")
    assert perfect_metrics[0].value == 1.0
    
    worst_metrics = metrics.get_metrics(name="worst")
    assert worst_metrics[0].value == 0.0


def test_failure_case_invalid_metric_level():
    """Test behavior with invalid metric level."""
    metrics = ReasoningQualityMetrics()
    
    # Should raise ValueError for invalid level
    with pytest.raises(ValueError):
        metrics.add_metric(name="test", level="invalid_level", value=0.5)


if __name__ == "__main__":
    # Run the tests manually if needed
    pytest.main(["-xvs", __file__]) 