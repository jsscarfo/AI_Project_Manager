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
Unit tests for the EvaluationDashboard component.
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

from beeai_framework.visualization.components.evaluation_dashboard import (
    DashboardConfig,
    EvaluationDashboard
)
from beeai_framework.visualization.components.reasoning_trace_visualizer import (
    ReasoningTrace,
    ReasoningStep
)
from beeai_framework.visualization.components.reasoning_quality_metrics import (
    ReasoningQualityMetrics,
    QualityMetric,
    MetricLevel
)
from beeai_framework.visualization.components.context_usage_analytics import (
    ContextUsageAnalytics,
    ContextUsageStats
)


@pytest.fixture
def mock_reasoning_trace():
    """Create a mock reasoning trace for testing."""
    trace = ReasoningTrace(
        trace_id="test-trace-001",
        task="Evaluate machine learning model performance",
        start_time=datetime.now(),
        overall_metrics={
            "completion_time_s": 6.5,
            "token_usage": 520,
            "total_context_items": 6
        }
    )
    
    # Add steps with context items
    step1 = ReasoningStep(
        step_number=1,
        step_type="information_gathering",
        content="This is a test step for information gathering.",
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
            }
        ],
        metrics={
            "tokens": 150,
            "confidence": 0.85
        }
    )
    trace.add_step(step1)
    
    step2 = ReasoningStep(
        step_number=2,
        step_type="analysis",
        content="Analyzing model performance metrics.",
        timestamp=datetime.now(),
        context_items=[
            {
                "content": "Context item 2",
                "similarity": 0.92,
                "metadata": {
                    "source": "document_2",
                    "level": "project",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ],
        metrics={
            "tokens": 180,
            "confidence": 0.91
        }
    )
    trace.add_step(step2)
    
    step3 = ReasoningStep(
        step_number=3,
        step_type="conclusion",
        content="Drawing conclusions based on model evaluation.",
        timestamp=datetime.now(),
        context_items=[
            {
                "content": "Context item 3",
                "similarity": 0.78,
                "metadata": {
                    "source": "document_3",
                    "level": "user_input",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ],
        metrics={
            "tokens": 190,
            "confidence": 0.89
        }
    )
    trace.add_step(step3)
    
    trace.end_time = datetime.now()
    
    return trace


@pytest.fixture
def mock_quality_metrics(mock_reasoning_trace):
    """Create mock reasoning quality metrics."""
    metrics = ReasoningQualityMetrics()
    
    # Add global metrics
    metrics.add_metric(
        QualityMetric(
            name="overall_coherence",
            value=0.87,
            level=MetricLevel.GLOBAL,
            description="Overall coherence of reasoning",
            metadata={"importance": "high"}
        )
    )
    
    metrics.add_metric(
        QualityMetric(
            name="factual_accuracy",
            value=0.92,
            level=MetricLevel.GLOBAL,
            description="Factual accuracy of conclusions",
            metadata={"importance": "critical"}
        )
    )
    
    # Add trace-level metrics
    metrics.add_metric(
        QualityMetric(
            name="logical_consistency",
            value=0.85,
            level=MetricLevel.TRACE,
            trace_id=mock_reasoning_trace.trace_id,
            description="Logical consistency within the trace",
            metadata={"importance": "high"}
        )
    )
    
    metrics.add_metric(
        QualityMetric(
            name="context_relevance",
            value=0.88,
            level=MetricLevel.TRACE,
            trace_id=mock_reasoning_trace.trace_id,
            description="Relevance of retrieved context",
            metadata={"importance": "medium"}
        )
    )
    
    # Add step-level metrics
    for step in mock_reasoning_trace.steps:
        metrics.add_metric(
            QualityMetric(
                name="step_clarity",
                value=0.80 + (step.step_number * 0.05),  # Increasing quality with steps
                level=MetricLevel.STEP,
                trace_id=mock_reasoning_trace.trace_id,
                step_number=step.step_number,
                description="Clarity of reasoning in this step",
                metadata={"importance": "medium"}
            )
        )
    
    return metrics


@pytest.fixture
def mock_context_analytics(mock_reasoning_trace):
    """Create mock context usage analytics."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    return analytics


def test_dashboard_config_creation():
    """Test creation of DashboardConfig."""
    config = DashboardConfig(
        title="Test Dashboard",
        description="A test dashboard for evaluation",
        show_trace_visualization=True,
        show_metrics_summary=True,
        show_context_usage=True,
        show_performance_metrics=True,
        layout_template="plotly_white",
        chart_height=500,
        chart_width=800
    )
    
    assert config.title == "Test Dashboard"
    assert config.description == "A test dashboard for evaluation"
    assert config.show_trace_visualization is True
    assert config.show_metrics_summary is True
    assert config.show_context_usage is True
    assert config.show_performance_metrics is True
    assert config.layout_template == "plotly_white"
    assert config.chart_height == 500
    assert config.chart_width == 800


def test_dashboard_config_defaults():
    """Test default values for DashboardConfig."""
    config = DashboardConfig(
        title="Test Dashboard"
    )
    
    assert config.title == "Test Dashboard"
    assert config.description is not None
    assert config.show_trace_visualization is True
    assert config.show_metrics_summary is True
    assert config.show_context_usage is True
    assert config.show_performance_metrics is True


def test_dashboard_config_to_dict():
    """Test converting DashboardConfig to dictionary."""
    config = DashboardConfig(
        title="Test Dashboard",
        description="A test dashboard for evaluation",
        show_metrics_summary=False
    )
    
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict["title"] == "Test Dashboard"
    assert config_dict["description"] == "A test dashboard for evaluation"
    assert config_dict["show_metrics_summary"] is False
    assert config_dict["show_trace_visualization"] is True  # Default value


def test_evaluation_dashboard_initialization():
    """Test initialization of EvaluationDashboard."""
    config = DashboardConfig(title="Test Dashboard")
    dashboard = EvaluationDashboard(config=config)
    
    assert dashboard is not None
    assert dashboard.config.title == "Test Dashboard"
    assert hasattr(dashboard, "traces") and len(dashboard.traces) == 0
    assert hasattr(dashboard, "quality_metrics") and dashboard.quality_metrics is None
    assert hasattr(dashboard, "context_analytics") and dashboard.context_analytics is None


def test_add_reasoning_trace(mock_reasoning_trace):
    """Test adding a reasoning trace to the dashboard."""
    dashboard = EvaluationDashboard()
    
    # Add trace
    dashboard.add_reasoning_trace(mock_reasoning_trace)
    
    assert len(dashboard.traces) == 1
    assert dashboard.traces[0].trace_id == mock_reasoning_trace.trace_id
    assert len(dashboard.traces[0].steps) == len(mock_reasoning_trace.steps)


def test_add_quality_metrics(mock_quality_metrics):
    """Test adding quality metrics to the dashboard."""
    dashboard = EvaluationDashboard()
    
    # Add metrics
    dashboard.add_quality_metrics(mock_quality_metrics)
    
    assert dashboard.quality_metrics is not None
    assert len(dashboard.quality_metrics.metrics) == len(mock_quality_metrics.metrics)


def test_add_context_analytics(mock_context_analytics):
    """Test adding context analytics to the dashboard."""
    dashboard = EvaluationDashboard()
    
    # Add context analytics
    dashboard.add_context_analytics(mock_context_analytics)
    
    assert dashboard.context_analytics is not None
    assert len(dashboard.context_analytics.usage_stats) == len(mock_context_analytics.usage_stats)


def test_create_dashboard_with_trace_only(mock_reasoning_trace):
    """Test creating dashboard with only trace visualization."""
    config = DashboardConfig(
        title="Trace-Only Dashboard",
        show_metrics_summary=False,
        show_context_usage=False,
        show_performance_metrics=False
    )
    dashboard = EvaluationDashboard(config=config)
    dashboard.add_reasoning_trace(mock_reasoning_trace)
    
    # Create dashboard
    fig = dashboard.create_dashboard()
    
    assert isinstance(fig, go.Figure)
    # Should have a figure with at least one subplot for the trace
    assert len(fig.data) > 0


def test_create_dashboard_with_all_components(mock_reasoning_trace, mock_quality_metrics, mock_context_analytics):
    """Test creating dashboard with all components."""
    dashboard = EvaluationDashboard()
    dashboard.add_reasoning_trace(mock_reasoning_trace)
    dashboard.add_quality_metrics(mock_quality_metrics)
    dashboard.add_context_analytics(mock_context_analytics)
    
    # Create dashboard
    fig = dashboard.create_dashboard()
    
    assert isinstance(fig, go.Figure)
    # Should have a figure with multiple subplots
    assert len(fig.data) > 0
    assert hasattr(fig.layout, 'template')


def test_create_metrics_summary(mock_reasoning_trace, mock_quality_metrics):
    """Test creating metrics summary."""
    dashboard = EvaluationDashboard()
    dashboard.add_reasoning_trace(mock_reasoning_trace)
    dashboard.add_quality_metrics(mock_quality_metrics)
    
    # Create metrics summary
    fig = dashboard.create_metrics_summary()
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Should have data for metrics


def test_create_context_usage_summary(mock_reasoning_trace, mock_context_analytics):
    """Test creating context usage summary."""
    dashboard = EvaluationDashboard()
    dashboard.add_reasoning_trace(mock_reasoning_trace)
    dashboard.add_context_analytics(mock_context_analytics)
    
    # Create context usage summary
    fig = dashboard.create_context_usage_summary()
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Should have data for context usage


def test_create_performance_metrics(mock_reasoning_trace):
    """Test creating performance metrics summary."""
    dashboard = EvaluationDashboard()
    dashboard.add_reasoning_trace(mock_reasoning_trace)
    
    # Create performance metrics
    fig = dashboard.create_performance_metrics()
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Should have data for performance metrics


def test_save_dashboard_html():
    """Test saving dashboard to HTML."""
    dashboard = EvaluationDashboard()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save dashboard to HTML
        output_path = os.path.join(tmpdir, "test_dashboard.html")
        dashboard.save_dashboard_html(output_path)
        
        # Check that file exists
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_export_dashboard_data(mock_reasoning_trace, mock_quality_metrics, mock_context_analytics):
    """Test exporting dashboard data to JSON."""
    dashboard = EvaluationDashboard()
    dashboard.add_reasoning_trace(mock_reasoning_trace)
    dashboard.add_quality_metrics(mock_quality_metrics)
    dashboard.add_context_analytics(mock_context_analytics)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export dashboard data
        output_path = os.path.join(tmpdir, "test_dashboard_data.json")
        dashboard.export_dashboard_data(output_path)
        
        # Check that file exists and has content
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Load the JSON and check content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert "config" in data
        assert "traces" in data
        assert len(data["traces"]) > 0
        assert "quality_metrics" in data


def test_configure_layout():
    """Test configuring dashboard layout."""
    config = DashboardConfig(
        title="Custom Layout Dashboard",
        layout_template="plotly_dark",
        chart_height=600,
        chart_width=900
    )
    dashboard = EvaluationDashboard(config=config)
    
    # Create a simple figure to test layout configuration
    fig = go.Figure()
    dashboard._configure_layout(fig)
    
    assert fig.layout.template == "plotly_dark"
    assert fig.layout.height == 600
    assert fig.layout.width == 900
    assert fig.layout.title.text == "Custom Layout Dashboard"


def test_multiple_traces():
    """Test adding and visualizing multiple traces."""
    dashboard = EvaluationDashboard()
    
    # Create two simple traces
    trace1 = ReasoningTrace(
        trace_id="trace-001",
        task="Task 1",
        start_time=datetime.now()
    )
    trace1.add_step(ReasoningStep(
        step_number=1,
        step_type="analysis",
        content="Step for trace 1",
        timestamp=datetime.now()
    ))
    trace1.end_time = datetime.now()
    
    trace2 = ReasoningTrace(
        trace_id="trace-002",
        task="Task 2",
        start_time=datetime.now()
    )
    trace2.add_step(ReasoningStep(
        step_number=1,
        step_type="analysis",
        content="Step for trace 2",
        timestamp=datetime.now()
    ))
    trace2.end_time = datetime.now()
    
    # Add both traces
    dashboard.add_reasoning_trace(trace1)
    dashboard.add_reasoning_trace(trace2)
    
    assert len(dashboard.traces) == 2
    assert dashboard.traces[0].trace_id == "trace-001"
    assert dashboard.traces[1].trace_id == "trace-002"
    
    # Create dashboard (should handle multiple traces)
    fig = dashboard.create_dashboard()
    assert isinstance(fig, go.Figure)


def test_filter_traces_by_id():
    """Test filtering traces by ID."""
    dashboard = EvaluationDashboard()
    
    # Add three traces with different IDs
    for i in range(1, 4):
        trace = ReasoningTrace(
            trace_id=f"trace-00{i}",
            task=f"Task {i}",
            start_time=datetime.now()
        )
        trace.add_step(ReasoningStep(
            step_number=1,
            step_type="analysis",
            content=f"Step for trace {i}",
            timestamp=datetime.now()
        ))
        trace.end_time = datetime.now()
        dashboard.add_reasoning_trace(trace)
    
    # Filter by ID
    filtered = dashboard.filter_traces_by_id("trace-002")
    
    assert len(filtered) == 1
    assert filtered[0].trace_id == "trace-002"


def test_filter_traces_by_property():
    """Test filtering traces by property."""
    dashboard = EvaluationDashboard()
    
    # Add traces with different properties
    trace1 = ReasoningTrace(
        trace_id="trace-001",
        task="ML model training",
        start_time=datetime.now()
    )
    trace1.overall_metrics = {"success": True, "score": 0.85}
    trace1.end_time = datetime.now()
    
    trace2 = ReasoningTrace(
        trace_id="trace-002",
        task="Data preprocessing",
        start_time=datetime.now()
    )
    trace2.overall_metrics = {"success": False, "score": 0.65}
    trace2.end_time = datetime.now()
    
    dashboard.add_reasoning_trace(trace1)
    dashboard.add_reasoning_trace(trace2)
    
    # Filter by property
    successful_traces = dashboard.filter_traces_by_property("overall_metrics.success", True)
    
    assert len(successful_traces) == 1
    assert successful_traces[0].trace_id == "trace-001"


def test_get_aggregate_metrics(mock_reasoning_trace, mock_quality_metrics):
    """Test getting aggregate metrics."""
    dashboard = EvaluationDashboard()
    dashboard.add_reasoning_trace(mock_reasoning_trace)
    dashboard.add_quality_metrics(mock_quality_metrics)
    
    # Get aggregate metrics
    metrics = dashboard.get_aggregate_metrics()
    
    assert isinstance(metrics, dict)
    assert "global_metrics" in metrics
    assert "trace_metrics" in metrics
    assert len(metrics["global_metrics"]) > 0
    assert len(metrics["trace_metrics"]) > 0


def test_edge_case_empty_dashboard():
    """Test creating dashboard with no data."""
    dashboard = EvaluationDashboard()
    
    # Create dashboard with no data
    fig = dashboard.create_dashboard()
    
    assert isinstance(fig, go.Figure)
    # Should have a message about no data
    assert hasattr(fig, 'layout')


def test_edge_case_missing_components():
    """Test creating dashboard with some components missing."""
    dashboard = EvaluationDashboard()
    
    # Create trace
    trace = ReasoningTrace(
        trace_id="trace-001",
        task="Test task",
        start_time=datetime.now()
    )
    trace.add_step(ReasoningStep(
        step_number=1,
        step_type="analysis",
        content="Test step",
        timestamp=datetime.now()
    ))
    trace.end_time = datetime.now()
    
    # Add only trace, no metrics or context analytics
    dashboard.add_reasoning_trace(trace)
    
    # Create dashboard (should handle missing components)
    fig = dashboard.create_dashboard()
    assert isinstance(fig, go.Figure)


def test_failure_case_invalid_config():
    """Test behavior with invalid configuration values."""
    # This should not raise errors but handle defaults gracefully
    config = DashboardConfig(
        title="Test Dashboard",
        chart_height=-100,  # Invalid height
        chart_width=-200    # Invalid width
    )
    
    dashboard = EvaluationDashboard(config=config)
    fig = dashboard.create_dashboard()
    
    assert isinstance(fig, go.Figure)
    # Heights should be adjusted to valid values
    assert fig.layout.height != -100
    assert fig.layout.width != -200


if __name__ == "__main__":
    # Run the tests manually if needed
    pytest.main(["-xvs", __file__]) 