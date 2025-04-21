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
Unit tests for QualityMetricsVisualizer component.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from unittest.mock import patch, MagicMock

from beeai_framework.visualization.components.quality_metrics_visualizer import (
    QualityMetrics,
    QualityMetricsVisualizer
)


@pytest.fixture
def sample_metrics():
    """Create a sample QualityMetrics object for testing."""
    return QualityMetrics(
        task_id="task_123",
        task_type="text_generation",
        metrics={
            "coherence": 0.85,
            "relevance": 0.92,
            "factual_accuracy": 0.78,
            "completeness": 0.88,
            "readability": 0.95
        },
        timestamp=datetime.now(),
        metadata={
            "model": "gpt-4",
            "difficulty": "medium",
            "prompt_tokens": 250,
            "completion_tokens": 750
        }
    )


@pytest.fixture
def multiple_metrics():
    """Create a list of multiple QualityMetrics objects for testing."""
    metrics_list = []
    base_time = datetime.now()
    
    models = ["gpt-3.5", "gpt-4", "llama-2", "claude-2"]
    task_types = ["text_generation", "summarization", "question_answering"]
    
    for i in range(20):
        model = models[i % len(models)]
        task_type = task_types[i % len(task_types)]
        
        # Create metrics with slight variations
        metrics = {
            "coherence": 0.7 + np.random.random() * 0.3,
            "relevance": 0.75 + np.random.random() * 0.25,
            "factual_accuracy": 0.6 + np.random.random() * 0.4,
            "completeness": 0.65 + np.random.random() * 0.35,
            "readability": 0.8 + np.random.random() * 0.2
        }
        
        if task_type == "summarization":
            metrics["conciseness"] = 0.7 + np.random.random() * 0.3
        
        if task_type == "question_answering":
            metrics["answer_precision"] = 0.75 + np.random.random() * 0.25
        
        metrics_list.append(
            QualityMetrics(
                task_id=f"task_{i}",
                task_type=task_type,
                metrics=metrics,
                timestamp=base_time,
                metadata={
                    "model": model,
                    "difficulty": ["easy", "medium", "hard"][i % 3],
                    "prompt_tokens": 100 + i * 20,
                    "completion_tokens": 500 + i * 50
                }
            )
        )
    
    return metrics_list


@pytest.fixture
def visualizer():
    """Create a QualityMetricsVisualizer instance."""
    return QualityMetricsVisualizer()


def test_quality_metrics_initialization():
    """Test initialization of QualityMetrics."""
    now = datetime.now()
    metrics = QualityMetrics(
        task_id="test_task",
        task_type="summarization",
        metrics={"accuracy": 0.9, "fluency": 0.8},
        timestamp=now,
        metadata={"model": "test_model"}
    )
    
    assert metrics.task_id == "test_task"
    assert metrics.task_type == "summarization"
    assert metrics.metrics == {"accuracy": 0.9, "fluency": 0.8}
    assert metrics.timestamp == now
    assert metrics.metadata == {"model": "test_model"}


def test_visualizer_initialization(visualizer):
    """Test initialization of QualityMetricsVisualizer."""
    assert isinstance(visualizer, QualityMetricsVisualizer)
    assert hasattr(visualizer, "create_radar_chart")
    assert hasattr(visualizer, "create_bar_chart")
    assert hasattr(visualizer, "create_trend_line")
    assert hasattr(visualizer, "create_comparison_chart")
    assert hasattr(visualizer, "create_heatmap")
    assert hasattr(visualizer, "create_distribution_chart")


def test_create_radar_chart(visualizer, sample_metrics):
    """Test radar chart creation."""
    fig = visualizer.create_radar_chart(sample_metrics)
    
    assert isinstance(fig, go.Figure)
    # Check that the radar chart has data
    assert len(fig.data) > 0
    # The radar chart should have one trace for the metrics
    assert len(fig.data) == 1
    # The trace should be of type 'scatterpolar'
    assert fig.data[0].type == "scatterpolar"


def test_create_bar_chart(visualizer, sample_metrics):
    """Test bar chart creation."""
    fig = visualizer.create_bar_chart(sample_metrics)
    
    assert isinstance(fig, go.Figure)
    # Check that the bar chart has data
    assert len(fig.data) > 0
    # The bar chart should have one trace for the metrics
    assert len(fig.data) == 1
    # The trace should be of type 'bar'
    assert fig.data[0].type == "bar"


def test_create_trend_line(visualizer, multiple_metrics):
    """Test trend line chart creation."""
    # Create a trend line for a specific metric across multiple data points
    fig = visualizer.create_trend_line(multiple_metrics, "coherence")
    
    assert isinstance(fig, go.Figure)
    # Should have at least one trace
    assert len(fig.data) > 0
    # The trace should be of type 'scatter'
    assert fig.data[0].type == "scatter"


def test_create_comparison_chart(visualizer, multiple_metrics):
    """Test comparison chart creation."""
    # Compare metrics across different models
    fig = visualizer.create_comparison_chart(
        multiple_metrics,
        group_by="metadata.model",
        metrics=["coherence", "relevance"]
    )
    
    assert isinstance(fig, go.Figure)
    # Should have data for each model
    assert len(fig.data) > 0


def test_create_heatmap(visualizer, multiple_metrics):
    """Test heatmap creation."""
    fig = visualizer.create_heatmap(
        multiple_metrics,
        x_axis="metadata.model",
        y_axis="task_type"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have a heatmap trace
    assert len(fig.data) > 0
    assert fig.data[0].type == "heatmap"


def test_create_distribution_chart(visualizer, multiple_metrics):
    """Test distribution chart creation."""
    fig = visualizer.create_distribution_chart(
        multiple_metrics,
        metric="coherence"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have at least one trace
    assert len(fig.data) > 0


def test_metrics_to_dataframe(visualizer, multiple_metrics):
    """Test conversion of metrics to DataFrame."""
    df = visualizer._metrics_to_dataframe(multiple_metrics)
    
    assert isinstance(df, pd.DataFrame)
    # DataFrame should have rows for each metrics object
    assert len(df) == len(multiple_metrics)
    # DataFrame should have columns for metrics and metadata
    assert "coherence" in df.columns
    assert "model" in df.columns


def test_get_metric_values(visualizer, multiple_metrics):
    """Test extracting metric values."""
    values = visualizer._get_metric_values(multiple_metrics, "coherence")
    
    assert isinstance(values, list)
    assert len(values) == len(multiple_metrics)
    # All values should be numeric
    assert all(isinstance(v, (int, float)) for v in values)


def test_validate_metrics():
    """Test validation of quality metrics."""
    # Valid metrics
    valid_metrics = QualityMetrics(
        task_id="valid_task",
        task_type="summarization",
        metrics={"accuracy": 0.9},
        timestamp=datetime.now()
    )
    
    # Should not raise any errors
    valid_metrics.validate()
    
    # Invalid metrics (missing required fields)
    invalid_metrics = QualityMetrics(
        task_id="",
        task_type="",
        metrics={},
        timestamp=None
    )
    
    # Should raise ValueError
    with pytest.raises(ValueError):
        invalid_metrics.validate()


def test_validate_metric_values():
    """Test validation of metric values."""
    # Valid metrics (values between 0 and 1)
    valid_metrics = QualityMetrics(
        task_id="valid_task",
        task_type="summarization",
        metrics={"accuracy": 0.9, "fluency": 0.8},
        timestamp=datetime.now()
    )
    
    # Should not raise any errors
    valid_metrics.validate()
    
    # Invalid metrics (values outside 0-1 range)
    invalid_metrics = QualityMetrics(
        task_id="invalid_task",
        task_type="summarization",
        metrics={"accuracy": 1.2, "fluency": -0.1},
        timestamp=datetime.now()
    )
    
    # Should raise ValueError
    with pytest.raises(ValueError):
        invalid_metrics.validate()


def test_visualizer_with_empty_metrics(visualizer):
    """Test visualizer with empty metrics."""
    empty_metrics = QualityMetrics(
        task_id="empty_task",
        task_type="text_generation",
        metrics={},
        timestamp=datetime.now()
    )
    
    # Should handle empty metrics gracefully
    radar_fig = visualizer.create_radar_chart(empty_metrics)
    assert isinstance(radar_fig, go.Figure)
    
    bar_fig = visualizer.create_bar_chart(empty_metrics)
    assert isinstance(bar_fig, go.Figure)


def test_export_visualizations(visualizer, sample_metrics, tmp_path):
    """Test exporting visualizations to files."""
    # Create a temporary directory
    output_dir = tmp_path / "test_exports"
    output_dir.mkdir()
    
    # Mock the save methods to avoid actual file creation
    with patch.object(go.Figure, 'write_html') as mock_write_html, \
         patch.object(go.Figure, 'write_image') as mock_write_image:
        
        # Export visualizations
        visualizer.export_visualizations(
            sample_metrics,
            str(output_dir),
            formats=["html", "png"]
        )
        
        # Check that write methods were called
        assert mock_write_html.call_count > 0
        assert mock_write_image.call_count > 0
        
        # Test with invalid format
        with pytest.raises(ValueError):
            visualizer.export_visualizations(
                sample_metrics,
                str(output_dir),
                formats=["invalid_format"]
            )


def test_create_aggregated_dashboard(visualizer, multiple_metrics):
    """Test creating an aggregated dashboard."""
    with patch('plotly.subplots.make_subplots') as mock_make_subplots:
        # Mock the return value of make_subplots
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig
        
        # Create dashboard
        dashboard = visualizer.create_aggregated_dashboard(multiple_metrics)
        
        # Verify subplot creation was called
        mock_make_subplots.assert_called()
        
        # Verify add_trace was called multiple times for different charts
        assert mock_fig.add_trace.call_count > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 