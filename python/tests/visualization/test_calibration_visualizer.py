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
Unit tests for CalibrationVisualizer component.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock

from beeai_framework.visualization.components.calibration_visualizer import (
    CalibrationVisualizer,
    CalibrationMethod
)


@pytest.fixture
def sample_data():
    """Create sample probability and actual data for testing."""
    np.random.seed(42)
    
    # Generate synthetic probabilities (50 samples, 3 classes)
    y_prob = np.random.rand(50, 3)
    # Normalize to make them proper probabilities
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    
    # Generate true labels (one-hot encoding)
    y_true = np.zeros_like(y_prob)
    y_true[np.arange(50), np.argmax(y_prob, axis=1)] = 1
    
    # Add some noise to make it more realistic
    noise_indices = np.random.choice(50, size=15, replace=False)
    for idx in noise_indices:
        true_class = np.argmax(y_true[idx])
        new_class = (true_class + 1) % 3
        y_true[idx] = np.zeros(3)
        y_true[idx, new_class] = 1
    
    # Class names
    class_names = ["Class A", "Class B", "Class C"]
    
    return y_prob, y_true, class_names


@pytest.fixture
def binary_data():
    """Create sample binary classification data for testing."""
    np.random.seed(42)
    
    # Generate binary probabilities (100 samples)
    y_prob = np.random.rand(100)
    
    # Generate binary true labels
    # More likely to be 1 when probability is high, but with some noise
    y_true = (y_prob > 0.5).astype(int)
    
    # Add some noise
    noise_indices = np.random.choice(100, size=20, replace=False)
    for idx in noise_indices:
        y_true[idx] = 1 - y_true[idx]
    
    return y_prob, y_true


@pytest.fixture
def visualizer():
    """Create a CalibrationVisualizer instance."""
    return CalibrationVisualizer()


def test_visualizer_initialization(visualizer):
    """Test initialization of CalibrationVisualizer."""
    assert isinstance(visualizer, CalibrationVisualizer)
    assert hasattr(visualizer, "plot_reliability_curve")
    assert hasattr(visualizer, "plot_calibration_curve")
    assert hasattr(visualizer, "plot_confidence_histogram")


def test_enum_calibration_method():
    """Test the CalibrationMethod enum."""
    assert CalibrationMethod.ISOTONIC.value == "isotonic"
    assert CalibrationMethod.SIGMOID.value == "sigmoid"
    
    # Test that the enum can be converted to string
    assert str(CalibrationMethod.ISOTONIC) == "isotonic"


def test_plot_reliability_curve_binary(visualizer, binary_data):
    """Test plotting reliability curve for binary classification."""
    y_prob, y_true = binary_data
    
    # Plot the reliability curve
    fig = visualizer.plot_reliability_curve(y_prob, y_true)
    
    assert isinstance(fig, go.Figure)
    # Should have at least 2 traces (perfect calibration line and actual calibration)
    assert len(fig.data) >= 2
    # First trace should be 'scatter' type
    assert fig.data[0].type == "scatter"


def test_plot_reliability_curve_multiclass(visualizer, sample_data):
    """Test plotting reliability curve for multiclass classification."""
    y_prob, y_true, class_names = sample_data
    
    # Plot the reliability curve
    fig = visualizer.plot_reliability_curve(
        y_prob, 
        y_true, 
        class_names=class_names,
        title="Multiclass Reliability Curve"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have multiple traces (at least one per class plus perfect calibration line)
    assert len(fig.data) >= len(class_names) + 1
    # Check title
    assert fig.layout.title.text == "Multiclass Reliability Curve"


def test_plot_calibration_curve_binary(visualizer, binary_data):
    """Test plotting calibration curve for binary classification with different methods."""
    y_prob, y_true = binary_data
    
    # Plot calibration curve with both methods
    fig = visualizer.plot_calibration_curve(
        y_prob, 
        y_true, 
        methods=[CalibrationMethod.ISOTONIC, CalibrationMethod.SIGMOID]
    )
    
    assert isinstance(fig, go.Figure)
    # Should have at least 3 traces (perfect calibration, isotonic, sigmoid)
    assert len(fig.data) >= 3
    
    # Check method names in the legend
    method_names_in_legend = [trace.name for trace in fig.data if trace.name not in ["Perfectly calibrated"]]
    assert "Isotonic" in method_names_in_legend
    assert "Sigmoid" in method_names_in_legend


def test_plot_calibration_curve_multiclass(visualizer, sample_data):
    """Test plotting calibration curve for multiclass classification."""
    y_prob, y_true, class_names = sample_data
    
    # Plot calibration curve
    fig = visualizer.plot_calibration_curve(
        y_prob, 
        y_true, 
        class_names=class_names,
        methods=[CalibrationMethod.ISOTONIC]
    )
    
    assert isinstance(fig, go.Figure)
    # Should have multiple traces
    assert len(fig.data) > 1


def test_plot_confidence_histogram_binary(visualizer, binary_data):
    """Test plotting confidence histogram for binary classification."""
    y_prob, y_true = binary_data
    
    # Plot confidence histogram
    fig = visualizer.plot_confidence_histogram(
        y_prob, 
        y_true, 
        bins=10,
        title="Binary Confidence Histogram"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have histograms
    assert len(fig.data) > 0
    # First trace should be 'bar' type
    assert fig.data[0].type == "bar"
    # Check title
    assert fig.layout.title.text == "Binary Confidence Histogram"


def test_plot_confidence_histogram_multiclass(visualizer, sample_data):
    """Test plotting confidence histogram for multiclass classification."""
    y_prob, y_true, class_names = sample_data
    
    # Plot confidence histogram
    fig = visualizer.plot_confidence_histogram(
        y_prob, 
        y_true, 
        class_names=class_names
    )
    
    assert isinstance(fig, go.Figure)
    # Should have multiple histograms
    assert len(fig.data) > 0


def test_calculate_calibration_binary(visualizer, binary_data):
    """Test calculating calibration metrics for binary classification."""
    y_prob, y_true = binary_data
    
    # Calculate calibration statistics
    results = visualizer.calculate_calibration_metrics(y_prob, y_true)
    
    assert "brier_score" in results
    assert isinstance(results["brier_score"], float)
    assert "ece" in results  # Expected Calibration Error
    assert isinstance(results["ece"], float)
    assert "mce" in results  # Maximum Calibration Error
    assert isinstance(results["mce"], float)


def test_calculate_calibration_multiclass(visualizer, sample_data):
    """Test calculating calibration metrics for multiclass classification."""
    y_prob, y_true, _ = sample_data
    
    # Calculate calibration statistics
    results = visualizer.calculate_calibration_metrics(y_prob, y_true)
    
    assert "brier_score" in results
    assert isinstance(results["brier_score"], float)
    assert "ece" in results
    assert isinstance(results["ece"], float)
    assert "mce" in results
    assert isinstance(results["mce"], float)


def test_generate_calibration_report(visualizer, binary_data):
    """Test generating a comprehensive calibration report."""
    y_prob, y_true = binary_data
    
    # Create a mock for subplots
    with patch('plotly.subplots.make_subplots') as mock_make_subplots:
        # Mock the return value of make_subplots
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig
        
        # Generate calibration report
        fig, metrics = visualizer.generate_calibration_report(
            y_prob, 
            y_true, 
            methods=[CalibrationMethod.ISOTONIC, CalibrationMethod.SIGMOID]
        )
        
        # Verify subplot creation was called
        mock_make_subplots.assert_called_once()
        
        # Verify add_trace was called multiple times
        assert mock_fig.add_trace.call_count > 0
        
        # Verify metrics
        assert isinstance(metrics, dict)
        assert "brier_score" in metrics
        assert "ece" in metrics
        assert "mce" in metrics


def test_export_calibration_plots(visualizer, binary_data, tmp_path):
    """Test exporting calibration plots to files."""
    y_prob, y_true = binary_data
    
    # Create a reliability curve
    fig = visualizer.plot_reliability_curve(y_prob, y_true)
    
    # Create a temporary file path
    output_file = tmp_path / "reliability_curve.html"
    
    # Mock the write_html method to avoid actual file creation
    with patch.object(go.Figure, 'write_html') as mock_write_html:
        # Export visualization
        visualizer.export_plot(fig, str(output_file))
        
        # Check that write_html was called with the correct file path
        mock_write_html.assert_called_once_with(str(output_file))


def test_invalid_input_dimensions(visualizer):
    """Test handling of invalid input dimensions."""
    # Create mismatched dimensions
    y_prob = np.random.rand(10, 3)
    y_true = np.zeros((5, 3))
    
    # Should raise ValueError due to dimension mismatch
    with pytest.raises(ValueError):
        visualizer.plot_reliability_curve(y_prob, y_true)


def test_invalid_probability_values(visualizer):
    """Test handling of invalid probability values."""
    # Create probabilities outside [0,1] range
    y_prob = np.random.rand(10) * 2  # Values up to 2
    y_true = np.zeros(10)
    
    # Should raise ValueError due to invalid probabilities
    with pytest.raises(ValueError):
        visualizer.plot_reliability_curve(y_prob, y_true)


def test_invalid_calibration_method(visualizer, binary_data):
    """Test handling of invalid calibration method."""
    y_prob, y_true = binary_data
    
    # Should raise ValueError due to invalid calibration method
    with pytest.raises(ValueError):
        visualizer.plot_calibration_curve(
            y_prob, 
            y_true, 
            methods=["invalid_method"]
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 