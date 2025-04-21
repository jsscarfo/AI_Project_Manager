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
Unit tests for MetricsVisualizer component.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock

from beeai_framework.visualization.components.metrics_visualizer import MetricsVisualizer


@pytest.fixture
def sample_classification_data():
    """Create sample classification data for testing."""
    np.random.seed(42)
    
    # True labels (3 classes, 100 samples)
    y_true = np.random.randint(0, 3, size=100)
    
    # Predicted labels with some errors
    y_pred = np.copy(y_true)
    error_indices = np.random.choice(100, size=20, replace=False)
    for idx in error_indices:
        y_pred[idx] = (y_true[idx] + np.random.randint(1, 3)) % 3
    
    # Prediction probabilities
    y_proba = np.zeros((100, 3))
    for i in range(100):
        # Base probabilities
        probs = np.random.dirichlet(np.ones(3) * 5)
        # Make the predicted class have higher probability
        max_idx = y_pred[i]
        probs = probs * 0.3
        probs[max_idx] += 0.7
        y_proba[i] = probs
    
    # Class names
    class_names = ["Class A", "Class B", "Class C"]
    
    return y_true, y_pred, y_proba, class_names


@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    np.random.seed(42)
    
    # Generate 100 data points
    X = np.linspace(0, 10, 100)
    
    # True values (follow a sine curve with some noise)
    y_true = np.sin(X) + 0.5 * X
    
    # Predicted values (with some error)
    y_pred = y_true + np.random.normal(0, 0.5, size=100)
    
    return X, y_true, y_pred


@pytest.fixture
def visualizer():
    """Create a MetricsVisualizer instance."""
    return MetricsVisualizer()


def test_visualizer_initialization(visualizer):
    """Test initialization of MetricsVisualizer."""
    assert isinstance(visualizer, MetricsVisualizer)
    assert hasattr(visualizer, "plot_confusion_matrix")
    assert hasattr(visualizer, "plot_roc_curve")
    assert hasattr(visualizer, "plot_precision_recall_curve")


def test_plot_confusion_matrix(visualizer, sample_classification_data):
    """Test plotting confusion matrix."""
    y_true, y_pred, _, class_names = sample_classification_data
    
    # Plot confusion matrix
    fig = visualizer.plot_confusion_matrix(
        y_true, 
        y_pred, 
        class_names=class_names,
        title="Test Confusion Matrix"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have a heatmap
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    # Check dimensions (3x3 for 3 classes)
    assert fig.data[0].z.shape == (3, 3)
    # Check title
    assert fig.layout.title.text == "Test Confusion Matrix"


def test_plot_confusion_matrix_normalized(visualizer, sample_classification_data):
    """Test plotting normalized confusion matrix."""
    y_true, y_pred, _, class_names = sample_classification_data
    
    # Plot normalized confusion matrix
    fig = visualizer.plot_confusion_matrix(
        y_true, 
        y_pred, 
        normalize=True,
        class_names=class_names
    )
    
    assert isinstance(fig, go.Figure)
    # Values should be between 0 and 1
    assert np.all(fig.data[0].z >= 0)
    assert np.all(fig.data[0].z <= 1)
    # Each row should sum to approximately 1
    for row in fig.data[0].z:
        assert np.isclose(np.sum(row), 1.0, atol=1e-6)


def test_plot_roc_curve_binary(visualizer, sample_classification_data):
    """Test plotting ROC curve for binary classification."""
    y_true, _, y_proba, _ = sample_classification_data
    
    # Convert to binary problem (class 0 vs rest)
    y_true_binary = (y_true == 0).astype(int)
    y_proba_binary = y_proba[:, 0]
    
    # Plot ROC curve
    fig = visualizer.plot_roc_curve(
        y_true_binary, 
        y_proba_binary,
        title="Binary ROC Curve"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have at least 2 traces (ROC curve and diagonal reference line)
    assert len(fig.data) >= 2
    # First trace should be 'scatter' type
    assert fig.data[0].type == "scatter"
    # Check title
    assert fig.layout.title.text == "Binary ROC Curve"


def test_plot_roc_curve_multiclass(visualizer, sample_classification_data):
    """Test plotting ROC curve for multiclass classification."""
    y_true, _, y_proba, class_names = sample_classification_data
    
    # Plot multiclass ROC curves
    fig = visualizer.plot_roc_curve(
        y_true, 
        y_proba,
        class_names=class_names,
        average="macro"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have multiple traces (one per class + average + diagonal)
    assert len(fig.data) >= len(class_names) + 2
    # Check class names in legend
    for class_name in class_names:
        found = False
        for trace in fig.data:
            if class_name in trace.name:
                found = True
                break
        assert found, f"Class {class_name} not found in ROC curve legend"


def test_plot_precision_recall_curve_binary(visualizer, sample_classification_data):
    """Test plotting precision-recall curve for binary classification."""
    y_true, _, y_proba, _ = sample_classification_data
    
    # Convert to binary problem (class 0 vs rest)
    y_true_binary = (y_true == 0).astype(int)
    y_proba_binary = y_proba[:, 0]
    
    # Plot precision-recall curve
    fig = visualizer.plot_precision_recall_curve(
        y_true_binary, 
        y_proba_binary,
        title="Binary Precision-Recall Curve"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have traces for the curve
    assert len(fig.data) > 0
    # First trace should be 'scatter' type
    assert fig.data[0].type == "scatter"
    # Check title
    assert fig.layout.title.text == "Binary Precision-Recall Curve"


def test_plot_precision_recall_curve_multiclass(visualizer, sample_classification_data):
    """Test plotting precision-recall curve for multiclass classification."""
    y_true, _, y_proba, class_names = sample_classification_data
    
    # Plot multiclass precision-recall curves
    fig = visualizer.plot_precision_recall_curve(
        y_true, 
        y_proba,
        class_names=class_names,
        average="macro"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have multiple traces (one per class + average)
    assert len(fig.data) >= len(class_names) + 1


def test_plot_residuals(visualizer, sample_regression_data):
    """Test plotting residuals for regression models."""
    _, y_true, y_pred = sample_regression_data
    
    # Plot residuals
    fig = visualizer.plot_residuals(
        y_true, 
        y_pred,
        title="Residuals Plot"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have scatter points
    assert len(fig.data) > 0
    assert fig.data[0].type == "scatter"
    # Check title
    assert fig.layout.title.text == "Residuals Plot"


def test_plot_prediction_error(visualizer, sample_regression_data):
    """Test plotting prediction error for regression models."""
    _, y_true, y_pred = sample_regression_data
    
    # Plot prediction error
    fig = visualizer.plot_prediction_error(
        y_true, 
        y_pred,
        title="Prediction Error"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have scatter points and a diagonal line
    assert len(fig.data) >= 2
    assert fig.data[0].type == "scatter"
    # Check title
    assert fig.layout.title.text == "Prediction Error"


def test_plot_feature_importance(visualizer):
    """Test plotting feature importance."""
    # Create some feature importance data
    feature_names = ["Feature A", "Feature B", "Feature C", "Feature D"]
    importance_values = [0.35, 0.25, 0.2, 0.2]
    
    # Plot feature importance
    fig = visualizer.plot_feature_importance(
        importance_values,
        feature_names=feature_names,
        title="Feature Importance"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have a bar chart
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    # Check number of bars
    assert len(fig.data[0].x) == len(feature_names)
    assert len(fig.data[0].y) == len(importance_values)
    # Check title
    assert fig.layout.title.text == "Feature Importance"


def test_plot_lift_curve(visualizer, sample_classification_data):
    """Test plotting lift curve."""
    y_true, _, y_proba, _ = sample_classification_data
    
    # Convert to binary problem (class 0 vs rest)
    y_true_binary = (y_true == 0).astype(int)
    y_proba_binary = y_proba[:, 0]
    
    # Plot lift curve
    fig = visualizer.plot_lift_curve(
        y_true_binary, 
        y_proba_binary,
        title="Lift Curve"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have traces for the curve and baseline
    assert len(fig.data) >= 2
    # First trace should be 'scatter' type
    assert fig.data[0].type == "scatter"
    # Check title
    assert fig.layout.title.text == "Lift Curve"


def test_plot_cumulative_gain(visualizer, sample_classification_data):
    """Test plotting cumulative gain curve."""
    y_true, _, y_proba, _ = sample_classification_data
    
    # Convert to binary problem (class 0 vs rest)
    y_true_binary = (y_true == 0).astype(int)
    y_proba_binary = y_proba[:, 0]
    
    # Plot cumulative gain curve
    fig = visualizer.plot_cumulative_gain(
        y_true_binary, 
        y_proba_binary,
        title="Cumulative Gain"
    )
    
    assert isinstance(fig, go.Figure)
    # Should have traces for the curve and baseline
    assert len(fig.data) >= 2
    # First trace should be 'scatter' type
    assert fig.data[0].type == "scatter"
    # Check title
    assert fig.layout.title.text == "Cumulative Gain"


def test_generate_classification_report(visualizer, sample_classification_data):
    """Test generating comprehensive classification report."""
    y_true, y_pred, y_proba, class_names = sample_classification_data
    
    # Create a mock for subplots
    with patch('plotly.subplots.make_subplots') as mock_make_subplots:
        # Mock the return value of make_subplots
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig
        
        # Generate classification report
        fig, metrics = visualizer.generate_classification_report(
            y_true, 
            y_pred, 
            y_proba,
            class_names=class_names
        )
        
        # Verify subplot creation was called
        mock_make_subplots.assert_called_once()
        
        # Verify add_trace was called multiple times
        assert mock_fig.add_trace.call_count > 0
        
        # Verify metrics dictionary
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics


def test_generate_regression_report(visualizer, sample_regression_data):
    """Test generating comprehensive regression report."""
    _, y_true, y_pred = sample_regression_data
    
    # Create a mock for subplots
    with patch('plotly.subplots.make_subplots') as mock_make_subplots:
        # Mock the return value of make_subplots
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig
        
        # Generate regression report
        fig, metrics = visualizer.generate_regression_report(
            y_true, 
            y_pred
        )
        
        # Verify subplot creation was called
        mock_make_subplots.assert_called_once()
        
        # Verify add_trace was called multiple times
        assert mock_fig.add_trace.call_count > 0
        
        # Verify metrics dictionary
        assert isinstance(metrics, dict)
        assert "mae" in metrics  # Mean Absolute Error
        assert "mse" in metrics  # Mean Squared Error
        assert "rmse" in metrics  # Root Mean Squared Error
        assert "r2" in metrics  # R-squared


def test_export_metrics_plot(visualizer, sample_classification_data, tmp_path):
    """Test exporting metrics plots to files."""
    y_true, y_pred, _, class_names = sample_classification_data
    
    # Create a confusion matrix
    fig = visualizer.plot_confusion_matrix(y_true, y_pred, class_names=class_names)
    
    # Create a temporary file path
    output_file = tmp_path / "confusion_matrix.html"
    
    # Mock the write_html method to avoid actual file creation
    with patch.object(go.Figure, 'write_html') as mock_write_html:
        # Export visualization
        visualizer.export_plot(fig, str(output_file))
        
        # Check that write_html was called with the correct file path
        mock_write_html.assert_called_once_with(str(output_file))


def test_invalid_input_shapes(visualizer):
    """Test handling of invalid input shapes."""
    # Create mismatched shapes
    y_true = np.random.randint(0, 2, size=10)
    y_pred = np.random.randint(0, 2, size=15)
    
    # Should raise ValueError due to shape mismatch
    with pytest.raises(ValueError):
        visualizer.plot_confusion_matrix(y_true, y_pred)


def test_invalid_probability_format(visualizer):
    """Test handling of invalid probability format."""
    # Create binary classification data
    y_true = np.random.randint(0, 2, size=10)
    
    # Invalid probabilities (not between 0 and 1)
    y_proba = np.random.rand(10) * 2
    
    # Should raise ValueError due to invalid probabilities
    with pytest.raises(ValueError):
        visualizer.plot_roc_curve(y_true, y_proba)


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 