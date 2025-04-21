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
Unit tests for DimensionReductionVisualizer component.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

from beeai_framework.visualization.components.dimension_reduction_visualizer import (
    DimensionReductionVisualizer,
    DimensionReductionMethod
)


@pytest.fixture
def sample_data():
    """Create sample high-dimensional data for testing."""
    # Generate synthetic data with 50 samples and 20 dimensions
    np.random.seed(42)
    X = np.random.randn(50, 20)
    
    # Create labels for the data points (3 classes)
    labels = np.array([i // 17 for i in range(50)])
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(20)]
    
    return X, labels, feature_names


@pytest.fixture
def visualizer():
    """Create a DimensionReductionVisualizer instance."""
    return DimensionReductionVisualizer()


def test_visualizer_initialization(visualizer):
    """Test initialization of DimensionReductionVisualizer."""
    assert isinstance(visualizer, DimensionReductionVisualizer)
    assert hasattr(visualizer, "reduce_dimensions")
    assert hasattr(visualizer, "visualize_2d")
    assert hasattr(visualizer, "visualize_3d")
    assert hasattr(visualizer, "visualize_feature_importance")


def test_enum_dimension_reduction_method():
    """Test the DimensionReductionMethod enum."""
    assert DimensionReductionMethod.PCA.value == "pca"
    assert DimensionReductionMethod.TSNE.value == "tsne"
    assert DimensionReductionMethod.MDS.value == "mds"
    
    # Test that the enum can be converted to string
    assert str(DimensionReductionMethod.PCA) == "pca"


def test_reduce_dimensions_pca(visualizer, sample_data):
    """Test dimension reduction using PCA."""
    X, labels, feature_names = sample_data
    
    # Reduce to 2 dimensions
    reduced_data, model = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.PCA,
        n_components=2
    )
    
    # Check output dimensions
    assert reduced_data.shape == (X.shape[0], 2)
    # Check returned model
    assert isinstance(model, PCA)
    

def test_reduce_dimensions_tsne(visualizer, sample_data):
    """Test dimension reduction using t-SNE."""
    X, labels, feature_names = sample_data
    
    # Reduce to 2 dimensions
    reduced_data, model = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.TSNE,
        n_components=2
    )
    
    # Check output dimensions
    assert reduced_data.shape == (X.shape[0], 2)
    # Check returned model
    assert isinstance(model, TSNE)


def test_reduce_dimensions_mds(visualizer, sample_data):
    """Test dimension reduction using MDS."""
    X, labels, feature_names = sample_data
    
    # Reduce to 2 dimensions
    reduced_data, model = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.MDS,
        n_components=2
    )
    
    # Check output dimensions
    assert reduced_data.shape == (X.shape[0], 2)
    # Check returned model
    assert isinstance(model, MDS)


def test_visualize_2d(visualizer, sample_data):
    """Test 2D visualization of reduced data."""
    X, labels, feature_names = sample_data
    
    # First reduce dimensions
    reduced_data, _ = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.PCA,
        n_components=2
    )
    
    # Visualize the reduced data
    fig = visualizer.visualize_2d(reduced_data, labels=labels)
    
    assert isinstance(fig, go.Figure)
    # Should have at least one trace
    assert len(fig.data) > 0
    # The trace should be of type 'scatter'
    assert fig.data[0].type == "scatter"


def test_visualize_3d(visualizer, sample_data):
    """Test 3D visualization of reduced data."""
    X, labels, feature_names = sample_data
    
    # First reduce dimensions
    reduced_data, _ = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.PCA,
        n_components=3
    )
    
    # Visualize the reduced data in 3D
    fig = visualizer.visualize_3d(reduced_data, labels=labels)
    
    assert isinstance(fig, go.Figure)
    # Should have at least one trace
    assert len(fig.data) > 0
    # The trace should be of type 'scatter3d'
    assert fig.data[0].type == "scatter3d"


def test_visualize_feature_importance(visualizer, sample_data):
    """Test visualization of feature importance from PCA."""
    X, labels, feature_names = sample_data
    
    # First reduce dimensions with PCA
    _, pca_model = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.PCA,
        n_components=2
    )
    
    # Visualize feature importance
    fig = visualizer.visualize_feature_importance(pca_model, feature_names=feature_names)
    
    assert isinstance(fig, go.Figure)
    # Should have at least one trace
    assert len(fig.data) > 0
    # The trace should be of type 'bar'
    assert fig.data[0].type == "bar"


def test_reduce_dimensions_invalid_method(visualizer, sample_data):
    """Test reduce_dimensions with invalid method."""
    X, _, _ = sample_data
    
    # Try to reduce dimensions with an invalid method
    with pytest.raises(ValueError):
        visualizer.reduce_dimensions(X, method="invalid_method")


def test_reduce_dimensions_invalid_components(visualizer, sample_data):
    """Test reduce_dimensions with invalid number of components."""
    X, _, _ = sample_data
    
    # Try to reduce dimensions with too many components
    with pytest.raises(ValueError):
        visualizer.reduce_dimensions(
            X, 
            method=DimensionReductionMethod.PCA,
            n_components=X.shape[1] + 1
        )


def test_visualize_2d_without_labels(visualizer, sample_data):
    """Test 2D visualization without labels."""
    X, _, _ = sample_data
    
    # First reduce dimensions
    reduced_data, _ = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.PCA,
        n_components=2
    )
    
    # Visualize without labels
    fig = visualizer.visualize_2d(reduced_data)
    
    assert isinstance(fig, go.Figure)
    # Should have one trace (all points in a single color)
    assert len(fig.data) == 1


def test_visualize_3d_without_labels(visualizer, sample_data):
    """Test 3D visualization without labels."""
    X, _, _ = sample_data
    
    # First reduce dimensions
    reduced_data, _ = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.PCA,
        n_components=3
    )
    
    # Visualize without labels
    fig = visualizer.visualize_3d(reduced_data)
    
    assert isinstance(fig, go.Figure)
    # Should have one trace (all points in a single color)
    assert len(fig.data) == 1


def test_convert_to_dataframe(visualizer, sample_data):
    """Test conversion of reduced data to DataFrame."""
    X, labels, _ = sample_data
    
    # First reduce dimensions
    reduced_data, _ = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.PCA,
        n_components=2
    )
    
    # Convert to DataFrame
    df = visualizer._convert_to_dataframe(reduced_data, labels=labels)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == X.shape[0]
    assert "Component 1" in df.columns
    assert "Component 2" in df.columns
    assert "Label" in df.columns


def test_add_annotations(visualizer, sample_data):
    """Test adding annotations to plots."""
    X, labels, _ = sample_data
    
    # First reduce dimensions
    reduced_data, _ = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.PCA,
        n_components=2
    )
    
    # Create a basic plot
    fig = visualizer.visualize_2d(reduced_data, labels=labels)
    
    # Add annotations for specific points
    indices = [0, 10, 20]
    annotations = [f"Point {i}" for i in indices]
    
    # Get the number of annotations before adding
    initial_annotations = len(fig.layout.annotations) if fig.layout.annotations else 0
    
    # Add annotations
    fig = visualizer._add_annotations(fig, reduced_data, indices, annotations)
    
    # Check that annotations were added
    assert len(fig.layout.annotations) == initial_annotations + len(indices)


def test_export_visualization(visualizer, sample_data, tmp_path):
    """Test exporting visualizations to files."""
    X, labels, _ = sample_data
    
    # First reduce dimensions
    reduced_data, _ = visualizer.reduce_dimensions(
        X, 
        method=DimensionReductionMethod.PCA,
        n_components=2
    )
    
    # Create visualization
    fig = visualizer.visualize_2d(reduced_data, labels=labels)
    
    # Create a temporary file path
    output_file = tmp_path / "pca_visualization.html"
    
    # Mock the write_html method to avoid actual file creation
    with patch.object(go.Figure, 'write_html') as mock_write_html:
        # Export visualization
        visualizer.export_visualization(fig, str(output_file))
        
        # Check that write_html was called with the correct file path
        mock_write_html.assert_called_once_with(str(output_file))


def test_compare_methods(visualizer, sample_data):
    """Test comparing different dimension reduction methods."""
    X, labels, _ = sample_data
    
    # Create a mock for subplots
    with patch('plotly.subplots.make_subplots') as mock_make_subplots:
        # Mock the return value of make_subplots
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig
        
        # Compare methods
        methods = [
            DimensionReductionMethod.PCA,
            DimensionReductionMethod.TSNE,
            DimensionReductionMethod.MDS
        ]
        
        fig = visualizer.compare_methods(X, methods=methods, labels=labels)
        
        # Verify subplot creation was called
        mock_make_subplots.assert_called_once()
        
        # Verify add_trace was called multiple times (once for each method and label combination)
        assert mock_fig.add_trace.call_count > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 