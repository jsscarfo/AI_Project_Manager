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
Tests for the Visualization Service.

This module contains tests for the VisualizationService class, verifying
its functionality in coordinating visualization components.
"""

import os
import tempfile
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, ANY

from beeai_framework.visualization import (
    VisualizationService, 
    ReasoningTrace,
    ReasoningStep
)
from beeai_framework.visualization.core.visualization_service import VECTOR_AVAILABLE


def create_test_trace():
    """Helper function to create a test reasoning trace."""
    trace = ReasoningTrace(
        trace_id="test-trace-001",
        task="Test task",
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now()
    )
    
    step = ReasoningStep(
        step_number=1,
        step_type="test_step",
        content="This is a test step content",
        timestamp=datetime.now() - timedelta(minutes=4),
        context_items=[
            {
                "id": "ctx-001",
                "source_type": "test",
                "title": "Test context",
                "relevance_score": 0.9,
                "content_snippet": "Test context content"
            }
        ]
    )
    
    trace.add_step(step)
    return trace


class TestVisualizationService:
    """Tests for the VisualizationService."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.service = VisualizationService(output_dir=self.output_dir)
        self.test_trace = create_test_trace()
    
    def teardown_method(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test service initialization with various parameters."""
        # Test default initialization
        service = VisualizationService()
        assert service.output_dir is None
        assert service.default_height == 600
        assert service.default_width == 800
        assert service.cache_traces is True
        
        # Test custom initialization
        custom_service = VisualizationService(
            output_dir=self.output_dir,
            default_height=700,
            default_width=900,
            cache_traces=False
        )
        assert custom_service.output_dir == self.output_dir
        assert custom_service.default_height == 700
        assert custom_service.default_width == 900
        assert custom_service.cache_traces is False
        
        # Test components initialization
        assert hasattr(custom_service, "reasoning_trace_visualizer")
        assert hasattr(custom_service, "steps_visualizer")
        assert hasattr(custom_service, "context_visualizer")
        assert hasattr(custom_service, "metrics_visualizer")
        assert hasattr(custom_service, "knowledge_graph_visualizer")
    
    @patch("beeai_framework.visualization.components.reasoning_trace_visualizer.ReasoningTraceVisualizer.create_step_visualization")
    def test_visualize_reasoning_trace(self, mock_create_step):
        """Test visualize_reasoning_trace method."""
        # Setup mock
        mock_fig = MagicMock()
        mock_create_step.return_value = mock_fig
        
        # Test with default parameters
        result = self.service.visualize_reasoning_trace(self.test_trace)
        
        # Verify calls
        mock_create_step.assert_called_with(
            trace=self.test_trace,
            selected_step=None,
            highlight_context=True,
            show_metrics=True,
            height=600,
            width=800
        )
        assert result == mock_fig
        
        # Test with custom parameters
        export_path = str(self.output_dir / "test_trace.html")
        result = self.service.visualize_reasoning_trace(
            trace=self.test_trace,
            selected_step=1,
            highlight_context=False,
            show_metrics=False,
            export_path=export_path,
            height=700,
            width=900
        )
        
        # Verify calls with custom parameters
        mock_create_step.assert_called_with(
            trace=self.test_trace,
            selected_step=1,
            highlight_context=False,
            show_metrics=False,
            height=700,
            width=900
        )
    
    @patch("beeai_framework.visualization.components.reasoning_trace_visualizer.ReasoningTraceVisualizer.create_knowledge_graph_visualization")
    def test_visualize_knowledge_graph(self, mock_create_graph):
        """Test visualize_knowledge_graph method."""
        # Setup mock
        mock_fig = MagicMock()
        mock_create_graph.return_value = mock_fig
        
        # Test with default parameters
        result = self.service.visualize_knowledge_graph(self.test_trace)
        
        # Verify calls
        mock_create_graph.assert_called_with(
            trace=self.test_trace,
            height=600,
            width=800
        )
        assert result == mock_fig
    
    @patch("beeai_framework.visualization.components.reasoning_trace_visualizer.ReasoningTraceVisualizer.create_context_relevance_visualization")
    def test_visualize_context_relevance(self, mock_create_relevance):
        """Test visualize_context_relevance method."""
        # Setup mock
        mock_fig = MagicMock()
        mock_create_relevance.return_value = mock_fig
        
        # Test with default parameters
        result = self.service.visualize_context_relevance(self.test_trace)
        
        # Verify calls
        mock_create_relevance.assert_called_with(
            trace=self.test_trace,
            selected_step=None,
            height=600,
            width=800
        )
        assert result == mock_fig
    
    @patch("beeai_framework.visualization.components.reasoning_trace_visualizer.ReasoningTraceVisualizer.create_context_evolution_timeline")
    def test_visualize_context_evolution(self, mock_create_timeline):
        """Test visualize_context_evolution method."""
        # Setup mock
        mock_fig = MagicMock()
        mock_create_timeline.return_value = mock_fig
        
        # Test with default parameters
        result = self.service.visualize_context_evolution(self.test_trace)
        
        # Verify calls
        mock_create_timeline.assert_called_with(
            trace=self.test_trace,
            height=600,
            width=800
        )
        assert result == mock_fig
    
    @patch("beeai_framework.visualization.components.steps_visualizer.StepsVisualizer.generate_transition_data")
    @patch("beeai_framework.visualization.components.steps_visualizer.StepsVisualizer.generate_step_flow_chart")
    def test_visualize_step_transitions(self, mock_flow_chart, mock_transition_data):
        """Test visualize_step_transitions method."""
        # Setup mocks
        mock_data = {"transitions": []}
        mock_fig = MagicMock()
        mock_transition_data.return_value = mock_data
        mock_flow_chart.return_value = mock_fig
        
        # Test with default parameters
        result = self.service.visualize_step_transitions(self.test_trace)
        
        # Verify calls
        mock_transition_data.assert_called_with(self.test_trace)
        mock_flow_chart.assert_called_with(
            mock_data,
            height=600,
            width=800
        )
        assert result == mock_fig
    
    @patch("beeai_framework.visualization.components.reasoning_quality_metrics.ReasoningQualityMetrics.compute_all_metrics")
    def test_compute_quality_metrics(self, mock_compute_metrics):
        """Test compute_quality_metrics method."""
        # Setup mock
        mock_metrics = {"quality": 0.85}
        mock_compute_metrics.return_value = mock_metrics
        
        # Test with trace object
        result = self.service.compute_quality_metrics(self.test_trace)
        
        # Verify calls
        mock_compute_metrics.assert_called_with(self.test_trace)
        assert result == mock_metrics
    
    @patch("beeai_framework.visualization.components.metrics_visualizer.MetricsVisualizer.create_metrics_dashboard")
    def test_visualize_quality_metrics(self, mock_dashboard):
        """Test visualize_quality_metrics method."""
        # Setup mock
        mock_fig = MagicMock()
        mock_dashboard.return_value = mock_fig
        
        # Test with mock metrics
        metrics = {"quality": 0.85}
        result = self.service.visualize_quality_metrics(metrics)
        
        # Verify calls
        mock_dashboard.assert_called_with(
            metrics,
            height=600,
            width=800
        )
        assert result == mock_fig
    
    @patch("beeai_framework.visualization.components.reasoning_trace_visualizer.ReasoningTraceVisualizer.export_trace_to_json")
    def test_export_trace_to_json(self, mock_export):
        """Test export_trace_to_json method."""
        # Setup mock
        mock_export.return_value = True
        
        # Test with trace object and explicit path
        file_path = str(self.output_dir / "test_trace.json")
        result = self.service.export_trace_to_json(
            trace=self.test_trace,
            file_path=file_path
        )
        
        # Verify calls
        mock_export.assert_called_with(self.test_trace.trace_id, file_path)
        assert result == file_path
        
        # Test with trace ID from cache
        self.service.trace_cache[self.test_trace.trace_id] = self.test_trace
        result = self.service.export_trace_to_json(
            trace=self.test_trace.trace_id,
            file_path=file_path
        )
        
        # Verify calls
        mock_export.assert_called_with(self.test_trace.trace_id, file_path)
        assert result == file_path
        
        # Test with auto-generated path
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, 0)
            mock_export.return_value = True
            
            result = self.service.export_trace_to_json(trace=self.test_trace)
            
            expected_path = str(self.output_dir / f"trace_{self.test_trace.trace_id}_20250101_120000.json")
            mock_export.assert_called_with(self.test_trace.trace_id, expected_path)
            assert result == expected_path
    
    @patch("beeai_framework.visualization.components.reasoning_trace_visualizer.ReasoningTraceVisualizer.import_trace_from_json")
    def test_import_trace_from_json(self, mock_import):
        """Test import_trace_from_json method."""
        # Setup mock
        mock_import.return_value = self.test_trace
        
        # Test import
        file_path = str(self.output_dir / "test_trace.json")
        result = self.service.import_trace_from_json(file_path)
        
        # Verify calls
        mock_import.assert_called_with(file_path)
        assert result == self.test_trace
        assert self.test_trace.trace_id in self.service.trace_cache
        
        # Test import failure
        mock_import.return_value = None
        result = self.service.import_trace_from_json(file_path)
        assert result is None
    
    def test_export_visualization(self):
        """Test _export_visualization method."""
        # Create mock figure
        mock_fig = MagicMock()
        
        # Test with different file types
        for ext in ['.html', '.json', '.png', '.jpg', '.svg', '.pdf']:
            file_path = str(self.output_dir / f"test_viz{ext}")
            
            with patch.object(mock_fig, f'write_{ext[1:]}' if ext != '.json' else 'to_dict') as mock_write:
                if ext == '.json':
                    mock_write.return_value = {}
                
                self.service._export_visualization(mock_fig, file_path)
                
                if ext == '.json':
                    # Special case for JSON
                    mock_write.assert_called_once()
                elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.svg', '.pdf']:
                    # Image formats use write_image
                    mock_fig.write_image.assert_called_with(file_path)
                else:
                    # HTML uses write_html
                    mock_fig.write_html.assert_called_with(file_path)
        
        # Test with unsupported extension
        file_path = str(self.output_dir / "test_viz.unsupported")
        self.service._export_visualization(mock_fig, file_path)
        mock_fig.write_html.assert_called_with(f"{os.path.splitext(file_path)[0]}.html")


# Run tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 