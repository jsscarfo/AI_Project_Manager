#!/usr/bin/env python
"""
Tests for metrics visualization components.

This module contains unit tests for the metrics visualization
components of the reasoning trace visualization framework.
"""

import pytest
import json
from datetime import datetime, timedelta
import statistics
from typing import Dict, List, Any

from ..core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData,
    ContextSourceVisualizationData,
    MetricsVisualizationData,
    VisualizationMetadata
)

from ..components.metrics_visualizer import (
    TimeSeriesGenerator,
    ComparisonMetricsGenerator,
    RadarChartGenerator,
    HeatMapGenerator,
    MetricsVisualizer
)


@pytest.fixture
def sample_trace_data():
    """Create sample trace data for testing."""
    # Create metadata
    metadata = VisualizationMetadata(
        title="Test Trace",
        description="Test trace for visualization testing",
        creator="test_suite"
    )
    
    # Create steps
    steps = []
    base_timestamp = datetime.now().timestamp()
    
    for i in range(1, 6):
        # Create step with varying durations and context references
        step = StepVisualizationData(
            step_id=f"step_{i}",
            step_number=i,
            title=f"Step {i}",
            content=f"This is the content for step {i}",
            step_type="analysis" if i % 2 == 0 else "synthesis",
            timestamp=base_timestamp + (i - 1) * 60,  # Each step is 60 seconds apart
            duration=2.0 if i % 2 == 0 else 4.0,  # Alternate durations
            requires_next_step=i < 5,  # Last step doesn't require next
            metrics={"complexity": 0.5 * i},
            context_references=[
                {
                    "context_id": f"ctx_{i}_{j}",
                    "source": f"source_{j % 3}",
                    "relevance_score": 0.7 if j % 2 == 0 else 0.4,
                    "usage_type": "reference"
                }
                for j in range(i)  # Number of references increases with step number
            ]
        )
        steps.append(step)
    
    # Create context sources
    context_sources = []
    for i in range(3):
        source = ContextSourceVisualizationData(
            source_id=f"source_{i}",
            name=f"Source {i}",
            source_type="document" if i == 0 else "code" if i == 1 else "comment",
            usage_count=i + 3,
            relevance_scores=[0.7, 0.5, 0.8] if i == 0 else [0.6, 0.4] if i == 1 else [0.3, 0.5, 0.7, 0.9],
            steps_referenced=[1, 3, 5] if i == 0 else [2, 4] if i == 1 else [1, 2, 3, 4, 5]
        )
        context_sources.append(source)
    
    # Create the trace data
    return TraceVisualizationData(
        trace_id="test_trace_001",
        task="Test reasoning task",
        metadata=metadata,
        steps=steps,
        context_sources=context_sources,
        final_result="This is the final result of the reasoning process."
    )


class TestTimeSeriesGenerator:
    """Tests for TimeSeriesGenerator class."""
    
    def test_generate_step_timing_series(self, sample_trace_data):
        """Test generation of step timing time series."""
        # Create generator
        generator = TimeSeriesGenerator()
        
        # Generate time series
        result = generator.generate_step_timing_series(sample_trace_data)
        
        # Verify result
        assert "step_numbers" in result
        assert "durations" in result
        assert "step_types" in result
        assert "timestamps" in result
        assert "cumulative_time" in result
        assert "series_data" in result
        
        # Verify series data
        assert len(result["series_data"]) == 2
        assert result["series_data"][0]["name"] == "Step Duration"
        assert result["series_data"][1]["name"] == "Cumulative Time"
        
        # Verify data values
        assert len(result["step_numbers"]) == 5
        assert result["step_numbers"] == [1, 2, 3, 4, 5]
        
        # Verify cumulative time is calculated correctly
        assert result["cumulative_time"][-1] == sum(result["durations"])
    
    def test_generate_context_usage_series(self, sample_trace_data):
        """Test generation of context usage time series."""
        # Create generator
        generator = TimeSeriesGenerator()
        
        # Generate time series
        result = generator.generate_context_usage_series(sample_trace_data)
        
        # Verify result
        assert "step_numbers" in result
        assert "context_counts" in result
        assert "type_averages" in result
        assert "total_context_references" in result
        assert "series_data" in result
        assert "average_by_step_type" in result
        
        # Verify context counts - step i has i references
        assert result["context_counts"] == [0, 1, 2, 3, 4]
        
        # Verify total matches sum of counts
        assert result["total_context_references"] == sum(result["context_counts"])
    
    def test_generate_relevance_series(self, sample_trace_data):
        """Test generation of relevance score time series."""
        # Create generator
        generator = TimeSeriesGenerator()
        
        # Generate time series
        result = generator.generate_relevance_series(sample_trace_data)
        
        # Verify result - steps without context references are skipped
        assert "step_numbers" in result
        assert "avg_relevance_scores" in result
        assert "max_relevance_scores" in result
        assert "min_relevance_scores" in result
        assert "overall_avg_relevance" in result
        assert "series_data" in result
        
        # Steps without context refs are skipped
        assert len(result["step_numbers"]) == 4  # 4 steps with context references
        
        # Verify data types
        assert all(isinstance(score, float) for score in result["avg_relevance_scores"])
        assert all(0 <= score <= 1 for score in result["avg_relevance_scores"])


class TestComparisonMetricsGenerator:
    """Tests for ComparisonMetricsGenerator class."""
    
    def test_compare_with_baseline(self, sample_trace_data):
        """Test comparison with baseline metrics."""
        # Create generator
        generator = ComparisonMetricsGenerator()
        
        # Define baseline
        baseline_metrics = {
            "step_count": 4,
            "total_time": 10.0,
            "avg_time_per_step": 2.5,
            "context_references": 8,
            "avg_context_per_step": 2.0
        }
        
        # Generate comparison
        result = generator.compare_with_baseline(sample_trace_data, baseline_metrics)
        
        # Verify result
        assert "trace_id" in result
        assert "metrics_comparison" in result
        assert "overall_assessment" in result
        
        # Check specific metrics
        comparison = result["metrics_comparison"]
        assert "step_count" in comparison
        assert "total_time" in comparison
        assert "avg_time_per_step" in comparison
        assert "context_references" in comparison
        assert "avg_context_per_step" in comparison
        
        # Verify each metric has required fields
        for metric_name, metric_data in comparison.items():
            assert "baseline" in metric_data
            assert "current" in metric_data
            assert "absolute_diff" in metric_data
            assert "percent_diff" in metric_data
            assert "improved" in metric_data
        
        # Check overall assessment
        assessment = result["overall_assessment"]
        assert "score" in assessment
        assert "improvements" in assessment
        assert "regressions" in assessment
        assert "assessment" in assessment


class TestRadarChartGenerator:
    """Tests for RadarChartGenerator class."""
    
    def test_generate_quality_radar(self, sample_trace_data):
        """Test generation of quality radar chart."""
        # Create generator
        generator = RadarChartGenerator()
        
        # Generate radar chart
        result = generator.generate_quality_radar(sample_trace_data)
        
        # Verify result
        assert "trace_id" in result
        assert "dimensions" in result
        assert "scores" in result
        assert "scores_object" in result
        assert "average_score" in result
        assert "chart_data" in result
        
        # Verify dimensions
        dimensions = [
            "step_count_efficiency",
            "time_efficiency",
            "context_relevance",
            "context_utilization",
            "reasoning_consistency",
            "conclusion_quality"
        ]
        assert set(result["dimensions"]) == set(dimensions)
        
        # Verify each dimension has a score
        assert len(result["scores"]) == len(dimensions)
        assert all(dim in result["scores_object"] for dim in dimensions)
        
        # Verify average score calculation
        assert result["average_score"] == sum(result["scores"]) / len(result["scores"])
        
        # Verify chart data
        assert len(result["chart_data"]) == 1
        assert result["chart_data"][0]["name"] == "Quality Score"
    
    def test_generate_step_type_radar(self, sample_trace_data):
        """Test generation of step type radar chart."""
        # Create generator
        generator = RadarChartGenerator()
        
        # Generate radar chart
        result = generator.generate_step_type_radar(sample_trace_data)
        
        # Verify result
        assert "trace_id" in result
        assert "dimensions" in result
        assert "counts" in result
        assert "percentages" in result
        assert "total_steps" in result
        assert "chart_data" in result
        
        # Verify step types count
        assert len(result["dimensions"]) == 2  # 'analysis' and 'synthesis'
        assert result["total_steps"] == 5
        
        # Verify percentages sum to 100%
        assert abs(sum(result["percentages"]) - 100.0) < 0.01


class TestHeatMapGenerator:
    """Tests for HeatMapGenerator class."""
    
    def test_generate_context_usage_heatmap(self, sample_trace_data):
        """Test generation of context usage heat map."""
        # Create generator
        generator = HeatMapGenerator()
        
        # Generate heat map
        result = generator.generate_context_usage_heatmap(sample_trace_data)
        
        # Verify result
        assert "trace_id" in result
        assert "x_labels" in result
        assert "y_labels" in result
        assert "data" in result
        assert "max_value" in result
        assert "chart_data" in result
        
        # Verify dimensions
        assert len(result["x_labels"]) == 3  # 3 sources
        assert len(result["y_labels"]) == 2  # 2 step types
        
        # Verify data shape
        assert len(result["data"]) == 2  # 2 rows (step types)
        for row in result["data"]:
            assert len(row) == 3  # 3 columns (sources)
    
    def test_generate_step_timing_heatmap(self, sample_trace_data):
        """Test generation of step timing heat map."""
        # Create generator
        generator = HeatMapGenerator()
        
        # Generate heat map
        result = generator.generate_step_timing_heatmap(sample_trace_data)
        
        # Verify result
        assert "trace_id" in result
        assert "x_labels" in result
        assert "y_labels" in result
        assert "data" in result
        assert "max_value" in result
        assert "chart_data" in result
        
        # Verify dimensions
        assert len(result["x_labels"]) == 5  # 5 time segments
        assert len(result["y_labels"]) == 2  # 2 step types
        
        # Verify data shape
        assert len(result["data"]) == 2  # 2 rows (step types)
        for row in result["data"]:
            assert len(row) == 5  # 5 columns (time segments)
            
            # Check if percentages sum to 100%
            assert abs(sum(row) - 100.0) < 0.01 or sum(row) == 0


class TestMetricsVisualizer:
    """Tests for MetricsVisualizer class."""
    
    def test_generate_visualization_data(self, sample_trace_data):
        """Test generation of complete metrics visualization data."""
        # Create visualizer
        visualizer = MetricsVisualizer()
        
        # Generate visualization data
        result = visualizer.generate_visualization_data(sample_trace_data)
        
        # Verify result
        assert "trace_id" in result
        assert "summary" in result
        assert "time_series" in result
        assert "radar_charts" in result
        assert "heat_maps" in result
        assert "comparison" in result
        
        # Verify time series
        assert "timing" in result["time_series"]
        assert "context" in result["time_series"]
        assert "relevance" in result["time_series"]
        
        # Verify radar charts
        assert "quality" in result["radar_charts"]
        assert "step_type" in result["radar_charts"]
        
        # Verify heat maps
        assert "context_usage" in result["heat_maps"]
        assert "timing" in result["heat_maps"]
    
    def test_to_json(self, sample_trace_data):
        """Test JSON serialization of visualization data."""
        # Create visualizer
        visualizer = MetricsVisualizer()
        
        # Generate JSON
        json_str = visualizer.to_json(sample_trace_data)
        
        # Verify result is valid JSON
        try:
            data = json.loads(json_str)
            assert isinstance(data, dict)
            assert "trace_id" in data
        except json.JSONDecodeError:
            pytest.fail("to_json did not produce valid JSON")
    
    def test_generate_dashboard_data(self, sample_trace_data):
        """Test generation of dashboard data."""
        # Create visualizer
        visualizer = MetricsVisualizer()
        
        # Generate dashboard data
        result = visualizer.generate_dashboard_data(sample_trace_data)
        
        # Verify result
        assert "trace_id" in result
        assert "task" in result
        assert "summary" in result
        assert "quality_score" in result
        assert "key_metrics" in result
        assert "top_performing_dimension" in result
        assert "improvement_area" in result
        
        # Verify key metrics
        assert len(result["key_metrics"]) == 5
        
        # Verify that top_performing_dimension and improvement_area are strings
        assert isinstance(result["top_performing_dimension"], str)
        assert isinstance(result["improvement_area"], str)


# Run basic tests with sample data if module is executed directly
if __name__ == "__main__":
    # Create sample data
    data = sample_trace_data()
    
    # Create metrics visualizer
    visualizer = MetricsVisualizer()
    
    # Generate visualization data
    viz_data = visualizer.generate_visualization_data(data)
    
    # Print a summary
    print(f"Trace ID: {viz_data['trace_id']}")
    print(f"Step Count: {viz_data['summary']['step_count']}")
    print(f"Total Time: {viz_data['summary']['total_time']:.2f}s")
    print(f"Quality Score: {viz_data['radar_charts']['quality']['average_score']:.1f}/100")
    
    # Test JSON serialization
    json_data = visualizer.to_json(data, indent=2)
    print(f"JSON Size: {len(json_data)} bytes")
    
    # Test dashboard data
    dashboard = visualizer.generate_dashboard_data(data)
    print(f"Top Performing Dimension: {dashboard['top_performing_dimension']}")
    print(f"Area for Improvement: {dashboard['improvement_area']}") 