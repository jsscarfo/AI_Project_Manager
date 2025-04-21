#!/usr/bin/env python
"""
Tests for context visualizer component.

This module contains unit tests for the context visualizer
component of the visualization framework.
"""

import pytest
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any
import re

import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ..core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData,
    ContextSourceVisualizationData,
    VisualizationMetadata
)

from ..components.context_visualizer import (
    ContextVisualizer,
    ContextHighlightingService,
    SourceAttributionVisualization,
    ContextUsageAnalytics,
    ContextHeatmapGenerator
)


@pytest.fixture
def sample_trace_data():
    """Create sample trace data for testing."""
    metadata = VisualizationMetadata(
        title="Test Trace",
        description="Test trace for context visualizer testing",
        creator="test_suite"
    )
    
    steps = []
    for i in range(1, 4):
        content = (
            f"In step {i} we analyze the document data. "
            f"According to the research paper, the analysis shows "
            f"that using deep learning techniques can improve performance. "
            f"The code snippets indicate that we should use TensorFlow for this purpose."
        )
        
        context_refs = [
            {"source_id": f"doc_{j}", "relevance": 0.7 + j * 0.1, "spans": [(10, 20), (50, 60)]}
            for j in range(1, 3)
        ]
        
        step = StepVisualizationData(
            step_id=f"step_{i}",
            step_number=i,
            title=f"Step {i}",
            content=content,
            step_type="analysis" if i % 2 == 0 else "synthesis",
            timestamp=datetime.now().timestamp() - (3-i) * 60,
            duration=2.0,
            requires_next_step=i < 3,
            metrics={"complexity": 0.5 * i},
            context_references=context_refs
        )
        steps.append(step)
    
    context_sources = []
    source_types = ["document", "code", "research_paper", "api_doc"]
    for i in range(1, 5):
        source = ContextSourceVisualizationData(
            source_id=f"doc_{i}",
            name=f"Source {i}",
            source_type=source_types[i-1],
            content=f"This is content from source {i}. It contains information about deep learning and performance optimization.",
            usage_count=i + 2,
            relevance_scores=[0.5 + i * 0.1] * 3,
            steps_referenced=[j for j in range(1, 4) if j % i != 0]
        )
        context_sources.append(source)
    
    return TraceVisualizationData(
        trace_id="test_trace_001",
        task="Test context usage in an AI system",
        metadata=metadata,
        steps=steps,
        context_sources=context_sources,
        final_result="The context analysis is complete."
    )


@pytest.fixture
def highlighting_service():
    """Create a context highlighting service instance for testing."""
    return ContextHighlightingService()


@pytest.fixture
def attribution_visualization():
    """Create a source attribution visualization instance for testing."""
    return SourceAttributionVisualization()


@pytest.fixture
def usage_analytics():
    """Create a context usage analytics instance for testing."""
    return ContextUsageAnalytics()


@pytest.fixture
def heatmap_generator():
    """Create a context heatmap generator instance for testing."""
    return ContextHeatmapGenerator()


@pytest.fixture
def visualizer():
    """Create a context visualizer instance for testing."""
    return ContextVisualizer()


class TestContextHighlightingService:
    """Tests for the ContextHighlightingService class."""
    
    def test_generate_highlights(self, highlighting_service):
        """Test generating text highlights based on relevance scores."""
        text = "This is a sample text for testing highlighting functionality."
        spans = [(5, 12), (20, 25)]
        score = 0.8
        
        highlights = highlighting_service.generate_highlights(text, spans, score)
        
        assert isinstance(highlights, dict)
        assert "text" in highlights
        assert "highlights" in highlights
        assert len(highlights["highlights"]) == len(spans)
        
        for highlight in highlights["highlights"]:
            assert "start" in highlight
            assert "end" in highlight
            assert "score" in highlight
            assert "text" in highlight
            assert 0 <= highlight["score"] <= 1
    
    def test_merge_overlapping_highlights(self, highlighting_service):
        """Test merging overlapping text highlights."""
        highlights = [
            {"start": 5, "end": 15, "score": 0.7, "text": "sample"},
            {"start": 10, "end": 20, "score": 0.8, "text": "text"},
            {"start": 30, "end": 40, "score": 0.9, "text": "another"}
        ]
        
        merged = highlighting_service.merge_overlapping_highlights(highlights)
        
        assert isinstance(merged, list)
        assert len(merged) <= len(highlights)
        
        # Check no overlaps in the result
        for i in range(len(merged) - 1):
            assert merged[i]["end"] <= merged[i + 1]["start"]
    
    def test_highlight_text_with_sources(self, highlighting_service, sample_trace_data):
        """Test highlighting text with source attribution."""
        step = sample_trace_data.steps[0]
        sources = sample_trace_data.context_sources
        
        result = highlighting_service.highlight_text_with_sources(step, sources)
        
        assert isinstance(result, dict)
        assert "text" in result
        assert "highlights" in result
        assert len(result["highlights"]) > 0
        
        for highlight in result["highlights"]:
            assert "start" in highlight
            assert "end" in highlight
            assert "score" in highlight
            assert "source_id" in highlight
            assert "text" in highlight


class TestSourceAttributionVisualization:
    """Tests for the SourceAttributionVisualization class."""
    
    def test_generate_source_attribution(self, attribution_visualization, sample_trace_data):
        """Test generating source attribution visualization data."""
        result = attribution_visualization.generate_source_attribution(sample_trace_data)
        
        assert isinstance(result, dict)
        assert "steps" in result
        assert "sources" in result
        assert "attribution_matrix" in result
        
        # Check that the attribution matrix has the right dimensions
        matrix = result["attribution_matrix"]
        assert len(matrix) == len(sample_trace_data.steps)
        
        for row in matrix:
            assert len(row) == len(sample_trace_data.context_sources)
    
    def test_calculate_source_impact(self, attribution_visualization, sample_trace_data):
        """Test calculating source impact across steps."""
        impact_result = attribution_visualization.calculate_source_impact(sample_trace_data)
        
        assert isinstance(impact_result, dict)
        assert "sources" in impact_result
        assert "impact_scores" in impact_result
        assert len(impact_result["sources"]) == len(sample_trace_data.context_sources)
        assert len(impact_result["impact_scores"]) == len(sample_trace_data.context_sources)
        
        # Check score range
        for score in impact_result["impact_scores"]:
            assert 0 <= score <= 1


class TestContextUsageAnalytics:
    """Tests for the ContextUsageAnalytics class."""
    
    def test_analyze_context_usage(self, usage_analytics, sample_trace_data):
        """Test analyzing context usage patterns."""
        result = usage_analytics.analyze_context_usage(sample_trace_data)
        
        assert isinstance(result, dict)
        assert "usage_by_type" in result
        assert "usage_over_time" in result
        assert "most_relevant_sources" in result
        
        # Check that source types are accounted for
        source_types = set(source.source_type for source in sample_trace_data.context_sources)
        assert set(result["usage_by_type"].keys()) == source_types
    
    def test_generate_source_statistics(self, usage_analytics, sample_trace_data):
        """Test generating source usage statistics."""
        stats = usage_analytics.generate_source_statistics(sample_trace_data)
        
        assert isinstance(stats, list)
        assert len(stats) == len(sample_trace_data.context_sources)
        
        for source_stat in stats:
            assert "source_id" in source_stat
            assert "name" in source_stat
            assert "type" in source_stat
            assert "usage_count" in source_stat
            assert "avg_relevance" in source_stat
            assert "steps_count" in source_stat
    
    def test_identify_key_context_transitions(self, usage_analytics, sample_trace_data):
        """Test identifying key context transitions between steps."""
        transitions = usage_analytics.identify_key_context_transitions(sample_trace_data)
        
        assert isinstance(transitions, list)
        
        for transition in transitions:
            assert "from_step" in transition
            assert "to_step" in transition
            assert "changed_sources" in transition
            assert "new_sources" in transition
            assert "removed_sources" in transition


class TestContextHeatmapGenerator:
    """Tests for the ContextHeatmapGenerator class."""
    
    def test_generate_relevance_heatmap(self, heatmap_generator, sample_trace_data):
        """Test generating relevance heatmap data."""
        heatmap_data = heatmap_generator.generate_relevance_heatmap(sample_trace_data)
        
        assert isinstance(heatmap_data, dict)
        assert "x_labels" in heatmap_data
        assert "y_labels" in heatmap_data
        assert "z_values" in heatmap_data
        
        assert len(heatmap_data["x_labels"]) == len(sample_trace_data.steps)
        assert len(heatmap_data["y_labels"]) == len(sample_trace_data.context_sources)
        assert len(heatmap_data["z_values"]) == len(sample_trace_data.context_sources)
        
        for row in heatmap_data["z_values"]:
            assert len(row) == len(sample_trace_data.steps)
            for value in row:
                assert 0 <= value <= 1
    
    def test_generate_source_usage_heatmap(self, heatmap_generator, sample_trace_data):
        """Test generating source usage heatmap data."""
        heatmap_data = heatmap_generator.generate_source_usage_heatmap(sample_trace_data)
        
        assert isinstance(heatmap_data, dict)
        assert "x_labels" in heatmap_data
        assert "y_labels" in heatmap_data
        assert "z_values" in heatmap_data
        
        source_types = set(source.source_type for source in sample_trace_data.context_sources)
        assert len(heatmap_data["y_labels"]) == len(source_types)


class TestContextVisualizer:
    """Tests for the ContextVisualizer class."""
    
    def test_generate_context_visualization(self, visualizer, sample_trace_data):
        """Test generating context visualization data."""
        viz_data = visualizer.generate_context_visualization(sample_trace_data)
        
        assert isinstance(viz_data, dict)
        assert "highlights" in viz_data
        assert "attribution" in viz_data
        assert "analytics" in viz_data
        assert "heatmaps" in viz_data
        
        # Check that each step has highlight data
        assert len(viz_data["highlights"]) == len(sample_trace_data.steps)
    
    def test_to_json(self, visualizer, sample_trace_data):
        """Test converting visualization data to JSON."""
        json_result = visualizer.to_json(sample_trace_data)
        
        assert isinstance(json_result, str)
        
        # Check that result is valid JSON
        parsed_result = json.loads(json_result)
        assert isinstance(parsed_result, dict)
        assert "highlights" in parsed_result
    
    def test_generate_source_attribution_chart(self, visualizer, sample_trace_data):
        """Test generating source attribution chart."""
        # First generate the visualization data
        viz_data = visualizer.generate_context_visualization(sample_trace_data)
        
        chart = visualizer.generate_source_attribution_chart(viz_data["attribution"])
        
        assert isinstance(chart, dict)
        assert "data" in chart
        assert "layout" in chart
        assert isinstance(chart["data"], list)
        assert len(chart["data"]) > 0
    
    def test_highlight_source_usage(self, visualizer, sample_trace_data):
        """Test highlighting specific source usage."""
        source_id = sample_trace_data.context_sources[0].source_id
        
        highlight_result = visualizer.highlight_source_usage(source_id, sample_trace_data)
        
        assert isinstance(highlight_result, dict)
        assert "source_id" in highlight_result
        assert "source_name" in highlight_result
        assert "usage_in_steps" in highlight_result
        assert "relevance_by_step" in highlight_result 