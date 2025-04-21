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
Unit tests for the ContextUsageAnalytics component.
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

from beeai_framework.visualization.components.context_usage_analytics import (
    ContextUsageStats,
    ContextUsageAnalytics
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
    
    # Add first step with context items
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
            "confidence": 0.85
        }
    )
    trace.add_step(step1)
    
    # Add second step with context items
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
            },
            {
                "content": "Context item 1", # Duplicated to test reuse tracking
                "similarity": 0.83,
                "metadata": {
                    "source": "document_1",
                    "level": "domain",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ],
        metrics={
            "tokens": 200,
            "confidence": 0.88
        }
    )
    trace.add_step(step2)
    
    # Add third step with context items
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
            },
            {
                "content": "Context item 5",
                "similarity": 0.76,
                "metadata": {
                    "source": "document_5",
                    "level": "user_input",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ],
        metrics={
            "tokens": 180,
            "confidence": 0.92
        }
    )
    trace.add_step(step3)
    
    # Set end time
    trace.end_time = datetime.now()
    
    return trace


def test_context_usage_stats_creation():
    """Test creation of ContextUsageStats."""
    stats = ContextUsageStats(
        context_id="doc-123",
        source="document_1",
        level="domain",
        usage_count=3,
        avg_similarity=0.85,
        steps_used=[1, 2, 3],
        metadata={"author": "Test Author"}
    )
    
    assert stats.context_id == "doc-123"
    assert stats.source == "document_1"
    assert stats.level == "domain"
    assert stats.usage_count == 3
    assert stats.avg_similarity == 0.85
    assert stats.steps_used == [1, 2, 3]
    assert stats.metadata.get("author") == "Test Author"


def test_context_usage_stats_to_dict():
    """Test converting ContextUsageStats to dictionary."""
    stats = ContextUsageStats(
        context_id="doc-123",
        source="document_1",
        level="domain",
        usage_count=3,
        avg_similarity=0.85,
        steps_used=[1, 2, 3]
    )
    
    stats_dict = stats.to_dict()
    
    assert isinstance(stats_dict, dict)
    assert stats_dict["context_id"] == "doc-123"
    assert stats_dict["source"] == "document_1"
    assert stats_dict["level"] == "domain"
    assert stats_dict["usage_count"] == 3
    assert stats_dict["avg_similarity"] == 0.85
    assert stats_dict["steps_used"] == [1, 2, 3]


def test_context_usage_analytics_initialization():
    """Test initialization of ContextUsageAnalytics."""
    analytics = ContextUsageAnalytics()
    
    assert analytics is not None
    assert hasattr(analytics, "analyze_trace")
    assert hasattr(analytics, "get_usage_stats")
    assert len(analytics.usage_stats) == 0


def test_analyze_trace(mock_reasoning_trace):
    """Test analyzing a trace for context usage."""
    analytics = ContextUsageAnalytics()
    
    # Analyze the trace
    results = analytics.analyze_trace(mock_reasoning_trace)
    
    # Verify results
    assert isinstance(results, dict)
    assert "context_items" in results
    assert "usage_by_source" in results
    assert "usage_by_level" in results
    
    # Check context items
    context_items = results["context_items"]
    assert len(context_items) >= 5  # At least 5 unique context items
    
    # Check stats in analytics object
    assert len(analytics.usage_stats) >= 5
    
    # Verify that document_1 appears twice (in steps 1 and 2)
    doc1_stats = next((s for s in analytics.usage_stats if s.source == "document_1"), None)
    assert doc1_stats is not None
    assert doc1_stats.usage_count >= 2
    assert 1 in doc1_stats.steps_used
    assert 2 in doc1_stats.steps_used


def test_get_usage_stats(mock_reasoning_trace):
    """Test retrieving usage stats."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    # Get all stats
    all_stats = analytics.get_usage_stats()
    assert len(all_stats) >= 5
    
    # Get stats by source
    doc1_stats = analytics.get_usage_stats(source="document_1")
    assert len(doc1_stats) == 1
    assert doc1_stats[0].source == "document_1"
    
    # Get stats by level
    domain_stats = analytics.get_usage_stats(level="domain")
    assert len(domain_stats) > 0
    assert all(s.level == "domain" for s in domain_stats)
    
    # Get stats by usage count
    high_usage_stats = analytics.get_usage_stats(min_usage_count=2)
    assert all(s.usage_count >= 2 for s in high_usage_stats)


def test_create_usage_visualization(mock_reasoning_trace):
    """Test creating usage visualization."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    # Create visualization
    fig = analytics.create_usage_visualization()
    
    assert isinstance(fig, go.Figure)
    # Confirm that the figure has at least some traces
    assert len(fig.data) > 0


def test_create_timeline_visualization(mock_reasoning_trace):
    """Test creating timeline visualization."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    # Create timeline visualization
    fig = analytics.create_timeline_visualization()
    
    assert isinstance(fig, go.Figure)
    # Should have data for the timeline
    assert len(fig.data) > 0


def test_create_source_distribution_visualization(mock_reasoning_trace):
    """Test creating source distribution visualization."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    # Create source distribution visualization
    fig = analytics.create_source_distribution_visualization()
    
    assert isinstance(fig, go.Figure)
    # Should have data for the distribution
    assert len(fig.data) > 0


def test_create_level_distribution_visualization(mock_reasoning_trace):
    """Test creating level distribution visualization."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    # Create level distribution visualization
    fig = analytics.create_level_distribution_visualization()
    
    assert isinstance(fig, go.Figure)
    # Should have data for the distribution
    assert len(fig.data) > 0


def test_create_similarity_distribution_visualization(mock_reasoning_trace):
    """Test creating similarity distribution visualization."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    # Create similarity distribution visualization
    fig = analytics.create_similarity_distribution_visualization()
    
    assert isinstance(fig, go.Figure)
    # Should have data for the distribution
    assert len(fig.data) > 0


def test_calculate_reuse_metrics(mock_reasoning_trace):
    """Test calculating context reuse metrics."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    # Calculate reuse metrics
    reuse_metrics = analytics.calculate_reuse_metrics()
    
    assert isinstance(reuse_metrics, dict)
    assert "reuse_rate" in reuse_metrics
    assert "unique_contexts" in reuse_metrics
    assert "total_context_uses" in reuse_metrics
    
    # Check reuse rate calculation
    assert 0 <= reuse_metrics["reuse_rate"] <= 1


def test_save_and_load_analytics():
    """Test saving and loading analytics to/from JSON."""
    analytics = ContextUsageAnalytics()
    
    # Add some sample stats
    stats1 = ContextUsageStats(
        context_id="doc-123",
        source="document_1",
        level="domain",
        usage_count=3,
        avg_similarity=0.85,
        steps_used=[1, 2, 3]
    )
    
    stats2 = ContextUsageStats(
        context_id="doc-456",
        source="document_2",
        level="project",
        usage_count=2,
        avg_similarity=0.77,
        steps_used=[2, 4]
    )
    
    analytics.usage_stats.append(stats1)
    analytics.usage_stats.append(stats2)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save analytics to JSON
        output_path = os.path.join(tmpdir, "test_analytics.json")
        analytics.save_analytics(output_path)
        
        # Check that file exists and has content
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Load analytics into a new instance
        new_analytics = ContextUsageAnalytics()
        new_analytics.load_analytics(output_path)
        
        # Verify analytics were loaded correctly
        assert len(new_analytics.usage_stats) == 2
        
        loaded_stats1 = next((s for s in new_analytics.usage_stats if s.context_id == "doc-123"), None)
        assert loaded_stats1 is not None
        assert loaded_stats1.source == "document_1"
        assert loaded_stats1.usage_count == 3
        
        loaded_stats2 = next((s for s in new_analytics.usage_stats if s.context_id == "doc-456"), None)
        assert loaded_stats2 is not None
        assert loaded_stats2.source == "document_2"
        assert loaded_stats2.usage_count == 2


def test_create_dashboard(mock_reasoning_trace):
    """Test creating a dashboard with multiple visualizations."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    # Create dashboard
    dashboard = analytics.create_dashboard()
    
    assert isinstance(dashboard, go.Figure)
    # Should have a figure with multiple subplots
    assert hasattr(dashboard, 'layout')
    assert hasattr(dashboard.layout, 'template')


def test_export_to_dataframe(mock_reasoning_trace):
    """Test exporting analytics to pandas DataFrame."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    # Export to DataFrame
    df = analytics.export_to_dataframe()
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "context_id" in df.columns
    assert "source" in df.columns
    assert "level" in df.columns
    assert "usage_count" in df.columns
    assert "avg_similarity" in df.columns


def test_export_to_csv(mock_reasoning_trace):
    """Test exporting analytics to CSV file."""
    analytics = ContextUsageAnalytics()
    analytics.analyze_trace(mock_reasoning_trace)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export to CSV
        output_path = os.path.join(tmpdir, "test_analytics.csv")
        analytics.export_to_csv(output_path)
        
        # Check that file exists and has content
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Load the CSV and check content
        df = pd.read_csv(output_path)
        assert not df.empty
        assert "context_id" in df.columns
        assert "source" in df.columns
        assert "level" in df.columns


def test_edge_case_empty_trace():
    """Test analyzing an empty trace."""
    analytics = ContextUsageAnalytics()
    empty_trace = ReasoningTrace(
        trace_id="empty-trace",
        task="Empty test task",
        start_time=datetime.now()
    )
    empty_trace.end_time = datetime.now()
    
    # Analyze empty trace
    results = analytics.analyze_trace(empty_trace)
    
    assert isinstance(results, dict)
    assert "context_items" in results
    assert len(results["context_items"]) == 0
    
    # Visualizations should handle empty data gracefully
    fig = analytics.create_usage_visualization()
    assert isinstance(fig, go.Figure)


def test_edge_case_no_context_items():
    """Test analyzing a trace with steps but no context items."""
    analytics = ContextUsageAnalytics()
    trace = ReasoningTrace(
        trace_id="no-context-trace",
        task="Test task without context",
        start_time=datetime.now()
    )
    
    # Add a step without context items
    step = ReasoningStep(
        step_number=1,
        step_type="analysis",
        content="Step with no context items",
        timestamp=datetime.now(),
        context_items=[]  # Empty context items
    )
    trace.add_step(step)
    trace.end_time = datetime.now()
    
    # Analyze trace
    results = analytics.analyze_trace(trace)
    
    assert isinstance(results, dict)
    assert "context_items" in results
    assert len(results["context_items"]) == 0
    
    # Reuse metrics should handle no context items
    reuse_metrics = analytics.calculate_reuse_metrics()
    assert reuse_metrics["unique_contexts"] == 0
    assert reuse_metrics["total_context_uses"] == 0


def test_failure_case_invalid_trace():
    """Test behavior with invalid trace object."""
    analytics = ContextUsageAnalytics()
    
    # Should raise TypeError or AttributeError for invalid trace
    with pytest.raises((TypeError, AttributeError)):
        analytics.analyze_trace("not a trace object")


if __name__ == "__main__":
    # Run the tests manually if needed
    pytest.main(["-xvs", __file__]) 