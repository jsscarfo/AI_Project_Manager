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
Tests for the ContextUsageAnalytics component.

This module contains tests for the context usage analytics functionality,
including token usage analysis, knowledge source utilization metrics,
context relevance heatmaps, information density tracking, and overlap detection.
"""

import json
import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from collections import Counter
import plotly.graph_objects as go

from beeai_framework.visualization.components.context_usage_analytics import (
    ContextUsageAnalytics,
    ContextUsageStats
)
from beeai_framework.visualization.components.reasoning_trace_visualizer import (
    ReasoningTrace,
    ReasoningStep
)

# Fixtures for test data
@pytest.fixture
def sample_trace() -> ReasoningTrace:
    """Create a sample reasoning trace for testing."""
    steps = [
        ReasoningStep(
            step_number=1,
            step_type="analysis",
            content="Initial analysis of the problem",
            context_items=[
                {
                    "id": "ctx1",
                    "content": "This is context item 1",
                    "metadata": {
                        "source": "knowledge_base",
                        "level": "basic",
                        "relevance_score": 0.8
                    }
                },
                {
                    "id": "ctx2",
                    "content": "This is context item 2 with different content",
                    "metadata": {
                        "source": "document",
                        "level": "detailed",
                        "relevance_score": 0.6
                    }
                }
            ]
        ),
        ReasoningStep(
            step_number=2,
            step_type="reasoning",
            content="Reasoning about the problem using context",
            context_items=[
                {
                    "id": "ctx1",  # Reused from step 1
                    "content": "This is context item 1",
                    "metadata": {
                        "source": "knowledge_base",
                        "level": "basic",
                        "relevance_score": 0.7
                    }
                },
                {
                    "id": "ctx3",
                    "content": "New context information for step 2",
                    "metadata": {
                        "source": "web_search",
                        "level": "advanced",
                        "relevance_score": 0.9
                    }
                }
            ]
        ),
        ReasoningStep(
            step_number=3,
            step_type="conclusion",
            content="Final conclusion of the analysis",
            context_items=[
                {
                    "id": "ctx3",  # Reused from step 2
                    "content": "New context information for step 2",
                    "metadata": {
                        "source": "web_search",
                        "level": "advanced",
                        "relevance_score": 0.85
                    }
                },
                {
                    "id": "ctx4",
                    "content": "Additional context for conclusion",
                    "metadata": {
                        "source": "document",
                        "level": "detailed",
                        "relevance_score": 0.75
                    }
                }
            ]
        )
    ]
    
    return ReasoningTrace(
        trace_id="test-trace-123",
        steps=steps,
        metadata={
            "created_at": "2025-01-01T12:00:00Z",
            "model": "beeai-test-model"
        }
    )

@pytest.fixture
def analytics() -> ContextUsageAnalytics:
    """Create a ContextUsageAnalytics instance for testing."""
    return ContextUsageAnalytics(
        default_height=400,
        default_width=600,
        tokenizer_name="cl100k_base",
        cache_enabled=False
    )


class TestContextUsageAnalytics:
    """Tests for the ContextUsageAnalytics class."""
    
    def test_init(self, analytics):
        """Test initialization of the analytics component."""
        assert analytics.default_height == 400
        assert analytics.default_width == 600
        assert analytics.cache_enabled is False
        assert isinstance(analytics.cache, dict)
        assert len(analytics.cache) == 0
    
    def test_analyze_token_usage(self, analytics, sample_trace):
        """Test token usage analysis across a reasoning trace."""
        token_stats = analytics.analyze_token_usage(sample_trace)
        
        # Check basic structure
        assert isinstance(token_stats, dict)
        assert "total_tokens" in token_stats
        assert "tokens_per_step" in token_stats
        assert "tokens_by_source" in token_stats
        assert "tokens_by_level" in token_stats
        
        # Check tokens per step
        assert len(token_stats["tokens_per_step"]) == 3
        assert 1 in token_stats["tokens_per_step"]
        assert 2 in token_stats["tokens_per_step"]
        assert 3 in token_stats["tokens_per_step"]
        
        # Check tokens by source
        assert "knowledge_base" in token_stats["tokens_by_source"]
        assert "document" in token_stats["tokens_by_source"]
        assert "web_search" in token_stats["tokens_by_source"]
        
        # Check tokens by level
        assert "basic" in token_stats["tokens_by_level"]
        assert "detailed" in token_stats["tokens_by_level"]
        assert "advanced" in token_stats["tokens_by_level"]
        
        # Ensure total tokens is the sum of step tokens and context tokens
        assert token_stats["total_tokens"] == token_stats["tokens_in_steps"] + token_stats["tokens_in_context"]
    
    def test_analyze_knowledge_source_utilization(self, analytics, sample_trace):
        """Test knowledge source utilization analysis."""
        source_stats = analytics.analyze_knowledge_source_utilization(sample_trace)
        
        # Check basic structure
        assert isinstance(source_stats, dict)
        assert "source_counts" in source_stats
        assert "level_counts" in source_stats
        assert "source_by_step" in source_stats
        assert "level_by_step" in source_stats
        
        # Check source counts
        assert source_stats["source_counts"]["knowledge_base"] == 2
        assert source_stats["source_counts"]["document"] == 2
        assert source_stats["source_counts"]["web_search"] == 2
        
        # Check level counts
        assert source_stats["level_counts"]["basic"] == 2
        assert source_stats["level_counts"]["detailed"] == 2
        assert source_stats["level_counts"]["advanced"] == 2
        
        # Check step distribution
        assert len(source_stats["source_by_step"]) == 3
        assert len(source_stats["level_by_step"]) == 3
    
    def test_create_context_relevance_heatmap(self, analytics, sample_trace):
        """Test creation of context relevance heatmap."""
        heatmap = analytics.create_context_relevance_heatmap(sample_trace)
        
        # Check return type
        assert isinstance(heatmap, go.Figure)
        
        # Check figure dimensions
        assert heatmap.layout.height == 400
        assert heatmap.layout.width == 600
        
        # Check that the figure has data
        assert len(heatmap.data) > 0
        
        # Check custom dimensions work
        custom_heatmap = analytics.create_context_relevance_heatmap(
            sample_trace, height=500, width=700
        )
        assert custom_heatmap.layout.height == 500
        assert custom_heatmap.layout.width == 700
    
    def test_calculate_information_density(self, analytics, sample_trace):
        """Test calculation of information density metrics."""
        density = analytics.calculate_information_density(sample_trace)
        
        # Check return type
        assert isinstance(density, dict)
        
        # Check all steps are included
        assert 1 in density
        assert 2 in density
        assert 3 in density
        
        # Check values are within expected range (0-1)
        for step_num, value in density.items():
            assert 0 <= value <= 1
    
    def test_detect_context_overlap(self, analytics, sample_trace):
        """Test detection of context overlap between steps."""
        overlap = analytics.detect_context_overlap(sample_trace)
        
        # Check return type
        assert isinstance(overlap, dict)
        
        # Check basic structure
        assert "overlap_matrix" in overlap
        assert "unique_context_percentage" in overlap
        assert "duplicated_context_percentage" in overlap
        
        # Check matrix dimensions match step count
        matrix = overlap["overlap_matrix"]
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)
        
        # Check percentages are within expected range (0-100)
        assert 0 <= overlap["unique_context_percentage"] <= 100
        assert 0 <= overlap["duplicated_context_percentage"] <= 100
        
        # Sum should be 100
        assert abs(overlap["unique_context_percentage"] + 
                  overlap["duplicated_context_percentage"] - 100) < 0.01
    
    def test_create_token_usage_chart(self, analytics, sample_trace):
        """Test creation of token usage chart."""
        chart = analytics.create_token_usage_chart(sample_trace)
        
        # Check return type
        assert isinstance(chart, go.Figure)
        
        # Check dimensions
        assert chart.layout.height == 400
        assert chart.layout.width == 600
        
        # Check that chart has data
        assert len(chart.data) > 0
        
        # Check custom dimensions work
        custom_chart = analytics.create_token_usage_chart(
            sample_trace, height=500, width=700
        )
        assert custom_chart.layout.height == 500
        assert custom_chart.layout.width == 700
    
    def test_create_knowledge_source_chart(self, analytics, sample_trace):
        """Test creation of knowledge source utilization chart."""
        chart = analytics.create_knowledge_source_chart(sample_trace)
        
        # Check return type
        assert isinstance(chart, go.Figure)
        
        # Check dimensions
        assert chart.layout.height == 400
        assert chart.layout.width == 600
        
        # Check that chart has data
        assert len(chart.data) > 0
        
        # Check custom dimensions work
        custom_chart = analytics.create_knowledge_source_chart(
            sample_trace, height=500, width=700
        )
        assert custom_chart.layout.height == 500
        assert custom_chart.layout.width == 700
    
    def test_create_information_density_chart(self, analytics, sample_trace):
        """Test creation of information density chart."""
        chart = analytics.create_information_density_chart(sample_trace)
        
        # Check return type
        assert isinstance(chart, go.Figure)
        
        # Check dimensions
        assert chart.layout.height == 400
        assert chart.layout.width == 600
        
        # Check that chart has data
        assert len(chart.data) > 0
        
        # Check custom dimensions work
        custom_chart = analytics.create_information_density_chart(
            sample_trace, height=500, width=700
        )
        assert custom_chart.layout.height == 500
        assert custom_chart.layout.width == 700
    
    def test_analyze_all_metrics(self, analytics, sample_trace):
        """Test comprehensive analysis of all metrics."""
        stats = analytics.analyze_all_metrics(sample_trace)
        
        # Check return type
        assert isinstance(stats, ContextUsageStats)
        
        # Check basic fields are populated
        assert stats.total_tokens > 0
        assert len(stats.tokens_per_step) == 3
        assert len(stats.source_usage) >= 3
        assert len(stats.level_usage) >= 3
        assert len(stats.relevance_scores) > 0
        assert 0 <= stats.avg_relevance <= 1
        assert len(stats.info_density_scores) == 3
        assert 0 <= stats.avg_info_density <= 1
        assert 0 <= stats.overlap_ratio <= 1
        assert 0 <= stats.unique_context_ratio <= 1
    
    def test_token_overlap_calculation(self, analytics):
        """Test calculation of token overlap between context items."""
        contexts_a = [
            {"content": "This is a test context with some overlap"},
            {"content": "This is a unique context A"}
        ]
        
        contexts_b = [
            {"content": "This is a test context with some overlap"},
            {"content": "This is a unique context B"}
        ]
        
        overlap = analytics._calculate_token_overlap(contexts_a, contexts_b)
        
        # Check return type
        assert isinstance(overlap, float)
        
        # Check value is within expected range (0-1)
        assert 0 <= overlap <= 1
        
        # Should be around 0.5 for this example (half of tokens overlap)
        assert 0.4 <= overlap <= 0.6
        
        # Test with completely different contexts
        contexts_c = [
            {"content": "Completely different text here"}
        ]
        
        overlap_zero = analytics._calculate_token_overlap(contexts_a, contexts_c)
        assert overlap_zero < 0.1  # Should be close to zero
        
        # Test with identical contexts
        overlap_identical = analytics._calculate_token_overlap(contexts_a, contexts_a)
        assert overlap_identical > 0.9  # Should be close to 1.0


if __name__ == "__main__":
    # Simple test with dummy data
    analytics = ContextUsageAnalytics()
    
    # Create a minimal test trace
    test_trace = ReasoningTrace(
        trace_id="test-123",
        steps=[
            ReasoningStep(
                step_number=1,
                step_type="analysis",
                content="Test content",
                context_items=[
                    {
                        "id": "ctx1",
                        "content": "Test context",
                        "metadata": {
                            "source": "test",
                            "level": "basic",
                            "relevance_score": 0.5
                        }
                    }
                ]
            )
        ],
        metadata={}
    )
    
    # Run a basic analysis
    token_stats = analytics.analyze_token_usage(test_trace)
    print(f"Token stats: {token_stats}")
    
    source_stats = analytics.analyze_knowledge_source_utilization(test_trace)
    print(f"Source stats: {source_stats}")
    
    all_metrics = analytics.analyze_all_metrics(test_trace)
    print(f"All metrics: {all_metrics}") 