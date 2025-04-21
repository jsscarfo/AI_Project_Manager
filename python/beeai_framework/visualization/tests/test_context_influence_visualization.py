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
Tests for the ContextInfluenceVisualization component.

This module contains tests for visualizing the influence of context 
on reasoning steps and the flow of source usage across reasoning steps.
"""

import pytest
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any

import plotly.graph_objects as go

from beeai_framework.visualization.components.context_visualizer import (
    ContextInfluenceVisualization
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
def influence_vis() -> ContextInfluenceVisualization:
    """Create a ContextInfluenceVisualization instance for testing."""
    return ContextInfluenceVisualization(
        default_height=500,
        default_width=800,
        highlight_threshold=0.2
    )


class TestContextInfluenceVisualization:
    """Tests for the ContextInfluenceVisualization class."""
    
    def test_init(self, influence_vis):
        """Test initialization of the visualization component."""
        assert influence_vis.default_height == 500
        assert influence_vis.default_width == 800
        assert influence_vis.highlight_threshold == 0.2
    
    def test_extract_context_usage_pattern(self, influence_vis, sample_trace):
        """Test extraction of context usage patterns from a trace."""
        usage_patterns = influence_vis._extract_context_usage_pattern(sample_trace)
        
        # Check the basic structure of the result
        assert isinstance(usage_patterns, dict)
        assert "step_context_mapping" in usage_patterns
        assert "context_step_mapping" in usage_patterns
        assert "context_metadata" in usage_patterns
        
        # Check step_context_mapping
        step_context = usage_patterns["step_context_mapping"]
        assert len(step_context) == 3  # 3 steps
        assert len(step_context[1]) == 2  # Step 1 has 2 contexts
        assert len(step_context[2]) == 2  # Step 2 has 2 contexts
        assert len(step_context[3]) == 2  # Step 3 has 2 contexts
        
        # Check context_step_mapping
        context_step = usage_patterns["context_step_mapping"]
        assert "ctx1" in context_step
        assert "ctx2" in context_step
        assert "ctx3" in context_step
        assert "ctx4" in context_step
        assert len(context_step["ctx1"]) == 2  # Used in 2 steps
        assert len(context_step["ctx2"]) == 1  # Used in 1 step
        assert len(context_step["ctx3"]) == 2  # Used in 2 steps
        assert len(context_step["ctx4"]) == 1  # Used in 1 step
        
        # Check context_metadata
        metadata = usage_patterns["context_metadata"]
        assert "ctx1" in metadata
        assert "ctx2" in metadata
        assert "ctx3" in metadata
        assert "ctx4" in metadata
        assert metadata["ctx1"]["source"] == "knowledge_base"
        assert metadata["ctx2"]["source"] == "document"
        assert metadata["ctx3"]["source"] == "web_search"
    
    def test_generate_influence_data(self, influence_vis, sample_trace):
        """Test generation of context influence data."""
        influence_data = influence_vis.generate_influence_data(sample_trace)
        
        # Check basic structure
        assert isinstance(influence_data, dict)
        assert "nodes" in influence_data
        assert "links" in influence_data
        assert "step_types" in influence_data
        
        # Check nodes
        nodes = influence_data["nodes"]
        assert len(nodes) > 0
        for node in nodes:
            assert "id" in node
            assert "label" in node
            assert "type" in node
            assert "metadata" in node
        
        # Check that all steps and contexts are represented in nodes
        node_ids = [node["id"] for node in nodes]
        assert "step-1" in node_ids
        assert "step-2" in node_ids
        assert "step-3" in node_ids
        assert "ctx1" in node_ids
        assert "ctx2" in node_ids
        assert "ctx3" in node_ids
        assert "ctx4" in node_ids
        
        # Check links
        links = influence_data["links"]
        assert len(links) > 0
        for link in links:
            assert "source" in link
            assert "target" in link
            assert "value" in link
            assert "metadata" in link
    
    def test_generate_source_flow(self, influence_vis, sample_trace):
        """Test generation of source usage flow data."""
        flow_data = influence_vis.generate_source_flow(sample_trace)
        
        # Check basic structure
        assert isinstance(flow_data, dict)
        assert "nodes" in flow_data
        assert "links" in flow_data
        assert "step_count" in flow_data
        assert "source_count" in flow_data
        
        # Check nodes
        nodes = flow_data["nodes"]
        assert len(nodes) > 0
        
        # Check that all steps and sources are represented
        source_nodes = [node for node in nodes if node["type"] == "source"]
        step_nodes = [node for node in nodes if node["type"] == "step"]
        
        assert len(step_nodes) == 3  # 3 steps
        assert len(source_nodes) == 3  # 3 unique sources
        
        # Check links
        links = flow_data["links"]
        assert len(links) > 0
        
        # Verify flow connection structure
        source_to_step_links = [link for link in links if link["source"].startswith("source-")]
        assert len(source_to_step_links) > 0
    
    def test_create_influence_network(self, influence_vis, sample_trace):
        """Test creation of influence network visualization."""
        figure = influence_vis.create_influence_network(sample_trace)
        
        # Check return type
        assert isinstance(figure, go.Figure)
        
        # Check figure dimensions
        assert figure.layout.height == 500
        assert figure.layout.width == 800
        
        # Check that the figure has at least one trace
        assert len(figure.data) > 0
        
        # Test with custom dimensions
        custom_figure = influence_vis.create_influence_network(
            sample_trace, height=600, width=900
        )
        assert custom_figure.layout.height == 600
        assert custom_figure.layout.width == 900
    
    def test_create_source_flow_sankey(self, influence_vis, sample_trace):
        """Test creation of source flow Sankey diagram."""
        figure = influence_vis.create_source_flow_sankey(sample_trace)
        
        # Check return type
        assert isinstance(figure, go.Figure)
        
        # Check figure dimensions
        assert figure.layout.height == 500
        assert figure.layout.width == 800
        
        # Check that the figure has a Sankey trace
        assert len(figure.data) > 0
        assert isinstance(figure.data[0], go.Sankey)
        
        # Test with custom dimensions
        custom_figure = influence_vis.create_source_flow_sankey(
            sample_trace, height=600, width=900
        )
        assert custom_figure.layout.height == 600
        assert custom_figure.layout.width == 900
    
    def test_generate_influence_json(self, influence_vis, sample_trace):
        """Test generation of context influence JSON data."""
        json_data = influence_vis.generate_influence_json(sample_trace)
        
        # Check that the result is valid JSON
        assert isinstance(json_data, str)
        parsed_data = json.loads(json_data)
        
        # Check basic structure
        assert "nodes" in parsed_data
        assert "links" in parsed_data
        assert "step_types" in parsed_data
    
    def test_context_influence_calculation(self, influence_vis):
        """Test calculation of context influence scores."""
        # Create test context items with different relevance scores
        context_items = [
            {"id": "ctx1", "metadata": {"relevance_score": 0.8}},
            {"id": "ctx2", "metadata": {"relevance_score": 0.5}},
            {"id": "ctx3", "metadata": {"relevance_score": 0.3}}
        ]
        
        # Calculate influence
        influence_scores = influence_vis._calculate_influence_scores(context_items)
        
        # Check basic structure
        assert isinstance(influence_scores, dict)
        assert "ctx1" in influence_scores
        assert "ctx2" in influence_scores
        assert "ctx3" in influence_scores
        
        # Highest relevance should have highest influence
        assert influence_scores["ctx1"] > influence_scores["ctx2"]
        assert influence_scores["ctx2"] > influence_scores["ctx3"]
        
        # Sum of normalized influences should be close to 1
        influence_sum = sum(influence_scores.values())
        assert 0.99 < influence_sum < 1.01
    
    def test_tracking_context_reuse(self, influence_vis, sample_trace):
        """Test tracking of context reuse across steps."""
        usage_patterns = influence_vis._extract_context_usage_pattern(sample_trace)
        context_reuse = usage_patterns["context_step_mapping"]
        
        # Check contexts used in multiple steps
        assert len(context_reuse["ctx1"]) == 2  # Used in steps 1 and 2
        assert 1 in context_reuse["ctx1"]
        assert 2 in context_reuse["ctx1"]
        
        assert len(context_reuse["ctx3"]) == 2  # Used in steps 2 and 3
        assert 2 in context_reuse["ctx3"]
        assert 3 in context_reuse["ctx3"]
        
        # Check contexts used in only one step
        assert len(context_reuse["ctx2"]) == 1  # Used in step 1 only
        assert 1 in context_reuse["ctx2"]
        
        assert len(context_reuse["ctx4"]) == 1  # Used in step 3 only
        assert 3 in context_reuse["ctx4"]


if __name__ == "__main__":
    # Simple test with dummy data
    influence_vis = ContextInfluenceVisualization()
    
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
                            "source": "test_source",
                            "relevance_score": 0.7
                        }
                    }
                ]
            )
        ],
        metadata={}
    )
    
    # Generate influence data
    influence_data = influence_vis.generate_influence_data(test_trace)
    print(f"Influence data nodes: {len(influence_data['nodes'])}")
    print(f"Influence data links: {len(influence_data['links'])}") 