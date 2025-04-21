#!/usr/bin/env python
"""
Tests for knowledge graph visualization component.

This module contains unit tests for the knowledge graph visualization
component of the visualization framework.
"""

import pytest
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any

import plotly.graph_objects as go
import networkx as nx

from ..core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData,
    ContextSourceVisualizationData,
    VisualizationMetadata,
    KnowledgeGraphNode,
    KnowledgeGraphEdge,
    KnowledgeGraphVisualizationData
)

from ..components.knowledge_graph import (
    KnowledgeGraphVisualizer,
    KnowledgeGraphBuilder,
    ConceptExtractor,
    GraphLayoutService,
    GraphAnalysisService,
    ConceptPathVisualizer
)


@pytest.fixture
def sample_trace_data():
    """Create sample trace data for testing."""
    metadata = VisualizationMetadata(
        title="Test Trace",
        description="Test trace for knowledge graph testing",
        creator="test_suite"
    )
    
    steps = []
    for i in range(1, 4):
        step = StepVisualizationData(
            step_id=f"step_{i}",
            step_number=i,
            title=f"Step {i}",
            content=(
                f"In this step we analyze the Machine Learning model performance. "
                f"The RandomForest classifier shows better accuracy than the Neural Network. "
                f"We need to perform Hyperparameter Tuning to improve model.accuracy."
            ),
            step_type="analysis" if i % 2 == 0 else "synthesis",
            timestamp=datetime.now().timestamp() - (3-i) * 60,
            duration=2.0,
            requires_next_step=i < 3,
            metrics={"complexity": 0.5 * i},
            context_references=[]
        )
        steps.append(step)
    
    context_sources = []
    for i in range(2):
        source = ContextSourceVisualizationData(
            source_id=f"source_{i}",
            name=f"Source {i}",
            source_type="document" if i == 0 else "code",
            usage_count=i + 3,
            relevance_scores=[0.7, 0.8],
            steps_referenced=[1, 2]
        )
        context_sources.append(source)
    
    return TraceVisualizationData(
        trace_id="test_trace_001",
        task="Test ML model performance",
        metadata=metadata,
        steps=steps,
        context_sources=context_sources,
        final_result="The model performance analysis is complete."
    )


@pytest.fixture
def concept_extractor():
    """Create a concept extractor instance for testing."""
    return ConceptExtractor()


@pytest.fixture
def graph_builder():
    """Create a knowledge graph builder instance for testing."""
    return KnowledgeGraphBuilder()


@pytest.fixture
def layout_service():
    """Create a graph layout service instance for testing."""
    return GraphLayoutService()


@pytest.fixture
def analysis_service():
    """Create a graph analysis service instance for testing."""
    return GraphAnalysisService()


@pytest.fixture
def path_visualizer():
    """Create a concept path visualizer instance for testing."""
    return ConceptPathVisualizer()


@pytest.fixture
def visualizer():
    """Create a knowledge graph visualizer instance for testing."""
    return KnowledgeGraphVisualizer()


@pytest.fixture
def sample_graph_data():
    """Create sample knowledge graph data for testing."""
    nodes = [
        KnowledgeGraphNode(
            node_id="node_1",
            label="Machine Learning",
            type="domain_concept",
            properties={"importance": 0.9},
            steps=[1, 2, 3]
        ),
        KnowledgeGraphNode(
            node_id="node_2",
            label="RandomForest",
            type="technical_concept",
            properties={"importance": 0.8},
            steps=[1, 2]
        ),
        KnowledgeGraphNode(
            node_id="node_3",
            label="Neural Network",
            type="technical_concept",
            properties={"importance": 0.7},
            steps=[2]
        ),
        KnowledgeGraphNode(
            node_id="node_4",
            label="Hyperparameter Tuning",
            type="technical_concept",
            properties={"importance": 0.85},
            steps=[3]
        )
    ]
    
    edges = [
        KnowledgeGraphEdge(
            edge_id="edge_1",
            source="node_1",
            target="node_2",
            type="contains",
            properties={"confidence": 0.9}
        ),
        KnowledgeGraphEdge(
            edge_id="edge_2",
            source="node_1",
            target="node_3",
            type="contains",
            properties={"confidence": 0.8}
        ),
        KnowledgeGraphEdge(
            edge_id="edge_3",
            source="node_2",
            target="node_4",
            type="related_to",
            properties={"confidence": 0.7}
        ),
        KnowledgeGraphEdge(
            edge_id="edge_4",
            source="node_3",
            target="node_4",
            type="related_to",
            properties={"confidence": 0.6}
        )
    ]
    
    return KnowledgeGraphVisualizationData(
        graph_id="test_graph_001",
        nodes=nodes,
        edges=edges,
        metadata={
            "trace_id": "test_trace_001",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "concept_types": ["domain_concept", "technical_concept"]
        }
    )


class TestConceptExtractor:
    """Tests for the ConceptExtractor class."""
    
    def test_extract_concepts(self, concept_extractor):
        """Test extracting concepts from text."""
        text = "The RandomForest classifier shows better accuracy than the Neural Network."
        concepts = concept_extractor.extract_concepts(text)
        
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        
        # Check that it found at least some of the key concepts
        concept_texts = [c["text"] for c in concepts]
        assert any(c for c in concept_texts if "RandomForest" in c)
        assert any(c for c in concept_texts if "Neural Network" in c)
        
        # Check concept structure
        for concept in concepts:
            assert "text" in concept
            assert "type" in concept
            assert "confidence" in concept
            assert "position" in concept
    
    def test_infer_relationships(self, concept_extractor):
        """Test inferring relationships between concepts."""
        text = "The RandomForest classifier shows better accuracy than the Neural Network."
        concepts = [
            {
                "text": "RandomForest",
                "type": "technical_concept",
                "confidence": 0.8,
                "position": text.find("RandomForest")
            },
            {
                "text": "Neural Network",
                "type": "technical_concept",
                "confidence": 0.8,
                "position": text.find("Neural Network")
            }
        ]
        
        relationships = concept_extractor.infer_relationships(concepts, text)
        
        assert isinstance(relationships, list)
        assert len(relationships) > 0
        
        # Check relationship structure
        relationship = relationships[0]
        assert "source" in relationship
        assert "target" in relationship
        assert "type" in relationship
        assert "confidence" in relationship
        assert "distance" in relationship
        
        # Check relationship content
        assert relationship["source"] == "RandomForest"
        assert relationship["target"] == "Neural Network"


class TestKnowledgeGraphBuilder:
    """Tests for the KnowledgeGraphBuilder class."""
    
    def test_build_knowledge_graph(self, graph_builder, sample_trace_data):
        """Test building a knowledge graph from trace data."""
        graph_data = graph_builder.build_knowledge_graph(sample_trace_data)
        
        assert isinstance(graph_data, KnowledgeGraphVisualizationData)
        assert graph_data.graph_id == f"graph_{sample_trace_data.trace_id}"
        assert len(graph_data.nodes) > 0
        assert len(graph_data.edges) > 0
        
        # Check that nodes have proper structure
        for node in graph_data.nodes:
            assert node.node_id is not None
            assert node.label is not None
            assert node.type is not None
            assert isinstance(node.properties, dict)
            assert isinstance(node.steps, list)
        
        # Check that edges have proper structure
        for edge in graph_data.edges:
            assert edge.edge_id is not None
            assert edge.source is not None
            assert edge.target is not None
            assert edge.type is not None
            assert isinstance(edge.properties, dict)


class TestGraphLayoutService:
    """Tests for the GraphLayoutService class."""
    
    def test_apply_force_directed_layout(self, layout_service, sample_graph_data):
        """Test applying force-directed layout to graph data."""
        layout_result = layout_service.apply_force_directed_layout(sample_graph_data)
        
        assert isinstance(layout_result, dict)
        assert "positions" in layout_result
        assert len(layout_result["positions"]) == len(sample_graph_data.nodes)
        
        # Check that positions are 2D coordinates
        for node_id, position in layout_result["positions"].items():
            assert node_id in [node.node_id for node in sample_graph_data.nodes]
            assert len(position) == 2
            assert isinstance(position[0], (int, float))
            assert isinstance(position[1], (int, float))


class TestGraphAnalysisService:
    """Tests for the GraphAnalysisService class."""
    
    def test_analyze_centrality(self, analysis_service, sample_graph_data):
        """Test analyzing centrality of graph nodes."""
        centrality_result = analysis_service.analyze_centrality(sample_graph_data)
        
        assert isinstance(centrality_result, dict)
        assert "centrality_scores" in centrality_result
        assert len(centrality_result["centrality_scores"]) == len(sample_graph_data.nodes)
        
        # Check that scores are properly formatted
        for node_id, score in centrality_result["centrality_scores"].items():
            assert node_id in [node.node_id for node in sample_graph_data.nodes]
            assert 0 <= score <= 1
    
    def test_detect_communities(self, analysis_service, sample_graph_data):
        """Test detecting communities in the graph."""
        communities_result = analysis_service.detect_communities(sample_graph_data)
        
        assert isinstance(communities_result, dict)
        assert "communities" in communities_result
        assert isinstance(communities_result["communities"], list)
        
        # Check that each node is assigned to a community
        nodes_in_communities = []
        for community in communities_result["communities"]:
            assert isinstance(community, list)
            nodes_in_communities.extend(community)
        
        assert set(nodes_in_communities) == set(node.node_id for node in sample_graph_data.nodes)


class TestConceptPathVisualizer:
    """Tests for the ConceptPathVisualizer class."""
    
    def test_visualize_concept_flow(self, path_visualizer, sample_graph_data, sample_trace_data):
        """Test visualizing concept flow through steps."""
        flow_result = path_visualizer.visualize_concept_flow(sample_graph_data, sample_trace_data)
        
        assert isinstance(flow_result, dict)
        assert "steps" in flow_result
        assert "concepts" in flow_result
        assert "paths" in flow_result
        
        # Check structure of flow data
        assert len(flow_result["steps"]) == len(sample_trace_data.steps)
        assert len(flow_result["concepts"]) > 0
        assert len(flow_result["paths"]) > 0


class TestKnowledgeGraphVisualizer:
    """Tests for the KnowledgeGraphVisualizer class."""
    
    def test_generate_visualization_data(self, visualizer, sample_trace_data):
        """Test generating visualization data."""
        viz_data = visualizer.generate_visualization_data(sample_trace_data)
        
        assert isinstance(viz_data, dict)
        assert "graph" in viz_data
        assert "layout" in viz_data
        assert "analysis" in viz_data
        
        # Check graph data
        assert "nodes" in viz_data["graph"]
        assert "edges" in viz_data["graph"]
        assert len(viz_data["graph"]["nodes"]) > 0
        assert len(viz_data["graph"]["edges"]) > 0
    
    def test_to_json(self, visualizer, sample_trace_data):
        """Test converting visualization data to JSON."""
        json_result = visualizer.to_json(sample_trace_data)
        
        assert isinstance(json_result, str)
        
        # Check that result is valid JSON
        parsed_result = json.loads(json_result)
        assert isinstance(parsed_result, dict)
        assert "graph" in parsed_result
    
    def test_highlight_concept_usage(self, visualizer, sample_trace_data):
        """Test highlighting concept usage across steps."""
        # First generate the graph data
        viz_data = visualizer.generate_visualization_data(sample_trace_data)
        
        # Get a concept from the graph to highlight
        concepts = [node["label"] for node in viz_data["graph"]["nodes"]]
        if concepts:
            highlight_result = visualizer.highlight_concept_usage(concepts[0], sample_trace_data)
            
            assert isinstance(highlight_result, dict)
            assert "concept" in highlight_result
            assert "steps" in highlight_result
            assert "usage_count" in highlight_result 