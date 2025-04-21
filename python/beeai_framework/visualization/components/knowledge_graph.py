#!/usr/bin/env python
"""
Knowledge Graph Visualization Component.

This module implements visualization components for exploring concept
relationships in reasoning traces using knowledge graph representations.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union, Set, Tuple
import uuid
import networkx as nx
from pydantic import BaseModel, Field

from ..core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData,
    KnowledgeGraphNode,
    KnowledgeGraphEdge,
    KnowledgeGraphVisualizationData
)


class ConceptExtractor:
    """
    Service for extracting concepts from reasoning text.
    
    This component analyzes reasoning text to identify key concepts
    and their relationships for knowledge graph visualization.
    """
    
    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract concepts from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of extracted concepts
        """
        # In a real implementation, this would use NLP techniques
        # Here we'll use a simplified approach for demonstration
        
        # Simple regex-based extraction of potential concepts
        # Look for noun phrases that might be concepts
        # This is a very simplified approach - real implementation would use NLP
        
        # Find capitalized phrases that might be proper noun concepts
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        
        # Find technical terms using common patterns
        tech_terms = re.findall(r'\b([a-z]+(?:\.[a-z]+)+)\b', text)  # dot notation
        tech_terms += re.findall(r'\b([a-z]+_[a-z]+(?:_[a-z]+)*)\b', text)  # snake case
        tech_terms += re.findall(r'\b([a-z]+[A-Z][a-z]+(?:[A-Z][a-z]+)*)\b', text)  # camelCase
        
        # Combine and deduplicate
        all_concepts = []
        seen = set()
        
        # Process proper nouns
        for noun in proper_nouns:
            if noun.lower() not in seen and len(noun) > 3:
                seen.add(noun.lower())
                all_concepts.append({
                    "text": noun,
                    "type": "domain_concept",
                    "confidence": 0.7,
                    "position": text.find(noun)
                })
        
        # Process technical terms
        for term in tech_terms:
            if term.lower() not in seen and len(term) > 3:
                seen.add(term.lower())
                all_concepts.append({
                    "text": term,
                    "type": "technical_concept",
                    "confidence": 0.8,
                    "position": text.find(term)
                })
        
        return all_concepts
    
    def infer_relationships(self, concepts: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Infer relationships between concepts based on text proximity.
        
        Args:
            concepts: List of concepts
            text: Original text content
            
        Returns:
            List of inferred relationships
        """
        relationships = []
        
        # Sort concepts by position in text
        sorted_concepts = sorted(concepts, key=lambda c: c["position"])
        
        # Infer relationships based on proximity
        for i in range(len(sorted_concepts) - 1):
            current = sorted_concepts[i]
            next_concept = sorted_concepts[i + 1]
            
            # Calculate distance (in characters) between concepts
            distance = next_concept["position"] - (current["position"] + len(current["text"]))
            
            # Only infer relationship if concepts are reasonably close
            if distance < 100:  # Arbitrary threshold
                # Determine relationship type based on concept types
                if current["type"] == next_concept["type"]:
                    rel_type = "related_to"
                else:
                    rel_type = "associated_with"
                
                # Create relationship
                relationships.append({
                    "source": current["text"],
                    "target": next_concept["text"],
                    "type": rel_type,
                    "confidence": 0.5,  # Lower confidence for inferred relationships
                    "distance": distance
                })
        
        return relationships


class KnowledgeGraphBuilder:
    """
    Builder for knowledge graph representations.
    
    This component constructs knowledge graph data structures
    from concepts and relationships extracted from reasoning steps.
    """
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        self.concept_extractor = ConceptExtractor()
    
    def build_knowledge_graph(self, data: TraceVisualizationData) -> KnowledgeGraphVisualizationData:
        """
        Build a knowledge graph from trace data.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Knowledge graph visualization data
        """
        # Create graph ID based on trace ID
        graph_id = f"graph_{data.trace_id}"
        
        # Extract concepts from each step
        all_concepts = []
        all_relationships = []
        
        for step in data.steps:
            # Extract concepts from step content
            step_concepts = self.concept_extractor.extract_concepts(step.content)
            
            # Add step number to concepts
            for concept in step_concepts:
                concept["steps"] = [step.step_number]
            
            # Infer relationships within this step
            step_relationships = self.concept_extractor.infer_relationships(step_concepts, step.content)
            
            # Merge with existing concepts
            self._merge_concepts(all_concepts, step_concepts)
            all_relationships.extend(step_relationships)
        
        # Convert to standardized graph nodes and edges
        nodes = self._create_nodes(all_concepts)
        edges = self._create_edges(all_relationships)
        
        # Create the knowledge graph
        return KnowledgeGraphVisualizationData(
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
            metadata={
                "trace_id": data.trace_id,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "concept_types": list(set(concept["type"] for concept in all_concepts))
            }
        )
    
    def _merge_concepts(self, existing_concepts: List[Dict[str, Any]], new_concepts: List[Dict[str, Any]]):
        """
        Merge new concepts into existing concepts list.
        
        Args:
            existing_concepts: List of existing concepts
            new_concepts: List of new concepts to merge
        """
        # Create map of existing concepts by text
        concept_map = {c["text"].lower(): c for c in existing_concepts}
        
        for new_concept in new_concepts:
            # Check if concept already exists
            key = new_concept["text"].lower()
            if key in concept_map:
                # Update existing concept
                existing = concept_map[key]
                
                # Merge steps
                existing["steps"] = list(set(existing.get("steps", []) + new_concept["steps"]))
                
                # Update confidence (use max)
                existing["confidence"] = max(existing["confidence"], new_concept["confidence"])
            else:
                # Add new concept
                existing_concepts.append(new_concept)
                concept_map[key] = new_concept
    
    def _create_nodes(self, concepts: List[Dict[str, Any]]) -> List[KnowledgeGraphNode]:
        """
        Create graph nodes from concepts.
        
        Args:
            concepts: List of concepts
            
        Returns:
            List of graph nodes
        """
        nodes = []
        
        for concept in concepts:
            # Create a unique ID for the node
            node_id = f"node_{uuid.uuid4().hex[:8]}"
            
            # Calculate node weight based on confidence and step count
            weight = concept["confidence"] * (1.0 + 0.1 * len(concept.get("steps", [])))
            
            # Create the node
            nodes.append(KnowledgeGraphNode(
                node_id=node_id,
                label=concept["text"],
                type=concept["type"],
                weight=min(weight, 2.0),  # Cap weight at 2.0
                attributes={
                    "confidence": concept["confidence"],
                    "position": concept.get("position", 0)
                },
                steps=concept.get("steps", [])
            ))
        
        return nodes
    
    def _create_edges(self, relationships: List[Dict[str, Any]]) -> List[KnowledgeGraphEdge]:
        """
        Create graph edges from relationships.
        
        Args:
            relationships: List of relationships
            
        Returns:
            List of graph edges
        """
        edges = []
        
        for rel in relationships:
            # Create the edge
            edges.append(KnowledgeGraphEdge(
                source=rel["source"],
                target=rel["target"],
                type=rel["type"],
                weight=rel["confidence"],
                directed=True,
                attributes={
                    "confidence": rel["confidence"],
                    "distance": rel.get("distance", 0)
                }
            ))
        
        return edges


class GraphLayoutService:
    """
    Service for generating graph layout data.
    
    This component applies layout algorithms to knowledge graphs
    to prepare them for visualization.
    """
    
    def apply_force_directed_layout(self, graph_data: KnowledgeGraphVisualizationData) -> Dict[str, Any]:
        """
        Apply force-directed layout to the graph.
        
        Args:
            graph_data: Knowledge graph data
            
        Returns:
            Graph with layout position data
        """
        # Create a NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in graph_data.nodes:
            G.add_node(node.node_id, 
                        label=node.label, 
                        type=node.type, 
                        weight=node.weight,
                        steps=node.steps,
                        **node.attributes)
        
        # Add edges with attributes
        for edge in graph_data.edges:
            # Find node IDs for source and target
            source_id = next((n.node_id for n in graph_data.nodes if n.label == edge.source), None)
            target_id = next((n.node_id for n in graph_data.nodes if n.label == edge.target), None)
            
            if source_id and target_id:
                G.add_edge(source_id, target_id, 
                          type=edge.type, 
                          weight=edge.weight,
                          directed=edge.directed,
                          **edge.attributes)
        
        # Apply force-directed layout
        try:
            pos = nx.spring_layout(G, k=0.15, iterations=50)
        except Exception:
            # Fallback to simpler layout if spring_layout fails
            pos = nx.circular_layout(G)
        
        # Convert layout to visualization data
        layout_data = {
            "nodes": [
                {
                    "id": node.node_id,
                    "label": node.label,
                    "type": node.type,
                    "weight": node.weight,
                    "x": float(pos[node.node_id][0]),
                    "y": float(pos[node.node_id][1]),
                    "attributes": node.attributes,
                    "steps": node.steps
                }
                for node in graph_data.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "weight": edge.weight,
                    "directed": edge.directed,
                    "attributes": edge.attributes
                }
                for edge in graph_data.edges
            ]
        }
        
        return layout_data


class GraphAnalysisService:
    """
    Service for analyzing graph properties.
    
    This component analyzes knowledge graphs to identify
    important concepts, clusters, and patterns.
    """
    
    def analyze_centrality(self, graph_data: KnowledgeGraphVisualizationData) -> Dict[str, Any]:
        """
        Analyze node centrality in the graph.
        
        Args:
            graph_data: Knowledge graph data
            
        Returns:
            Centrality analysis results
        """
        # Create a NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in graph_data.nodes:
            G.add_node(node.node_id, 
                        label=node.label, 
                        type=node.type, 
                        weight=node.weight)
        
        # Add edges with attributes
        for edge in graph_data.edges:
            # Find node IDs for source and target
            source_id = next((n.node_id for n in graph_data.nodes if n.label == edge.source), None)
            target_id = next((n.node_id for n in graph_data.nodes if n.label == edge.target), None)
            
            if source_id and target_id:
                G.add_edge(source_id, target_id, weight=edge.weight)
        
        # Calculate centrality metrics
        degree_centrality = nx.degree_centrality(G)
        
        try:
            # These might fail on disconnected graphs
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
        except Exception:
            # Fallback values
            betweenness_centrality = {node.node_id: 0.0 for node in graph_data.nodes}
            closeness_centrality = {node.node_id: 0.0 for node in graph_data.nodes}
        
        # Combine metrics
        centrality_data = []
        for node in graph_data.nodes:
            centrality_data.append({
                "node_id": node.node_id,
                "label": node.label,
                "type": node.type,
                "degree_centrality": degree_centrality.get(node.node_id, 0.0),
                "betweenness_centrality": betweenness_centrality.get(node.node_id, 0.0),
                "closeness_centrality": closeness_centrality.get(node.node_id, 0.0)
            })
        
        # Sort by degree centrality
        centrality_data.sort(key=lambda x: x["degree_centrality"], reverse=True)
        
        return {
            "centrality_by_node": centrality_data,
            "top_concepts": centrality_data[:min(5, len(centrality_data))]
        }
    
    def detect_communities(self, graph_data: KnowledgeGraphVisualizationData) -> Dict[str, Any]:
        """
        Detect communities in the graph.
        
        Args:
            graph_data: Knowledge graph data
            
        Returns:
            Community detection results
        """
        # Create an undirected NetworkX graph for community detection
        G = nx.Graph()
        
        # Add nodes with attributes
        for node in graph_data.nodes:
            G.add_node(node.node_id, 
                        label=node.label, 
                        type=node.type, 
                        weight=node.weight)
        
        # Add edges with attributes
        for edge in graph_data.edges:
            # Find node IDs for source and target
            source_id = next((n.node_id for n in graph_data.nodes if n.label == edge.source), None)
            target_id = next((n.node_id for n in graph_data.nodes if n.label == edge.target), None)
            
            if source_id and target_id:
                G.add_edge(source_id, target_id, weight=edge.weight)
        
        # Detect communities
        try:
            communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        except Exception:
            # Fallback to connected components if community detection fails
            communities = list(nx.connected_components(G))
        
        # Format results
        community_data = []
        for i, community in enumerate(communities):
            # Get nodes in this community
            nodes = []
            for node_id in community:
                node = next((n for n in graph_data.nodes if n.node_id == node_id), None)
                if node:
                    nodes.append({
                        "node_id": node.node_id,
                        "label": node.label,
                        "type": node.type
                    })
            
            community_data.append({
                "community_id": i,
                "size": len(community),
                "nodes": nodes
            })
        
        # Sort by size
        community_data.sort(key=lambda x: x["size"], reverse=True)
        
        return {
            "community_count": len(community_data),
            "communities": community_data,
            "modularity": len(community_data) / max(1, len(graph_data.nodes))
        }


class ConceptPathVisualizer:
    """
    Visualizer for concept paths in reasoning.
    
    This component generates visualization data for concept flow
    through reasoning steps.
    """
    
    def visualize_concept_flow(self, graph_data: KnowledgeGraphVisualizationData, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Visualize concept flow through reasoning steps.
        
        Args:
            graph_data: Knowledge graph data
            data: Trace visualization data
            
        Returns:
            Concept flow visualization data
        """
        # Build a map of which steps each concept appears in
        concept_steps = {}
        for node in graph_data.nodes:
            concept_steps[node.label] = node.steps
        
        # Build path data for each step
        step_paths = []
        for step_number in range(1, len(data.steps) + 1):
            # Find concepts in this step
            concepts_in_step = [node.label for node in graph_data.nodes if step_number in node.steps]
            
            # Find concepts that carry over from previous step
            prev_concepts = set()
            if step_number > 1:
                prev_concepts = set(node.label for node in graph_data.nodes if step_number - 1 in node.steps)
            
            # Find continuing concepts
            continuing_concepts = [c for c in concepts_in_step if c in prev_concepts]
            
            # Find new concepts
            new_concepts = [c for c in concepts_in_step if c not in prev_concepts]
            
            step_paths.append({
                "step_number": step_number,
                "concepts": concepts_in_step,
                "concept_count": len(concepts_in_step),
                "continuing_concepts": continuing_concepts,
                "new_concepts": new_concepts
            })
        
        # Build concept journey data
        concept_journeys = []
        for node in graph_data.nodes:
            # Skip concepts that only appear in one step
            if len(node.steps) <= 1:
                continue
            
            journey = {
                "concept": node.label,
                "type": node.type,
                "first_step": min(node.steps),
                "last_step": max(node.steps),
                "step_sequence": sorted(node.steps),
                "persistence": len(node.steps) / len(data.steps)
            }
            
            concept_journeys.append(journey)
        
        # Sort by persistence
        concept_journeys.sort(key=lambda x: x["persistence"], reverse=True)
        
        return {
            "step_paths": step_paths,
            "concept_journeys": concept_journeys,
            "persistent_concepts": [j for j in concept_journeys if j["persistence"] > 0.5],
            "total_concepts": len(graph_data.nodes)
        }


class KnowledgeGraphVisualizer:
    """
    Main visualizer for knowledge graphs.
    
    This class integrates various knowledge graph visualization components
    to provide a comprehensive graph visualization system.
    """
    
    def __init__(self):
        """Initialize the knowledge graph visualizer."""
        self.graph_builder = KnowledgeGraphBuilder()
        self.layout_service = GraphLayoutService()
        self.analysis_service = GraphAnalysisService()
        self.path_visualizer = ConceptPathVisualizer()
    
    def generate_visualization_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate complete visualization data for knowledge graph.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Complete knowledge graph visualization data
        """
        # Build the knowledge graph if not already present
        graph_data = data.knowledge_graph or self.graph_builder.build_knowledge_graph(data)
        
        # Apply layout
        layout_data = self.layout_service.apply_force_directed_layout(graph_data)
        
        # Analyze graph properties
        centrality_data = self.analysis_service.analyze_centrality(graph_data)
        community_data = self.analysis_service.detect_communities(graph_data)
        
        # Visualize concept flow
        flow_data = self.path_visualizer.visualize_concept_flow(graph_data, data)
        
        # Create complete visualization data
        result = {
            "graph_id": graph_data.graph_id,
            "trace_id": data.trace_id,
            "layout": layout_data,
            "centrality": centrality_data,
            "communities": community_data,
            "concept_flow": flow_data,
            "metadata": graph_data.metadata
        }
        
        return result
    
    def to_json(self, data: TraceVisualizationData, **kwargs) -> str:
        """
        Convert visualization data to JSON.
        
        Args:
            data: Trace visualization data
            **kwargs: Additional arguments for json.dumps
            
        Returns:
            JSON representation of visualization data
        """
        visualization_data = self.generate_visualization_data(data)
        return json.dumps(visualization_data, **kwargs)
    
    def highlight_concept_usage(self, concept: str, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate visualization data highlighting a specific concept.
        
        Args:
            concept: Concept to highlight
            data: Trace visualization data
            
        Returns:
            Concept highlight visualization data
        """
        # Build the knowledge graph if not already present
        graph_data = data.knowledge_graph or self.graph_builder.build_knowledge_graph(data)
        
        # Find the concept node
        concept_node = next((node for node in graph_data.nodes if node.label.lower() == concept.lower()), None)
        
        if not concept_node:
            return {"error": f"Concept '{concept}' not found in the knowledge graph"}
        
        # Find related concepts (direct connections)
        related_concepts = []
        for edge in graph_data.edges:
            if edge.source == concept_node.label:
                related_concepts.append({
                    "concept": edge.target,
                    "relationship": edge.type,
                    "strength": edge.weight
                })
            elif edge.target == concept_node.label:
                related_concepts.append({
                    "concept": edge.source,
                    "relationship": edge.type,
                    "strength": edge.weight
                })
        
        # Find steps where concept appears
        steps_with_concept = []
        for step in data.steps:
            if step.step_number in concept_node.steps:
                steps_with_concept.append({
                    "step_number": step.step_number,
                    "title": step.title,
                    "content_preview": step.content_preview,
                    "step_type": step.step_type
                })
        
        return {
            "concept": concept_node.label,
            "type": concept_node.type,
            "steps": concept_node.steps,
            "related_concepts": related_concepts,
            "steps_with_concept": steps_with_concept,
            "first_appearance": min(concept_node.steps) if concept_node.steps else None,
            "persistence": len(concept_node.steps) / len(data.steps) if data.steps else 0
        } 