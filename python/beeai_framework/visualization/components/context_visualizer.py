#!/usr/bin/env python
"""
Context Source Visualizer Component.

This module implements visualization components for context sources,
including relevance scoring, highlighting, and source attribution.
"""

import json
from typing import Dict, List, Optional, Any, Union, Callable
import uuid
import re
from pydantic import BaseModel, Field

from ..core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData,
    ContextSourceVisualizationData
)


class ContextHighlightingService:
    """
    Service for generating text highlights based on relevance.
    
    This component analyzes context text and generates metadata
    for highlighting segments based on relevance scores.
    """
    
    def generate_highlights(self, text: str, relevance_score: float, usage_type: str) -> Dict[str, Any]:
        """
        Generate highlight metadata for text segments.
        
        Args:
            text: Text content to highlight
            relevance_score: Overall relevance score
            usage_type: How the context was used
            
        Returns:
            Highlight metadata
        """
        # For demonstration purposes, we'll simulate an analysis of the text
        # In a real implementation, this would use more sophisticated NLP techniques
        
        # Split text into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Assign varying relevance to sentences
        highlights = []
        base_score = relevance_score * 0.7  # Base score from overall relevance
        
        for i, sentence in enumerate(sentences):
            # Simulate varying relevance with some randomness
            # In a real implementation, this would use semantic analysis
            import random
            variance = random.uniform(-0.2, 0.2)
            sentence_score = min(1.0, max(0.0, base_score + variance))
            
            # Determine highlight level based on score
            highlight_level = "none"
            if sentence_score > 0.8:
                highlight_level = "high"
            elif sentence_score > 0.5:
                highlight_level = "medium"
            elif sentence_score > 0.2:
                highlight_level = "low"
            
            # Find start and end positions
            start_pos = text.find(sentence)
            end_pos = start_pos + len(sentence)
            
            highlights.append({
                "id": f"highlight_{i}",
                "text": sentence,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "relevance_score": sentence_score,
                "highlight_level": highlight_level
            })
        
        return {
            "text": text,
            "overall_relevance": relevance_score,
            "usage_type": usage_type,
            "highlights": highlights,
            "has_high_relevance": any(h["highlight_level"] == "high" for h in highlights)
        }
    
    def analyze_keyword_relevance(self, text: str, query: str) -> Dict[str, Any]:
        """
        Analyze text for keyword relevance to a query.
        
        Args:
            text: Text to analyze
            query: Query to check relevance against
            
        Returns:
            Keyword relevance analysis
        """
        # Extract keywords from query (simple approach)
        # In a real implementation, use more sophisticated NLP
        query_words = set(re.findall(r'\w+', query.lower()))
        query_words = {w for w in query_words if len(w) > 3}  # Filter short words
        
        # Find keyword matches in text
        keyword_matches = []
        
        for word in query_words:
            # Find all occurrences of the word in text
            for match in re.finditer(r'\b' + re.escape(word) + r'\b', text.lower()):
                keyword_matches.append({
                    "keyword": word,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "context": text[max(0, match.start()-20):min(len(text), match.end()+20)]
                })
        
        # Calculate keyword density
        word_count = len(re.findall(r'\w+', text))
        keyword_density = len(keyword_matches) / word_count if word_count > 0 else 0
        
        return {
            "keyword_matches": keyword_matches,
            "keyword_count": len(keyword_matches),
            "keyword_density": keyword_density,
            "unique_keywords_matched": len({m["keyword"] for m in keyword_matches}),
            "total_query_keywords": len(query_words)
        }


class SourceAttributionVisualization:
    """
    Visualizer for context source attribution.
    
    This component generates visualization data for attribution
    of context to its original sources, with metadata display.
    """
    
    def generate_source_attribution(self, source: ContextSourceVisualizationData) -> Dict[str, Any]:
        """
        Generate attribution visualization data for a context source.
        
        Args:
            source: Context source to visualize
            
        Returns:
            Source attribution visualization data
        """
        # Create source attribution data
        attribution = {
            "source_id": source.source_id,
            "name": source.name,
            "source_type": source.source_type,
            "usage_count": source.usage_count,
            "average_relevance": source.average_relevance,
            "steps_referenced": source.steps_referenced,
            "metadata": source.metadata,
            # Add visualization-specific fields
            "color": self._get_source_color(source.source_type),
            "icon": self._get_source_icon(source.source_type),
            "importance": self._calculate_importance(source)
        }
        
        return attribution
    
    def _get_source_color(self, source_type: str) -> str:
        """Get color for visualization based on source type."""
        # Map source types to colors
        color_map = {
            "document": "#3498db",
            "code": "#2ecc71",
            "comment": "#e74c3c",
            "documentation": "#9b59b6",
            "web": "#e67e22",
            "database": "#f1c40f"
        }
        return color_map.get(source_type, "#95a5a6")  # Default gray for unknown
    
    def _get_source_icon(self, source_type: str) -> str:
        """Get icon identifier for visualization based on source type."""
        # Map source types to icon identifiers
        icon_map = {
            "document": "file-text",
            "code": "code",
            "comment": "message-square",
            "documentation": "book",
            "web": "globe",
            "database": "database"
        }
        return icon_map.get(source_type, "help-circle")  # Default icon for unknown
    
    def _calculate_importance(self, source: ContextSourceVisualizationData) -> float:
        """Calculate importance score for visualization prominence."""
        # Simple importance calculation based on usage and relevance
        importance = (source.usage_count * 0.6) + (source.average_relevance * 0.4)
        return min(1.0, importance / 10.0)  # Normalize to 0-1


class ContextFilteringVisualization:
    """
    Visualizer for context filtering capabilities.
    
    This component generates visualization data for interactive
    filtering of context sources by relevance, type, etc.
    """
    
    def generate_filter_options(self, sources: List[ContextSourceVisualizationData]) -> Dict[str, Any]:
        """
        Generate filter options for context visualization.
        
        Args:
            sources: Context sources to analyze
            
        Returns:
            Filter options visualization data
        """
        # Get unique source types
        source_types = list(set(source.source_type for source in sources))
        
        # Calculate relevance ranges
        relevance_scores = [source.average_relevance for source in sources]
        min_relevance = min(relevance_scores) if relevance_scores else 0.0
        max_relevance = max(relevance_scores) if relevance_scores else 1.0
        
        # Calculate usage ranges
        usage_counts = [source.usage_count for source in sources]
        min_usage = min(usage_counts) if usage_counts else 0
        max_usage = max(usage_counts) if usage_counts else 0
        
        # Create filter options
        return {
            "source_types": [
                {"value": source_type, "label": source_type.capitalize(), "count": sum(1 for s in sources if s.source_type == source_type)}
                for source_type in source_types
            ],
            "relevance_range": {
                "min": min_relevance,
                "max": max_relevance,
                "steps": 10  # Number of steps in slider
            },
            "usage_range": {
                "min": min_usage,
                "max": max_usage,
                "steps": max(5, min(max_usage, 20))  # Adaptive steps
            },
            "steps": list(set(step for source in sources for step in source.steps_referenced))
        }
    
    def apply_filters(self, 
                     sources: List[ContextSourceVisualizationData], 
                     type_filter: Optional[List[str]] = None,
                     min_relevance: float = 0.0,
                     min_usage: int = 0) -> List[ContextSourceVisualizationData]:
        """
        Apply filters to context sources and return visualization data.
        
        Args:
            sources: Context sources to filter
            type_filter: Optional list of source types to include
            min_relevance: Minimum relevance score
            min_usage: Minimum usage count
            
        Returns:
            Filtered context sources
        """
        filtered_sources = []
        
        for source in sources:
            # Apply type filter
            if type_filter and source.source_type not in type_filter:
                continue
            
            # Apply relevance filter
            if source.average_relevance < min_relevance:
                continue
            
            # Apply usage filter
            if source.usage_count < min_usage:
                continue
            
            # Source passed all filters
            filtered_sources.append(source)
        
        return filtered_sources


class ContextInfluenceVisualization:
    """
    Visualizer for context influence on reasoning.
    
    This component analyzes and visualizes how context sources
    influenced the reasoning process across steps.
    """
    
    def generate_influence_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate visualization data for context influence.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Context influence visualization data
        """
        # Analyze influence by step
        step_influence = []
        
        for step in data.steps:
            # Collect references for this step
            references = []
            
            for ref in step.context_references:
                # Find the source for this reference
                source = next((s for s in data.context_sources if s.source_id == ref["source"]), None)
                if source:
                    references.append({
                        "context_id": ref["context_id"],
                        "source": ref["source"],
                        "source_type": source.source_type,
                        "relevance_score": ref["relevance_score"],
                        "usage_type": ref["usage_type"]
                    })
            
            # Add to step influence data
            step_influence.append({
                "step_number": step.step_number,
                "step_type": step.step_type,
                "references": references,
                "reference_count": len(references),
                "average_relevance": sum(ref["relevance_score"] for ref in references) / len(references) if references else 0.0
            })
        
        # Calculate influence by source type
        influence_by_type = {}
        
        for source in data.context_sources:
            if source.source_type not in influence_by_type:
                influence_by_type[source.source_type] = {
                    "type": source.source_type,
                    "total_usage": 0,
                    "sources": [],
                    "step_distribution": {}
                }
            
            influence_by_type[source.source_type]["total_usage"] += source.usage_count
            influence_by_type[source.source_type]["sources"].append(source.source_id)
            
            # Track usage by step
            for step in source.steps_referenced:
                if step not in influence_by_type[source.source_type]["step_distribution"]:
                    influence_by_type[source.source_type]["step_distribution"][step] = 0
                influence_by_type[source.source_type]["step_distribution"][step] += 1
        
        return {
            "trace_id": data.trace_id,
            "step_influence": step_influence,
            "influence_by_type": list(influence_by_type.values()),
            "total_references": sum(source.usage_count for source in data.context_sources)
        }
    
    def generate_source_flow_visualization(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate visualization data for source usage flow across steps.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Source flow visualization data
        """
        nodes = []
        links = []
        
        # Create nodes for steps
        for step in data.steps:
            nodes.append({
                "id": f"step_{step.step_number}",
                "name": f"Step {step.step_number}",
                "type": "step",
                "value": 1  # Base size for steps
            })
        
        # Create nodes for sources
        for source in data.context_sources:
            nodes.append({
                "id": f"source_{source.source_id}",
                "name": source.name,
                "type": "source",
                "source_type": source.source_type,
                "value": source.usage_count  # Size based on usage
            })
            
            # Create links from sources to steps
            for step_number in source.steps_referenced:
                links.append({
                    "source": f"source_{source.source_id}",
                    "target": f"step_{step_number}",
                    "value": 1  # We could weight this based on relevance
                })
        
        return {
            "nodes": nodes,
            "links": links
        }


class ContextVisualizer:
    """
    Main visualizer for context sources.
    
    This class integrates various context visualization components
    to provide a comprehensive context visualization system.
    """
    
    def __init__(self):
        """Initialize the context visualizer."""
        self.highlighting_service = ContextHighlightingService()
        self.source_attribution = SourceAttributionVisualization()
        self.filtering = ContextFilteringVisualization()
        self.influence = ContextInfluenceVisualization()
    
    def generate_visualization_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate complete visualization data for context sources.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Complete context visualization data
        """
        # Generate source attributions
        attributions = [self.source_attribution.generate_source_attribution(source)
                        for source in data.context_sources]
        
        # Generate influence data
        influence_data = self.influence.generate_influence_data(data)
        
        # Generate filter options
        filter_options = self.filtering.generate_filter_options(data.context_sources)
        
        # Generate source flow
        source_flow = self.influence.generate_source_flow_visualization(data)
        
        # Create complete context visualization data
        result = {
            "trace_id": data.trace_id,
            "source_count": len(data.context_sources),
            "sources": attributions,
            "filter_options": filter_options,
            "influence": influence_data,
            "source_flow": source_flow
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
    
    def visualize_source(self, source: ContextSourceVisualizationData) -> Dict[str, Any]:
        """
        Generate visualization for a specific context source.
        
        Args:
            source: Source to visualize
            
        Returns:
            Source visualization data
        """
        return self.source_attribution.generate_source_attribution(source)
    
    def generate_text_highlights(self, text: str, relevance_score: float, usage_type: str) -> Dict[str, Any]:
        """
        Generate text highlights for a context item.
        
        Args:
            text: Text content to highlight
            relevance_score: Overall relevance score
            usage_type: How the context was used
            
        Returns:
            Highlight visualization data
        """
        return self.highlighting_service.generate_highlights(text, relevance_score, usage_type) 