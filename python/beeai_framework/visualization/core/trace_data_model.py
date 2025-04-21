#!/usr/bin/env python
"""
Trace Data Model for visualizations.

This module defines the data models and conversion utilities
for reasoning trace visualizations.
"""

import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

class VisualizationMetadata(BaseModel):
    """Metadata for visualization components."""
    title: str = Field(..., description="Title of the visualization")
    description: Optional[str] = Field(None, description="Description of the visualization")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    creator: Optional[str] = Field(None, description="Creator identifier")
    version: str = Field("1.0.0", description="Visualization format version")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    sources: Dict[str, Any] = Field(default_factory=dict, description="Data source information")


class StepVisualizationData(BaseModel):
    """Data structure for step visualization."""
    step_id: str = Field(..., description="Step identifier")
    step_number: int = Field(..., description="Position in sequence")
    title: str = Field(..., description="Step title or summary")
    content: str = Field(..., description="Step content")
    step_type: str = Field(..., description="Type of reasoning step")
    timestamp: float = Field(..., description="Unix timestamp of creation")
    duration: float = Field(0.0, description="Duration in seconds")
    requires_next_step: bool = Field(..., description="Whether another step is needed")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Step-specific metrics")
    context_references: List[Dict[str, Any]] = Field(default_factory=list, description="References to context")
    annotations: Dict[str, Any] = Field(default_factory=dict, description="Visualization annotations")
    
    @property
    def formatted_timestamp(self) -> str:
        """Return formatted timestamp for display."""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    @property
    def content_preview(self) -> str:
        """Return a preview of content for compact displays."""
        max_length = 100
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."


class ContextSourceVisualizationData(BaseModel):
    """Data structure for context source visualization."""
    source_id: str = Field(..., description="Source identifier")
    name: str = Field(..., description="Source name")
    source_type: str = Field(..., description="Type of source")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Context items from this source")
    usage_count: int = Field(0, description="Total usage count across all steps")
    relevance_scores: List[float] = Field(default_factory=list, description="Relevance scores")
    steps_referenced: List[int] = Field(default_factory=list, description="Steps where this source is referenced")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")
    
    @property
    def average_relevance(self) -> float:
        """Calculate the average relevance score."""
        if not self.relevance_scores:
            return 0.0
        return sum(self.relevance_scores) / len(self.relevance_scores)


class KnowledgeGraphNode(BaseModel):
    """Node in the knowledge graph visualization."""
    node_id: str = Field(..., description="Node identifier")
    label: str = Field(..., description="Node label")
    type: str = Field(..., description="Node type")
    weight: float = Field(1.0, description="Node importance weight")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Node attributes")
    steps: List[int] = Field(default_factory=list, description="Steps where this node appears")


class KnowledgeGraphEdge(BaseModel):
    """Edge in the knowledge graph visualization."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: str = Field(..., description="Edge type")
    weight: float = Field(1.0, description="Edge weight")
    directed: bool = Field(True, description="Whether the edge is directed")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Edge attributes")


class KnowledgeGraphVisualizationData(BaseModel):
    """Data structure for knowledge graph visualization."""
    graph_id: str = Field(..., description="Graph identifier")
    nodes: List[KnowledgeGraphNode] = Field(default_factory=list, description="Nodes in the graph")
    edges: List[KnowledgeGraphEdge] = Field(default_factory=list, description="Edges in the graph")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph metadata")


class MetricsVisualizationData(BaseModel):
    """Data structure for metrics visualization."""
    trace_id: str = Field(..., description="Trace identifier")
    trace_metrics: Dict[str, Any] = Field(default_factory=dict, description="Overall trace metrics")
    step_metrics: List[Dict[str, Any]] = Field(default_factory=list, description="Per-step metrics")
    context_metrics: Dict[str, Any] = Field(default_factory=dict, description="Context usage metrics")
    comparison_metrics: Optional[Dict[str, Any]] = Field(None, description="Comparison to baseline or reference")
    time_series: Dict[str, List[Any]] = Field(default_factory=dict, description="Time series metrics data")


class TraceVisualizationData(BaseModel):
    """Complete data structure for trace visualization."""
    trace_id: str = Field(..., description="Trace identifier")
    task: str = Field(..., description="Task description")
    metadata: VisualizationMetadata = Field(..., description="Visualization metadata")
    steps: List[StepVisualizationData] = Field(default_factory=list, description="Step visualization data")
    context_sources: List[ContextSourceVisualizationData] = Field(default_factory=list, description="Context source data")
    knowledge_graph: Optional[KnowledgeGraphVisualizationData] = Field(None, description="Knowledge graph data")
    metrics: Optional[MetricsVisualizationData] = Field(None, description="Metrics visualization data")
    final_result: Optional[str] = Field(None, description="Final result of reasoning process")
    
    def to_json(self, **kwargs) -> str:
        """Convert to JSON representation."""
        return json.dumps(self.dict(), **kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TraceVisualizationData':
        """Create from JSON representation."""
        data = json.loads(json_str)
        return cls(**data)


def convert_reasoning_trace_to_visualization_data(trace: Any) -> TraceVisualizationData:
    """
    Convert a ReasoningTrace instance to visualization data.
    
    Args:
        trace: ReasoningTrace instance
        
    Returns:
        TraceVisualizationData instance
    """
    # Import here to avoid circular imports
    from V5.extensions.middleware.sequential.reasoning_trace import ReasoningTrace
    
    if not isinstance(trace, ReasoningTrace):
        raise TypeError("Expected ReasoningTrace instance")
    
    # Create metadata
    metadata = VisualizationMetadata(
        title=f"Reasoning Trace: {trace.task}",
        description=f"Visualization of reasoning process for task: {trace.task}",
        sources={"original_trace_id": trace.trace_id}
    )
    
    # Convert steps
    steps = []
    for step in trace.steps:
        # Calculate step duration
        duration = 0.0
        next_idx = trace.steps.index(step) + 1
        if next_idx < len(trace.steps):
            duration = trace.steps[next_idx].timestamp - step.timestamp
        
        # Create step visualization data
        steps.append(StepVisualizationData(
            step_id=step.step_id,
            step_number=step.thought_number,
            title=f"Step {step.thought_number}: {step.step_type.capitalize()}",
            content=step.content,
            step_type=step.step_type,
            timestamp=step.timestamp,
            duration=duration,
            requires_next_step=step.requires_next_step,
            metrics=step.metrics,
            context_references=[ref.dict() for ref in step.context_references]
        ))
    
    # Process context sources
    source_map = {}
    for step in trace.steps:
        for ref in step.context_references:
            source_id = ref.source
            if source_id not in source_map:
                source_map[source_id] = {
                    "source_id": source_id,
                    "name": ref.source,
                    "source_type": "unknown",  # We might need to infer this
                    "items": [],
                    "usage_count": 0,
                    "relevance_scores": [],
                    "steps_referenced": []
                }
            
            source_map[source_id]["usage_count"] += 1
            source_map[source_id]["relevance_scores"].append(ref.relevance_score)
            
            if step.thought_number not in source_map[source_id]["steps_referenced"]:
                source_map[source_id]["steps_referenced"].append(step.thought_number)
            
            # Add context item if not already added
            item = {
                "context_id": ref.context_id,
                "relevance_score": ref.relevance_score,
                "usage_type": ref.usage_type
            }
            if item not in source_map[source_id]["items"]:
                source_map[source_id]["items"].append(item)
    
    context_sources = [ContextSourceVisualizationData(**data) for data in source_map.values()]
    
    # Create metrics visualization data
    metrics = MetricsVisualizationData(
        trace_id=trace.trace_id,
        trace_metrics={
            "duration": trace.end_time - trace.start_time if trace.end_time else 0.0,
            "step_count": len(trace.steps),
            "is_complete": trace.end_time is not None
        },
        step_metrics=[
            {
                "step_number": s.thought_number,
                "context_count": len(s.context_references),
                "duration": steps[i].duration,
                **s.metrics
            }
            for i, s in enumerate(trace.steps)
        ],
        context_metrics={
            "total_references": sum(source.usage_count for source in context_sources),
            "sources_count": len(context_sources),
            "average_relevance": sum(source.average_relevance for source in context_sources) / len(context_sources) if context_sources else 0.0
        }
    )
    
    # Create the full visualization data
    return TraceVisualizationData(
        trace_id=trace.trace_id,
        task=trace.task,
        metadata=metadata,
        steps=steps,
        context_sources=context_sources,
        metrics=metrics,
        final_result=trace.final_result
    ) 