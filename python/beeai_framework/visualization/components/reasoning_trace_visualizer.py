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
Reasoning Trace Visualization Component

This component provides interactive visualization tools for reasoning traces,
including step exploration, context relevance highlighting, knowledge graph
visualization, and timeline views of context evolution during reasoning.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union, TYPE_CHECKING
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

# Handle type checking to avoid circular imports
if TYPE_CHECKING:
    from beeai_framework.vector.sequential_thinking_integration import SequentialKnowledgeIntegration
    from beeai_framework.vector.knowledge_retrieval import (
        RetrievedKnowledge, KnowledgeRetrievalResult, StepContextManager
    )

# BeeAI Imports
from beeai_framework.visualization.components.knowledge_graph import KnowledgeGraphVisualizer
from beeai_framework.visualization.core.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single step in the reasoning process with all relevant data."""
    
    step_number: int
    step_type: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    context_items: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    key_concepts: List[Dict[str, Any]] = field(default_factory=list)
    next_step_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step data to a dictionary."""
        return {
            "step_number": self.step_number,
            "step_type": self.step_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "context_items": self.context_items,
            "metrics": self.metrics,
            "key_concepts": self.key_concepts,
            "next_step_suggestions": self.next_step_suggestions
        }


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process with multiple steps."""
    
    trace_id: str
    task: str
    steps: List[ReasoningStep] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    overall_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a step to the reasoning trace."""
        self.steps.append(step)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace data to a dictionary."""
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "steps": [step.to_dict() for step in self.steps],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "overall_metrics": self.overall_metrics
        }


class ReasoningTraceVisualizer(BaseVisualizer):
    """
    Visualizer for reasoning traces in sequential thinking processes.
    
    This component provides a comprehensive set of visualization tools
    for exploring and analyzing reasoning traces, including interactive
    step exploration, context relevance highlighting, knowledge graph
    visualization, and timeline views of context evolution.
    """
    
    def __init__(
        self,
        knowledge_integration: Optional[Any] = None,
        context_manager: Optional[Any] = None,
        knowledge_graph_visualizer: Optional[KnowledgeGraphVisualizer] = None,
        cache_traces: bool = True,
        default_height: int = 600,
        default_width: int = 800
    ):
        """
        Initialize the reasoning trace visualizer.
        
        Args:
            knowledge_integration: Integration between Sequential Thinking and Knowledge Retrieval
            context_manager: Manager for context across reasoning steps
            knowledge_graph_visualizer: Visualizer for knowledge graphs
            cache_traces: Whether to cache traces in memory
            default_height: Default height for visualizations
            default_width: Default width for visualizations
        """
        super().__init__(default_height=default_height, default_width=default_width)
        self.knowledge_integration = knowledge_integration
        self.context_manager = context_manager
        self.knowledge_graph_visualizer = knowledge_graph_visualizer or KnowledgeGraphVisualizer()
        
        # Cache for reasoning traces
        self.cache_traces = cache_traces
        self.trace_cache: Dict[str, ReasoningTrace] = {}
    
    def create_step_visualization(
        self,
        trace: Union[ReasoningTrace, Dict[str, Any]],
        selected_step: Optional[int] = None,
        highlight_context: bool = True,
        show_metrics: bool = True,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Create an interactive visualization for exploring reasoning steps.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            selected_step: Index of the initially selected step (1-based)
            highlight_context: Whether to highlight context references
            show_metrics: Whether to show step metrics
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with interactive step visualization
        """
        # Convert dictionary to trace if needed
        if isinstance(trace, dict):
            trace_obj = self._dict_to_trace(trace)
        else:
            trace_obj = trace
        
        # Cache trace if enabled
        if self.cache_traces:
            self.trace_cache[trace_obj.trace_id] = trace_obj
        
        # Default to first step if none selected
        if selected_step is None:
            selected_step = 1
        elif selected_step > len(trace_obj.steps):
            selected_step = len(trace_obj.steps)
        
        # Create figure with subplots for step content and metrics
        fig = make_subplots(
            rows=2, 
            cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Reasoning Step Content", "Step Metrics"),
            vertical_spacing=0.1
        )
        
        # Add step content
        step = trace_obj.steps[selected_step - 1]
        step_content = step.content
        
        # Highlight context references if enabled
        if highlight_context and step.context_items:
            step_content = self._highlight_context_references(step_content, step.context_items)
        
        # Add step content as text
        fig.add_trace(
            go.Scatter(
                x=[0], 
                y=[0],
                mode="text",
                text=step_content,
                textposition="top left",
                hoverinfo="none"
            ),
            row=1, col=1
        )
        
        # Add step metrics if enabled
        if show_metrics and step.metrics:
            metrics_df = pd.DataFrame({
                "Metric": list(step.metrics.keys()),
                "Value": list(step.metrics.values())
            })
            
            fig.add_trace(
                go.Bar(
                    x=metrics_df["Metric"],
                    y=metrics_df["Value"],
                    marker=dict(color="#4389EA"),
                    name="Step Metrics"
                ),
                row=2, col=1
            )
        
        # Add navigation buttons for step selection
        step_buttons = []
        for i, s in enumerate(trace_obj.steps):
            step_buttons.append(
                dict(
                    method="update",
                    args=[
                        {"visible": [True] * len(fig.data)},  # Update visibility
                        {"title": f"Step {s.step_number}: {s.step_type.capitalize()}"}  # Update title
                    ],
                    label=f"Step {s.step_number}"
                )
            )
        
        # Add button menu for step navigation
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=step_buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    y=1.15,
                    xanchor="left",
                    yanchor="top"
                )
            ],
            height=height or self.default_height,
            width=width or self.default_width,
            title=f"Step {selected_step}: {step.step_type.capitalize()}",
            title_x=0.5
        )
        
        # Update axis settings
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        return fig
    
    def create_context_relevance_visualization(
        self,
        trace: Union[ReasoningTrace, Dict[str, Any]],
        selected_step: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Create a visualization showing context relevance for reasoning steps.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            selected_step: Index of the step to visualize (1-based)
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with context relevance visualization
        """
        # Convert dictionary to trace if needed
        if isinstance(trace, dict):
            trace_obj = self._dict_to_trace(trace)
        else:
            trace_obj = trace
        
        # Default to first step if none selected
        if selected_step is None:
            selected_step = 1
        elif selected_step > len(trace_obj.steps):
            selected_step = len(trace_obj.steps)
        
        step = trace_obj.steps[selected_step - 1]
        
        # Extract context items and their relevance scores
        context_items = step.context_items
        if not context_items:
            # Create empty visualization if no context items
            fig = go.Figure()
            fig.add_annotation(
                text="No context items available for this step",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                height=height or self.default_height,
                width=width or self.default_width,
                title=f"Context Relevance for Step {selected_step}",
                title_x=0.5
            )
            return fig
        
        # Prepare data for heatmap visualization
        context_texts = []
        context_sources = []
        similarity_scores = []
        
        for item in context_items:
            # Get abbreviated context text (first 50 characters)
            ctx_text = item.get("content", "")
            if len(ctx_text) > 50:
                ctx_text = ctx_text[:47] + "..."
            
            context_texts.append(ctx_text)
            context_sources.append(item.get("metadata", {}).get("source", "unknown"))
            similarity_scores.append(item.get("similarity", 0.0))
        
        # Create figure with context relevance heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[similarity_scores],
            x=context_texts,
            y=["Relevance"],
            colorscale="Viridis",
            hoverongaps=False,
            text=[context_sources],
            hovertemplate="<b>Context:</b> %{x}<br><b>Source:</b> %{text}<br><b>Relevance:</b> %{z:.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            height=height or self.default_height // 2,
            width=width or self.default_width,
            title=f"Context Relevance for Step {selected_step}",
            title_x=0.5,
            xaxis_title="Context Items",
            yaxis_title="",
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_knowledge_graph_visualization(
        self,
        trace: Union[ReasoningTrace, Dict[str, Any]],
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Create a knowledge graph visualization showing concept relationships.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with knowledge graph visualization
        """
        # Convert dictionary to trace if needed
        if isinstance(trace, dict):
            trace_obj = self._dict_to_trace(trace)
        else:
            trace_obj = trace
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Extract all concepts from all steps
        concepts_by_step = {}
        for step in trace_obj.steps:
            step_concepts = []
            for concept in step.key_concepts:
                concept_text = concept.get("concept", "")
                if concept_text:
                    step_concepts.append(concept_text)
                    # Add concept as node if it doesn't exist
                    if not G.has_node(concept_text):
                        G.add_node(concept_text, size=10, group=step.step_type)
            
            concepts_by_step[step.step_number] = step_concepts
        
        # Add edges between concepts in consecutive steps
        for i in range(1, len(trace_obj.steps)):
            current_step = i + 1
            prev_step = i
            
            if prev_step in concepts_by_step and current_step in concepts_by_step:
                prev_concepts = concepts_by_step[prev_step]
                current_concepts = concepts_by_step[current_step]
                
                # Connect concepts between steps
                for prev_concept in prev_concepts:
                    for current_concept in current_concepts:
                        # Simple edge weight based on concept co-occurrence
                        weight = 1.0
                        G.add_edge(prev_concept, current_concept, weight=weight)
        
        # If no edges, add a default node
        if not G.nodes():
            G.add_node("No concepts identified", size=15, group="unknown")
        
        # Use knowledge graph visualizer to create the plot
        fig = self.knowledge_graph_visualizer.visualize_graph(
            G,
            title=f"Knowledge Graph for Reasoning Trace",
            height=height or self.default_height,
            width=width or self.default_width
        )
        
        return fig
    
    def create_context_evolution_timeline(
        self,
        trace: Union[ReasoningTrace, Dict[str, Any]],
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Create a timeline visualization showing context evolution during reasoning.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with context evolution timeline
        """
        # Convert dictionary to trace if needed
        if isinstance(trace, dict):
            trace_obj = self._dict_to_trace(trace)
        else:
            trace_obj = trace
        
        # Prepare data for timeline visualization
        timeline_data = []
        
        for step in trace_obj.steps:
            # Process context items for this step
            for i, context_item in enumerate(step.context_items):
                content = context_item.get("content", "")
                # Truncate content for display
                if len(content) > 60:
                    content = content[:57] + "..."
                
                source = context_item.get("metadata", {}).get("source", "unknown")
                level = context_item.get("metadata", {}).get("level", "unknown")
                similarity = context_item.get("similarity", 0.0)
                
                # Add context item to timeline data
                timeline_data.append({
                    "Step": f"Step {step.step_number}",
                    "Type": step.step_type,
                    "Context": content,
                    "Source": source,
                    "Level": level,
                    "Relevance": similarity,
                    "Order": i  # To maintain order within step
                })
        
        # Create DataFrame from timeline data
        df = pd.DataFrame(timeline_data)
        
        if df.empty:
            # Create empty visualization if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No context evolution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                height=height or self.default_height,
                width=width or self.default_width,
                title="Context Evolution Timeline",
                title_x=0.5
            )
            return fig
        
        # Create timeline visualization
        fig = px.timeline(
            df,
            x_start="Step",
            y="Context",
            color="Relevance",
            color_continuous_scale="Viridis",
            hover_name="Context",
            hover_data=["Source", "Level", "Type", "Relevance"],
            title="Context Evolution Timeline"
        )
        
        # Update layout
        fig.update_layout(
            height=height or self.default_height,
            width=width or self.default_width,
            title="Context Evolution Timeline",
            title_x=0.5,
            xaxis_title="Reasoning Steps",
            yaxis_title="Context Items",
            coloraxis_colorbar=dict(
                title="Relevance Score",
                tickvals=[0.0, 0.5, 1.0],
                ticktext=["Low", "Medium", "High"]
            )
        )
        
        return fig

    def export_trace_to_json(self, trace_id: str, file_path: str) -> bool:
        """
        Export a reasoning trace to a JSON file.
        
        Args:
            trace_id: ID of the trace to export
            file_path: Path to save the JSON file
            
        Returns:
            True if export was successful, False otherwise
        """
        if trace_id not in self.trace_cache:
            logger.error(f"Trace ID {trace_id} not found in cache")
            return False
        
        trace = self.trace_cache[trace_id]
        
        try:
            with open(file_path, 'w') as f:
                json.dump(trace.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error exporting trace to JSON: {str(e)}")
            return False
    
    def import_trace_from_json(self, file_path: str) -> Optional[ReasoningTrace]:
        """
        Import a reasoning trace from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Imported reasoning trace or None if import failed
        """
        try:
            with open(file_path, 'r') as f:
                trace_dict = json.load(f)
            
            trace = self._dict_to_trace(trace_dict)
            
            # Cache trace if enabled
            if self.cache_traces:
                self.trace_cache[trace.trace_id] = trace
                
            return trace
        except Exception as e:
            logger.error(f"Error importing trace from JSON: {str(e)}")
            return None
    
    def _dict_to_trace(self, trace_dict: Dict[str, Any]) -> ReasoningTrace:
        """Convert a dictionary to a ReasoningTrace object."""
        # Create empty trace
        trace = ReasoningTrace(
            trace_id=trace_dict.get("trace_id", "unknown"),
            task=trace_dict.get("task", "unknown"),
            start_time=datetime.fromisoformat(trace_dict.get("start_time", datetime.now().isoformat())),
            overall_metrics=trace_dict.get("overall_metrics", {})
        )
        
        # Set end time if available
        if trace_dict.get("end_time"):
            trace.end_time = datetime.fromisoformat(trace_dict["end_time"])
        
        # Add steps
        for step_dict in trace_dict.get("steps", []):
            step = ReasoningStep(
                step_number=step_dict.get("step_number", 0),
                step_type=step_dict.get("step_type", "unknown"),
                content=step_dict.get("content", ""),
                timestamp=datetime.fromisoformat(step_dict.get("timestamp", datetime.now().isoformat())),
                context_items=step_dict.get("context_items", []),
                metrics=step_dict.get("metrics", {}),
                key_concepts=step_dict.get("key_concepts", []),
                next_step_suggestions=step_dict.get("next_step_suggestions", [])
            )
            trace.add_step(step)
        
        return trace
    
    def _highlight_context_references(
        self,
        text: str,
        context_items: List[Dict[str, Any]]
    ) -> str:
        """Highlight references to context items in the text."""
        # Simple highlighting approach - find exact substring matches
        highlighted_text = text
        
        for item in context_items:
            content = item.get("content", "")
            if not content or len(content) < 10:
                continue
            
            # Extract key phrases from context (simplified approach)
            phrases = content.split('.')
            for phrase in phrases:
                phrase = phrase.strip()
                if len(phrase) < 10:
                    continue
                
                # Look for this phrase in the text
                if phrase in highlighted_text:
                    # Highlight with markdown-style bold
                    highlighted_text = highlighted_text.replace(phrase, f"**{phrase}**")
        
        return highlighted_text 