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
Context Usage Analytics Component

This module provides tools for analyzing how context is used during
reasoning processes, including token usage tracking, knowledge source
utilization metrics, context relevance heatmaps, information density
metrics, and overlap/redundancy detection.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Counter as CounterType
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import Counter

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# NLP libraries for tokenization
import tiktoken

# BeeAI Imports
from beeai_framework.vector.sequential_thinking_integration import SequentialKnowledgeIntegration
from beeai_framework.vector.knowledge_retrieval import StepContextManager
from beeai_framework.visualization.components.reasoning_trace_visualizer import (
    ReasoningTrace, ReasoningStep
)
from beeai_framework.visualization.core.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


@dataclass
class ContextUsageStats:
    """Statistics about context usage within a reasoning trace."""
    
    # Token usage stats
    total_tokens: int = 0
    tokens_per_step: Dict[int, int] = field(default_factory=dict)
    
    # Knowledge source stats
    source_usage: Dict[str, int] = field(default_factory=dict)
    level_usage: Dict[str, int] = field(default_factory=dict)
    
    # Relevance stats
    relevance_scores: List[float] = field(default_factory=list)
    avg_relevance: float = 0.0
    
    # Information density
    info_density_scores: Dict[int, float] = field(default_factory=dict)
    avg_info_density: float = 0.0
    
    # Overlap stats
    overlap_ratio: float = 0.0
    unique_context_ratio: float = 0.0


class ContextUsageAnalytics(BaseVisualizer):
    """
    Analytics system for context usage in reasoning processes.
    
    This component provides tools for analyzing how context is used during
    reasoning, including token usage tracking, knowledge source utilization,
    context relevance heatmaps, information density metrics, and overlap detection.
    """
    
    def __init__(
        self, 
        default_height: int = 600,
        default_width: int = 800,
        tokenizer_name: str = "cl100k_base",
        cache_enabled: bool = True
    ):
        """
        Initialize the context usage analytics system.
        
        Args:
            default_height: Default height for visualizations
            default_width: Default width for visualizations
            tokenizer_name: Name of the tokenizer to use for token counting
            cache_enabled: Whether to cache analysis results
        """
        super().__init__(default_height=default_height, default_width=default_width)
        
        # Set up tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # Cache for analysis results
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, ContextUsageStats] = {}
    
    def analyze_token_usage(
        self, 
        trace: ReasoningTrace,
        include_prompts: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze token usage across a reasoning trace.
        
        Args:
            trace: Reasoning trace to analyze
            include_prompts: Whether to include system/user prompts in token count
            
        Returns:
            Dictionary of token usage statistics
        """
        # Initialize token stats
        token_stats = {
            "total_tokens": 0,
            "tokens_per_step": {},
            "tokens_by_source": {},
            "tokens_by_level": {},
            "tokens_in_steps": 0,
            "tokens_in_context": 0
        }
        
        # Process each step in the trace
        for step in trace.steps:
            # Count tokens in step content
            step_tokens = len(self.tokenizer.encode(step.content))
            token_stats["tokens_in_steps"] += step_tokens
            token_stats["tokens_per_step"][step.step_number] = step_tokens
            
            # Count tokens in context items
            step_context_tokens = 0
            for item in step.context_items:
                content = item.get("content", "")
                if not content:
                    continue
                    
                # Count tokens in this context item
                item_tokens = len(self.tokenizer.encode(content))
                step_context_tokens += item_tokens
                
                # Track by source
                source = item.get("metadata", {}).get("source", "unknown")
                if source not in token_stats["tokens_by_source"]:
                    token_stats["tokens_by_source"][source] = 0
                token_stats["tokens_by_source"][source] += item_tokens
                
                # Track by level
                level = item.get("metadata", {}).get("level", "unknown")
                if level not in token_stats["tokens_by_level"]:
                    token_stats["tokens_by_level"][level] = 0
                token_stats["tokens_by_level"][level] += item_tokens
            
            token_stats["tokens_in_context"] += step_context_tokens
        
        # Calculate total tokens
        token_stats["total_tokens"] = token_stats["tokens_in_steps"] + token_stats["tokens_in_context"]
        
        return token_stats
    
    def analyze_knowledge_source_utilization(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """
        Analyze the utilization of different knowledge sources in a reasoning trace.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            Dictionary of knowledge source utilization statistics
        """
        # Initialize source stats
        source_stats = {
            "source_counts": Counter(),
            "level_counts": Counter(),
            "source_by_step": {},
            "level_by_step": {}
        }
        
        # Process each step in the trace
        for step in trace.steps:
            step_sources = Counter()
            step_levels = Counter()
            
            for item in step.context_items:
                metadata = item.get("metadata", {})
                
                # Track source
                source = metadata.get("source", "unknown")
                step_sources[source] += 1
                source_stats["source_counts"][source] += 1
                
                # Track level
                level = metadata.get("level", "unknown")
                step_levels[level] += 1
                source_stats["level_counts"][level] += 1
            
            # Store stats for this step
            source_stats["source_by_step"][step.step_number] = dict(step_sources)
            source_stats["level_by_step"][step.step_number] = dict(step_levels)
        
        # Calculate percentage distributions
        total_sources = sum(source_stats["source_counts"].values())
        total_levels = sum(source_stats["level_counts"].values())
        
        if total_sources > 0:
            source_stats["source_percentages"] = {
                source: (count / total_sources * 100) 
                for source, count in source_stats["source_counts"].items()
            }
        else:
            source_stats["source_percentages"] = {}
            
        if total_levels > 0:
            source_stats["level_percentages"] = {
                level: (count / total_levels * 100)
                for level, count in source_stats["level_counts"].items()
            }
        else:
            source_stats["level_percentages"] = {}
        
        return source_stats
    
    def create_context_relevance_heatmap(
        self, 
        trace: ReasoningTrace,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Create a heatmap visualization of context relevance across reasoning steps.
        
        Args:
            trace: Reasoning trace to visualize
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with context relevance heatmap
        """
        # Prepare data for heatmap
        steps = []
        sources = set()
        data = []
        
        # Extract data
        for step in trace.steps:
            steps.append(f"Step {step.step_number}")
            
            step_data = {}
            for item in step.context_items:
                source = item.get("metadata", {}).get("source", "unknown")
                similarity = item.get("similarity", 0.0)
                
                # Aggregate by source (taking highest similarity if multiple)
                if source not in step_data or similarity > step_data[source]:
                    step_data[source] = similarity
                
                sources.add(source)
            
            data.append(step_data)
        
        # Convert to matrix format for heatmap
        sources = sorted(list(sources))
        z = []
        for source in sources:
            row = [step_data.get(source, 0.0) for step_data in data]
            z.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=steps,
            y=sources,
            colorscale="Viridis",
            colorbar=dict(
                title="Relevance",
                titleside="right",
                ticks="outside"
            )
        ))
        
        # Update layout
        fig.update_layout(
            title="Context Relevance by Source Across Steps",
            title_x=0.5,
            height=height or self.default_height,
            width=width or self.default_width,
            xaxis_title="Reasoning Steps",
            yaxis_title="Knowledge Sources"
        )
        
        return fig
    
    def calculate_information_density(self, trace: ReasoningTrace) -> Dict[int, float]:
        """
        Calculate information density for context used in each reasoning step.
        
        Information density is a measure of how much unique, non-redundant
        information is contained in the context for each step.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            Dictionary mapping step numbers to information density scores
        """
        density_scores = {}
        
        for step in trace.steps:
            step_number = step.step_number
            
            # Skip if no context items
            if not step.context_items:
                density_scores[step_number] = 0.0
                continue
            
            # Get all context content for this step
            context_texts = [item.get("content", "") for item in step.context_items]
            context_texts = [text for text in context_texts if text]
            
            if not context_texts:
                density_scores[step_number] = 0.0
                continue
            
            # Tokenize all contexts
            all_tokens = []
            for text in context_texts:
                all_tokens.extend(self.tokenizer.encode(text))
            
            # Count unique tokens
            unique_tokens = len(set(all_tokens))
            total_tokens = len(all_tokens)
            
            # Calculate density as ratio of unique to total tokens
            if total_tokens > 0:
                density = unique_tokens / total_tokens
            else:
                density = 0.0
            
            density_scores[step_number] = density
        
        return density_scores
    
    def detect_context_overlap(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """
        Detect and analyze overlap and redundancy in context across steps.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            Dictionary with overlap and redundancy metrics
        """
        overlap_stats = {
            "overlap_by_step_pair": {},
            "redundant_context_items": [],
            "unique_contexts_ratio": 0.0,
            "total_overlap_ratio": 0.0
        }
        
        # Extract context from all steps
        all_contexts = []
        context_by_step = {}
        
        for step in trace.steps:
            step_contexts = []
            for item in step.context_items:
                content = item.get("content", "")
                if not content:
                    continue
                    
                # Store tokenized context
                tokens = self.tokenizer.encode(content)
                token_set = set(tokens)
                
                context_info = {
                    "step": step.step_number,
                    "content": content,
                    "tokens": tokens,
                    "token_set": token_set,
                    "metadata": item.get("metadata", {})
                }
                
                step_contexts.append(context_info)
                all_contexts.append(context_info)
            
            context_by_step[step.step_number] = step_contexts
        
        # Calculate pairwise overlap between steps
        for i in range(len(trace.steps)):
            step_i = trace.steps[i].step_number
            contexts_i = context_by_step.get(step_i, [])
            
            for j in range(i+1, len(trace.steps)):
                step_j = trace.steps[j].step_number
                contexts_j = context_by_step.get(step_j, [])
                
                # Skip if either step has no context
                if not contexts_i or not contexts_j:
                    continue
                
                # Calculate token overlap between steps
                overlap_ratio = self._calculate_token_overlap(contexts_i, contexts_j)
                
                # Store overlap
                pair_key = f"{step_i}-{step_j}"
                overlap_stats["overlap_by_step_pair"][pair_key] = overlap_ratio
        
        # Identify redundant context items (high similarity with other items)
        seen_token_sets = []
        unique_contexts = []
        redundant_count = 0
        
        for context in all_contexts:
            # Check for high overlap with any previous context
            is_redundant = False
            for seen_tokens in seen_token_sets:
                overlap = len(context["token_set"].intersection(seen_tokens)) / len(context["token_set"])
                if overlap > 0.7:  # Threshold for redundancy
                    is_redundant = True
                    redundant_count += 1
                    
                    # Add to redundant items list
                    overlap_stats["redundant_context_items"].append({
                        "step": context["step"],
                        "content_preview": context["content"][:100] + "..." if len(context["content"]) > 100 else context["content"],
                        "source": context["metadata"].get("source", "unknown"),
                        "overlap_ratio": overlap
                    })
                    
                    break
            
            if not is_redundant:
                unique_contexts.append(context)
                seen_token_sets.append(context["token_set"])
        
        # Calculate overall metrics
        if all_contexts:
            overlap_stats["unique_contexts_ratio"] = len(unique_contexts) / len(all_contexts)
            overlap_stats["total_overlap_ratio"] = redundant_count / len(all_contexts)
        
        return overlap_stats
    
    def create_token_usage_chart(
        self,
        trace: ReasoningTrace,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Create a chart visualizing token usage per reasoning step.
        
        Args:
            trace: Reasoning trace to visualize
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with token usage chart
        """
        # Get token usage stats
        token_stats = self.analyze_token_usage(trace)
        
        # Prepare data for visualization
        steps = []
        step_tokens = []
        context_tokens = []
        
        for step in trace.steps:
            step_num = step.step_number
            steps.append(f"Step {step_num}")
            step_tokens.append(token_stats["tokens_per_step"].get(step_num, 0))
            
            # Count context tokens for this step
            ctx_tokens = 0
            for item in step.context_items:
                content = item.get("content", "")
                if content:
                    ctx_tokens += len(self.tokenizer.encode(content))
            
            context_tokens.append(ctx_tokens)
        
        # Create stacked bar chart
        fig = go.Figure(data=[
            go.Bar(
                name="Step Content",
                x=steps,
                y=step_tokens,
                marker_color="#4389EA"
            ),
            go.Bar(
                name="Context Content",
                x=steps,
                y=context_tokens,
                marker_color="#7AC36A"
            )
        ])
        
        # Stack the bars
        fig.update_layout(barmode="stack")
        
        # Add token efficiency line (ratio of step to context tokens)
        token_efficiency = []
        for s, c in zip(step_tokens, context_tokens):
            if c > 0:
                efficiency = s / c
            else:
                efficiency = 0
            token_efficiency.append(min(5, efficiency))  # Cap at 5 for visualization
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=token_efficiency,
            mode="lines+markers",
            name="Token Efficiency",
            line=dict(color="#FD7F0C", width=2),
            yaxis="y2"
        ))
        
        # Update layout
        fig.update_layout(
            title="Token Usage by Reasoning Step",
            title_x=0.5,
            height=height or self.default_height,
            width=width or self.default_width,
            xaxis_title="Reasoning Steps",
            yaxis_title="Token Count",
            yaxis2=dict(
                title="Token Efficiency",
                titlefont=dict(color="#FD7F0C"),
                tickfont=dict(color="#FD7F0C"),
                overlaying="y",
                side="right",
                range=[0, 5]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_knowledge_source_chart(
        self,
        trace: ReasoningTrace,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Create a chart visualizing knowledge source utilization.
        
        Args:
            trace: Reasoning trace to visualize
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with knowledge source utilization chart
        """
        # Get source utilization stats
        source_stats = self.analyze_knowledge_source_utilization(trace)
        
        # Create two subplot figures: sources and levels
        fig = make_subplots(
            rows=1, 
            cols=2,
            subplot_titles=("Knowledge Sources", "Knowledge Levels"),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        # Add source pie chart
        if source_stats["source_counts"]:
            fig.add_trace(
                go.Pie(
                    labels=list(source_stats["source_counts"].keys()),
                    values=list(source_stats["source_counts"].values()),
                    textinfo="label+percent",
                    insidetextorientation="radial"
                ),
                row=1, col=1
            )
        
        # Add level pie chart
        if source_stats["level_counts"]:
            fig.add_trace(
                go.Pie(
                    labels=list(source_stats["level_counts"].keys()),
                    values=list(source_stats["level_counts"].values()),
                    textinfo="label+percent",
                    insidetextorientation="radial"
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Knowledge Source Utilization",
            title_x=0.5,
            height=height or self.default_height,
            width=width or self.default_width
        )
        
        return fig
    
    def create_information_density_chart(
        self,
        trace: ReasoningTrace,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Create a chart visualizing information density across reasoning steps.
        
        Args:
            trace: Reasoning trace to visualize
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with information density chart
        """
        # Calculate information density
        density_scores = self.calculate_information_density(trace)
        
        # Prepare data for visualization
        steps = [f"Step {step}" for step in sorted(density_scores.keys())]
        densities = [density_scores[step] for step in sorted(density_scores.keys())]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=steps,
                y=densities,
                marker_color="#4389EA",
                text=[f"{d:.2f}" for d in densities],
                textposition="auto"
            )
        ])
        
        # Add reference line for optimal density
        fig.add_shape(
            type="line",
            x0=steps[0],
            y0=0.8,
            x1=steps[-1],
            y1=0.8,
            line=dict(
                color="red",
                width=2,
                dash="dash"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Information Density by Reasoning Step",
            title_x=0.5,
            height=height or self.default_height,
            width=width or self.default_width,
            xaxis_title="Reasoning Steps",
            yaxis_title="Information Density",
            yaxis=dict(
                range=[0, 1]
            ),
            annotations=[
                dict(
                    x=steps[-1],
                    y=0.8,
                    xshift=10,
                    text="Optimal Density",
                    showarrow=False,
                    font=dict(color="red")
                )
            ]
        )
        
        return fig
    
    def analyze_all_metrics(self, trace: ReasoningTrace) -> ContextUsageStats:
        """
        Run a complete analysis of all context usage metrics for a trace.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            ContextUsageStats object with all metrics
        """
        # Check cache first
        if self.cache_enabled and trace.trace_id in self.cache:
            return self.cache[trace.trace_id]
        
        # Initialize stats object
        stats = ContextUsageStats()
        
        # Token usage analysis
        token_stats = self.analyze_token_usage(trace)
        stats.total_tokens = token_stats["total_tokens"]
        stats.tokens_per_step = token_stats["tokens_per_step"]
        
        # Knowledge source utilization
        source_stats = self.analyze_knowledge_source_utilization(trace)
        stats.source_usage = dict(source_stats["source_counts"])
        stats.level_usage = dict(source_stats["level_counts"])
        
        # Information density
        info_density = self.calculate_information_density(trace)
        stats.info_density_scores = info_density
        if info_density:
            stats.avg_info_density = sum(info_density.values()) / len(info_density)
        
        # Context overlap
        overlap_stats = self.detect_context_overlap(trace)
        stats.overlap_ratio = overlap_stats["total_overlap_ratio"]
        stats.unique_context_ratio = overlap_stats["unique_contexts_ratio"]
        
        # Relevance scores
        all_relevance = []
        for step in trace.steps:
            for item in step.context_items:
                similarity = item.get("similarity", 0.0)
                all_relevance.append(similarity)
        
        stats.relevance_scores = all_relevance
        if all_relevance:
            stats.avg_relevance = sum(all_relevance) / len(all_relevance)
        
        # Cache results
        if self.cache_enabled:
            self.cache[trace.trace_id] = stats
        
        return stats
    
    def _calculate_token_overlap(
        self, 
        contexts_a: List[Dict[str, Any]], 
        contexts_b: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate token overlap between two sets of context items.
        
        Args:
            contexts_a: First set of context items
            contexts_b: Second set of context items
            
        Returns:
            Overlap ratio between 0.0 and 1.0
        """
        # Combine all tokens from each set
        tokens_a = set()
        for ctx in contexts_a:
            tokens_a.update(ctx["token_set"])
            
        tokens_b = set()
        for ctx in contexts_b:
            tokens_b.update(ctx["token_set"])
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        
        return intersection / union if union > 0 else 0.0 