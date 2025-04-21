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
Evaluation Dashboard Component

This module provides a comprehensive dashboard for visualizing and analyzing
the performance of reasoning systems, including historical performance tracking,
example comparison, and filtering by task type and complexity.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# BeeAI Imports
from beeai_framework.visualization.components.reasoning_trace_visualizer import (
    ReasoningTrace, ReasoningStep, ReasoningTraceVisualizer
)
from beeai_framework.visualization.components.reasoning_quality_metrics import (
    ReasoningQualityMetrics
)
from beeai_framework.visualization.components.context_usage_analytics import (
    ContextUsageAnalytics
)
from beeai_framework.visualization.core.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the evaluation dashboard."""
    
    title: str = "Reasoning Evaluation Dashboard"
    port: int = 8050
    debug: bool = False
    theme: str = "plotly"
    cache_dir: str = "./dashboard_cache"
    max_traces: int = 100
    enable_export: bool = True


class EvaluationDashboard:
    """
    Dashboard for visualizing and analyzing reasoning system performance.
    
    This dashboard provides a comprehensive view of reasoning quality, context
    usage, and system performance metrics, with interactive filtering,
    historical tracking, and export capabilities.
    """
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        trace_visualizer: Optional[ReasoningTraceVisualizer] = None,
        quality_metrics: Optional[ReasoningQualityMetrics] = None,
        context_analytics: Optional[ContextUsageAnalytics] = None
    ):
        """
        Initialize the evaluation dashboard.
        
        Args:
            config: Dashboard configuration
            trace_visualizer: Visualizer for reasoning traces
            quality_metrics: Metrics for reasoning quality
            context_analytics: Analytics for context usage
        """
        self.config = config or DashboardConfig()
        
        # Initialize component dependencies
        self.trace_visualizer = trace_visualizer or ReasoningTraceVisualizer()
        self.quality_metrics = quality_metrics or ReasoningQualityMetrics()
        self.context_analytics = context_analytics or ContextUsageAnalytics()
        
        # Create cache directory if needed
        if self.config.enable_export:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Initialize trace store
        self.traces: Dict[str, ReasoningTrace] = {}
        self.trace_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize dashboard application
        self.app = dash.Dash(
            __name__,
            title=self.config.title,
            external_stylesheets=[f"https://cdn.jsdelivr.net/npm/bootswatch@5.3.1/dist/{self.config.theme}/bootstrap.min.css"]
        )
        
        # Set up dashboard layout
        self._setup_layout()
        
        # Set up dashboard callbacks
        self._setup_callbacks()
    
    def add_trace(
        self, 
        trace: ReasoningTrace,
        compute_metrics: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a reasoning trace to the dashboard.
        
        Args:
            trace: Reasoning trace to add
            compute_metrics: Whether to compute metrics immediately
            metadata: Additional metadata for the trace
        """
        trace_id = trace.trace_id
        
        # Store trace
        self.traces[trace_id] = trace
        
        # Initialize or update metadata
        if trace_id not in self.trace_metadata:
            self.trace_metadata[trace_id] = {
                "added_at": datetime.now().isoformat(),
                "task_type": "unknown",
                "complexity": "medium",
                "tags": [],
                "baseline": False
            }
        
        # Update with provided metadata
        if metadata:
            self.trace_metadata[trace_id].update(metadata)
        
        # Compute metrics if requested
        if compute_metrics:
            # Get baseline traces for comparative metrics
            baseline_traces = []
            for tid, meta in self.trace_metadata.items():
                if meta.get("baseline", False) and tid in self.traces:
                    baseline_traces.append(self.traces[tid])
            
            # Compute quality metrics
            self.quality_metrics.calculate_all_metrics(
                trace=trace,
                baseline_traces=baseline_traces
            )
            
            # Compute context usage metrics
            self.context_analytics.analyze_all_metrics(trace)
        
        # Limit number of traces if needed
        if len(self.traces) > self.config.max_traces:
            # Remove oldest trace
            oldest_id = min(
                self.trace_metadata,
                key=lambda tid: self.trace_metadata[tid]["added_at"]
            )
            del self.traces[oldest_id]
            del self.trace_metadata[oldest_id]
        
        logger.info(f"Added trace {trace_id} to dashboard with {len(trace.steps)} steps")
    
    def export_trace_to_file(self, trace_id: str) -> Optional[str]:
        """
        Export a trace to a JSON file.
        
        Args:
            trace_id: ID of the trace to export
            
        Returns:
            Path to the exported file or None if export failed
        """
        if not self.config.enable_export:
            logger.warning("Export is disabled in dashboard configuration")
            return None
            
        if trace_id not in self.traces:
            logger.error(f"Trace {trace_id} not found")
            return None
        
        # Get trace
        trace = self.traces[trace_id]
        
        # Create export file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = Path(self.config.cache_dir) / f"trace_{trace_id}_{timestamp}.json"
        
        # Export trace
        try:
            with open(export_path, 'w') as f:
                json.dump(trace.to_dict(), f, indent=2)
            
            logger.info(f"Exported trace {trace_id} to {export_path}")
            return str(export_path)
        except Exception as e:
            logger.error(f"Error exporting trace: {str(e)}")
            return None
    
    def import_trace_from_file(self, file_path: str) -> Optional[str]:
        """
        Import a trace from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            ID of the imported trace or None if import failed
        """
        try:
            trace = self.trace_visualizer.import_trace_from_json(file_path)
            if not trace:
                return None
                
            # Add trace to dashboard
            self.add_trace(trace)
            
            return trace.trace_id
        except Exception as e:
            logger.error(f"Error importing trace: {str(e)}")
            return None
    
    def run_server(self, **kwargs) -> None:
        """
        Run the dashboard server.
        
        Args:
            **kwargs: Additional keyword arguments for Dash run_server
        """
        # Default parameters
        server_kwargs = {
            "port": self.config.port,
            "debug": self.config.debug
        }
        
        # Override with provided kwargs
        server_kwargs.update(kwargs)
        
        # Run server
        logger.info(f"Starting dashboard server on port {server_kwargs['port']}")
        self.app.run_server(**server_kwargs)
    
    def _setup_layout(self) -> None:
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(self.config.title, className="display-4"),
                html.P("Interactive visualization and analysis of reasoning system performance", 
                       className="lead")
            ], className="container my-4"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.H4("Filters and Controls"),
                    html.Div([
                        html.Label("Select Trace"),
                        dcc.Dropdown(id="trace-selector", placeholder="Select a trace...")
                    ], className="mb-3"),
                    
                    html.Div([
                        html.Label("Task Type"),
                        dcc.Dropdown(
                            id="task-type-filter",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "Planning", "value": "planning"},
                                {"label": "Research", "value": "research"},
                                {"label": "Analysis", "value": "analysis"},
                                {"label": "Coding", "value": "coding"},
                                {"label": "Debugging", "value": "debugging"}
                            ],
                            value="all"
                        )
                    ], className="mb-3"),
                    
                    html.Div([
                        html.Label("Complexity"),
                        dcc.Dropdown(
                            id="complexity-filter",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "Simple", "value": "simple"},
                                {"label": "Medium", "value": "medium"},
                                {"label": "Complex", "value": "complex"}
                            ],
                            value="all"
                        )
                    ], className="mb-3"),
                    
                    html.Div([
                        html.Label("Date Range"),
                        dcc.DatePickerRange(
                            id="date-range-filter",
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date"
                        )
                    ], className="mb-3"),
                    
                    html.Div([
                        html.Button("Export Results", id="export-button", className="btn btn-primary me-2"),
                        html.Button("Import Trace", id="import-button", className="btn btn-secondary")
                    ], className="mt-4")
                ], className="col-3"),
                
                # Main Content Area
                html.Div([
                    # Summary Metrics
                    html.Div([
                        html.H4("Performance Summary"),
                        html.Div(id="metrics-summary", className="row")
                    ], className="mb-4"),
                    
                    # Tabs for different visualizations
                    dcc.Tabs([
                        dcc.Tab(label="Reasoning Trace", children=[
                            html.Div(id="reasoning-trace-container", className="mt-3")
                        ]),
                        dcc.Tab(label="Quality Metrics", children=[
                            html.Div(id="quality-metrics-container", className="mt-3")
                        ]),
                        dcc.Tab(label="Context Usage", children=[
                            html.Div(id="context-usage-container", className="mt-3")
                        ]),
                        dcc.Tab(label="Comparative Analysis", children=[
                            html.Div(id="comparative-analysis-container", className="mt-3")
                        ]),
                        dcc.Tab(label="Historical Trends", children=[
                            html.Div(id="historical-trends-container", className="mt-3")
                        ])
                    ])
                ], className="col-9")
            ], className="container row")
        ], className="container-fluid")
    
    def _setup_callbacks(self) -> None:
        """Set up the dashboard callbacks."""
        # Update trace selector options
        @self.app.callback(
            Output("trace-selector", "options"),
            [
                Input("task-type-filter", "value"),
                Input("complexity-filter", "value"),
                Input("date-range-filter", "start_date"),
                Input("date-range-filter", "end_date")
            ]
        )
        def update_trace_selector(task_type, complexity, start_date, end_date):
            # Filter traces based on selection
            filtered_traces = self._filter_traces(task_type, complexity, start_date, end_date)
            
            # Create options
            options = []
            for trace_id in filtered_traces:
                trace = self.traces[trace_id]
                options.append({
                    "label": f"{trace.task[:30]}... (ID: {trace_id[:8]})",
                    "value": trace_id
                })
            
            return options
        
        # Update metrics summary
        @self.app.callback(
            Output("metrics-summary", "children"),
            [Input("trace-selector", "value")]
        )
        def update_metrics_summary(trace_id):
            if not trace_id or trace_id not in self.traces:
                return [html.P("Select a trace to view metrics")]
                
            # Get trace
            trace = self.traces[trace_id]
            
            # Get metrics
            quality_metrics = self.quality_metrics.calculate_all_metrics(trace)
            context_stats = self.context_analytics.analyze_all_metrics(trace)
            
            # Create summary cards
            summary_cards = []
            
            # Quality score
            if "trace_metrics" in quality_metrics:
                avg_quality = sum(quality_metrics["trace_metrics"].values()) / len(quality_metrics["trace_metrics"])
                summary_cards.append(
                    html.Div([
                        html.H5("Quality Score"),
                        html.P(f"{avg_quality:.2f}", className="display-6 text-primary")
                    ], className="col-md-3 border rounded p-3 m-2")
                )
            
            # Context relevance
            summary_cards.append(
                html.Div([
                    html.H5("Avg Context Relevance"),
                    html.P(f"{context_stats.avg_relevance:.2f}", className="display-6 text-success")
                ], className="col-md-3 border rounded p-3 m-2")
            )
            
            # Step count
            summary_cards.append(
                html.Div([
                    html.H5("Reasoning Steps"),
                    html.P(f"{len(trace.steps)}", className="display-6 text-info")
                ], className="col-md-3 border rounded p-3 m-2")
            )
            
            # Token usage
            summary_cards.append(
                html.Div([
                    html.H5("Total Tokens"),
                    html.P(f"{context_stats.total_tokens:,}", className="display-6 text-warning")
                ], className="col-md-3 border rounded p-3 m-2")
            )
            
            return summary_cards
        
        # Update reasoning trace visualization
        @self.app.callback(
            Output("reasoning-trace-container", "children"),
            [Input("trace-selector", "value")]
        )
        def update_reasoning_trace(trace_id):
            if not trace_id or trace_id not in self.traces:
                return [html.P("Select a trace to view visualization")]
                
            # Get trace
            trace = self.traces[trace_id]
            
            # Create visualizations
            step_viz = self.trace_visualizer.create_step_visualization(trace)
            context_viz = self.trace_visualizer.create_context_relevance_visualization(trace)
            timeline_viz = self.trace_visualizer.create_context_evolution_timeline(trace)
            graph_viz = self.trace_visualizer.create_knowledge_graph_visualization(trace)
            
            # Create container
            return [
                html.Div([
                    html.H5("Reasoning Steps"),
                    dcc.Graph(figure=step_viz)
                ], className="mb-4"),
                
                html.Div([
                    html.H5("Context Relevance"),
                    dcc.Graph(figure=context_viz)
                ], className="mb-4"),
                
                html.Div([
                    html.H5("Context Evolution Timeline"),
                    dcc.Graph(figure=timeline_viz)
                ], className="mb-4"),
                
                html.Div([
                    html.H5("Knowledge Graph"),
                    dcc.Graph(figure=graph_viz)
                ])
            ]
        
        # Update quality metrics visualization
        @self.app.callback(
            Output("quality-metrics-container", "children"),
            [Input("trace-selector", "value")]
        )
        def update_quality_metrics(trace_id):
            if not trace_id or trace_id not in self.traces:
                return [html.P("Select a trace to view quality metrics")]
                
            # Get trace
            trace = self.traces[trace_id]
            
            # Get metrics
            metrics = self.quality_metrics.calculate_all_metrics(trace)
            
            # Create visualizations based on metrics
            step_metrics = metrics.get("step_metrics", {})
            trace_metrics = metrics.get("trace_metrics", {})
            
            # Step-level metrics chart
            if step_metrics:
                step_data = {}
                for step_key, step_metric in step_metrics.items():
                    step_num = int(step_key.split("_")[1])
                    for metric_name, value in step_metric.items():
                        if metric_name not in step_data:
                            step_data[metric_name] = {}
                        step_data[metric_name][step_num] = value
                
                step_fig = go.Figure()
                for metric_name, values in step_data.items():
                    steps = list(values.keys())
                    metric_values = list(values.values())
                    step_fig.add_trace(go.Scatter(
                        x=[f"Step {s}" for s in steps],
                        y=metric_values,
                        mode="lines+markers",
                        name=metric_name.replace("_", " ").title()
                    ))
                
                step_fig.update_layout(
                    title="Step-level Quality Metrics",
                    xaxis_title="Reasoning Steps",
                    yaxis_title="Metric Value",
                    yaxis=dict(range=[0, 1])
                )
                
                # Trace-level metrics chart
                trace_fig = go.Figure(data=[
                    go.Bar(
                        x=list(trace_metrics.keys()),
                        y=list(trace_metrics.values()),
                        marker_color="#4389EA"
                    )
                ])
                
                trace_fig.update_layout(
                    title="Trace-level Quality Metrics",
                    xaxis_title="Metrics",
                    yaxis_title="Value",
                    yaxis=dict(range=[0, 1])
                )
                
                return [
                    html.Div([
                        html.H5("Step-level Metrics"),
                        dcc.Graph(figure=step_fig)
                    ], className="mb-4"),
                    
                    html.Div([
                        html.H5("Trace-level Metrics"),
                        dcc.Graph(figure=trace_fig)
                    ])
                ]
            else:
                return [html.P("No metrics data available for this trace")]
        
        # Update context usage visualization
        @self.app.callback(
            Output("context-usage-container", "children"),
            [Input("trace-selector", "value")]
        )
        def update_context_usage(trace_id):
            if not trace_id or trace_id not in self.traces:
                return [html.P("Select a trace to view context usage")]
                
            # Get trace
            trace = self.traces[trace_id]
            
            # Create visualizations
            token_chart = self.context_analytics.create_token_usage_chart(trace)
            source_chart = self.context_analytics.create_knowledge_source_chart(trace)
            relevance_heatmap = self.context_analytics.create_context_relevance_heatmap(trace)
            density_chart = self.context_analytics.create_information_density_chart(trace)
            
            return [
                html.Div([
                    html.H5("Token Usage"),
                    dcc.Graph(figure=token_chart)
                ], className="mb-4"),
                
                html.Div([
                    html.H5("Knowledge Sources"),
                    dcc.Graph(figure=source_chart)
                ], className="mb-4"),
                
                html.Div([
                    html.H5("Context Relevance Heatmap"),
                    dcc.Graph(figure=relevance_heatmap)
                ], className="mb-4"),
                
                html.Div([
                    html.H5("Information Density"),
                    dcc.Graph(figure=density_chart)
                ])
            ]
    
    def _filter_traces(
        self,
        task_type: str,
        complexity: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[str]:
        """
        Filter traces based on criteria.
        
        Args:
            task_type: Task type filter
            complexity: Complexity filter
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            List of trace IDs matching the filters
        """
        filtered_ids = []
        
        for trace_id, metadata in self.trace_metadata.items():
            # Skip if trace not in store
            if trace_id not in self.traces:
                continue
                
            # Apply task type filter
            if task_type != "all" and metadata.get("task_type") != task_type:
                continue
                
            # Apply complexity filter
            if complexity != "all" and metadata.get("complexity") != complexity:
                continue
                
            # Apply date filters if specified
            added_at = datetime.fromisoformat(metadata.get("added_at", datetime.now().isoformat()))
            
            if start_date:
                start = datetime.fromisoformat(start_date)
                if added_at < start:
                    continue
                    
            if end_date:
                end = datetime.fromisoformat(end_date)
                if added_at > end:
                    continue
            
            # Add to filtered list
            filtered_ids.append(trace_id)
        
        return filtered_ids 