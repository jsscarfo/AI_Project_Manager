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
Real-Time Evaluation Dashboard Example

This script demonstrates how to create a real-time dashboard for evaluating
reasoning traces using the BeeAI Framework visualization components.
"""

import os
import json
import time
import random
import logging
import argparse
import threading
import webbrowser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

from beeai_framework.visualization import (
    ReasoningTraceVisualizer,
    ReasoningTrace,
    ReasoningStep,
    ReasoningQualityMetrics,
    ContextUsageAnalytics,
    EvaluationDashboard,
    DashboardConfig,
    QualityMetric,
    MetricLevel
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for simulated data
TRACES: Dict[str, ReasoningTrace] = {}
METRICS_HISTORY: List[Dict[str, Any]] = []
CONTEXT_STATS_HISTORY: List[Dict[str, Any]] = []
ACTIVE_TRACES: List[str] = []
SIMULATION_RUNNING = False


def create_mock_trace(trace_id: str, task: str, num_steps: int = 10) -> ReasoningTrace:
    """
    Create a mock reasoning trace with the specified number of steps.
    
    Args:
        trace_id: Unique trace identifier
        task: Task description
        num_steps: Number of steps to generate
        
    Returns:
        Mock reasoning trace
    """
    start_time = datetime.now() - timedelta(minutes=num_steps * 2)
    
    trace = ReasoningTrace(
        trace_id=trace_id,
        task=task,
        start_time=start_time,
        overall_metrics={
            "completion_time_s": 0,
            "token_usage": 0,
            "total_context_items": 0
        }
    )
    
    # Define possible step types
    step_types = [
        "problem_definition",
        "information_gathering",
        "analysis",
        "solution_formulation",
        "verification"
    ]
    
    return trace


def add_step_to_trace(trace_id: str) -> Optional[ReasoningStep]:
    """
    Add a new step to an existing trace.
    
    Args:
        trace_id: Trace identifier
        
    Returns:
        Newly added step or None if trace not found
    """
    if trace_id not in TRACES:
        logger.warning(f"Trace {trace_id} not found")
        return None
    
    trace = TRACES[trace_id]
    
    # Define possible step types
    step_types = [
        "problem_definition",
        "information_gathering",
        "analysis",
        "solution_formulation",
        "verification"
    ]
    
    # Use appropriate step type based on current step count
    step_number = len(trace.steps) + 1
    step_type = step_types[min(step_number - 1, len(step_types) - 1)]
    
    # Create timestamp (2 minutes after previous step or start time)
    if len(trace.steps) > 0:
        prev_timestamp = trace.steps[-1].timestamp
    else:
        prev_timestamp = trace.start_time
    
    step_time = prev_timestamp + timedelta(minutes=2)
    
    # Create context items
    num_contexts = random.randint(2, 5)
    context_items = []
    
    for j in range(num_contexts):
        similarity = max(0.5, 0.9 - (j * 0.1) + random.uniform(-0.1, 0.1))
        
        context_items.append({
            "content": f"This is context item {j+1} for step {step_number}. " +
                     f"It contains information about {trace.task.split()[j % len(trace.task.split())]}.",
            "similarity": similarity,
            "metadata": {
                "source": f"source_{(step_number+j) % 3 + 1}",
                "level": ["domain", "techstack", "project"][j % 3],
                "timestamp": step_time.isoformat()
            }
        })
    
    # Create step metrics
    confidence = min(0.95, 0.6 + (step_number * 0.05) + random.uniform(-0.1, 0.1))
    relevance = min(0.95, 0.7 + (step_number * 0.02) + random.uniform(-0.1, 0.1))
    
    metrics = {
        "tokens": 100 + (step_number * 20),
        "confidence": confidence,
        "relevance": relevance,
        "novelty": max(0.1, 0.8 - (step_number * 0.05) + random.uniform(-0.1, 0.1))
    }
    
    # Create key concepts
    key_concepts = [
        {"concept": f"concept_{(step_number*2)}", "importance": 0.9},
        {"concept": f"concept_{(step_number*2)+1}", "importance": 0.7}
    ]
    
    # Create suggestions for next steps
    next_steps = [
        f"Explore {step_types[min(step_number, len(step_types)-1)]} in more detail",
        f"Investigate the relationship between {trace.task.split()[0]} and performance"
    ]
    
    # Create step
    step = ReasoningStep(
        step_number=step_number,
        step_type=step_type,
        content=f"Step {step_number} ({step_type}): Analysis of the {trace.task}. " + 
               f"Based on the context information, we can see that this requires understanding of {step_type}. " +
               f"The key aspects to consider are X, Y, and Z.",
        timestamp=step_time,
        context_items=context_items,
        metrics=metrics,
        key_concepts=key_concepts,
        next_step_suggestions=next_steps
    )
    
    # Add step to trace
    trace.add_step(step)
    
    # Update overall metrics
    trace.overall_metrics["completion_time_s"] = (step_time - trace.start_time).total_seconds()
    trace.overall_metrics["token_usage"] += metrics["tokens"]
    trace.overall_metrics["total_context_items"] += len(context_items)
    
    # If this is the last step, set end time
    if step_number >= 10:
        trace.end_time = step_time
        ACTIVE_TRACES.remove(trace_id) if trace_id in ACTIVE_TRACES else None
    
    return step


def analyze_and_log_metrics(trace_id: str) -> Dict[str, Any]:
    """
    Analyze and log metrics for a trace.
    
    Args:
        trace_id: Trace identifier
        
    Returns:
        Dictionary with metrics data
    """
    if trace_id not in TRACES:
        logger.warning(f"Trace {trace_id} not found")
        return {}
    
    trace = TRACES[trace_id]
    
    # Calculate quality metrics
    quality_metrics = ReasoningQualityMetrics()
    metrics_result = quality_metrics.calculate_all_metrics(trace)
    
    # Calculate context usage
    context_analytics = ContextUsageAnalytics()
    usage_stats = context_analytics.analyze_all_metrics(trace)
    
    # Create metrics data
    timestamp = datetime.now()
    metrics_data = {
        "timestamp": timestamp,
        "trace_id": trace_id,
        "step_count": len(trace.steps),
        "coherence": metrics_result["trace_metrics"].get("coherence", 0),
        "goal_alignment": metrics_result["trace_metrics"].get("goal_alignment", 0),
        "completeness": metrics_result["trace_metrics"].get("completeness", 0),
        "quality_score": np.mean([
            metrics_result["trace_metrics"].get("coherence", 0),
            metrics_result["trace_metrics"].get("goal_alignment", 0),
            metrics_result["trace_metrics"].get("completeness", 0)
        ]),
        "token_usage": trace.overall_metrics.get("token_usage", 0),
        "completion_time_s": trace.overall_metrics.get("completion_time_s", 0)
    }
    
    # Append to history
    METRICS_HISTORY.append(metrics_data)
    
    # Create context stats data
    context_data = {
        "timestamp": timestamp,
        "trace_id": trace_id,
        "step_count": len(trace.steps),
        "avg_relevance": usage_stats.avg_relevance,
        "total_context_items": trace.overall_metrics.get("total_context_items", 0),
        "relevant_items_ratio": usage_stats.relevant_items_ratio
    }
    
    # Append to history
    CONTEXT_STATS_HISTORY.append(context_data)
    
    return metrics_data


def start_simulation():
    """Start the real-time simulation by generating traces and steps."""
    global SIMULATION_RUNNING
    SIMULATION_RUNNING = True
    
    # Create three traces
    tasks = [
        "Design a scalable microservice architecture",
        "Implement a secure authentication system",
        "Optimize database performance for high-traffic website"
    ]
    
    for i, task in enumerate(tasks):
        trace_id = f"trace_{i+1}"
        TRACES[trace_id] = create_mock_trace(trace_id, task)
        ACTIVE_TRACES.append(trace_id)
    
    # Start simulation thread
    def simulation_loop():
        while SIMULATION_RUNNING and len(ACTIVE_TRACES) > 0:
            # Select a random active trace
            if not ACTIVE_TRACES:
                break
                
            trace_id = random.choice(ACTIVE_TRACES)
            
            # Add a step to the trace
            add_step_to_trace(trace_id)
            
            # Analyze and log metrics
            analyze_and_log_metrics(trace_id)
            
            # Wait for 1-3 seconds before next update
            time.sleep(random.uniform(1, 3))
    
    # Start simulation thread
    sim_thread = threading.Thread(target=simulation_loop)
    sim_thread.daemon = True
    sim_thread.start()


def stop_simulation():
    """Stop the real-time simulation."""
    global SIMULATION_RUNNING
    SIMULATION_RUNNING = False


def create_dashboard():
    """Create the real-time dashboard."""
    # Create the Dash application
    app = dash.Dash(__name__, title="BeeAI Real-time Evaluation Dashboard")
    
    # Define the layout
    app.layout = html.Div([
        html.H1("BeeAI Real-time Reasoning Evaluation Dashboard", 
                style={'textAlign': 'center', 'margin': '20px'}),
        
        html.Div([
            html.Button("Start Simulation", id="start-btn", className="control-btn"),
            html.Button("Stop Simulation", id="stop-btn", className="control-btn"),
            html.Button("Reset", id="reset-btn", className="control-btn"),
        ], style={'textAlign': 'center', 'margin': '20px'}),
        
        html.Div([
            html.Div([
                html.H3("Active Traces", style={'textAlign': 'center'}),
                html.Div(id="active-traces-container")
            ], className="dashboard-card"),
            
            html.Div([
                html.H3("Quality Metrics Trend", style={'textAlign': 'center'}),
                dcc.Graph(id="quality-metrics-chart")
            ], className="dashboard-card"),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
        
        html.Div([
            html.Div([
                html.H3("Context Usage", style={'textAlign': 'center'}),
                dcc.Graph(id="context-usage-chart")
            ], className="dashboard-card"),
            
            html.Div([
                html.H3("Performance Metrics", style={'textAlign': 'center'}),
                dcc.Graph(id="performance-metrics-chart")
            ], className="dashboard-card"),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
        
        # Interval component for updates
        dcc.Interval(
            id='interval-component',
            interval=1000,  # in milliseconds (1 second)
            n_intervals=0
        ),
        
        # CSS for styling
        html.Style('''
            .dashboard-card {
                width: 45%;
                min-width: 500px;
                margin: 10px;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                background-color: white;
            }
            .control-btn {
                margin: 0 10px;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
            }
            #start-btn {
                background-color: #4CAF50;
                color: white;
            }
            #stop-btn {
                background-color: #f44336;
                color: white;
            }
            #reset-btn {
                background-color: #2196F3;
                color: white;
            }
            body {
                background-color: #f5f5f5;
                font-family: Arial, sans-serif;
            }
        ''')
    ])
    
    # Callback for active traces
    @app.callback(
        Output('active-traces-container', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_active_traces(_):
        trace_cards = []
        
        for trace_id, trace in TRACES.items():
            is_active = trace_id in ACTIVE_TRACES
            step_count = len(trace.steps)
            
            # Get the latest metrics for this trace
            latest_metrics = next(
                (m for m in reversed(METRICS_HISTORY) if m["trace_id"] == trace_id), 
                None
            )
            quality_score = latest_metrics.get("quality_score", 0) if latest_metrics else 0
            
            # Calculate color based on quality score
            color = f"rgb({int(255 * (1 - quality_score))}, {int(255 * quality_score)}, 0)"
            
            card = html.Div([
                html.H4(f"Trace: {trace_id}", style={'marginBottom': '5px'}),
                html.P(f"Task: {trace.task}", style={'marginBottom': '5px'}),
                html.P(f"Steps: {step_count}/10", style={'marginBottom': '5px'}),
                html.P(f"Status: {'Active' if is_active else 'Completed'}", 
                       style={'color': 'green' if is_active else 'blue', 'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div([
                    html.Span("Quality: "),
                    html.Span(f"{quality_score:.2f}", style={'color': color, 'fontWeight': 'bold'})
                ])
            ], style={
                'border': '1px solid #ddd',
                'borderRadius': '5px',
                'padding': '10px',
                'margin': '10px',
                'width': '200px',
                'display': 'inline-block',
                'backgroundColor': '#f9f9f9' if is_active else '#eaeaea'
            })
            
            trace_cards.append(card)
        
        return trace_cards
    
    # Callback for quality metrics chart
    @app.callback(
        Output('quality-metrics-chart', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_quality_metrics_chart(_):
        if not METRICS_HISTORY:
            return go.Figure()
        
        # Convert to DataFrame
        df = pd.DataFrame(METRICS_HISTORY)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure
        fig = go.Figure()
        
        for trace_id in TRACES.keys():
            trace_data = df[df['trace_id'] == trace_id]
            if not trace_data.empty:
                fig.add_trace(go.Scatter(
                    x=trace_data['step_count'],
                    y=trace_data['quality_score'],
                    mode='lines+markers',
                    name=f"{trace_id} - Quality",
                    line=dict(width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=trace_data['step_count'],
                    y=trace_data['coherence'],
                    mode='lines',
                    name=f"{trace_id} - Coherence",
                    line=dict(dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=trace_data['step_count'],
                    y=trace_data['goal_alignment'],
                    mode='lines',
                    name=f"{trace_id} - Goal Alignment",
                    line=dict(dash='dot')
                ))
        
        fig.update_layout(
            xaxis_title="Step Count",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=30, b=40, l=40, r=40),
            height=350
        )
        
        return fig
    
    # Callback for context usage chart
    @app.callback(
        Output('context-usage-chart', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_context_usage_chart(_):
        if not CONTEXT_STATS_HISTORY:
            return go.Figure()
        
        # Convert to DataFrame
        df = pd.DataFrame(CONTEXT_STATS_HISTORY)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure
        fig = go.Figure()
        
        for trace_id in TRACES.keys():
            trace_data = df[df['trace_id'] == trace_id]
            if not trace_data.empty:
                fig.add_trace(go.Scatter(
                    x=trace_data['step_count'],
                    y=trace_data['avg_relevance'],
                    mode='lines+markers',
                    name=f"{trace_id} - Avg Relevance",
                    line=dict(width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=trace_data['step_count'],
                    y=trace_data['relevant_items_ratio'],
                    mode='lines',
                    name=f"{trace_id} - Relevant Items Ratio",
                    line=dict(dash='dash')
                ))
        
        fig.update_layout(
            xaxis_title="Step Count",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=30, b=40, l=40, r=40),
            height=350
        )
        
        return fig
    
    # Callback for performance metrics chart
    @app.callback(
        Output('performance-metrics-chart', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_performance_metrics_chart(_):
        if not METRICS_HISTORY:
            return go.Figure()
        
        # Convert to DataFrame
        df = pd.DataFrame(METRICS_HISTORY)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure
        fig = go.Figure()
        
        for trace_id in TRACES.keys():
            trace_data = df[df['trace_id'] == trace_id]
            if not trace_data.empty:
                # Primary Y-axis - Token Usage
                fig.add_trace(go.Scatter(
                    x=trace_data['step_count'],
                    y=trace_data['token_usage'],
                    mode='lines+markers',
                    name=f"{trace_id} - Token Usage",
                    line=dict(width=3)
                ))
                
                # Secondary Y-axis - Completion Time
                fig.add_trace(go.Scatter(
                    x=trace_data['step_count'],
                    y=trace_data['completion_time_s'] / 60,  # Convert to minutes
                    mode='lines',
                    name=f"{trace_id} - Time (min)",
                    line=dict(dash='dash'),
                    yaxis="y2"
                ))
        
        fig.update_layout(
            xaxis_title="Step Count",
            yaxis_title="Token Count",
            yaxis2=dict(
                title="Time (minutes)",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=30, b=40, l=40, r=40),
            height=350
        )
        
        return fig
    
    # Callback for buttons
    @app.callback(
        Output('start-btn', 'disabled'),
        Output('stop-btn', 'disabled'),
        Output('reset-btn', 'disabled'),
        Input('start-btn', 'n_clicks'),
        Input('stop-btn', 'n_clicks'),
        Input('reset-btn', 'n_clicks')
    )
    def handle_button_clicks(start_clicks, stop_clicks, reset_clicks):
        ctx = dash.callback_context
        
        if not ctx.triggered:
            return False, True, False
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'start-btn' and start_clicks:
            start_simulation()
            return True, False, True
        
        elif button_id == 'stop-btn' and stop_clicks:
            stop_simulation()
            return False, True, False
        
        elif button_id == 'reset-btn' and reset_clicks:
            # Reset all data
            global TRACES, METRICS_HISTORY, CONTEXT_STATS_HISTORY, ACTIVE_TRACES
            TRACES = {}
            METRICS_HISTORY = []
            CONTEXT_STATS_HISTORY = []
            ACTIVE_TRACES = []
            return False, True, False
        
        return dash.no_update, dash.no_update, dash.no_update
    
    return app


def main():
    """Main function to run the dashboard."""
    parser = argparse.ArgumentParser(description="Real-time Evaluation Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port for the dashboard server")
    parser.add_argument("--auto-start", action="store_true", help="Auto-start the simulation")
    args = parser.parse_args()
    
    app = create_dashboard()
    
    if args.auto_start:
        start_simulation()
    
    # Open browser
    url = f"http://localhost:{args.port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    
    # Run the app
    app.run_server(debug=False, port=args.port)


if __name__ == "__main__":
    main() 