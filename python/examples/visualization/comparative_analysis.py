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
Comparative Analysis Example

This script demonstrates how to perform comparative analysis of reasoning traces
using multiple visualization components from the BeeAI Framework.
"""

import os
import json
import logging
import asyncio
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from beeai_framework.visualization import (
    ReasoningTraceVisualizer,
    ReasoningTrace,
    ReasoningStep,
    ReasoningQualityMetrics,
    ContextUsageAnalytics,
    ABTestingFramework,
    TestCase,
    TestStrategy,
    TestResult,
    MetricsVisualizer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mock_traces(file_path):
    """
    Load mock traces from JSON file or create them if file doesn't exist.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of reasoning traces
    """
    if os.path.exists(file_path):
        logger.info(f"Loading traces from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        traces = []
        for trace_data in data:
            trace = ReasoningTrace(
                trace_id=trace_data["trace_id"],
                task=trace_data["task"],
                start_time=datetime.fromisoformat(trace_data["start_time"]),
                end_time=datetime.fromisoformat(trace_data["end_time"]) if trace_data["end_time"] else None,
                overall_metrics=trace_data["overall_metrics"]
            )
            
            for step_data in trace_data["steps"]:
                step = ReasoningStep(
                    step_number=step_data["step_number"],
                    step_type=step_data["step_type"],
                    content=step_data["content"],
                    timestamp=datetime.fromisoformat(step_data["timestamp"]),
                    context_items=step_data["context_items"],
                    metrics=step_data["metrics"],
                    key_concepts=step_data.get("key_concepts", []),
                    next_step_suggestions=step_data.get("next_step_suggestions", [])
                )
                trace.add_step(step)
                
            traces.append(trace)
        
        return traces
    else:
        # Create mock traces with different strategies
        logger.info(f"Creating mock traces")
        strategies = [
            {"name": "Baseline", "context_count": 3, "step_count": 5},
            {"name": "Enhanced", "context_count": 5, "step_count": 7},
            {"name": "Minimal", "context_count": 2, "step_count": 4}
        ]
        
        tasks = [
            "Design a scalable microservice architecture",
            "Implement a secure authentication system",
            "Optimize database performance for high-traffic website"
        ]
        
        traces = []
        for task in tasks:
            for strategy in strategies:
                trace = create_mock_trace(
                    task, 
                    strategy["name"],
                    strategy["step_count"],
                    strategy["context_count"]
                )
                traces.append(trace)
        
        # Save traces to file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json_data = [trace.to_dict() for trace in traces]
            json.dump(json_data, f, indent=2, default=str)
            
        return traces


def create_mock_trace(task, strategy_name, steps=5, contexts_per_step=3):
    """
    Create a mock reasoning trace.
    
    Args:
        task: Task description
        strategy_name: Name of the strategy used
        steps: Number of steps to generate
        contexts_per_step: Number of context items per step
        
    Returns:
        Mock reasoning trace
    """
    trace_id = f"{strategy_name.lower()}_{task.split()[0].lower()}_{hash(task) % 1000}"
    
    # Create trace with strategy-specific characteristics
    start_time = datetime.now() - timedelta(minutes=steps * 2)
    
    # Strategy-specific metrics
    if strategy_name == "Baseline":
        overall_metrics = {
            "completion_time_s": steps * 60,
            "token_usage": steps * 500,
            "total_context_items": steps * contexts_per_step,
            "strategy": strategy_name
        }
    elif strategy_name == "Enhanced":
        overall_metrics = {
            "completion_time_s": steps * 45,  # Faster
            "token_usage": steps * 650,       # More tokens
            "total_context_items": steps * contexts_per_step,
            "strategy": strategy_name
        }
    else:  # Minimal
        overall_metrics = {
            "completion_time_s": steps * 40,  # Fastest
            "token_usage": steps * 300,       # Fewer tokens
            "total_context_items": steps * contexts_per_step,
            "strategy": strategy_name
        }
    
    trace = ReasoningTrace(
        trace_id=trace_id,
        task=task,
        start_time=start_time,
        overall_metrics=overall_metrics
    )
    
    # Define step types for typical reasoning flow
    step_types = [
        "problem_definition",
        "information_gathering",
        "analysis",
        "solution_formulation",
        "verification"
    ]
    
    # Create steps
    for i in range(1, steps + 1):
        step_type = step_types[min(i-1, len(step_types)-1)]
        step_time = start_time + timedelta(minutes=i * 2)
        
        # Create context items for this step
        context_items = []
        for j in range(contexts_per_step):
            # Strategy-specific relevance patterns
            if strategy_name == "Baseline":
                similarity = 0.7 + (0.3 * (1 - j/contexts_per_step))
            elif strategy_name == "Enhanced":
                similarity = 0.8 + (0.2 * (1 - j/contexts_per_step))
            else:  # Minimal
                similarity = 0.6 + (0.4 * (1 - j/contexts_per_step))
                
            context_items.append({
                "content": f"This is context item {j+1} for step {i} using {strategy_name} strategy. " +
                          f"It contains relevant information about {task.split()[j % len(task.split())]}.",
                "similarity": similarity,
                "metadata": {
                    "source": f"source_{(i+j) % 3 + 1}",
                    "level": ["domain", "techstack", "project"][j % 3],
                    "timestamp": step_time.isoformat(),
                    "strategy": strategy_name
                }
            })
        
        # Strategy-specific metrics
        if strategy_name == "Baseline":
            step_metrics = {
                "tokens": 150 + (i * 20),
                "confidence": 0.7 + (i * 0.05),
                "relevance": 0.75,
                "novelty": 0.6
            }
        elif strategy_name == "Enhanced":
            step_metrics = {
                "tokens": 200 + (i * 25),
                "confidence": 0.8 + (i * 0.02),
                "relevance": 0.85,
                "novelty": 0.7
            }
        else:  # Minimal
            step_metrics = {
                "tokens": 100 + (i * 15),
                "confidence": 0.65 + (i * 0.05),
                "relevance": 0.7,
                "novelty": 0.5
            }
        
        # Create step
        step = ReasoningStep(
            step_number=i,
            step_type=step_type,
            content=f"Step {i} ({step_type}): Analysis of the {task} using {strategy_name} strategy. " + 
                   f"Based on the context information, we can see that this requires understanding of {step_type}. " +
                   f"The key aspects to consider are X, Y, and Z.",
            timestamp=step_time,
            context_items=context_items,
            metrics=step_metrics,
            key_concepts=[
                {"concept": f"concept_{(i*2)}", "importance": 0.9},
                {"concept": f"concept_{(i*2)+1}", "importance": 0.7},
                {"concept": f"concept_{(i*2)+2}", "importance": 0.5}
            ],
            next_step_suggestions=[
                f"Consider exploring {step_types[min(i, len(step_types)-1)]} in more detail",
                f"Investigate the relationship between {task.split()[0]} and performance",
                f"Analyze the impact of {task.split()[-1]} on the solution"
            ]
        )
        
        trace.add_step(step)
    
    # Set end time
    trace.end_time = start_time + timedelta(minutes=steps * 2)
    
    return trace


def create_comparative_dashboard(traces, output_dir):
    """
    Create a comparative dashboard of multiple traces.
    
    Args:
        traces: List of reasoning traces
        output_dir: Output directory for saving visualizations
        
    Returns:
        Dictionary with analysis results
    """
    # Initialize components
    trace_visualizer = ReasoningTraceVisualizer()
    quality_metrics = ReasoningQualityMetrics()
    context_analytics = ContextUsageAnalytics()
    metrics_viz = MetricsVisualizer()
    
    # Initialize AB testing framework
    ab_testing = ABTestingFramework(quality_metrics=quality_metrics)
    
    # Group traces by task and strategy
    task_groups = {}
    strategies = set()
    
    for trace in traces:
        task = trace.task
        strategy = trace.overall_metrics.get("strategy", "Unknown")
        strategies.add(strategy)
        
        if task not in task_groups:
            task_groups[task] = {}
            
        task_groups[task][strategy] = trace
    
    # Add test cases and strategies
    for task in task_groups:
        case = TestCase(
            case_id=f"case_{hash(task) % 1000}",
            description=f"Test case for {task}",
            task=task,
            difficulty="medium",
            tags=["test", "demo"]
        )
        ab_testing.add_test_case(case)
    
    for strategy in strategies:
        strat = TestStrategy(
            strategy_id=f"strategy_{strategy.lower()}",
            name=strategy,
            description=f"{strategy} reasoning approach",
            config={"name": strategy}
        )
        ab_testing.add_strategy(strat)
    
    # Add traces to AB testing framework
    for task, strategies_dict in task_groups.items():
        case_id = f"case_{hash(task) % 1000}"
        
        for strategy, trace in strategies_dict.items():
            strategy_id = f"strategy_{strategy.lower()}"
            
            # Add trace
            ab_testing.traces[trace.trace_id] = trace
            
            # Calculate metrics
            metrics_result = quality_metrics.calculate_all_metrics(trace)
            
            # Create result
            overall_quality = np.mean([
                metrics_result["trace_metrics"].get("coherence", 0),
                metrics_result["trace_metrics"].get("goal_alignment", 0),
                metrics_result["trace_metrics"].get("completeness", 0)
            ])
            
            # Context usage
            usage_stats = context_analytics.analyze_all_metrics(trace)
            
            # Create test result
            result = TestResult(
                case_id=case_id,
                strategy_id=strategy_id,
                trace_id=trace.trace_id,
                metrics={
                    "quality_score": overall_quality,
                    "coherence": metrics_result["trace_metrics"].get("coherence", 0),
                    "goal_alignment": metrics_result["trace_metrics"].get("goal_alignment", 0),
                    "completeness": metrics_result["trace_metrics"].get("completeness", 0),
                    "context_relevance": usage_stats.avg_relevance,
                    "token_efficiency": 1.0 / (trace.overall_metrics.get("token_usage", 1) / 1000 + 0.1),
                    "completion_time": trace.overall_metrics.get("completion_time_s", 0) / 60
                },
                timestamp=datetime.now()
            )
            
            if case_id not in ab_testing.results:
                ab_testing.results[case_id] = []
                
            ab_testing.results[case_id].append(result)
    
    # Analyze results
    logger.info("Analyzing results...")
    analysis = ab_testing.analyze_results()
    
    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Comparative visualization
    logger.info("Creating comparative visualizations...")
    comp_viz = ab_testing.create_comparative_visualization(analysis)
    comp_viz.write_html(os.path.join(output_dir, "strategy_comparison.html"))
    
    # Performance matrix
    for metric in ["quality_score", "coherence", "goal_alignment", "completeness", "context_relevance"]:
        matrix_viz = ab_testing.create_performance_matrix(analysis, metric)
        matrix_viz.write_html(os.path.join(output_dir, f"{metric}_matrix.html"))
    
    # Create trace-level visualizations for each task and strategy
    for task, strategies_dict in task_groups.items():
        task_dir = os.path.join(output_dir, f"task_{hash(task) % 1000}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Create comparison subplot for this task
        fig = sp.make_subplots(
            rows=len(strategies_dict), 
            cols=1,
            subplot_titles=[f"{strategy} Strategy" for strategy in strategies_dict.keys()],
            vertical_spacing=0.1
        )
        
        row = 1
        token_usage_data = []
        step_metrics_data = []
        
        for strategy, trace in strategies_dict.items():
            # Calculate metrics
            metrics_result = quality_metrics.calculate_all_metrics(trace)
            
            # Create visualizations
            context_viz = trace_visualizer.create_context_relevance_visualization(trace)
            context_viz.write_html(os.path.join(task_dir, f"{strategy.lower()}_context.html"))
            
            timeline_viz = trace_visualizer.create_context_evolution_timeline(trace)
            timeline_viz.write_html(os.path.join(task_dir, f"{strategy.lower()}_timeline.html"))
            
            # Add to comparison subplot
            for trace_data in context_viz.data:
                fig.add_trace(trace_data, row=row, col=1)
            
            row += 1
            
            # Collect token usage data
            token_data = {
                "strategy": strategy,
                "total_tokens": trace.overall_metrics.get("token_usage", 0),
                "completion_time": trace.overall_metrics.get("completion_time_s", 0) / 60
            }
            token_usage_data.append(token_data)
            
            # Collect step metrics data
            for step in trace.steps:
                step_data = {
                    "strategy": strategy,
                    "step": step.step_number,
                    "step_type": step.step_type
                }
                step_data.update(step.metrics)
                step_metrics_data.append(step_data)
        
        # Update layout for comparison subplot
        fig.update_layout(
            title=f"Context Relevance Comparison for Task: {task}",
            height=300 * len(strategies_dict),
            width=800,
            showlegend=False
        )
        fig.write_html(os.path.join(task_dir, "context_comparison.html"))
        
        # Create token usage comparison
        token_df = pd.DataFrame(token_usage_data)
        token_fig = go.Figure(data=[
            go.Bar(
                name="Total Tokens",
                x=token_df["strategy"],
                y=token_df["total_tokens"],
                marker_color="blue"
            ),
            go.Bar(
                name="Completion Time (min)",
                x=token_df["strategy"],
                y=token_df["completion_time"],
                marker_color="red",
                yaxis="y2"
            )
        ])
        
        token_fig.update_layout(
            title=f"Token Usage vs. Completion Time for Task: {task}",
            yaxis=dict(title="Token Count"),
            yaxis2=dict(title="Time (min)", overlaying="y", side="right"),
            barmode='group'
        )
        token_fig.write_html(os.path.join(task_dir, "token_comparison.html"))
        
        # Create step metrics comparison
        step_df = pd.DataFrame(step_metrics_data)
        
        for metric in ["confidence", "relevance"]:
            if metric in step_df.columns:
                metric_fig = go.Figure()
                
                for strategy in strategies_dict.keys():
                    strategy_data = step_df[step_df["strategy"] == strategy]
                    metric_fig.add_trace(go.Scatter(
                        x=strategy_data["step"],
                        y=strategy_data[metric],
                        mode="lines+markers",
                        name=strategy
                    ))
                
                metric_fig.update_layout(
                    title=f"{metric.capitalize()} Across Steps for Task: {task}",
                    xaxis=dict(title="Step Number"),
                    yaxis=dict(title=metric.capitalize()),
                    legend=dict(x=0, y=1, traceorder="normal")
                )
                metric_fig.write_html(os.path.join(task_dir, f"{metric}_comparison.html"))
    
    # Save analysis to JSON
    with open(os.path.join(output_dir, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    return {
        "analysis": analysis,
        "traces": len(traces),
        "tasks": len(task_groups),
        "strategies": len(strategies)
    }


async def main():
    """Main function for comparative analysis."""
    parser = argparse.ArgumentParser(description="Comparative Analysis of Reasoning Traces")
    parser.add_argument("--data-file", default="data/mock_traces.json", help="Path to trace data file")
    parser.add_argument("--output", default="visualizations/comparative", help="Output directory")
    args = parser.parse_args()
    
    logger.info("Starting comparative analysis...")
    
    # Load or create mock traces
    traces = load_mock_traces(args.data_file)
    logger.info(f"Loaded {len(traces)} traces")
    
    # Create comparative dashboard
    results = create_comparative_dashboard(traces, args.output)
    
    logger.info(f"Analysis complete. Processed {results['traces']} traces across " +
               f"{results['tasks']} tasks and {results['strategies']} strategies.")
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main()) 