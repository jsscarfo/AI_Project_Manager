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
Reasoning Evaluation Demo

This script demonstrates the usage of visualization and evaluation tools
for reasoning quality analysis in the Sequential Thinking system.
"""

import logging
import asyncio
import json
import os
from datetime import datetime
import uuid
from typing import Dict, List, Any
import argparse

# BeeAI Framework imports
from beeai_framework.visualization import (
    ReasoningTraceVisualizer,
    ReasoningTrace,
    ReasoningStep,
    ReasoningQualityMetrics,
    ContextUsageAnalytics,
    EvaluationDashboard,
    DashboardConfig,
    ABTestingFramework,
    TestCase,
    TestStrategy
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_reasoning_trace(task_description: str, steps: int = 5) -> ReasoningTrace:
    """
    Create a mock reasoning trace for demonstration purposes.
    
    Args:
        task_description: Description of the reasoning task
        steps: Number of steps to generate
        
    Returns:
        Mock reasoning trace
    """
    trace_id = f"trace_{uuid.uuid4().hex[:8]}"
    
    # Create trace
    trace = ReasoningTrace(
        trace_id=trace_id,
        task=task_description,
        start_time=datetime.now(),
        overall_metrics={
            "completion_time_s": 5.2,
            "token_usage": 3450,
            "total_context_items": 15
        }
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
        
        # Create context items for this step
        context_items = []
        for j in range(3):  # 3 context items per step
            similarity = 0.7 + (0.3 * (1 - j/3))  # Higher relevance for first items
            
            context_items.append({
                "content": f"This is context item {j+1} for step {i}. It contains relevant information about {task_description.split()[j if j < len(task_description.split()) else 0]}.",
                "similarity": similarity,
                "metadata": {
                    "source": f"source_{(i+j) % 3 + 1}",
                    "level": ["domain", "techstack", "project"][j % 3],
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        # Create step
        step = ReasoningStep(
            step_number=i,
            step_type=step_type,
            content=f"Step {i} ({step_type}): Analysis of the {task_description}. " + 
                    f"Based on the context information, we can see that this requires understanding of {step_type}. " +
                    f"The key aspects to consider are X, Y, and Z.",
            timestamp=datetime.now(),
            context_items=context_items,
            metrics={
                "tokens": 150 + (i * 20),
                "confidence": 0.7 + (i * 0.05),
                "relevance": 0.8 - (0.1 * abs(i - 3) / 3)
            },
            key_concepts=[
                {"concept": f"concept_{(i*2)}", "importance": 0.9},
                {"concept": f"concept_{(i*2)+1}", "importance": 0.7},
                {"concept": f"concept_{(i*2)+2}", "importance": 0.5}
            ],
            next_step_suggestions=[
                f"Consider exploring {step_types[min(i, len(step_types)-1)]} in more detail",
                f"Investigate the relationship between {task_description.split()[0]} and performance",
                f"Analyze the impact of {task_description.split()[-1]} on the solution"
            ]
        )
        
        trace.add_step(step)
    
    # Set end time
    trace.end_time = datetime.now()
    
    return trace


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Reasoning Evaluation Demo")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive dashboard")
    parser.add_argument("--output", default="visualizations", help="Output directory for visualization files")
    parser.add_argument("--traces", type=int, default=3, help="Number of mock traces to generate")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize components
    trace_visualizer = ReasoningTraceVisualizer()
    quality_metrics = ReasoningQualityMetrics()
    context_analytics = ContextUsageAnalytics()
    
    # Create mock traces
    tasks = [
        "Optimize database queries for a high-traffic web application",
        "Design a caching strategy for API responses",
        "Implement a secure authentication system with JWT",
        "Create a scalable file storage service",
        "Develop a recommendation algorithm for product suggestions"
    ]
    
    traces = []
    for i in range(min(args.traces, len(tasks))):
        trace = create_mock_reasoning_trace(tasks[i])
        traces.append(trace)
        logger.info(f"Created mock reasoning trace: {trace.trace_id}")
    
    # Generate visualizations
    for i, trace in enumerate(traces):
        logger.info(f"Generating visualizations for trace {i+1}/{len(traces)}")
        
        # Create step visualization
        step_viz = trace_visualizer.create_step_visualization(trace)
        step_viz.write_html(os.path.join(args.output, f"trace_{i+1}_steps.html"))
        
        # Create context relevance visualization
        context_viz = trace_visualizer.create_context_relevance_visualization(trace)
        context_viz.write_html(os.path.join(args.output, f"trace_{i+1}_context_relevance.html"))
        
        # Create knowledge graph visualization
        graph_viz = trace_visualizer.create_knowledge_graph_visualization(trace)
        graph_viz.write_html(os.path.join(args.output, f"trace_{i+1}_knowledge_graph.html"))
        
        # Create timeline visualization
        timeline_viz = trace_visualizer.create_context_evolution_timeline(trace)
        timeline_viz.write_html(os.path.join(args.output, f"trace_{i+1}_timeline.html"))
        
        # Calculate quality metrics
        metrics = quality_metrics.calculate_all_metrics(trace)
        
        # Save metrics to JSON
        with open(os.path.join(args.output, f"trace_{i+1}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create context usage analytics
        usage_stats = context_analytics.analyze_all_metrics(trace)
        
        # Generate context usage visualizations
        token_viz = context_analytics.create_token_usage_chart(trace)
        token_viz.write_html(os.path.join(args.output, f"trace_{i+1}_token_usage.html"))
        
        source_viz = context_analytics.create_knowledge_source_chart(trace)
        source_viz.write_html(os.path.join(args.output, f"trace_{i+1}_source_usage.html"))
    
    # Create A/B testing example
    logger.info("Creating A/B testing example")
    ab_testing = ABTestingFramework(quality_metrics=quality_metrics)
    
    # Add test cases
    for i, task in enumerate(tasks[:3]):
        case = TestCase(
            case_id=f"case_{i+1}",
            description=f"Test case for {task}",
            task=task,
            difficulty="medium",
            tags=["api", "performance"]
        )
        ab_testing.add_test_case(case)
    
    # Add strategies
    strategies = [
        TestStrategy(
            strategy_id="strategy_1",
            name="Baseline Strategy",
            description="Standard retrieval approach",
            config={
                "similarity_threshold": 0.7,
                "max_results": 5
            }
        ),
        TestStrategy(
            strategy_id="strategy_2",
            name="Enhanced Strategy",
            description="Improved retrieval with contextual boosting",
            config={
                "similarity_threshold": 0.6,
                "max_results": 8,
                "context_boost": True
            }
        )
    ]
    
    for strategy in strategies:
        ab_testing.add_strategy(strategy)
    
    # Mock test results
    for i, trace in enumerate(traces):
        # Associate trace with test cases and strategies
        if i < len(strategies):
            case_id = f"case_{(i % 3) + 1}"
            strategy_id = f"strategy_{(i % 2) + 1}"
            
            # Add trace to results
            ab_testing.traces[trace.trace_id] = trace
            
            # Create mock test result
            result = {
                "case_id": case_id,
                "strategy_id": strategy_id,
                "trace_id": trace.trace_id,
                "metrics": {
                    "quality_score": 0.75 + (0.1 * (i % 2)),
                    "context_relevance": 0.8 + (0.15 * (i % 2)),
                    "token_efficiency": 0.6 + (0.2 * (i % 2)),
                    "completion_time": 2.5 - (0.5 * (i % 2))
                },
                "timestamp": datetime.now()
            }
            
            if case_id not in ab_testing.results:
                ab_testing.results[case_id] = []
                
            ab_testing.results[case_id].append(TestResult(**result))
    
    # Analyze results
    analysis = ab_testing.analyze_results()
    
    # Save analysis to JSON
    with open(os.path.join(args.output, "ab_testing_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Create comparative visualization
    comp_viz = ab_testing.create_comparative_visualization(analysis)
    comp_viz.write_html(os.path.join(args.output, "strategy_comparison.html"))
    
    # Create performance matrix
    matrix_viz = ab_testing.create_performance_matrix(analysis, "quality_score")
    matrix_viz.write_html(os.path.join(args.output, "performance_matrix.html"))
    
    logger.info(f"All visualizations saved to {args.output} directory")
    
    # Launch dashboard if requested
    if args.dashboard:
        logger.info("Launching interactive dashboard")
        
        # Configure dashboard
        config = DashboardConfig(
            title="Reasoning Quality Evaluation Dashboard",
            port=8050,
            debug=True,
            cache_dir=args.output
        )
        
        # Create dashboard
        dashboard = EvaluationDashboard(
            config=config,
            trace_visualizer=trace_visualizer,
            quality_metrics=quality_metrics,
            context_analytics=context_analytics
        )
        
        # Add traces to dashboard
        for trace in traces:
            dashboard.add_trace(
                trace=trace,
                compute_metrics=True,
                metadata={
                    "task_type": "analysis" if "algorithm" in trace.task else "implementation",
                    "complexity": "medium",
                    "tags": ["demo", "mock"],
                    "baseline": False
                }
            )
        
        # Run dashboard
        dashboard.run_server()
    else:
        logger.info("Dashboard not launched. Use --dashboard flag to launch interactive dashboard.")


if __name__ == "__main__":
    asyncio.run(main()) 