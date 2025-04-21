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
A/B Testing Framework for the BeeAI Reasoning System.

This module provides a framework for comparing different retrieval strategies
and reasoning approaches, evaluating their performance metrics, and visualizing
the results of comparative tests.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

from beeai_framework.visualization.base_visualizer import BaseVisualizer
from beeai_framework.visualization.components.reasoning_trace_visualizer import (
    ReasoningTrace
)
from beeai_framework.visualization.components.reasoning_quality_metrics import (
    ReasoningQualityMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a test case for evaluating retrieval strategies."""
    case_id: str
    description: str
    task: str
    reference_content: Optional[str] = None
    difficulty: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestStrategy:
    """Represents a retrieval strategy configuration to be tested."""
    strategy_id: str
    name: str
    description: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Represents the result of running a test case with a specific strategy."""
    strategy_id: str
    case_id: str
    trace_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingFramework(BaseVisualizer):
    """
    A framework for conducting A/B testing of different retrieval strategies.
    
    This framework allows users to define test cases and strategies, run comparative
    tests, analyze results, and visualize performance differences.
    """
    
    def __init__(self, quality_metrics: Optional[ReasoningQualityMetrics] = None):
        """Initialize the A/B testing framework."""
        super().__init__()
        self.strategies: Dict[str, TestStrategy] = {}
        self.test_cases: Dict[str, TestCase] = {}
        self.results: Dict[str, List[TestResult]] = {}
        self.traces: Dict[str, ReasoningTrace] = {}
        self.quality_metrics = quality_metrics or ReasoningQualityMetrics()
    
    def add_strategy(self, strategy: TestStrategy) -> None:
        """Add a strategy to the testing framework."""
        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"Added strategy: {strategy.name} (ID: {strategy.strategy_id})")
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the testing framework."""
        self.test_cases[test_case.case_id] = test_case
        logger.info(f"Added test case: {test_case.description} (ID: {test_case.case_id})")
        
        # Initialize results container for this test case
        if test_case.case_id not in self.results:
            self.results[test_case.case_id] = []
    
    async def run_test(
        self,
        case_id: str,
        strategy_id: str,
        retrieval_service: Any,
        run_func: Callable,
        iterations: int = 1,
    ) -> Optional[TestResult]:
        """
        Run a single test with the specified strategy and test case.
        
        Args:
            case_id: ID of the test case to use.
            strategy_id: ID of the strategy to test.
            retrieval_service: Service for handling retrievals.
            run_func: Async function that executes the test with signature:
                async def run_func(task: str, config: Dict) -> Tuple[ReasoningTrace, Dict[str, float]]
            iterations: Number of times to run the test (default: 1).
        
        Returns:
            TestResult object with the results, or None if the test case or strategy is invalid.
        """
        # Validate inputs
        if strategy_id not in self.strategies or case_id not in self.test_cases:
            logger.error(f"Strategy '{strategy_id}' or test case '{case_id}' not found")
            return None
        
        strategy = self.strategies[strategy_id]
        test_case = self.test_cases[case_id]
        
        logger.info(f"Running test with strategy '{strategy.name}' on test case '{test_case.description}'")
        
        try:
            # Execute the test using the provided function
            trace, metrics = await run_func(test_case.task, strategy.config)
            
            # Store the reasoning trace
            self.traces[trace.trace_id] = trace
            
            # Create and store test result
            result = TestResult(
                strategy_id=strategy_id,
                case_id=case_id,
                trace_id=trace.trace_id,
                metrics=metrics,
                timestamp=datetime.now()
            )
            
            if case_id not in self.results:
                self.results[case_id] = []
            self.results[case_id].append(result)
            
            logger.info(f"Test completed successfully with metrics: {metrics}")
            return result
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            return None
    
    async def run_comparative_test(
        self,
        case_ids: List[str],
        strategy_ids: List[str],
        retrieval_service: Any,
        run_func: Callable,
        iterations: int = 3,
    ) -> Dict[str, Dict[str, List[TestResult]]]:
        """
        Run a comparative test with multiple strategies on multiple test cases.
        
        Args:
            case_ids: List of test case IDs to use.
            strategy_ids: List of strategy IDs to test.
            retrieval_service: Service for handling retrievals.
            run_func: Async function that executes the test
            iterations: Number of times to run each test for statistical reliability (default: 3).
        
        Returns:
            Nested dictionary of results organized by case_id and strategy_id.
        """
        results_map: Dict[str, Dict[str, List[TestResult]]] = {}
        
        # Validate inputs
        valid_case_ids = [case_id for case_id in case_ids if case_id in self.test_cases]
        valid_strategy_ids = [strategy_id for strategy_id in strategy_ids if strategy_id in self.strategies]
        
        if not valid_case_ids or not valid_strategy_ids:
            logger.error("No valid test cases or strategies provided")
            return results_map
        
        # Initialize results map
        for case_id in valid_case_ids:
            results_map[case_id] = {strategy_id: [] for strategy_id in valid_strategy_ids}
        
        # Run tests for each combination
        for case_id in valid_case_ids:
            for strategy_id in valid_strategy_ids:
                for i in range(iterations):
                    logger.info(f"Running iteration {i+1}/{iterations} for case '{case_id}' with strategy '{strategy_id}'")
                    result = await self.run_test(
                        case_id=case_id,
                        strategy_id=strategy_id,
                        retrieval_service=retrieval_service,
                        run_func=run_func
                    )
                    
                    if result:
                        results_map[case_id][strategy_id].append(result)
        
        return results_map
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the test results to compare strategies.
        
        Returns:
            Dictionary with analysis results including:
            - by_strategy: Strategy-level performance summaries
            - by_case: Test case-level comparisons
            - significance: Statistical significance tests between strategies
        """
        if not self.results:
            logger.warning("No test results to analyze")
            return {"by_strategy": {}, "by_case": {}, "significance": {}}
        
        # Extract unique strategy IDs from results
        strategy_ids = {result.strategy_id for results_list in self.results.values() 
                        for result in results_list}
        
        # Prepare data structures for analysis
        strategy_metrics: Dict[str, Dict[str, List[float]]] = {sid: {} for sid in strategy_ids}
        case_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
            case_id: {sid: {} for sid in strategy_ids} for case_id in self.results
        }
        
        # Populate metrics data
        for case_id, results_list in self.results.items():
            for result in results_list:
                strategy_id = result.strategy_id
                
                for metric_name, value in result.metrics.items():
                    # Update strategy-level metrics
                    if metric_name not in strategy_metrics[strategy_id]:
                        strategy_metrics[strategy_id][metric_name] = []
                    strategy_metrics[strategy_id][metric_name].append(value)
                    
                    # Update case-level metrics
                    case_metrics[case_id][strategy_id][metric_name] = value
        
        # Compute strategy-level summaries
        strategy_summary = {}
        for strategy_id, metrics in strategy_metrics.items():
            strategy_summary[strategy_id] = {
                metric_name: {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                } for metric_name, values in metrics.items() if values
            }
        
        # Compute statistical significance
        significance_tests = {}
        if len(strategy_ids) > 1:
            all_metrics = {m for metrics in strategy_metrics.values() for m in metrics.keys()}
            
            for metric_name in all_metrics:
                metric_data = {
                    sid: strategy_metrics[sid].get(metric_name, [])
                    for sid in strategy_ids
                    if metric_name in strategy_metrics[sid] and strategy_metrics[sid][metric_name]
                }
                
                if len(metric_data) >= 2:
                    significance_tests[metric_name] = self._run_significance_tests(metric_data)
        
        return {
            "by_strategy": strategy_summary,
            "by_case": case_metrics,
            "significance": significance_tests
        }
    
    def _run_significance_tests(self, metric_data: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        """Run statistical significance tests between strategies for a single metric."""
        results = {}
        strategy_ids = list(metric_data.keys())
        
        # Run pairwise t-tests
        for i, strategy_a in enumerate(strategy_ids):
            for strategy_b in strategy_ids[i+1:]:
                data_a = metric_data[strategy_a]
                data_b = metric_data[strategy_b]
                
                if not (data_a and data_b):
                    continue
                    
                try:
                    # Run two-sample t-test
                    t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
                    
                    # Determine which strategy has higher mean
                    mean_a = np.mean(data_a)
                    mean_b = np.mean(data_b)
                    better = strategy_a if mean_a > mean_b else strategy_b
                    
                    # Store results
                    results[f"{strategy_a}_vs_{strategy_b}"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "better": better,
                        "difference": abs(mean_a - mean_b),
                        "n1": len(data_a),
                        "n2": len(data_b)
                    }
                except Exception as e:
                    logger.warning(f"Error running t-test: {str(e)}")
        
        return results
    
    def create_comparative_visualization(self, analysis_results: Dict[str, Any]) -> go.Figure:
        """Create a visualization comparing metrics across different strategies."""
        if not analysis_results or not analysis_results.get("by_strategy"):
            return go.Figure()
        
        strategy_results = analysis_results["by_strategy"]
        strategies = list(strategy_results.keys())
        all_metrics = sorted({m for s in strategy_results.values() for m in s.keys()})
        
        # Create subplots for each metric
        fig = make_subplots(
            rows=len(all_metrics), 
            cols=1,
            subplot_titles=[metric.replace('_', ' ').title() for metric in all_metrics],
            vertical_spacing=0.1
        )
        
        # Add a bar chart for each metric
        for i, metric in enumerate(all_metrics):
            metric_values = []
            error_bars = []
            
            for strategy in strategies:
                if metric in strategy_results[strategy]:
                    metric_values.append(strategy_results[strategy][metric]["mean"])
                    error_bars.append(strategy_results[strategy][metric]["std"])
                else:
                    metric_values.append(0)
                    error_bars.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=metric_values,
                    error_y=dict(type='data', array=error_bars, visible=True),
                    name=metric.replace('_', ' ').title()
                ),
                row=i+1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Strategy Performance Comparison",
            showlegend=False,
            height=300 * len(all_metrics),
            width=800
        )
        
        return fig
    
    def create_performance_matrix(self, analysis_results: Dict[str, Any], metric: str) -> go.Figure:
        """Create a heatmap visualizing strategy performance across different test cases."""
        if not analysis_results or not analysis_results.get("by_case"):
            return go.Figure()
        
        case_results = analysis_results["by_case"]
        test_cases = list(case_results.keys())
        strategies = sorted({s for case_data in case_results.values() for s in case_data.keys()})
        
        # Prepare data for heatmap
        z_data = [
            [
                case_results[case_id].get(strategy_id, {}).get(metric) 
                for strategy_id in strategies
            ] 
            for case_id in test_cases
        ]
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=strategies,
            y=test_cases,
            colorscale='Viridis',
            zmin=0,
            zmax=1 if metric in ["precision", "recall", "f1_score"] else None,
            colorbar=dict(title=metric.replace('_', ' ').title())
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Performance Matrix: {metric.replace('_', ' ').title()}",
            xaxis_title="Strategies",
            yaxis_title="Test Cases",
            height=max(500, 50 * len(test_cases)),
            width=max(700, 100 * len(strategies))
        )
        
        return fig
    
    def export_results_to_json(self, filepath: str) -> bool:
        """Export test results, strategies, and test cases to a JSON file."""
        try:
            # Convert results to dictionary format
            results_dict = {case_id: [asdict(r) for r in results] 
                           for case_id, results in self.results.items()}
            
            # Prepare complete data for export
            export_data = {
                "strategies": {id: asdict(s) for id, s in self.strategies.items()},
                "test_cases": {id: asdict(c) for id, c in self.test_cases.items()},
                "results": results_dict
            }
            
            # Write to file
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(export_data, default=str, indent=2, fp=f)
            
            logger.info(f"Results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return False
    
    def import_results_from_json(self, filepath: str) -> bool:
        """Import test results, strategies, and test cases from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Import strategies
            for strategy_id, strategy_data in data.get("strategies", {}).items():
                self.strategies[strategy_id] = TestStrategy(**strategy_data)
            
            # Import test cases
            for case_id, case_data in data.get("test_cases", {}).items():
                self.test_cases[case_id] = TestCase(**case_data)
            
            # Import results
            for case_id, results_list in data.get("results", {}).items():
                self.results[case_id] = []
                for result_data in results_list:
                    # Convert timestamp string back to datetime
                    if isinstance(result_data.get("timestamp"), str):
                        try:
                            result_data["timestamp"] = datetime.fromisoformat(
                                result_data["timestamp"].replace('Z', '+00:00')
                            )
                        except ValueError:
                            result_data["timestamp"] = datetime.now()
                    
                    self.results[case_id].append(TestResult(**result_data))
            
            logger.info(f"Results imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing results: {str(e)}")
            return False
    
    def visualize_trace_comparison(
        self, 
        case_id: str, 
        strategy_ids: List[str]
    ) -> Dict[str, go.Figure]:
        """Create visualizations comparing reasoning traces from different strategies."""
        if case_id not in self.results:
            logger.error(f"No results found for test case {case_id}")
            return {}
        
        # Find results for the specified strategies and case
        strategy_results = {r.strategy_id: r for r in self.results[case_id] 
                           if r.strategy_id in strategy_ids}
        
        if not strategy_results:
            logger.error(f"No results found for the specified strategies")
            return {}
        
        # Create visualizations
        figures = {}
        
        if self.quality_metrics:
            try:
                # Extract traces for each strategy
                trace_ids = {s_id: r.trace_id for s_id, r in strategy_results.items()}
                traces = {s_id: self.traces.get(t_id) 
                         for s_id, t_id in trace_ids.items() if t_id in self.traces}
                
                if traces:
                    # Generate comparison visualizations
                    figures["reasoning_steps"] = self._create_step_comparison(traces)
                    figures["reasoning_quality"] = self._create_quality_comparison(traces)
                
            except Exception as e:
                logger.error(f"Error creating trace comparison visualization: {str(e)}")
        
        return figures
    
    def _create_step_comparison(self, traces: Dict[str, ReasoningTrace]) -> go.Figure:
        """Create a comparison of reasoning steps across strategies."""
        # Count steps per strategy
        steps_data = [
            {
                "strategy": strategy_id,
                "steps": len(trace.steps) if hasattr(trace, "steps") else 0,
                "duration": (trace.end_time - trace.start_time).total_seconds()
            }
            for strategy_id, trace in traces.items() if trace
        ]
        
        if not steps_data:
            return go.Figure()
        
        # Create DataFrame and subplot
        df = pd.DataFrame(steps_data)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add steps bars and duration line
        fig.add_trace(
            go.Bar(x=df["strategy"], y=df["steps"], name="Number of Steps", marker_color='blue'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=df["strategy"], y=df["duration"], name="Duration (s)", 
                      marker_color='red', mode='lines+markers'),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Reasoning Process Comparison",
            xaxis_title="Strategy",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Number of Steps", secondary_y=False)
        fig.update_yaxes(title_text="Duration (seconds)", secondary_y=True)
        
        return fig
    
    def _create_quality_comparison(self, traces: Dict[str, ReasoningTrace]) -> go.Figure:
        """Create a comparison of reasoning quality metrics across strategies."""
        if not self.quality_metrics:
            return go.Figure()
        
        # Calculate quality metrics for each trace
        quality_data = []
        for strategy_id, trace in traces.items():
            if trace:
                # Get trace-level metrics
                trace_metrics = self.quality_metrics.evaluate_trace(trace).get_trace_metrics()
                
                for metric_name, value in trace_metrics.items():
                    quality_data.append({
                        "strategy": strategy_id,
                        "metric": metric_name,
                        "value": value
                    })
        
        if not quality_data:
            return go.Figure()
        
        # Create grouped bar chart
        fig = px.bar(
            pd.DataFrame(quality_data),
            x="strategy",
            y="value",
            color="metric",
            barmode="group",
            title="Reasoning Quality Comparison"
        )
        
        fig.update_layout(
            xaxis_title="Strategy",
            yaxis_title="Score",
            legend_title="Quality Metric"
        )
        
        return fig


# Example usage
if __name__ == "__main__":
    # Example setup
    framework = ABTestingFramework()
    
    # Add test strategies
    framework.add_strategy(TestStrategy(
        strategy_id="strategy-a",
        name="BM25 Retrieval",
        description="Basic keyword-based retrieval",
        config={"retrieval_method": "bm25", "top_k": 5}
    ))
    
    framework.add_strategy(TestStrategy(
        strategy_id="strategy-b",
        name="Embedding Retrieval",
        description="Vector embedding-based retrieval",
        config={"retrieval_method": "embedding", "top_k": 5}
    ))
    
    # Add test case
    framework.add_test_case(TestCase(
        case_id="case-001",
        description="Climate change impacts",
        task="What are the impacts of climate change on biodiversity?",
        reference_content="Climate change affects biodiversity through temperature changes, habitat loss...",
        difficulty="medium",
        tags=["climate", "science"]
    )) 