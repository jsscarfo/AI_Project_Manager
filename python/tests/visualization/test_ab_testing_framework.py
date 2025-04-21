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
Unit tests for the A/B Testing Framework.
"""

import os
import json
import asyncio
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from unittest import mock

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from beeai_framework.visualization.components.ab_testing_framework import (
    ABTestingFramework, TestCase, TestStrategy, TestResult
)
from beeai_framework.visualization.components.reasoning_trace_visualizer import (
    ReasoningTrace, ReasoningStep
)


@pytest.fixture
def mock_test_cases() -> List[TestCase]:
    """Fixture providing test cases for testing."""
    return [
        TestCase(
            case_id="case-001",
            description="Simple query test",
            task="What is the capital of France?",
            reference_content="The capital of France is Paris.",
            difficulty="easy",
            tags=["geography", "europe"]
        ),
        TestCase(
            case_id="case-002",
            description="Complex reasoning test",
            task="Explain quantum entanglement",
            reference_content="Quantum entanglement is a physical phenomenon that occurs...",
            difficulty="hard",
            tags=["physics", "quantum"]
        )
    ]


@pytest.fixture
def mock_strategies() -> List[TestStrategy]:
    """Fixture providing strategies for testing."""
    return [
        TestStrategy(
            strategy_id="strategy-a",
            name="BM25 Strategy",
            description="Basic keyword retrieval",
            config={"retrieval_method": "bm25", "top_k": 3}
        ),
        TestStrategy(
            strategy_id="strategy-b",
            name="Embedding Strategy",
            description="Vector embedding retrieval",
            config={"retrieval_method": "embedding", "top_k": 5}
        )
    ]


@pytest.fixture
def testing_framework() -> ABTestingFramework:
    """Fixture providing a configured ABTestingFramework."""
    return ABTestingFramework()


@pytest.fixture
def populated_framework(
    testing_framework: ABTestingFramework,
    mock_test_cases: List[TestCase],
    mock_strategies: List[TestStrategy]
) -> ABTestingFramework:
    """Fixture providing a framework with test cases and strategies."""
    for test_case in mock_test_cases:
        testing_framework.add_test_case(test_case)
    
    for strategy in mock_strategies:
        testing_framework.add_strategy(strategy)
    
    return testing_framework


@pytest.fixture
def mock_trace() -> ReasoningTrace:
    """Fixture providing a mock reasoning trace."""
    trace = ReasoningTrace(
        trace_id="trace-001",
        query="What is the capital of France?",
        start_time=datetime.now() - timedelta(seconds=10),
        end_time=datetime.now(),
    )
    
    # Add steps
    trace.steps = [
        ReasoningStep(
            step_id="step-1",
            content="Searching for information about France's capital",
            step_type="search",
            timestamp=trace.start_time + timedelta(seconds=2)
        ),
        ReasoningStep(
            step_id="step-2",
            content="Found that Paris is the capital of France",
            step_type="analysis",
            timestamp=trace.start_time + timedelta(seconds=4)
        ),
        ReasoningStep(
            step_id="step-3",
            content="The capital of France is Paris.",
            step_type="answer",
            timestamp=trace.start_time + timedelta(seconds=6)
        )
    ]
    
    return trace


@pytest.fixture
def mock_test_results(mock_test_cases: List[TestCase], mock_strategies: List[TestStrategy]) -> Dict[str, List[TestResult]]:
    """Fixture providing mock test results."""
    results = {}
    
    for test_case in mock_test_cases:
        results[test_case.case_id] = []
        for strategy in mock_strategies:
            # Create a couple of results for each strategy-case pair
            for i in range(2):
                result = TestResult(
                    strategy_id=strategy.strategy_id,
                    case_id=test_case.case_id,
                    trace_id=f"trace-{test_case.case_id}-{strategy.strategy_id}-{i}",
                    metrics={
                        "precision": 0.8 + 0.05 * i,
                        "recall": 0.7 + 0.1 * i,
                        "f1_score": 0.75 + 0.075 * i,
                        "latency": 350 - 50 * i
                    },
                    timestamp=datetime.now()
                )
                results[test_case.case_id].append(result)
    
    return results


@pytest.fixture
def framework_with_results(
    populated_framework: ABTestingFramework,
    mock_test_results: Dict[str, List[TestResult]],
    mock_trace: ReasoningTrace
) -> ABTestingFramework:
    """Fixture providing a framework with test results."""
    for case_id, results in mock_test_results.items():
        populated_framework.results[case_id] = results
    
    # Add a trace for at least one result
    first_result = next(iter(mock_test_results.values()))[0]
    populated_framework.traces[first_result.trace_id] = mock_trace
    
    return populated_framework


async def mock_run_func(task: str, config: Dict[str, Any]) -> Tuple[ReasoningTrace, Dict[str, float]]:
    """Mock function for testing run_test."""
    # Create a mock trace
    trace = ReasoningTrace(
        trace_id=f"trace-{hash(task) % 1000:03d}-{hash(str(config)) % 1000:03d}",
        query=task,
        start_time=datetime.now() - timedelta(seconds=5),
        end_time=datetime.now(),
    )
    
    # Add a step
    trace.steps = [
        ReasoningStep(
            step_id="step-1",
            content=f"Processing {task} with {config.get('retrieval_method', 'unknown')}",
            step_type="process",
            timestamp=trace.start_time + timedelta(seconds=2)
        )
    ]
    
    # Generate mock metrics based on the config
    metrics = {
        "precision": 0.8 if config.get("retrieval_method") == "embedding" else 0.7,
        "recall": 0.75 if config.get("retrieval_method") == "embedding" else 0.65,
        "f1_score": 0.77 if config.get("retrieval_method") == "embedding" else 0.67,
        "latency": 300 if config.get("top_k", 0) > 3 else 200
    }
    
    # Simulate some processing time
    await asyncio.sleep(0.01)
    
    return trace, metrics


class TestABTestingFramework:
    """Tests for the ABTestingFramework class."""
    
    def test_add_test_case(self, testing_framework: ABTestingFramework, mock_test_cases: List[TestCase]):
        """Test adding test cases to the framework."""
        for test_case in mock_test_cases:
            testing_framework.add_test_case(test_case)
        
        # Verify test cases were added
        assert len(testing_framework.test_cases) == len(mock_test_cases)
        for test_case in mock_test_cases:
            assert test_case.case_id in testing_framework.test_cases
    
    def test_add_strategy(self, testing_framework: ABTestingFramework, mock_strategies: List[TestStrategy]):
        """Test adding strategies to the framework."""
        for strategy in mock_strategies:
            testing_framework.add_strategy(strategy)
        
        # Verify strategies were added
        assert len(testing_framework.strategies) == len(mock_strategies)
        for strategy in mock_strategies:
            assert strategy.strategy_id in testing_framework.strategies
    
    @pytest.mark.asyncio
    async def test_run_test(self, populated_framework: ABTestingFramework):
        """Test running a single test."""
        # Get first case and strategy
        case_id = next(iter(populated_framework.test_cases.keys()))
        strategy_id = next(iter(populated_framework.strategies.keys()))
        
        # Run test with mock function
        result = await populated_framework.run_test(
            case_id=case_id,
            strategy_id=strategy_id,
            retrieval_service=None,  # Not used in this test
            run_func=mock_run_func
        )
        
        # Verify result
        assert result is not None
        assert result.case_id == case_id
        assert result.strategy_id == strategy_id
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0
        
        # Verify result was added to framework
        assert case_id in populated_framework.results
        assert len(populated_framework.results[case_id]) == 1
        
        # Verify trace was stored
        assert result.trace_id in populated_framework.traces
    
    @pytest.mark.asyncio
    async def test_run_comparative_test(self, populated_framework: ABTestingFramework):
        """Test running comparative tests."""
        # Get all cases and strategies
        case_ids = list(populated_framework.test_cases.keys())
        strategy_ids = list(populated_framework.strategies.keys())
        
        # Run comparative test
        results = await populated_framework.run_comparative_test(
            case_ids=case_ids,
            strategy_ids=strategy_ids,
            retrieval_service=None,  # Not used in this test
            run_func=mock_run_func,
            iterations=2  # Run each test twice for reliability
        )
        
        # Verify results structure
        for case_id in case_ids:
            assert case_id in results
            for strategy_id in strategy_ids:
                assert strategy_id in results[case_id]
                assert len(results[case_id][strategy_id]) == 2  # 2 iterations
        
        # Verify results were also added to framework
        for case_id in case_ids:
            assert case_id in populated_framework.results
            # Should have 2 results per strategy (2 iterations)
            assert len(populated_framework.results[case_id]) == len(strategy_ids) * 2
    
    def test_analyze_results(self, framework_with_results: ABTestingFramework):
        """Test analyzing test results."""
        # Analyze results
        analysis = framework_with_results.analyze_results()
        
        # Verify analysis structure
        assert "by_strategy" in analysis
        assert "by_case" in analysis
        assert "significance" in analysis
        
        # Check strategy summaries
        for strategy_id in framework_with_results.strategies:
            assert strategy_id in analysis["by_strategy"]
            strategy_summary = analysis["by_strategy"][strategy_id]
            # Should have metrics
            assert len(strategy_summary) > 0
            # Each metric should have statistics
            for metric_data in strategy_summary.values():
                assert "mean" in metric_data
                assert "std" in metric_data
        
        # Check case-level data
        for case_id in framework_with_results.test_cases:
            assert case_id in analysis["by_case"]
            case_data = analysis["by_case"][case_id]
            for strategy_id in framework_with_results.strategies:
                assert strategy_id in case_data
    
    def test_create_comparative_visualization(self, framework_with_results: ABTestingFramework):
        """Test creating comparative visualization."""
        # Analyze results
        analysis = framework_with_results.analyze_results()
        
        # Create visualization
        fig = framework_with_results.create_comparative_visualization(analysis)
        
        # Verify figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have at least one trace
    
    def test_create_performance_matrix(self, framework_with_results: ABTestingFramework):
        """Test creating performance matrix visualization."""
        # Analyze results
        analysis = framework_with_results.analyze_results()
        
        # Create visualization for a specific metric
        fig = framework_with_results.create_performance_matrix(analysis, "precision")
        
        # Verify figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have at least one trace
    
    def test_export_import_results(self, framework_with_results: ABTestingFramework):
        """Test exporting and importing results."""
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            export_path = temp_file.name
        
        try:
            # Export results
            success = framework_with_results.export_results_to_json(export_path)
            assert success
            
            # Create a new framework
            new_framework = ABTestingFramework()
            
            # Import results
            success = new_framework.import_results_from_json(export_path)
            assert success
            
            # Verify imported data
            assert len(new_framework.test_cases) == len(framework_with_results.test_cases)
            assert len(new_framework.strategies) == len(framework_with_results.strategies)
            
            # Verify results were imported
            for case_id in framework_with_results.results:
                assert case_id in new_framework.results
                assert len(new_framework.results[case_id]) == len(framework_with_results.results[case_id])
            
        finally:
            # Clean up
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_visualize_trace_comparison(self, framework_with_results: ABTestingFramework):
        """Test visualizing trace comparison."""
        # Get first case
        case_id = next(iter(framework_with_results.test_cases.keys()))
        
        # Get strategies
        strategy_ids = list(framework_with_results.strategies.keys())
        
        # Visualize trace comparison
        figures = framework_with_results.visualize_trace_comparison(case_id, strategy_ids)
        
        # We may not have figures if traces aren't available for all strategies
        # But the function should not raise an exception
        if figures:
            for fig in figures.values():
                assert isinstance(fig, go.Figure)


if __name__ == "__main__":
    # Run a quick test to verify framework functionality
    framework = ABTestingFramework()
    
    # Add strategies
    strategies = [
        TestStrategy(
            strategy_id="bm25",
            name="BM25 Retrieval",
            description="Basic keyword-based retrieval",
            config={"retrieval_method": "bm25", "top_k": 5}
        ),
        TestStrategy(
            strategy_id="embedding",
            name="Embedding Retrieval",
            description="Vector embedding-based retrieval",
            config={"retrieval_method": "embedding", "top_k": 5}
        )
    ]
    
    for strategy in strategies:
        framework.add_strategy(strategy)
    
    # Add test cases
    test_cases = [
        TestCase(
            case_id="case-001",
            description="Climate change impacts",
            task="What are the impacts of climate change on biodiversity?",
            reference_content="Climate change affects biodiversity through temperature changes, habitat loss...",
            difficulty="medium",
            tags=["climate", "science"]
        ),
        TestCase(
            case_id="case-002",
            description="COVID-19 vaccines",
            task="How do mRNA vaccines work?",
            reference_content="mRNA vaccines work by introducing a piece of mRNA that corresponds to...",
            difficulty="medium",
            tags=["medicine", "vaccines"]
        )
    ]
    
    for test_case in test_cases:
        framework.add_test_case(test_case)
    
    print("Framework initialized with test cases and strategies") 