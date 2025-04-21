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
Example demonstrating how to use the A/B Testing Framework to compare
different retrieval strategies for reasoning tasks.
"""

import os
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from beeai_framework.visualization.components.ab_testing_framework import (
    ABTestingFramework, TestCase, TestStrategy, TestResult
)
from beeai_framework.visualization.components.reasoning_trace_visualizer import (
    ReasoningTrace, ReasoningStep
)
from beeai_framework.visualization.components.reasoning_quality_metrics import (
    ReasoningQualityMetrics
)


class MockRetrievalService:
    """Mock service to simulate a retrieval system for testing purposes."""
    
    def __init__(self, knowledge_base: Dict[str, str] = None):
        """Initialize with optional knowledge base."""
        self.knowledge_base = knowledge_base or {
            "climate": "Climate change affects biodiversity through temperature changes, habitat loss, and ecosystem disruption.",
            "vaccines": "mRNA vaccines work by introducing a piece of mRNA that corresponds to a viral protein, usually the spike protein of SARS-CoV-2.",
            "quantum": "Quantum entanglement occurs when pairs or groups of particles interact in ways such that the quantum state of each particle cannot be described independently.",
            "france": "France is a country in Western Europe. Its capital is Paris.",
            "economics": "Inflation is a general increase in prices and fall in the purchasing value of money."
        }
    
    async def retrieve(self, query: str, method: str, top_k: int = 3) -> Dict[str, float]:
        """Simulate retrieving information using different methods."""
        results = {}
        
        # Simulate different behaviors for different retrieval methods
        if method == "bm25":
            # Keyword-based retrieval (simulated)
            for key, text in self.knowledge_base.items():
                # Simple word matching to simulate BM25
                query_words = set(query.lower().split())
                text_words = set(text.lower().split())
                overlap = len(query_words.intersection(text_words))
                
                if overlap > 0:
                    results[key] = overlap / len(query_words)
        
        elif method == "embedding":
            # Semantic retrieval (simulated)
            for key, text in self.knowledge_base.items():
                # Use length difference as a very crude proxy for semantic similarity
                # In a real system, this would use vector embeddings
                len_diff = abs(len(query) - len(text))
                max_len = max(len(query), len(text))
                
                # Add some randomness to simulate semantic matching
                similarity = 0.5 + 0.5 * (1 - len_diff / max_len)
                # Add noise to make embedding results different
                similarity += random.uniform(-0.1, 0.1)
                results[key] = min(max(similarity, 0), 1)
        
        else:
            # Default method
            for key, text in self.knowledge_base.items():
                results[key] = random.uniform(0.3, 0.9)
        
        # Sort and take top_k
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {k: v for k, v in sorted_results}


async def run_reasoning_process(
    task: str, 
    config: Dict, 
    retrieval_service: MockRetrievalService
) -> Tuple[ReasoningTrace, Dict[str, float]]:
    """
    Simulate a reasoning process using the provided configuration.
    
    Args:
        task: The query or task to process
        config: Configuration parameters for the retrieval
        retrieval_service: Service for retrieving information
    
    Returns:
        A tuple of (reasoning_trace, metrics)
    """
    start_time = datetime.now()
    trace_id = f"trace-{hash(task) % 1000:03d}-{hash(str(config)) % 1000:03d}"
    
    # Create a new reasoning trace
    trace = ReasoningTrace(
        trace_id=trace_id,
        query=task,
        start_time=start_time,
        end_time=None  # Will set this at the end
    )
    
    # Add a planning step
    trace.steps = [
        ReasoningStep(
            step_id=f"{trace_id}-step-1",
            content=f"Planning to answer: {task}",
            step_type="planning",
            timestamp=start_time + timedelta(milliseconds=100)
        )
    ]
    
    # Extract config parameters
    retrieval_method = config.get("retrieval_method", "default")
    top_k = config.get("top_k", 3)
    
    # Simulate retrieval
    step_time = trace.steps[-1].timestamp + timedelta(milliseconds=200)
    trace.steps.append(
        ReasoningStep(
            step_id=f"{trace_id}-step-2",
            content=f"Retrieving information using {retrieval_method} method with top_k={top_k}",
            step_type="retrieval",
            timestamp=step_time
        )
    )
    
    # Simulate the actual retrieval
    retrieved_results = await retrieval_service.retrieve(task, retrieval_method, top_k)
    
    # Add retrieved content to the trace
    step_time = trace.steps[-1].timestamp + timedelta(milliseconds=300)
    for doc_id, score in retrieved_results.items():
        trace.steps.append(
            ReasoningStep(
                step_id=f"{trace_id}-retrieve-{doc_id}",
                content=f"Retrieved document '{doc_id}' with score {score:.3f}: {retrieval_service.knowledge_base.get(doc_id, 'No content')}",
                step_type="context",
                timestamp=step_time,
                metadata={"score": score, "doc_id": doc_id}
            )
        )
        step_time += timedelta(milliseconds=50)
    
    # Simulate analysis
    step_time += timedelta(milliseconds=100)
    trace.steps.append(
        ReasoningStep(
            step_id=f"{trace_id}-step-3",
            content="Analyzing retrieved information to formulate response",
            step_type="analysis",
            timestamp=step_time
        )
    )
    
    # Simulate answer generation
    step_time += timedelta(milliseconds=500)
    
    # Add some strategy-specific variations to make the comparison interesting
    if retrieval_method == "embedding":
        answer_quality = 0.8 + random.uniform(-0.1, 0.1)  # Higher quality
        reasoning_steps = 5 + random.randint(0, 2)  # More steps
    else:
        answer_quality = 0.7 + random.uniform(-0.1, 0.1)  # Lower quality
        reasoning_steps = 3 + random.randint(0, 2)  # Fewer steps
    
    # Add some reasoning steps
    for i in range(reasoning_steps):
        trace.steps.append(
            ReasoningStep(
                step_id=f"{trace_id}-reason-{i+1}",
                content=f"Reasoning step {i+1}: Analyzing information from sources",
                step_type="reasoning",
                timestamp=step_time
            )
        )
        step_time += timedelta(milliseconds=100)
    
    # Generate a simulated answer
    answer_text = f"Based on the retrieved information, the answer to '{task}' is..."
    trace.steps.append(
        ReasoningStep(
            step_id=f"{trace_id}-step-final",
            content=answer_text,
            step_type="answer",
            timestamp=step_time
        )
    )
    
    # Set the end time
    trace.end_time = step_time + timedelta(milliseconds=50)
    
    # Calculate overall process time
    process_time = (trace.end_time - start_time).total_seconds()
    
    # Generate metrics based on the retrieval method to simulate different performance
    if retrieval_method == "embedding":
        # Simulate embedding method having better relevance but higher latency
        metrics = {
            "precision": 0.85 + random.uniform(-0.05, 0.05),
            "recall": 0.80 + random.uniform(-0.05, 0.05),
            "f1_score": 0.82 + random.uniform(-0.05, 0.05),
            "latency": process_time * 1000,  # ms
            "context_relevance": 0.90 + random.uniform(-0.05, 0.05),
            "answer_quality": answer_quality
        }
    else:
        # Simulate BM25 method having lower relevance but faster speed
        metrics = {
            "precision": 0.75 + random.uniform(-0.05, 0.05),
            "recall": 0.70 + random.uniform(-0.05, 0.05),
            "f1_score": 0.72 + random.uniform(-0.05, 0.05),
            "latency": process_time * 800,  # ms (20% faster)
            "context_relevance": 0.80 + random.uniform(-0.05, 0.05),
            "answer_quality": answer_quality
        }
    
    return trace, metrics


async def main():
    """Run the A/B testing example."""
    print("Starting A/B Testing Framework Example")
    
    # Create the testing framework
    framework = ABTestingFramework()
    
    # Create a mock retrieval service
    retrieval_service = MockRetrievalService()
    
    # Define test strategies
    strategies = [
        TestStrategy(
            strategy_id="bm25",
            name="BM25 Retrieval",
            description="Classical keyword-based retrieval",
            config={"retrieval_method": "bm25", "top_k": 3}
        ),
        TestStrategy(
            strategy_id="embedding",
            name="Embedding Retrieval",
            description="Semantic vector embedding retrieval",
            config={"retrieval_method": "embedding", "top_k": 3}
        ),
        TestStrategy(
            strategy_id="embedding-5",
            name="Embedding Top-5",
            description="Semantic retrieval with more context",
            config={"retrieval_method": "embedding", "top_k": 5}
        )
    ]
    
    # Add strategies to the framework
    for strategy in strategies:
        framework.add_strategy(strategy)
        print(f"Added strategy: {strategy.name}")
    
    # Define test cases
    test_cases = [
        TestCase(
            case_id="climate",
            description="Climate Change Impact",
            task="How does climate change affect biodiversity?",
            reference_content="Climate change impacts biodiversity through temperature shifts, habitat disruption, and altered ecosystem dynamics.",
            difficulty="medium",
            tags=["science", "environment"]
        ),
        TestCase(
            case_id="vaccine",
            description="mRNA Vaccine Mechanism",
            task="How do mRNA vaccines work?",
            reference_content="mRNA vaccines work by introducing a piece of genetic material that instructs cells to produce a viral protein, triggering an immune response.",
            difficulty="hard",
            tags=["health", "biology"]
        ),
        TestCase(
            case_id="quantum",
            description="Quantum Entanglement",
            task="Explain quantum entanglement",
            reference_content="Quantum entanglement is a phenomenon where particles become correlated such that the quantum state of each cannot be described independently.",
            difficulty="hard",
            tags=["physics", "quantum"]
        ),
        TestCase(
            case_id="france",
            description="Capital City",
            task="What is the capital of France?",
            reference_content="The capital of France is Paris.",
            difficulty="easy",
            tags=["geography", "europe"]
        )
    ]
    
    # Add test cases to the framework
    for test_case in test_cases:
        framework.add_test_case(test_case)
        print(f"Added test case: {test_case.description}")
    
    # Run comparative tests
    print("\nRunning comparative tests...")
    results = await framework.run_comparative_test(
        case_ids=[case.case_id for case in test_cases],
        strategy_ids=[strategy.strategy_id for strategy in strategies],
        retrieval_service=retrieval_service,
        run_func=lambda task, config: run_reasoning_process(task, config, retrieval_service),
        iterations=3  # Run each test 3 times for more reliable results
    )
    
    # Analyze the results
    print("Analyzing results...")
    analysis = framework.analyze_results()
    
    # Print a summary of the results
    print("\nResults Summary:")
    for strategy_id, metrics in analysis["by_strategy"].items():
        strategy = framework.strategies[strategy_id]
        print(f"\n{strategy.name}:")
        for metric_name, stats in metrics.items():
            print(f"  {metric_name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Look at significance test results
    if analysis["significance"]:
        print("\nStatistical Significance Tests:")
        for metric, tests in analysis["significance"].items():
            print(f"\n{metric}:")
            for comparison, result in tests.items():
                if result["significant"]:
                    better = framework.strategies[result["better"]].name
                    p_value = result["p_value"]
                    print(f"  {comparison}: {better} is significantly better (p={p_value:.4f})")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Comparative visualization
    fig1 = framework.create_comparative_visualization(analysis)
    
    # Performance matrix for precision
    fig2 = framework.create_performance_matrix(analysis, "precision")
    
    # Example of trace comparison visualization
    figs = framework.visualize_trace_comparison("climate", ["bm25", "embedding"])
    
    # Save the figures if pio is available
    try:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        fig1.write_html(f"{output_dir}/strategy_comparison.html")
        fig2.write_html(f"{output_dir}/precision_matrix.html")
        
        for name, fig in figs.items():
            fig.write_html(f"{output_dir}/trace_{name}.html")
        
        print(f"Visualizations saved to {output_dir} directory")
    except Exception as e:
        print(f"Error saving visualizations: {str(e)}")
    
    # Export results to JSON
    export_path = "output/ab_test_results.json"
    framework.export_results_to_json(export_path)
    print(f"Results exported to {export_path}")
    
    print("\nA/B Testing Framework Example completed")


if __name__ == "__main__":
    asyncio.run(main()) 