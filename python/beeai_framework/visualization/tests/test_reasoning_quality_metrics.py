#!/usr/bin/env python
"""
Tests for reasoning quality metrics component.

This module contains unit tests for the reasoning quality metrics
component of the visualization framework.
"""

import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..components.reasoning_quality_metrics import (
    ReasoningQualityMetrics,
    QualityMetric,
    MetricLevel
)

from ..components.reasoning_trace_visualizer import (
    ReasoningTrace,
    ReasoningStep
)


@pytest.fixture
def quality_metrics():
    """Create a reasoning quality metrics instance for testing."""
    return ReasoningQualityMetrics()


@pytest.fixture
def sample_steps():
    """Create sample reasoning steps for testing."""
    step1 = ReasoningStep(
        step_number=1,
        step_type="analysis",
        content="This is the first step of the reasoning process. We are analyzing the performance of an ML model.",
        timestamp=datetime.now() - timedelta(minutes=30),
        context_items=[
            {
                "source_id": "source_1",
                "text": "Model performance metrics include accuracy, precision, and recall.",
                "relevance_score": 0.85
            }
        ],
        metrics={},
        key_concepts=[
            {"concept": "ML model", "importance": 0.9},
            {"concept": "performance metrics", "importance": 0.8},
            {"concept": "analysis", "importance": 0.7}
        ],
        next_step_suggestions=["Examine accuracy metrics", "Review data quality"]
    )
    
    step2 = ReasoningStep(
        step_number=2,
        step_type="synthesis",
        content="In this step, we examine the accuracy metrics of the model. The accuracy is 85%, which is good but could be improved.",
        timestamp=datetime.now() - timedelta(minutes=25),
        context_items=[
            {
                "source_id": "source_2",
                "text": "The model achieved 85% accuracy on the test set.",
                "relevance_score": 0.9
            }
        ],
        metrics={},
        key_concepts=[
            {"concept": "accuracy", "importance": 0.95},
            {"concept": "ML model", "importance": 0.8},
            {"concept": "test set", "importance": 0.7}
        ],
        next_step_suggestions=["Analyze precision and recall", "Investigate misclassifications"]
    )
    
    step3 = ReasoningStep(
        step_number=3,
        step_type="analysis",
        content="Now we analyze precision and recall. The precision is 82% and recall is 79%, indicating some imbalance in false positives and negatives.",
        timestamp=datetime.now() - timedelta(minutes=20),
        context_items=[
            {
                "source_id": "source_3",
                "text": "Precision: 82%, Recall: 79%",
                "relevance_score": 0.95
            }
        ],
        metrics={},
        key_concepts=[
            {"concept": "precision", "importance": 0.9},
            {"concept": "recall", "importance": 0.9},
            {"concept": "false positives", "importance": 0.7},
            {"concept": "false negatives", "importance": 0.7}
        ],
        next_step_suggestions=["Analyze confusion matrix", "Suggest improvements"]
    )
    
    step4 = ReasoningStep(
        step_number=4,
        step_type="synthesis",
        content="Based on our analysis, we should adjust the classification threshold to improve recall, even if it slightly reduces precision.",
        timestamp=datetime.now() - timedelta(minutes=15),
        context_items=[
            {
                "source_id": "source_4",
                "text": "Adjusting the classification threshold can trade off precision vs. recall.",
                "relevance_score": 0.9
            }
        ],
        metrics={},
        key_concepts=[
            {"concept": "classification threshold", "importance": 0.95},
            {"concept": "precision-recall tradeoff", "importance": 0.9},
            {"concept": "model improvement", "importance": 0.8}
        ],
        next_step_suggestions=["Implement threshold adjustment", "Validate results"]
    )
    
    return [step1, step2, step3, step4]


@pytest.fixture
def sample_trace(sample_steps):
    """Create a sample reasoning trace for testing."""
    trace = ReasoningTrace(
        trace_id="test-trace-001",
        task="Analyze and improve ML model performance metrics",
        start_time=datetime.now() - timedelta(minutes=35)
    )
    
    # Add all steps to the trace
    for step in sample_steps:
        trace.add_step(step)
    
    # Set end time
    trace.end_time = datetime.now() - timedelta(minutes=10)
    
    return trace


@pytest.fixture
def baseline_trace(sample_steps):
    """Create a baseline trace with slightly different metrics for comparison."""
    trace = ReasoningTrace(
        trace_id="baseline-trace-001",
        task="Analyze ML model performance",
        start_time=datetime.now() - timedelta(minutes=60)
    )
    
    # Create modified steps with slightly lower quality
    for i, step in enumerate(sample_steps):
        modified_step = ReasoningStep(
            step_number=step.step_number,
            step_type=step.step_type,
            content=step.content.replace("improve", "analyze").replace("85%", "80%"),
            timestamp=step.timestamp - timedelta(minutes=20),
            context_items=[item for item in step.context_items],
            metrics={},
            key_concepts=[concept for concept in step.key_concepts],
            next_step_suggestions=[suggestion for suggestion in step.next_step_suggestions]
        )
        trace.add_step(modified_step)
    
    # Set end time
    trace.end_time = datetime.now() - timedelta(minutes=30)
    
    return trace


class TestQualityMetric:
    """Tests for the QualityMetric class."""
    
    def test_initialization(self):
        """Test initialization of QualityMetric."""
        metric = QualityMetric(
            name="test_metric",
            description="A test metric",
            level=MetricLevel.STEP,
            min_value=0.0,
            max_value=1.0,
            target_value=0.8,
            weight=1.5,
            tags=["test", "quality"]
        )
        
        assert metric.name == "test_metric"
        assert metric.description == "A test metric"
        assert metric.level == MetricLevel.STEP
        assert metric.min_value == 0.0
        assert metric.max_value == 1.0
        assert metric.target_value == 0.8
        assert metric.weight == 1.5
        assert metric.tags == ["test", "quality"]


class TestReasoningQualityMetrics:
    """Tests for the ReasoningQualityMetrics class."""
    
    def test_standard_metrics_defined(self, quality_metrics):
        """Test standard metrics are properly defined."""
        # Check that we have the expected standard metrics
        assert len(quality_metrics.metrics) > 0
        
        # Check for specific metrics
        assert "step_coherence" in quality_metrics.metrics
        assert "overall_coherence" in quality_metrics.metrics
        assert "factual_consistency" in quality_metrics.metrics
        assert "goal_alignment" in quality_metrics.metrics
        assert "context_relevance" in quality_metrics.metrics
        assert "solution_completeness" in quality_metrics.metrics
        
        # Check metric properties
        coherence_metric = quality_metrics.metrics["step_coherence"]
        assert coherence_metric.level == MetricLevel.STEP
        assert "coherence" in coherence_metric.tags
    
    def test_evaluate_step_coherence(self, quality_metrics, sample_steps):
        """Test evaluation of step coherence."""
        # Test first step coherence
        first_step_coherence = quality_metrics.evaluate_step_coherence(
            sample_steps[0], None
        )
        assert first_step_coherence == 1.0  # First step should be fully coherent by definition
        
        # Test coherence between steps with shared concepts
        coherence_score = quality_metrics.evaluate_step_coherence(
            sample_steps[1], sample_steps[0]
        )
        assert 0 <= coherence_score <= 1.0
        
        # Test coherence between consecutive steps with shared concepts
        coherence_score = quality_metrics.evaluate_step_coherence(
            sample_steps[2], sample_steps[1]
        )
        assert 0 <= coherence_score <= 1.0
    
    def test_evaluate_context_relevance(self, quality_metrics, sample_steps):
        """Test evaluation of context relevance."""
        for step in sample_steps:
            relevance_score = quality_metrics.evaluate_context_relevance(step)
            assert 0 <= relevance_score <= 1.0
    
    def test_evaluate_step_progress(self, quality_metrics, sample_steps):
        """Test evaluation of step progress."""
        task = "Analyze and improve ML model performance metrics"
        
        # Test progress of first step
        progress_score = quality_metrics.evaluate_step_progress(
            sample_steps[0], None, task
        )
        assert 0 <= progress_score <= 1.0
        
        # Test progress between steps
        progress_score = quality_metrics.evaluate_step_progress(
            sample_steps[1], sample_steps[0], task
        )
        assert 0 <= progress_score <= 1.0
    
    def test_evaluate_trace_coherence(self, quality_metrics, sample_trace):
        """Test evaluation of trace coherence."""
        coherence_score = quality_metrics.evaluate_trace_coherence(sample_trace)
        assert 0 <= coherence_score <= 1.0
    
    def test_evaluate_goal_alignment(self, quality_metrics, sample_trace):
        """Test evaluation of goal alignment."""
        alignment_score = quality_metrics.evaluate_goal_alignment(sample_trace)
        assert 0 <= alignment_score <= 1.0
    
    def test_evaluate_solution_completeness(self, quality_metrics, sample_trace):
        """Test evaluation of solution completeness."""
        completeness_score = quality_metrics.evaluate_solution_completeness(sample_trace)
        assert 0 <= completeness_score <= 1.0
    
    def test_evaluate_comparative_metrics(self, quality_metrics, sample_trace, baseline_trace):
        """Test evaluation of comparative metrics."""
        comparative_metrics = quality_metrics.evaluate_comparative_metrics(
            sample_trace, [baseline_trace]
        )
        
        assert "baseline_improvement" in comparative_metrics
        assert -1.0 <= comparative_metrics["baseline_improvement"] <= 1.0
    
    def test_calculate_all_metrics(self, quality_metrics, sample_trace, baseline_trace):
        """Test calculation of all metrics."""
        all_metrics = quality_metrics.calculate_all_metrics(
            sample_trace, [baseline_trace]
        )
        
        # Check structure of results
        assert "step_metrics" in all_metrics
        assert "trace_metrics" in all_metrics
        assert "comparative_metrics" in all_metrics
        
        # Check step metrics
        step_metrics = all_metrics["step_metrics"]
        assert len(step_metrics) == len(sample_trace.steps)
        for step_num in step_metrics:
            assert "step_coherence" in step_metrics[step_num]
            assert "context_relevance" in step_metrics[step_num]
        
        # Check trace metrics
        trace_metrics = all_metrics["trace_metrics"]
        assert "overall_coherence" in trace_metrics
        assert "goal_alignment" in trace_metrics
        assert "solution_completeness" in trace_metrics
        
        # Check comparative metrics
        comparative_metrics = all_metrics["comparative_metrics"]
        assert "baseline_improvement" in comparative_metrics
    
    def test_get_aggregate_score(self, quality_metrics, sample_trace):
        """Test getting aggregate score."""
        # First calculate metrics to populate cache
        quality_metrics.calculate_all_metrics(sample_trace)
        
        # Get aggregate score
        score = quality_metrics.get_aggregate_score(sample_trace.trace_id)
        
        assert score is not None
        assert 0 <= score <= 1.0 