#!/usr/bin/env python
"""
Tests for AB Testing Framework component.

This module contains unit tests for the AB Testing Framework
component of the visualization framework.
"""

import pytest
import json
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any

from ..components.ab_testing_framework import (
    ABTestingFramework,
    ExperimentConfig,
    ExperimentResult,
    ExperimentComparison,
    StatisticalAnalysisService,
    VisualizationGenerator
)


@pytest.fixture
def sample_experiment_data():
    """Create sample experiment data for testing."""
    # Create experiment A data
    experiment_a_metrics = {
        "accuracy": [0.82, 0.84, 0.83, 0.85, 0.84],
        "precision": [0.78, 0.80, 0.79, 0.81, 0.80],
        "recall": [0.75, 0.77, 0.76, 0.78, 0.77],
        "f1_score": [0.76, 0.78, 0.77, 0.79, 0.78],
        "latency_ms": [250, 245, 260, 240, 255]
    }
    
    # Create experiment B data
    experiment_b_metrics = {
        "accuracy": [0.86, 0.87, 0.85, 0.88, 0.87],
        "precision": [0.82, 0.83, 0.81, 0.85, 0.84],
        "recall": [0.79, 0.81, 0.78, 0.82, 0.81],
        "f1_score": [0.80, 0.82, 0.79, 0.83, 0.82],
        "latency_ms": [270, 275, 265, 280, 275]
    }
    
    # Convert to ExperimentResult objects
    experiment_a_result = ExperimentResult(
        experiment_id="exp_a",
        name="Baseline Model",
        description="Standard implementation with default parameters",
        metrics=experiment_a_metrics,
        timestamp=datetime.now().timestamp(),
        parameters={
            "model_type": "transformer",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        },
        sample_outputs=[
            {"input": "Sample input 1", "output": "Sample output A1", "metrics": {"confidence": 0.82}},
            {"input": "Sample input 2", "output": "Sample output A2", "metrics": {"confidence": 0.84}}
        ]
    )
    
    experiment_b_result = ExperimentResult(
        experiment_id="exp_b",
        name="Enhanced Model",
        description="Implementation with optimized parameters",
        metrics=experiment_b_metrics,
        timestamp=datetime.now().timestamp(),
        parameters={
            "model_type": "transformer",
            "learning_rate": 0.0005,
            "batch_size": 64,
            "epochs": 15
        },
        sample_outputs=[
            {"input": "Sample input 1", "output": "Sample output B1", "metrics": {"confidence": 0.86}},
            {"input": "Sample input 2", "output": "Sample output B2", "metrics": {"confidence": 0.87}}
        ]
    )
    
    # Create ExperimentConfig
    config = ExperimentConfig(
        metrics_of_interest=["accuracy", "precision", "recall", "f1_score", "latency_ms"],
        higher_is_better={"accuracy": True, "precision": True, "recall": True, "f1_score": True, "latency_ms": False},
        significance_level=0.05,
        visualization_preferences={
            "primary_metric": "accuracy",
            "chart_type": "bar",
            "color_scheme": ["blue", "red"]
        }
    )
    
    return {
        "experiment_a": experiment_a_result,
        "experiment_b": experiment_b_result,
        "config": config
    }


@pytest.fixture
def statistical_service():
    """Create a statistical analysis service instance for testing."""
    return StatisticalAnalysisService()


@pytest.fixture
def visualization_generator():
    """Create a visualization generator instance for testing."""
    return VisualizationGenerator()


@pytest.fixture
def ab_testing_framework():
    """Create an AB testing framework instance for testing."""
    return ABTestingFramework()


class TestExperimentConfig:
    """Tests for the ExperimentConfig class."""
    
    def test_validate_config(self, sample_experiment_data):
        """Test validating experiment configuration."""
        config = sample_experiment_data["config"]
        
        # Valid configuration
        assert config.validate_config() == True
        
        # Test invalid configuration
        invalid_config = ExperimentConfig(
            metrics_of_interest=["accuracy", "unknown_metric"],
            higher_is_better={"accuracy": True},  # Missing higher_is_better for unknown_metric
            significance_level=0.05,
            visualization_preferences={}
        )
        
        with pytest.raises(ValueError):
            invalid_config.validate_config()
    
    def test_get_metric_direction(self, sample_experiment_data):
        """Test getting metric optimization direction."""
        config = sample_experiment_data["config"]
        
        assert config.get_metric_direction("accuracy") == "higher"
        assert config.get_metric_direction("latency_ms") == "lower"
        
        with pytest.raises(KeyError):
            config.get_metric_direction("unknown_metric")


class TestExperimentResult:
    """Tests for the ExperimentResult class."""
    
    def test_get_metric_value(self, sample_experiment_data):
        """Test getting metric values."""
        result = sample_experiment_data["experiment_a"]
        
        # Test getting average metric value
        accuracy = result.get_metric_value("accuracy")
        assert isinstance(accuracy, float)
        assert abs(accuracy - np.mean(result.metrics["accuracy"])) < 1e-6
        
        # Test getting raw metric values
        accuracy_values = result.get_metric_value("accuracy", aggregated=False)
        assert isinstance(accuracy_values, list)
        assert accuracy_values == result.metrics["accuracy"]
        
        # Test getting non-existent metric
        with pytest.raises(KeyError):
            result.get_metric_value("unknown_metric")
    
    def test_get_parameter(self, sample_experiment_data):
        """Test getting parameter values."""
        result = sample_experiment_data["experiment_a"]
        
        assert result.get_parameter("learning_rate") == 0.001
        assert result.get_parameter("batch_size") == 32
        
        # Test default value for missing parameter
        assert result.get_parameter("unknown_param", default="default") == "default"


class TestStatisticalAnalysisService:
    """Tests for the StatisticalAnalysisService class."""
    
    def test_calculate_descriptive_statistics(self, statistical_service, sample_experiment_data):
        """Test calculating descriptive statistics."""
        result = sample_experiment_data["experiment_a"]
        
        stats = statistical_service.calculate_descriptive_statistics(result)
        
        assert isinstance(stats, dict)
        for metric in result.metrics:
            assert metric in stats
            assert "mean" in stats[metric]
            assert "median" in stats[metric]
            assert "std" in stats[metric]
            assert "min" in stats[metric]
            assert "max" in stats[metric]
    
    def test_perform_statistical_test(self, statistical_service, sample_experiment_data):
        """Test performing statistical test between experiments."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        
        results = statistical_service.perform_statistical_test(
            exp_a.metrics["accuracy"],
            exp_b.metrics["accuracy"],
            test_type="t-test"
        )
        
        assert isinstance(results, dict)
        assert "p_value" in results
        assert "test_statistic" in results
        assert "significant" in results
        assert 0 <= results["p_value"] <= 1
        
        # Test with invalid test type
        with pytest.raises(ValueError):
            statistical_service.perform_statistical_test(
                exp_a.metrics["accuracy"],
                exp_b.metrics["accuracy"],
                test_type="invalid_test"
            )
    
    def test_calculate_effect_size(self, statistical_service, sample_experiment_data):
        """Test calculating effect size between experiments."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        
        effect_size = statistical_service.calculate_effect_size(
            exp_a.metrics["accuracy"],
            exp_b.metrics["accuracy"]
        )
        
        assert isinstance(effect_size, float)


class TestVisualizationGenerator:
    """Tests for the VisualizationGenerator class."""
    
    def test_generate_metric_comparison_chart(self, visualization_generator, sample_experiment_data):
        """Test generating metric comparison chart."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        config = sample_experiment_data["config"]
        
        chart = visualization_generator.generate_metric_comparison_chart(
            [exp_a, exp_b],
            metric="accuracy",
            chart_type="bar"
        )
        
        assert isinstance(chart, dict)
        assert "data" in chart
        assert "layout" in chart
        assert isinstance(chart["data"], list)
        assert len(chart["data"]) == 2
    
    def test_generate_distribution_plot(self, visualization_generator, sample_experiment_data):
        """Test generating distribution plot."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        
        chart = visualization_generator.generate_distribution_plot(
            [exp_a, exp_b],
            metric="accuracy"
        )
        
        assert isinstance(chart, dict)
        assert "data" in chart
        assert "layout" in chart
        assert isinstance(chart["data"], list)
    
    def test_generate_parameter_impact_chart(self, visualization_generator, sample_experiment_data):
        """Test generating parameter impact chart."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        
        chart = visualization_generator.generate_parameter_impact_chart(
            [exp_a, exp_b],
            parameter="learning_rate",
            metric="accuracy"
        )
        
        assert isinstance(chart, dict)
        assert "data" in chart
        assert "layout" in chart


class TestExperimentComparison:
    """Tests for the ExperimentComparison class."""
    
    def test_compare_experiments(self, sample_experiment_data, statistical_service):
        """Test comparing experiments."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        config = sample_experiment_data["config"]
        
        comparison = ExperimentComparison(
            experiment_a=exp_a,
            experiment_b=exp_b,
            config=config,
            statistical_service=statistical_service
        )
        
        results = comparison.compare_experiments()
        
        assert isinstance(results, dict)
        for metric in config.metrics_of_interest:
            assert metric in results
            assert "difference" in results[metric]
            assert "p_value" in results[metric]
            assert "significant" in results[metric]
    
    def test_determine_winner(self, sample_experiment_data, statistical_service):
        """Test determining winner between experiments."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        config = sample_experiment_data["config"]
        
        comparison = ExperimentComparison(
            experiment_a=exp_a,
            experiment_b=exp_b,
            config=config,
            statistical_service=statistical_service
        )
        
        # Compare experiments
        comparison.compare_experiments()
        
        # Determine winner for accuracy (higher is better)
        winner = comparison.determine_winner("accuracy")
        assert winner in ["experiment_a", "experiment_b", "tie"]
        
        # For this sample data, experiment_b should have better accuracy
        assert winner == "experiment_b"
        
        # Determine winner across all metrics
        overall_winner = comparison.determine_overall_winner()
        assert isinstance(overall_winner, dict)
        assert "winner" in overall_winner
        assert "metrics_won" in overall_winner
        assert "summary" in overall_winner


class TestABTestingFramework:
    """Tests for the ABTestingFramework class."""
    
    def test_load_experiment_results(self, ab_testing_framework, sample_experiment_data):
        """Test loading experiment results."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        
        ab_testing_framework.load_experiment_results([exp_a, exp_b])
        
        assert len(ab_testing_framework.experiments) == 2
        assert ab_testing_framework.experiments[0].experiment_id == exp_a.experiment_id
        assert ab_testing_framework.experiments[1].experiment_id == exp_b.experiment_id
    
    def test_set_experiment_config(self, ab_testing_framework, sample_experiment_data):
        """Test setting experiment configuration."""
        config = sample_experiment_data["config"]
        
        ab_testing_framework.set_experiment_config(config)
        
        assert ab_testing_framework.config is not None
        assert ab_testing_framework.config.metrics_of_interest == config.metrics_of_interest
    
    def test_run_experiment_comparison(self, ab_testing_framework, sample_experiment_data):
        """Test running experiment comparison."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        config = sample_experiment_data["config"]
        
        # Setup framework
        ab_testing_framework.load_experiment_results([exp_a, exp_b])
        ab_testing_framework.set_experiment_config(config)
        
        # Run comparison
        comparison_results = ab_testing_framework.run_experiment_comparison(
            exp_a.experiment_id,
            exp_b.experiment_id
        )
        
        assert isinstance(comparison_results, dict)
        assert "metrics_comparison" in comparison_results
        assert "winner" in comparison_results
        assert "visualizations" in comparison_results
    
    def test_generate_experiment_report(self, ab_testing_framework, sample_experiment_data):
        """Test generating experiment report."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        config = sample_experiment_data["config"]
        
        # Setup framework
        ab_testing_framework.load_experiment_results([exp_a, exp_b])
        ab_testing_framework.set_experiment_config(config)
        
        # Run comparison
        ab_testing_framework.run_experiment_comparison(
            exp_a.experiment_id,
            exp_b.experiment_id
        )
        
        # Generate report
        report = ab_testing_framework.generate_experiment_report()
        
        assert isinstance(report, dict)
        assert "title" in report
        assert "experiments" in report
        assert "comparisons" in report
        assert "summary" in report
        assert "visualizations" in report
    
    def test_to_json(self, ab_testing_framework, sample_experiment_data):
        """Test exporting results to JSON."""
        exp_a = sample_experiment_data["experiment_a"]
        exp_b = sample_experiment_data["experiment_b"]
        config = sample_experiment_data["config"]
        
        # Setup framework
        ab_testing_framework.load_experiment_results([exp_a, exp_b])
        ab_testing_framework.set_experiment_config(config)
        
        # Run comparison
        ab_testing_framework.run_experiment_comparison(
            exp_a.experiment_id,
            exp_b.experiment_id
        )
        
        # Convert to JSON
        json_result = ab_testing_framework.to_json()
        
        assert isinstance(json_result, str)
        parsed = json.loads(json_result)
        assert isinstance(parsed, dict)
        assert "experiments" in parsed
        assert "comparisons" in parsed 