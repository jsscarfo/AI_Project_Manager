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
Tests for the EvaluationDashboard component.

This module contains tests for visualizing and analyzing evaluation results,
performance metrics, and model comparisons.
"""

import pytest
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from beeai_framework.visualization.components.evaluation_dashboard import (
    EvaluationDashboard,
    EvaluationResult,
    ModelEvaluation
)


@pytest.fixture
def sample_evaluation_results() -> List[EvaluationResult]:
    """Create sample evaluation results for testing."""
    return [
        EvaluationResult(
            id="eval1",
            model_id="model_a",
            dataset_id="dataset1",
            task_type="classification",
            timestamp=datetime.datetime(2025, 1, 1, 12, 0),
            metrics={
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.79,
                "f1_score": 0.80,
                "latency_ms": 120,
            },
            metadata={
                "evaluator": "automated_pipeline",
                "environment": "test",
                "version": "1.0.0"
            }
        ),
        EvaluationResult(
            id="eval2",
            model_id="model_b",
            dataset_id="dataset1",
            task_type="classification",
            timestamp=datetime.datetime(2025, 1, 1, 13, 0),
            metrics={
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.83,
                "f1_score": 0.84,
                "latency_ms": 150,
            },
            metadata={
                "evaluator": "automated_pipeline",
                "environment": "test",
                "version": "1.0.0"
            }
        ),
        EvaluationResult(
            id="eval3",
            model_id="model_a",
            dataset_id="dataset2",
            task_type="classification",
            timestamp=datetime.datetime(2025, 1, 2, 12, 0),
            metrics={
                "accuracy": 0.82,
                "precision": 0.80,
                "recall": 0.78,
                "f1_score": 0.79,
                "latency_ms": 125,
            },
            metadata={
                "evaluator": "automated_pipeline",
                "environment": "test",
                "version": "1.0.0"
            }
        ),
        EvaluationResult(
            id="eval4",
            model_id="model_c",
            dataset_id="dataset1",
            task_type="regression",
            timestamp=datetime.datetime(2025, 1, 3, 12, 0),
            metrics={
                "rmse": 0.25,
                "mae": 0.18,
                "r2_score": 0.76,
                "latency_ms": 90,
            },
            metadata={
                "evaluator": "manual",
                "environment": "prod",
                "version": "1.0.0"
            }
        )
    ]


@pytest.fixture
def sample_model_evaluations() -> List[ModelEvaluation]:
    """Create sample model evaluations for testing."""
    return [
        ModelEvaluation(
            model_id="model_a",
            model_name="Model A",
            version="1.0.0",
            task_type="classification",
            performance_over_time=[
                {
                    "timestamp": datetime.datetime(2025, 1, 1),
                    "accuracy": 0.85,
                    "latency_ms": 120
                },
                {
                    "timestamp": datetime.datetime(2025, 1, 8),
                    "accuracy": 0.86,
                    "latency_ms": 118
                },
                {
                    "timestamp": datetime.datetime(2025, 1, 15),
                    "accuracy": 0.87,
                    "latency_ms": 115
                }
            ],
            metrics_by_category={
                "category1": {"accuracy": 0.90, "f1_score": 0.88},
                "category2": {"accuracy": 0.82, "f1_score": 0.80},
                "category3": {"accuracy": 0.78, "f1_score": 0.75}
            },
            metadata={
                "training_dataset": "dataset_train_1",
                "architecture": "transformer",
                "parameters": "10B"
            }
        ),
        ModelEvaluation(
            model_id="model_b",
            model_name="Model B",
            version="2.0.0",
            task_type="classification",
            performance_over_time=[
                {
                    "timestamp": datetime.datetime(2025, 1, 1),
                    "accuracy": 0.87,
                    "latency_ms": 150
                },
                {
                    "timestamp": datetime.datetime(2025, 1, 8),
                    "accuracy": 0.88,
                    "latency_ms": 145
                },
                {
                    "timestamp": datetime.datetime(2025, 1, 15),
                    "accuracy": 0.89,
                    "latency_ms": 142
                }
            ],
            metrics_by_category={
                "category1": {"accuracy": 0.92, "f1_score": 0.90},
                "category2": {"accuracy": 0.85, "f1_score": 0.83},
                "category3": {"accuracy": 0.82, "f1_score": 0.80}
            },
            metadata={
                "training_dataset": "dataset_train_2",
                "architecture": "transformer",
                "parameters": "20B"
            }
        )
    ]


@pytest.fixture
def dashboard() -> EvaluationDashboard:
    """Create an EvaluationDashboard instance for testing."""
    return EvaluationDashboard(
        default_height=600,
        default_width=1000,
        theme="light"
    )


class TestEvaluationDashboard:
    """Tests for the EvaluationDashboard class."""
    
    def test_init(self, dashboard):
        """Test initialization of the dashboard component."""
        assert dashboard.default_height == 600
        assert dashboard.default_width == 1000
        assert dashboard.theme == "light"
    
    def test_prepare_evaluation_dataframe(self, dashboard, sample_evaluation_results):
        """Test preparation of evaluation results dataframe."""
        df = dashboard._prepare_evaluation_dataframe(sample_evaluation_results)
        
        # Check dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == len(sample_evaluation_results)
        
        # Check columns
        expected_columns = ["id", "model_id", "dataset_id", "task_type", "timestamp"]
        for col in expected_columns:
            assert col in df.columns
        
        # Check metrics expansion
        assert "accuracy" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1_score" in df.columns
        assert "latency_ms" in df.columns
        
        # Check metadata expansion
        assert "evaluator" in df.columns
        assert "environment" in df.columns
        assert "version" in df.columns
    
    def test_create_model_comparison_chart(self, dashboard, sample_evaluation_results):
        """Test creation of model comparison chart."""
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        fig = dashboard.create_model_comparison_chart(
            sample_evaluation_results[:2],  # Same dataset, different models
            metrics=metrics,
            group_by="model_id"
        )
        
        # Check figure type
        assert isinstance(fig, go.Figure)
        
        # Check figure dimensions
        assert fig.layout.height == 600
        assert fig.layout.width == 1000
        
        # Check custom dimensions
        custom_fig = dashboard.create_model_comparison_chart(
            sample_evaluation_results[:2],
            metrics=metrics,
            group_by="model_id",
            height=700,
            width=1200
        )
        assert custom_fig.layout.height == 700
        assert custom_fig.layout.width == 1200
        
        # Check data
        assert len(fig.data) > 0
        
        # Check that all specified metrics are included
        chart_data = [trace.name for trace in fig.data]
        for metric in metrics:
            assert any(metric in name for name in chart_data)
    
    def test_create_performance_over_time_chart(self, dashboard, sample_model_evaluations):
        """Test creation of performance over time chart."""
        metrics = ["accuracy", "latency_ms"]
        fig = dashboard.create_performance_over_time_chart(
            sample_model_evaluations,
            metrics=metrics
        )
        
        # Check figure type
        assert isinstance(fig, go.Figure)
        
        # Check figure dimensions
        assert fig.layout.height == 600
        assert fig.layout.width == 1000
        
        # Check data
        assert len(fig.data) > 0
        
        # There should be traces for each model and metric combination
        expected_trace_count = len(sample_model_evaluations) * len(metrics)
        assert len(fig.data) == expected_trace_count
    
    def test_create_category_performance_chart(self, dashboard, sample_model_evaluations):
        """Test creation of category performance chart."""
        fig = dashboard.create_category_performance_chart(
            sample_model_evaluations[0],
            metric="accuracy"
        )
        
        # Check figure type
        assert isinstance(fig, go.Figure)
        
        # Check figure dimensions
        assert fig.layout.height == 600
        assert fig.layout.width == 1000
        
        # Check data
        assert len(fig.data) == 1  # One trace for bar chart
        
        # There should be as many bars as categories
        bar_data = fig.data[0]
        assert len(bar_data.x) == len(sample_model_evaluations[0].metrics_by_category)
        
        # Test with multiple models
        multi_model_fig = dashboard.create_category_performance_chart(
            sample_model_evaluations,
            metric="accuracy"
        )
        assert len(multi_model_fig.data) == len(sample_model_evaluations)
    
    def test_create_metric_distribution_chart(self, dashboard, sample_evaluation_results):
        """Test creation of metric distribution chart."""
        fig = dashboard.create_metric_distribution_chart(
            sample_evaluation_results[:3],  # Only classification task
            metric="accuracy"
        )
        
        # Check figure type
        assert isinstance(fig, go.Figure)
        
        # Check data
        assert len(fig.data) > 0
        
        # Test with grouping
        grouped_fig = dashboard.create_metric_distribution_chart(
            sample_evaluation_results[:3],
            metric="accuracy",
            group_by="model_id"
        )
        
        # Should have a trace for each unique model_id
        unique_models = len(set(r.model_id for r in sample_evaluation_results[:3]))
        assert len(grouped_fig.data) == unique_models
    
    def test_create_metrics_radar_chart(self, dashboard, sample_evaluation_results):
        """Test creation of metrics radar chart."""
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        fig = dashboard.create_metrics_radar_chart(
            sample_evaluation_results[:2],  # Same dataset, different models
            metrics=metrics
        )
        
        # Check figure type
        assert isinstance(fig, go.Figure)
        
        # Check figure dimensions
        assert fig.layout.height == 600
        assert fig.layout.width == 1000
        
        # Check data
        assert len(fig.data) == 2  # Two models
        
        # Each trace should have the same number of points as metrics
        for trace in fig.data:
            assert len(trace.r) == len(metrics)
    
    def test_create_metric_heatmap(self, dashboard, sample_model_evaluations):
        """Test creation of metric heatmap across categories and models."""
        fig = dashboard.create_metric_heatmap(
            sample_model_evaluations,
            metric="accuracy"
        )
        
        # Check figure type
        assert isinstance(fig, go.Figure)
        
        # Check figure dimensions
        assert fig.layout.height == 600
        assert fig.layout.width == 1000
        
        # Check data
        assert len(fig.data) == 1  # One heatmap trace
        
        # Heatmap dimensions should match models and categories
        heatmap = fig.data[0]
        num_models = len(sample_model_evaluations)
        num_categories = len(sample_model_evaluations[0].metrics_by_category)
        assert len(heatmap.x) == num_categories  # Categories on x-axis
        assert len(heatmap.y) == num_models      # Models on y-axis
    
    def test_create_performance_vs_latency_chart(self, dashboard, sample_evaluation_results):
        """Test creation of performance vs latency scatter chart."""
        fig = dashboard.create_performance_vs_latency_chart(
            sample_evaluation_results[:3],  # Only classification task
            performance_metric="accuracy"
        )
        
        # Check figure type
        assert isinstance(fig, go.Figure)
        
        # Check data
        assert len(fig.data) == 1  # One scatter trace
        
        # Each point represents one evaluation result
        scatter = fig.data[0]
        assert len(scatter.x) == len(sample_evaluation_results[:3])
        assert len(scatter.y) == len(sample_evaluation_results[:3])
        
        # Test with grouping
        grouped_fig = dashboard.create_performance_vs_latency_chart(
            sample_evaluation_results[:3],
            performance_metric="accuracy",
            group_by="model_id"
        )
        
        # Should have a trace for each unique model_id
        unique_models = len(set(r.model_id for r in sample_evaluation_results[:3]))
        assert len(grouped_fig.data) == unique_models
    
    def test_create_dashboard(self, dashboard, sample_evaluation_results, sample_model_evaluations):
        """Test creation of complete dashboard."""
        fig = dashboard.create_dashboard(
            evaluation_results=sample_evaluation_results[:3],  # Only classification task
            model_evaluations=sample_model_evaluations
        )
        
        # Check figure type
        assert isinstance(fig, go.Figure)
        
        # It should be a subplot figure with multiple subplots
        assert hasattr(fig, 'layout')
        assert hasattr(fig.layout, 'annotations')
        
        # Check figure dimensions
        assert fig.layout.height > 600  # Should be taller for multiple plots
        assert fig.layout.width == 1000
        
        # Check custom dimensions and title
        custom_fig = dashboard.create_dashboard(
            evaluation_results=sample_evaluation_results[:3],
            model_evaluations=sample_model_evaluations,
            height=1200,
            width=1500,
            title="Custom Dashboard Title"
        )
        assert custom_fig.layout.height == 1200
        assert custom_fig.layout.width == 1500
        assert custom_fig.layout.title.text == "Custom Dashboard Title"
    
    def test_export_to_html(self, dashboard, sample_evaluation_results, tmp_path):
        """Test exporting dashboard to HTML."""
        # Create a simple figure
        fig = dashboard.create_model_comparison_chart(
            sample_evaluation_results[:2],
            metrics=["accuracy", "f1_score"],
            group_by="model_id"
        )
        
        # Export to HTML
        html_path = tmp_path / "test_dashboard.html"
        html_file = str(html_path)
        dashboard.export_to_html(fig, html_file)
        
        # Check that file exists and has content
        assert html_path.exists()
        content = html_path.read_text()
        assert len(content) > 0
        assert "<html>" in content.lower()
        assert "plotly" in content.lower()
    
    def test_export_evaluation_report(self, dashboard, sample_evaluation_results, 
                                     sample_model_evaluations, tmp_path):
        """Test exporting a complete evaluation report."""
        # Export report
        json_path = tmp_path / "evaluation_report.json"
        json_file = str(json_path)
        
        dashboard.export_evaluation_report(
            evaluation_results=sample_evaluation_results,
            model_evaluations=sample_model_evaluations,
            output_file=json_file
        )
        
        # Check that file exists
        assert json_path.exists()
        
        # Verify JSON content
        with open(json_file, 'r') as f:
            report = json.load(f)
        
        # Check report structure
        assert "report_metadata" in report
        assert "timestamp" in report["report_metadata"]
        assert "evaluation_summary" in report
        assert "model_comparisons" in report
        assert "category_performance" in report


if __name__ == "__main__":
    # Simple test with dummy data
    dashboard = EvaluationDashboard()
    
    # Create a minimal test evaluation result
    test_eval = EvaluationResult(
        id="test-eval",
        model_id="test-model",
        dataset_id="test-dataset",
        task_type="classification",
        timestamp=datetime.datetime.now(),
        metrics={
            "accuracy": 0.85,
            "f1_score": 0.82,
            "latency_ms": 120
        },
        metadata={"test": "true"}
    )
    
    # Create a simple chart
    fig = dashboard.create_metric_distribution_chart([test_eval], metric="accuracy")
    print(f"Created chart with {len(fig.data)} traces")