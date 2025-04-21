#!/usr/bin/env python
"""
Metrics Visualization Component.

This module implements visualization components for reasoning quality
metrics and comparative analysis of reasoning traces.
"""

import json
import math
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import statistics
from pydantic import BaseModel, Field

from ..core.trace_data_model import (
    TraceVisualizationData,
    MetricsVisualizationData
)


class TimeSeriesGenerator:
    """
    Generator for time series visualizations.
    
    This component analyzes metrics over time to generate
    time series data for visualizations.
    """
    
    def generate_step_timing_series(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate time series data for step timing.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Step timing time series data
        """
        # Extract timing data from steps
        step_numbers = []
        durations = []
        step_types = []
        timestamps = []
        
        for step in data.steps:
            step_numbers.append(step.step_number)
            durations.append(step.duration)
            step_types.append(step.step_type)
            timestamps.append(step.timestamp)
        
        # Calculate cumulative time
        cumulative_time = []
        current_cum = 0
        for duration in durations:
            current_cum += duration
            cumulative_time.append(current_cum)
        
        # Format timestamps
        formatted_timestamps = [datetime.fromtimestamp(ts).strftime("%H:%M:%S") for ts in timestamps]
        
        # Generate time series data
        return {
            "step_numbers": step_numbers,
            "durations": durations,
            "step_types": step_types,
            "timestamps": timestamps,
            "formatted_timestamps": formatted_timestamps,
            "cumulative_time": cumulative_time,
            "series_data": [
                {
                    "name": "Step Duration",
                    "type": "bar",
                    "data": list(zip(step_numbers, durations))
                },
                {
                    "name": "Cumulative Time",
                    "type": "line",
                    "data": list(zip(step_numbers, cumulative_time))
                }
            ]
        }
    
    def generate_context_usage_series(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate time series data for context usage.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Context usage time series data
        """
        # Extract context usage data from steps
        step_numbers = []
        context_counts = []
        
        for step in data.steps:
            step_numbers.append(step.step_number)
            context_counts.append(len(step.context_references))
        
        # Calculate averages by step type
        context_by_type = {}
        for step in data.steps:
            if step.step_type not in context_by_type:
                context_by_type[step.step_type] = []
            context_by_type[step.step_type].append(len(step.context_references))
        
        type_averages = {
            step_type: sum(counts) / len(counts) if counts else 0
            for step_type, counts in context_by_type.items()
        }
        
        # Generate time series data
        return {
            "step_numbers": step_numbers,
            "context_counts": context_counts,
            "type_averages": type_averages,
            "total_context_references": sum(context_counts),
            "series_data": [
                {
                    "name": "Context References",
                    "type": "bar",
                    "data": list(zip(step_numbers, context_counts))
                }
            ],
            "average_by_step_type": [
                {
                    "step_type": step_type,
                    "average_context_count": avg
                }
                for step_type, avg in type_averages.items()
            ]
        }
    
    def generate_relevance_series(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate time series data for relevance scores.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Relevance score time series data
        """
        # Extract relevance scores from steps
        step_numbers = []
        avg_relevance_scores = []
        max_relevance_scores = []
        min_relevance_scores = []
        
        for step in data.steps:
            if not step.context_references:
                continue
                
            relevance_scores = [ref.get("relevance_score", 0) for ref in step.context_references]
            
            if relevance_scores:
                step_numbers.append(step.step_number)
                avg_relevance_scores.append(sum(relevance_scores) / len(relevance_scores))
                max_relevance_scores.append(max(relevance_scores))
                min_relevance_scores.append(min(relevance_scores))
        
        # Generate time series data
        return {
            "step_numbers": step_numbers,
            "avg_relevance_scores": avg_relevance_scores,
            "max_relevance_scores": max_relevance_scores,
            "min_relevance_scores": min_relevance_scores,
            "overall_avg_relevance": sum(avg_relevance_scores) / len(avg_relevance_scores) if avg_relevance_scores else 0,
            "series_data": [
                {
                    "name": "Average Relevance",
                    "type": "line",
                    "data": list(zip(step_numbers, avg_relevance_scores))
                },
                {
                    "name": "Max Relevance",
                    "type": "line",
                    "data": list(zip(step_numbers, max_relevance_scores))
                },
                {
                    "name": "Min Relevance",
                    "type": "line",
                    "data": list(zip(step_numbers, min_relevance_scores))
                }
            ]
        }


class ComparisonMetricsGenerator:
    """
    Generator for comparison metrics visualizations.
    
    This component generates comparative metrics analyses between
    traces or against baselines.
    """
    
    def compare_with_baseline(self, data: TraceVisualizationData, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare trace metrics with baseline values.
        
        Args:
            data: Trace visualization data
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            Comparison visualization data
        """
        # Extract metrics from trace
        if not data.metrics:
            return {"error": "No metrics available in trace data"}
        
        trace_metrics = {
            "step_count": len(data.steps),
            "total_time": sum(step.duration for step in data.steps),
            "avg_time_per_step": statistics.mean([step.duration for step in data.steps]) if data.steps else 0,
            "context_references": sum(len(step.context_references) for step in data.steps),
            "avg_context_per_step": statistics.mean([len(step.context_references) for step in data.steps]) if data.steps else 0
        }
        
        # Calculate differences and percentages
        comparison = {}
        for key in trace_metrics:
            if key in baseline_metrics:
                baseline_value = baseline_metrics[key]
                current_value = trace_metrics[key]
                
                if baseline_value == 0:
                    pct_diff = 100.0 if current_value > 0 else 0.0
                else:
                    pct_diff = ((current_value - baseline_value) / baseline_value) * 100
                
                comparison[key] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "absolute_diff": current_value - baseline_value,
                    "percent_diff": pct_diff,
                    "improved": (pct_diff < 0 if key in ["total_time", "avg_time_per_step"] else pct_diff > 0)
                }
        
        return {
            "trace_id": data.trace_id,
            "metrics_comparison": comparison,
            "overall_assessment": self._generate_overall_assessment(comparison)
        }
    
    def _generate_overall_assessment(self, comparison: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an overall assessment from comparison metrics.
        
        Args:
            comparison: Comparison metrics
            
        Returns:
            Overall assessment
        """
        # Count improvements and regressions
        improvements = sum(1 for item in comparison.values() if item.get("improved", False))
        regressions = sum(1 for item in comparison.values() if not item.get("improved", True))
        
        # Calculate an overall score (-100 to 100)
        total_metrics = len(comparison)
        if total_metrics == 0:
            score = 0
        else:
            score = ((improvements - regressions) / total_metrics) * 100
        
        # Generate assessment text
        if score > 50:
            assessment = "Significant improvement over baseline"
        elif score > 20:
            assessment = "Moderate improvement over baseline"
        elif score > -20:
            assessment = "Similar performance to baseline"
        elif score > -50:
            assessment = "Moderate regression from baseline"
        else:
            assessment = "Significant regression from baseline"
        
        return {
            "score": score,
            "improvements": improvements,
            "regressions": regressions,
            "total_metrics": total_metrics,
            "assessment": assessment
        }
    
    def compare_traces(self, primary_data: TraceVisualizationData, comparison_data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Compare metrics between two traces.
        
        Args:
            primary_data: Primary trace visualization data
            comparison_data: Comparison trace visualization data
            
        Returns:
            Trace comparison visualization data
        """
        # Extract metrics from primary trace
        primary_metrics = {
            "step_count": len(primary_data.steps),
            "total_time": sum(step.duration for step in primary_data.steps),
            "avg_time_per_step": statistics.mean([step.duration for step in primary_data.steps]) if primary_data.steps else 0,
            "context_references": sum(len(step.context_references) for step in primary_data.steps),
            "avg_context_per_step": statistics.mean([len(step.context_references) for step in primary_data.steps]) if primary_data.steps else 0
        }
        
        # Use the comparison method
        return self.compare_with_baseline(primary_data, primary_metrics)


class RadarChartGenerator:
    """
    Generator for radar/spider chart visualizations.
    
    This component creates multi-dimensional quality assessment
    visualizations for reasoning traces.
    """
    
    def generate_quality_radar(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate radar chart data for quality assessment.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Quality radar chart data
        """
        # Define quality dimensions to evaluate
        dimensions = [
            "step_count_efficiency",
            "time_efficiency",
            "context_relevance",
            "context_utilization",
            "reasoning_consistency",
            "conclusion_quality"
        ]
        
        # Calculate scores for each dimension (0-1 scale)
        scores = {}
        
        # Step count efficiency - fewer steps is better, but at least 3
        step_count = len(data.steps)
        if step_count < 3:
            scores["step_count_efficiency"] = 0.3  # Too few steps
        elif step_count <= 7:
            scores["step_count_efficiency"] = 1.0 - ((step_count - 3) / 4)  # 3 steps = 1.0, 7 steps = 0.0
        else:
            scores["step_count_efficiency"] = max(0.0, 0.7 - ((step_count - 7) / 10))  # Diminishing penalty
        
        # Time efficiency - less time per step is better
        avg_time_per_step = statistics.mean([step.duration for step in data.steps]) if data.steps else 0
        if avg_time_per_step <= 1.0:
            scores["time_efficiency"] = 1.0
        elif avg_time_per_step <= 5.0:
            scores["time_efficiency"] = 1.0 - ((avg_time_per_step - 1.0) / 4.0)
        else:
            scores["time_efficiency"] = max(0.0, 0.5 - ((avg_time_per_step - 5.0) / 20.0))
        
        # Context relevance - higher average relevance scores are better
        all_relevance_scores = []
        for step in data.steps:
            for ref in step.context_references:
                relevance_score = ref.get("relevance_score", 0)
                if relevance_score > 0:
                    all_relevance_scores.append(relevance_score)
        
        avg_relevance = statistics.mean(all_relevance_scores) if all_relevance_scores else 0
        scores["context_relevance"] = min(1.0, avg_relevance)
        
        # Context utilization - using context is good, but not overwhelming
        avg_context_per_step = statistics.mean([len(step.context_references) for step in data.steps]) if data.steps else 0
        if avg_context_per_step == 0:
            scores["context_utilization"] = 0.0  # No context used
        elif avg_context_per_step <= 5:
            scores["context_utilization"] = 0.6 + (avg_context_per_step / 10.0)  # 0.6 to 1.0
        else:
            scores["context_utilization"] = max(0.6, 1.0 - ((avg_context_per_step - 5) / 15.0))  # Diminishing returns
        
        # Reasoning consistency - steps should build on each other
        # This is an approximation - ideally would do semantic analysis
        step_types = [step.step_type for step in data.steps]
        type_changes = sum(1 for i in range(1, len(step_types)) if step_types[i] != step_types[i-1])
        if type_changes <= step_count / 3:
            scores["reasoning_consistency"] = 1.0  # Good balance of consistency and progression
        else:
            scores["reasoning_consistency"] = max(0.0, 1.0 - ((type_changes - step_count/3) / step_count))
        
        # Conclusion quality - did the trace reach a conclusion?
        if data.final_result and data.final_result.strip():
            conclusion_length = len(data.final_result.strip())
            if conclusion_length > 300:
                scores["conclusion_quality"] = 1.0
            else:
                scores["conclusion_quality"] = min(1.0, conclusion_length / 300.0)
        else:
            scores["conclusion_quality"] = 0.0
        
        # Scale scores to 0-100 for visualization
        scaled_scores = {key: value * 100 for key, value in scores.items()}
        
        # Format data for radar chart
        return {
            "trace_id": data.trace_id,
            "dimensions": dimensions,
            "scores": [scaled_scores[dim] for dim in dimensions],
            "scores_object": scaled_scores,
            "average_score": sum(scaled_scores.values()) / len(scaled_scores),
            "chart_data": [
                {
                    "name": "Quality Score",
                    "data": list(zip(dimensions, [scaled_scores[dim] for dim in dimensions]))
                }
            ]
        }
    
    def generate_step_type_radar(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate radar chart data for step type distribution.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Step type radar chart data
        """
        # Count steps by type
        step_type_counts = {}
        
        for step in data.steps:
            if step.step_type not in step_type_counts:
                step_type_counts[step.step_type] = 0
            step_type_counts[step.step_type] += 1
        
        # Get all unique step types
        step_types = list(step_type_counts.keys())
        
        # Calculate percentages
        total_steps = len(data.steps)
        step_type_percentages = {
            step_type: (count / total_steps) * 100 
            for step_type, count in step_type_counts.items()
        }
        
        # Format data for radar chart
        return {
            "trace_id": data.trace_id,
            "dimensions": step_types,
            "counts": [step_type_counts[step_type] for step_type in step_types],
            "percentages": [step_type_percentages[step_type] for step_type in step_types],
            "total_steps": total_steps,
            "chart_data": [
                {
                    "name": "Step Distribution",
                    "data": list(zip(step_types, [step_type_percentages[st] for st in step_types]))
                }
            ]
        }


class HeatMapGenerator:
    """
    Generator for heat map visualizations.
    
    This component creates heat maps for context usage patterns
    and other multi-dimensional data.
    """
    
    def generate_context_usage_heatmap(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate heat map data for context usage patterns.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Context usage heat map data
        """
        # Get unique step types and context sources
        step_types = list(set(step.step_type for step in data.steps))
        context_sources = list(set(source.source_id for source in data.context_sources))
        
        # Initialize the heat map data
        heatmap_data = []
        
        # Build cell data for each step type and source
        for step_type in step_types:
            row = []
            for source_id in context_sources:
                # Count references to this source in steps of this type
                reference_count = 0
                steps_of_type = [step for step in data.steps if step.step_type == step_type]
                
                for step in steps_of_type:
                    reference_count += sum(1 for ref in step.context_references if ref.get("source") == source_id)
                
                # Calculate average references per step of this type
                if steps_of_type:
                    avg_references = reference_count / len(steps_of_type)
                else:
                    avg_references = 0
                
                row.append(avg_references)
            
            heatmap_data.append(row)
        
        # Get source names for better labels
        source_names = {}
        for source in data.context_sources:
            source_names[source.source_id] = source.name
        
        source_labels = [source_names.get(source_id, source_id) for source_id in context_sources]
        
        return {
            "trace_id": data.trace_id,
            "x_labels": source_labels,
            "y_labels": step_types,
            "data": heatmap_data,
            "max_value": max([max(row) for row in heatmap_data]) if heatmap_data and heatmap_data[0] else 0,
            "chart_data": {
                "data": heatmap_data,
                "x_labels": source_labels,
                "y_labels": step_types
            }
        }
    
    def generate_step_timing_heatmap(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate heat map data for step timing patterns.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Step timing heat map data
        """
        # Get unique step types
        step_types = list(set(step.step_type for step in data.steps))
        
        # Define time segments (in seconds)
        time_segments = ["0-1s", "1-3s", "3-5s", "5-10s", "10s+"]
        
        # Initialize the heat map data
        heatmap_data = []
        
        # Build cell data for each step type and time segment
        for step_type in step_types:
            row = [0, 0, 0, 0, 0]  # Counts for each time segment
            steps_of_type = [step for step in data.steps if step.step_type == step_type]
            
            for step in steps_of_type:
                duration = step.duration
                
                # Assign to the right time segment
                if duration <= 1.0:
                    row[0] += 1
                elif duration <= 3.0:
                    row[1] += 1
                elif duration <= 5.0:
                    row[2] += 1
                elif duration <= 10.0:
                    row[3] += 1
                else:
                    row[4] += 1
            
            # Convert to percentages
            total_steps = len(steps_of_type)
            if total_steps > 0:
                row = [count / total_steps * 100 for count in row]
            
            heatmap_data.append(row)
        
        return {
            "trace_id": data.trace_id,
            "x_labels": time_segments,
            "y_labels": step_types,
            "data": heatmap_data,
            "max_value": max([max(row) for row in heatmap_data]) if heatmap_data and heatmap_data[0] else 0,
            "chart_data": {
                "data": heatmap_data,
                "x_labels": time_segments,
                "y_labels": step_types
            }
        }


class MetricsVisualizer:
    """
    Main visualizer for metrics data.
    
    This class integrates various metrics visualization components
    to provide a comprehensive metrics visualization system.
    """
    
    def __init__(self):
        """Initialize the metrics visualizer."""
        self.time_series_generator = TimeSeriesGenerator()
        self.comparison_generator = ComparisonMetricsGenerator()
        self.radar_generator = RadarChartGenerator()
        self.heatmap_generator = HeatMapGenerator()
    
    def generate_visualization_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate complete visualization data for metrics.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Complete metrics visualization data
        """
        # Generate time series data
        timing_series = self.time_series_generator.generate_step_timing_series(data)
        context_series = self.time_series_generator.generate_context_usage_series(data)
        relevance_series = self.time_series_generator.generate_relevance_series(data)
        
        # Generate radar charts
        quality_radar = self.radar_generator.generate_quality_radar(data)
        step_type_radar = self.radar_generator.generate_step_type_radar(data)
        
        # Generate heat maps
        context_heatmap = self.heatmap_generator.generate_context_usage_heatmap(data)
        timing_heatmap = self.heatmap_generator.generate_step_timing_heatmap(data)
        
        # Create synthetic baseline for comparison (in a real implementation, this would come from a database)
        baseline_metrics = {
            "step_count": 5,
            "total_time": 12.0,
            "avg_time_per_step": 2.4,
            "context_references": 10,
            "avg_context_per_step": 2.0
        }
        
        # Generate comparison data
        comparison = self.comparison_generator.compare_with_baseline(data, baseline_metrics)
        
        # Create summary metrics
        summary_metrics = {
            "trace_id": data.trace_id,
            "step_count": len(data.steps),
            "total_time": sum(step.duration for step in data.steps),
            "avg_time_per_step": statistics.mean([step.duration for step in data.steps]) if data.steps else 0,
            "context_references": sum(len(step.context_references) for step in data.steps),
            "avg_context_per_step": statistics.mean([len(step.context_references) for step in data.steps]) if data.steps else 0,
            "quality_score": quality_radar["average_score"]
        }
        
        # Combine all metrics
        return {
            "trace_id": data.trace_id,
            "summary": summary_metrics,
            "time_series": {
                "timing": timing_series,
                "context": context_series,
                "relevance": relevance_series
            },
            "radar_charts": {
                "quality": quality_radar,
                "step_type": step_type_radar
            },
            "heat_maps": {
                "context_usage": context_heatmap,
                "timing": timing_heatmap
            },
            "comparison": comparison
        }
    
    def to_json(self, data: TraceVisualizationData, **kwargs) -> str:
        """
        Convert visualization data to JSON.
        
        Args:
            data: Trace visualization data
            **kwargs: Additional arguments for json.dumps
            
        Returns:
            JSON representation of visualization data
        """
        visualization_data = self.generate_visualization_data(data)
        return json.dumps(visualization_data, **kwargs)
    
    def generate_dashboard_data(self, data: TraceVisualizationData) -> Dict[str, Any]:
        """
        Generate data for metrics dashboard.
        
        Args:
            data: Trace visualization data
            
        Returns:
            Dashboard visualization data
        """
        # Get basic metrics
        metrics = self.generate_visualization_data(data)
        
        # Extract key metrics for dashboard
        dashboard_data = {
            "trace_id": data.trace_id,
            "task": data.task,
            "summary": metrics["summary"],
            "quality_score": metrics["radar_charts"]["quality"]["average_score"],
            "key_metrics": [
                {"name": "Step Count", "value": len(data.steps)},
                {"name": "Total Time", "value": f"{metrics['summary']['total_time']:.2f}s"},
                {"name": "Avg Time per Step", "value": f"{metrics['summary']['avg_time_per_step']:.2f}s"},
                {"name": "Context References", "value": metrics['summary']['context_references']},
                {"name": "Quality Score", "value": f"{metrics['radar_charts']['quality']['average_score']:.1f}/100"}
            ],
            "top_performing_dimension": max(
                metrics["radar_charts"]["quality"]["scores_object"].items(),
                key=lambda x: x[1]
            )[0],
            "improvement_area": min(
                metrics["radar_charts"]["quality"]["scores_object"].items(),
                key=lambda x: x[1]
            )[0]
        }
        
        return dashboard_data 