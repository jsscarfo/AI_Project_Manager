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
Reasoning Quality Metrics System

This module provides components for evaluating the quality of reasoning
in sequential thinking processes, including coherence, consistency,
context relevance, step effectiveness, and solution quality metrics.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime
import asyncio
import re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from enum import Enum

# BeeAI Imports
from beeai_framework.vector.sequential_thinking_integration import SequentialKnowledgeIntegration
from beeai_framework.vector.knowledge_retrieval import StepContextManager
from beeai_framework.visualization.components.reasoning_trace_visualizer import ReasoningTrace, ReasoningStep

logger = logging.getLogger(__name__)


class MetricLevel(Enum):
    """Levels for quality metrics."""
    
    STEP = "step"  # Metrics evaluated at individual step level
    TRACE = "trace"  # Metrics evaluated across entire reasoning trace
    COMPARATIVE = "comparative"  # Metrics comparing against baselines or other traces


@dataclass
class QualityMetric:
    """Definition of a quality metric with metadata."""
    
    name: str
    description: str
    level: MetricLevel
    min_value: float = 0.0
    max_value: float = 1.0
    target_value: Optional[float] = None
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)


class ReasoningQualityMetrics:
    """
    System for evaluating the quality of reasoning processes.
    
    This class provides methods for assessing reasoning quality across
    different dimensions including coherence, consistency, context relevance,
    step effectiveness, and final solution quality.
    """
    
    def __init__(self):
        """Initialize the reasoning quality metrics system."""
        # Define standard metrics
        self.metrics: Dict[str, QualityMetric] = {}
        self._define_standard_metrics()
        
        # Cache for computed metrics
        self.cache: Dict[str, Dict[str, float]] = {}
    
    def _define_standard_metrics(self) -> None:
        """Define the standard set of metrics for reasoning quality."""
        # Coherence metrics
        self.metrics["step_coherence"] = QualityMetric(
            name="step_coherence",
            description="Coherence between a step and its previous step",
            level=MetricLevel.STEP,
            tags=["coherence"]
        )
        
        self.metrics["overall_coherence"] = QualityMetric(
            name="overall_coherence",
            description="Overall coherence across all reasoning steps",
            level=MetricLevel.TRACE,
            tags=["coherence"]
        )
        
        # Consistency metrics
        self.metrics["factual_consistency"] = QualityMetric(
            name="factual_consistency",
            description="Consistency of facts and claims across steps",
            level=MetricLevel.TRACE,
            tags=["consistency"]
        )
        
        self.metrics["goal_alignment"] = QualityMetric(
            name="goal_alignment",
            description="Alignment of reasoning with the original goal",
            level=MetricLevel.TRACE,
            tags=["consistency", "relevance"]
        )
        
        # Context relevance metrics
        self.metrics["context_relevance"] = QualityMetric(
            name="context_relevance",
            description="Relevance of context used in a step",
            level=MetricLevel.STEP,
            tags=["context", "relevance"]
        )
        
        self.metrics["context_utilization"] = QualityMetric(
            name="context_utilization",
            description="Effective utilization of provided context",
            level=MetricLevel.STEP,
            tags=["context", "effectiveness"]
        )
        
        # Step effectiveness metrics
        self.metrics["step_progress"] = QualityMetric(
            name="step_progress",
            description="Progress made by a step toward the solution",
            level=MetricLevel.STEP,
            tags=["effectiveness", "progress"]
        )
        
        self.metrics["insight_generation"] = QualityMetric(
            name="insight_generation",
            description="Generation of novel insights in a step",
            level=MetricLevel.STEP,
            tags=["effectiveness", "creativity"]
        )
        
        # Solution quality metrics
        self.metrics["solution_completeness"] = QualityMetric(
            name="solution_completeness",
            description="Completeness of the final solution",
            level=MetricLevel.TRACE,
            tags=["solution", "completeness"]
        )
        
        self.metrics["solution_correctness"] = QualityMetric(
            name="solution_correctness",
            description="Correctness of the final solution",
            level=MetricLevel.TRACE,
            tags=["solution", "correctness"]
        )
        
        # Comparative metrics
        self.metrics["baseline_improvement"] = QualityMetric(
            name="baseline_improvement",
            description="Improvement over baseline approaches",
            level=MetricLevel.COMPARATIVE,
            min_value=-1.0,
            max_value=1.0,
            tags=["comparative"]
        )
    
    def evaluate_step_coherence(self, step: ReasoningStep, previous_step: Optional[ReasoningStep] = None) -> float:
        """
        Evaluate the coherence between a step and its previous step.
        
        Args:
            step: Current reasoning step
            previous_step: Previous reasoning step (if available)
            
        Returns:
            Coherence score between 0.0 and 1.0
        """
        if not previous_step:
            # First step is always coherent by definition
            return 1.0
        
        # Simple coherence evaluation based on shared concepts
        current_concepts = set()
        previous_concepts = set()
        
        # Extract concepts from current step
        for concept in step.key_concepts:
            concept_text = concept.get("concept", "").lower()
            if concept_text:
                current_concepts.add(concept_text)
        
        # Extract concepts from previous step
        for concept in previous_step.key_concepts:
            concept_text = concept.get("concept", "").lower()
            if concept_text:
                previous_concepts.add(concept_text)
        
        # Calculate Jaccard similarity between concept sets
        if not current_concepts and not previous_concepts:
            return 0.5  # Neutral score if no concepts found
        
        intersection = len(current_concepts.intersection(previous_concepts))
        union = len(current_concepts.union(previous_concepts))
        
        if union == 0:
            return 0.5  # Neutral score if no concepts found
        
        jaccard = intersection / union
        
        # Adjust scale to be more lenient
        # We don't expect perfect overlap between steps
        coherence = 0.5 + 0.5 * jaccard
        
        return coherence
    
    def evaluate_context_relevance(self, step: ReasoningStep) -> float:
        """
        Evaluate the relevance of context used in a reasoning step.
        
        Args:
            step: Reasoning step to evaluate
            
        Returns:
            Context relevance score between 0.0 and 1.0
        """
        if not step.context_items:
            return 0.0  # No context used
        
        # Get explicit relevance scores if available
        explicit_scores = [
            item.get("similarity", 0.0) 
            for item in step.context_items
        ]
        
        if explicit_scores:
            # Use average of explicit scores
            return min(1.0, sum(explicit_scores) / len(explicit_scores))
        
        # Fallback: Check for context mentions in the step content
        mentions = 0
        text = step.content.lower()
        
        for item in step.context_items:
            content = item.get("content", "").lower()
            if not content:
                continue
                
            # Check for substantial phrases from context in the content
            phrases = content.split('.')
            for phrase in phrases:
                phrase = phrase.strip()
                if len(phrase) >= 10 and phrase in text:
                    mentions += 1
        
        if not step.context_items:
            return 0.0
            
        mention_ratio = mentions / len(step.context_items)
        # Scale to 0.0-1.0 range
        return min(1.0, mention_ratio)
    
    def evaluate_step_progress(
        self, 
        step: ReasoningStep, 
        previous_step: Optional[ReasoningStep] = None,
        task_description: str = ""
    ) -> float:
        """
        Evaluate the progress made by a step toward the solution.
        
        Args:
            step: Current reasoning step
            previous_step: Previous reasoning step (if available)
            task_description: Description of the task being solved
            
        Returns:
            Progress score between 0.0 and 1.0
        """
        # Implementation would ideally use semantic analysis to determine
        # how much progress was made toward the final goal
        
        # Simple heuristic implementation for now
        
        # Check if step introduces concrete ideas, actions, or decisions
        concrete_indicators = [
            "should", "will", "must", "recommend", "propose", "implement",
            "solution", "approach", "strategy", "plan", "design", "architecture"
        ]
        
        concrete_score = 0
        for indicator in concrete_indicators:
            if indicator in step.content.lower():
                concrete_score += 1
        
        # Normalize concrete score
        concrete_score = min(1.0, concrete_score / 5)
        
        # Step position bonus - later steps should be more concrete
        position_factor = min(1.0, (step.step_number / 5) * 0.5)
        
        # Combine factors
        progress_score = 0.3 * concrete_score + 0.2 * position_factor
        
        # Add 0.5 baseline to avoid overly harsh scores
        return min(1.0, 0.5 + progress_score)
    
    def evaluate_trace_coherence(self, trace: ReasoningTrace) -> float:
        """
        Evaluate the overall coherence across all reasoning steps.
        
        Args:
            trace: Complete reasoning trace
            
        Returns:
            Overall coherence score between 0.0 and 1.0
        """
        steps = trace.steps
        if len(steps) <= 1:
            return 1.0  # Single step is coherent by definition
        
        # Calculate pairwise coherence between consecutive steps
        coherence_scores = []
        for i in range(1, len(steps)):
            current_step = steps[i]
            previous_step = steps[i-1]
            
            coherence = self.evaluate_step_coherence(current_step, previous_step)
            coherence_scores.append(coherence)
        
        # Average coherence across step transitions
        return sum(coherence_scores) / len(coherence_scores)
    
    def evaluate_goal_alignment(self, trace: ReasoningTrace) -> float:
        """
        Evaluate the alignment of reasoning with the original goal.
        
        Args:
            trace: Complete reasoning trace
            
        Returns:
            Goal alignment score between 0.0 and 1.0
        """
        if not trace.steps:
            return 0.0
        
        # Extract task description
        task = trace.task.lower()
        
        # Check final steps for alignment with the task
        final_steps = trace.steps[-min(3, len(trace.steps)):]
        final_content = " ".join([step.content.lower() for step in final_steps])
        
        # Extract key phrases from task
        task_phrases = []
        for phrase in task.split():
            if len(phrase) >= 5:
                task_phrases.append(phrase)
        
        # Count phrase occurrences in final steps
        matches = 0
        for phrase in task_phrases:
            if phrase in final_content:
                matches += 1
        
        # Calculate match ratio
        if not task_phrases:
            return 0.5  # Neutral if no phrases to match
            
        match_ratio = matches / len(task_phrases)
        
        # Scale to 0.0-1.0 range, with 0.5 baseline
        return min(1.0, 0.5 + (match_ratio * 0.5))
    
    def evaluate_solution_completeness(self, trace: ReasoningTrace) -> float:
        """
        Evaluate the completeness of the final solution.
        
        Args:
            trace: Complete reasoning trace
            
        Returns:
            Solution completeness score between 0.0 and 1.0
        """
        if not trace.steps:
            return 0.0
        
        # Get last step as the solution
        solution_step = trace.steps[-1]
        solution_text = solution_step.content.lower()
        
        # Check for completion indicators
        completion_indicators = [
            "conclusion", "finally", "in summary", "to summarize",
            "the solution is", "we recommend", "to conclude", 
            "in conclusion", "therefore"
        ]
        
        indicator_score = 0
        for indicator in completion_indicators:
            if indicator in solution_text:
                indicator_score += 1
        
        # Normalize indicator score
        indicator_score = min(1.0, indicator_score / 3)
        
        # Length-based heuristic - longer solutions tend to be more complete
        # (up to a reasonable limit)
        solution_length = len(solution_text)
        length_score = min(1.0, solution_length / 1000)
        
        # Step count heuristic - more steps suggest more thorough reasoning
        step_count_score = min(1.0, len(trace.steps) / 5)
        
        # Combine factors
        completeness_score = (0.4 * indicator_score) + (0.3 * length_score) + (0.3 * step_count_score)
        
        return completeness_score
    
    def evaluate_comparative_metrics(
        self, 
        trace: ReasoningTrace,
        baseline_traces: List[ReasoningTrace]
    ) -> Dict[str, float]:
        """
        Evaluate comparative metrics against baseline approaches.
        
        Args:
            trace: Current reasoning trace to evaluate
            baseline_traces: List of baseline traces for comparison
            
        Returns:
            Dictionary of comparative metric scores
        """
        if not baseline_traces:
            return {"baseline_improvement": 0.0}
        
        # Calculate core metrics for current trace
        current_metrics = {
            "coherence": self.evaluate_trace_coherence(trace),
            "goal_alignment": self.evaluate_goal_alignment(trace),
            "completeness": self.evaluate_solution_completeness(trace)
        }
        
        # Calculate same metrics for each baseline
        baseline_metrics = []
        for baseline in baseline_traces:
            metrics = {
                "coherence": self.evaluate_trace_coherence(baseline),
                "goal_alignment": self.evaluate_goal_alignment(baseline),
                "completeness": self.evaluate_solution_completeness(baseline)
            }
            baseline_metrics.append(metrics)
        
        # Calculate average metrics across baselines
        avg_baseline = {}
        for key in current_metrics.keys():
            avg_baseline[key] = sum(b[key] for b in baseline_metrics) / len(baseline_metrics)
        
        # Calculate improvement ratios
        improvements = {}
        for key in current_metrics.keys():
            if avg_baseline[key] > 0:
                improvements[key] = (current_metrics[key] - avg_baseline[key]) / avg_baseline[key]
            else:
                improvements[key] = 0.0 if current_metrics[key] == 0 else 1.0
        
        # Calculate overall improvement
        weights = {"coherence": 0.3, "goal_alignment": 0.4, "completeness": 0.3}
        weighted_improvement = sum(improvements[k] * weights[k] for k in weights.keys())
        
        # Normalize to -1.0 to 1.0 range
        baseline_improvement = max(-1.0, min(1.0, weighted_improvement))
        
        return {"baseline_improvement": baseline_improvement}
    
    def calculate_all_metrics(
        self, 
        trace: ReasoningTrace,
        baseline_traces: Optional[List[ReasoningTrace]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate all defined metrics for a reasoning trace.
        
        Args:
            trace: Reasoning trace to evaluate
            baseline_traces: Optional list of baseline traces for comparative metrics
            
        Returns:
            Dictionary of metric categories, each containing metric scores
        """
        # Skip if already in cache
        trace_id = trace.trace_id
        if trace_id in self.cache:
            return self.cache[trace_id]
        
        # Initialize results structure
        results = {
            "step_metrics": {},
            "trace_metrics": {},
            "comparative_metrics": {}
        }
        
        # Calculate step-level metrics
        for i, step in enumerate(trace.steps):
            step_metrics = {}
            
            # Get previous step if available
            previous_step = trace.steps[i-1] if i > 0 else None
            
            # Calculate step coherence
            step_metrics["step_coherence"] = self.evaluate_step_coherence(step, previous_step)
            
            # Calculate context relevance
            step_metrics["context_relevance"] = self.evaluate_context_relevance(step)
            
            # Calculate step progress
            step_metrics["step_progress"] = self.evaluate_step_progress(
                step, previous_step, trace.task
            )
            
            # Store in results
            results["step_metrics"][f"step_{step.step_number}"] = step_metrics
        
        # Calculate trace-level metrics
        trace_metrics = {}
        
        # Overall coherence
        trace_metrics["overall_coherence"] = self.evaluate_trace_coherence(trace)
        
        # Goal alignment
        trace_metrics["goal_alignment"] = self.evaluate_goal_alignment(trace)
        
        # Solution completeness
        trace_metrics["solution_completeness"] = self.evaluate_solution_completeness(trace)
        
        results["trace_metrics"] = trace_metrics
        
        # Calculate comparative metrics if baselines provided
        if baseline_traces:
            results["comparative_metrics"] = self.evaluate_comparative_metrics(
                trace, baseline_traces
            )
        
        # Cache results
        self.cache[trace_id] = results
        
        return results
    
    def get_aggregate_score(self, trace_id: str) -> Optional[float]:
        """
        Calculate an aggregate quality score for a trace.
        
        Args:
            trace_id: ID of the trace to score
            
        Returns:
            Aggregate score between 0.0 and 1.0, or None if metrics not calculated
        """
        if trace_id not in self.cache:
            return None
        
        metrics = self.cache[trace_id]
        
        # Extract trace-level metrics
        trace_metrics = metrics.get("trace_metrics", {})
        
        # Weighted average of trace metrics
        weights = {
            "overall_coherence": 0.3,
            "goal_alignment": 0.4,
            "solution_completeness": 0.3
        }
        
        # Only use available metrics
        available_weights = {k: v for k, v in weights.items() if k in trace_metrics}
        if not available_weights:
            return None
            
        # Normalize weights
        weight_sum = sum(available_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in available_weights.items()}
        
        # Calculate weighted score
        score = sum(trace_metrics[k] * normalized_weights[k] for k in normalized_weights.keys())
        
        return score 