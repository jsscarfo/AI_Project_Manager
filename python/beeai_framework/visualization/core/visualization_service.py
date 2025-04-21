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
Visualization Service

This module provides a unified service for accessing and coordinating
all visualization components in the BeeAI Framework. It serves as the main
entry point for visualization functionality, orchestrating the various
visualization components and providing a consistent interface.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Set, Tuple, TYPE_CHECKING
from pathlib import Path
import json
import os
from datetime import datetime

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from beeai_framework.visualization.components.reasoning_trace_visualizer import (
        ReasoningTraceVisualizer, ReasoningTrace, ReasoningStep
    )
    from beeai_framework.visualization.components.knowledge_graph import KnowledgeGraphVisualizer
    from beeai_framework.visualization.components.context_visualizer import ContextVisualizer
    from beeai_framework.visualization.components.steps_visualizer import StepsVisualizer
    from beeai_framework.visualization.components.metrics_visualizer import MetricsVisualizer
    from beeai_framework.visualization.components.reasoning_quality_metrics import (
        ReasoningQualityMetrics, QualityMetric, MetricLevel
    )
    # Define a type alias for ReasoningTrace to use it in method signatures
    ReasoningTrace = 'ReasoningTrace'

from beeai_framework.visualization.core.base_visualizer import BaseVisualizer

# Optional integrations
try:
    from beeai_framework.vector.sequential_thinking_integration import SequentialKnowledgeIntegration
    from beeai_framework.vector.knowledge_retrieval import StepContextManager
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualizationService:
    """
    Service for coordinating and accessing all visualization components.
    
    This service integrates all visualization components, providing a unified 
    interface for visualization functionality. It orchestrates the interactions 
    between different visualizers and maintains configuration settings.
    """
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        default_height: int = 600,
        default_width: int = 800,
        cache_traces: bool = True,
        context_manager: Optional[Any] = None,
        knowledge_integration: Optional[Any] = None
    ):
        """
        Initialize the visualization service with components and settings.
        
        Args:
            output_dir: Directory to save visualization outputs
            default_height: Default height for visualizations
            default_width: Default width for visualizations
            cache_traces: Whether to cache traces in memory
            context_manager: Optional context manager for integration
            knowledge_integration: Optional knowledge integration component
        """
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = None
        
        # Configuration
        self.default_height = default_height
        self.default_width = default_width
        self.cache_traces = cache_traces
        
        # Integration components
        self.context_manager = context_manager
        self.knowledge_integration = knowledge_integration
        
        # Initialize visualization components
        self._initialize_components()
        
        # Cache for reasoning traces
        self.trace_cache: Dict[str, ReasoningTrace] = {}
        
        logger.info("Visualization service initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all visualization components with consistent settings."""
        # Import components here to avoid circular imports
        from beeai_framework.visualization.components.knowledge_graph import KnowledgeGraphVisualizer
        from beeai_framework.visualization.components.reasoning_trace_visualizer import ReasoningTraceVisualizer
        from beeai_framework.visualization.components.context_visualizer import ContextVisualizer
        from beeai_framework.visualization.components.steps_visualizer import StepsVisualizer
        from beeai_framework.visualization.components.metrics_visualizer import MetricsVisualizer
        from beeai_framework.visualization.components.reasoning_quality_metrics import ReasoningQualityMetrics
        
        # Knowledge graph visualizer (used by other components)
        self.knowledge_graph_visualizer = KnowledgeGraphVisualizer(
            default_height=self.default_height,
            default_width=self.default_width
        )
        
        # Main visualization components
        self.reasoning_trace_visualizer = ReasoningTraceVisualizer(
            knowledge_integration=self.knowledge_integration if VECTOR_AVAILABLE else None,
            context_manager=self.context_manager if VECTOR_AVAILABLE else None,
            knowledge_graph_visualizer=self.knowledge_graph_visualizer,
            cache_traces=self.cache_traces,
            default_height=self.default_height,
            default_width=self.default_width
        )
        
        self.steps_visualizer = StepsVisualizer(
            default_height=self.default_height,
            default_width=self.default_width
        )
        
        self.context_visualizer = ContextVisualizer(
            default_height=self.default_height,
            default_width=self.default_width
        )
        
        self.metrics_visualizer = MetricsVisualizer(
            default_height=self.default_height,
            default_width=self.default_width
        )
        
        self.quality_metrics = ReasoningQualityMetrics()
    
    def visualize_reasoning_trace(
        self,
        trace: Union['ReasoningTrace', Dict[str, Any]],
        selected_step: Optional[int] = None,
        highlight_context: bool = True,
        show_metrics: bool = True,
        export_path: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """
        Create comprehensive visualization for a reasoning trace.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            selected_step: Index of the initially selected step (1-based)
            highlight_context: Whether to highlight context references
            show_metrics: Whether to show step metrics
            export_path: Path to export the visualization (if None, uses output_dir)
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with interactive trace visualization
        """
        figure = self.reasoning_trace_visualizer.create_step_visualization(
            trace=trace,
            selected_step=selected_step,
            highlight_context=highlight_context,
            show_metrics=show_metrics,
            height=height or self.default_height,
            width=width or self.default_width
        )
        
        # Cache trace if enabled
        if self.cache_traces and isinstance(trace, ReasoningTrace):
            self.trace_cache[trace.trace_id] = trace
        
        # Export if path provided
        if export_path:
            self._export_visualization(figure, export_path)
        
        return figure
    
    def visualize_knowledge_graph(
        self,
        trace: Union['ReasoningTrace', Dict[str, Any]],
        export_path: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """
        Create knowledge graph visualization from a reasoning trace.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            export_path: Path to export the visualization (if None, uses output_dir)
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with knowledge graph visualization
        """
        figure = self.reasoning_trace_visualizer.create_knowledge_graph_visualization(
            trace=trace,
            height=height or self.default_height,
            width=width or self.default_width
        )
        
        # Export if path provided
        if export_path:
            self._export_visualization(figure, export_path)
        
        return figure
    
    def visualize_context_relevance(
        self,
        trace: Union['ReasoningTrace', Dict[str, Any]],
        selected_step: Optional[int] = None,
        export_path: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """
        Create context relevance visualization for a reasoning trace.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            selected_step: Index of the step to analyze (1-based)
            export_path: Path to export the visualization (if None, uses output_dir)
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with context relevance visualization
        """
        figure = self.reasoning_trace_visualizer.create_context_relevance_visualization(
            trace=trace,
            selected_step=selected_step,
            height=height or self.default_height,
            width=width or self.default_width
        )
        
        # Export if path provided
        if export_path:
            self._export_visualization(figure, export_path)
        
        return figure
    
    def visualize_context_evolution(
        self,
        trace: Union['ReasoningTrace', Dict[str, Any]],
        export_path: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """
        Create timeline visualization of context evolution across steps.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            export_path: Path to export the visualization (if None, uses output_dir)
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with context evolution timeline
        """
        figure = self.reasoning_trace_visualizer.create_context_evolution_timeline(
            trace=trace,
            height=height or self.default_height,
            width=width or self.default_width
        )
        
        # Export if path provided
        if export_path:
            self._export_visualization(figure, export_path)
        
        return figure
    
    def visualize_step_transitions(
        self,
        trace: Union['ReasoningTrace', Dict[str, Any]],
        export_path: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """
        Create visualization of step transitions in a reasoning trace.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            export_path: Path to export the visualization (if None, uses output_dir)
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with step transition visualization
        """
        # Convert dictionary to trace if needed
        if isinstance(trace, dict):
            trace_obj = self.reasoning_trace_visualizer._dict_to_trace(trace)
        else:
            trace_obj = trace
        
        # Generate transition data
        transition_data = self.steps_visualizer.generate_transition_data(trace_obj)
        
        # Create flow chart visualization
        figure = self.steps_visualizer.generate_step_flow_chart(
            transition_data,
            height=height or self.default_height,
            width=width or self.default_width
        )
        
        # Export if path provided
        if export_path:
            self._export_visualization(figure, export_path)
        
        return figure
    
    def compute_quality_metrics(
        self,
        trace: Union['ReasoningTrace', Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute reasoning quality metrics for a trace.
        
        Args:
            trace: Reasoning trace or dictionary with trace data
            
        Returns:
            Dictionary with quality metrics
        """
        # Convert dictionary to trace if needed
        if isinstance(trace, dict):
            trace_obj = self.reasoning_trace_visualizer._dict_to_trace(trace)
        else:
            trace_obj = trace
            
        # Compute metrics
        metrics = self.quality_metrics.compute_all_metrics(trace_obj)
        
        return metrics
    
    def visualize_quality_metrics(
        self,
        metrics: Dict[str, Any],
        export_path: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """
        Create visualization of reasoning quality metrics.
        
        Args:
            metrics: Dictionary with quality metrics
            export_path: Path to export the visualization (if None, uses output_dir)
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Plotly figure with metrics visualization
        """
        figure = self.metrics_visualizer.create_metrics_dashboard(
            metrics,
            height=height or self.default_height,
            width=width or self.default_width
        )
        
        # Export if path provided
        if export_path:
            self._export_visualization(figure, export_path)
        
        return figure
    
    def export_trace_to_json(
        self, 
        trace: Union[str, 'ReasoningTrace', Dict[str, Any]],
        file_path: Optional[str] = None
    ) -> str:
        """
        Export reasoning trace to JSON file.
        
        Args:
            trace: Trace ID (if in cache), ReasoningTrace object, or trace dict
            file_path: Path to save the JSON file (if None, uses output_dir)
            
        Returns:
            Path to the saved file
        """
        # Get trace object
        if isinstance(trace, str):
            if trace not in self.trace_cache:
                raise ValueError(f"Trace ID '{trace}' not found in cache")
            trace_obj = self.trace_cache[trace]
        elif isinstance(trace, ReasoningTrace):
            trace_obj = trace
        else:
            trace_obj = self.reasoning_trace_visualizer._dict_to_trace(trace)
        
        # Generate file path if not provided
        if file_path is None:
            if self.output_dir is None:
                raise ValueError("No output_dir set and no file_path provided")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.output_dir / f"trace_{trace_obj.trace_id}_{timestamp}.json")
        
        # Export using the trace visualizer
        success = self.reasoning_trace_visualizer.export_trace_to_json(
            trace_obj.trace_id if isinstance(trace, ReasoningTrace) else trace_obj,
            file_path
        )
        
        if not success:
            raise RuntimeError(f"Failed to export trace to {file_path}")
        
        return file_path
    
    def import_trace_from_json(self, file_path: str) -> Optional['ReasoningTrace']:
        """
        Import reasoning trace from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Imported ReasoningTrace object or None if import failed
        """
        trace = self.reasoning_trace_visualizer.import_trace_from_json(file_path)
        
        # Cache if enabled
        if trace and self.cache_traces:
            self.trace_cache[trace.trace_id] = trace
        
        return trace
    
    def _export_visualization(self, figure: Any, file_path: str) -> None:
        """
        Export visualization to file.
        
        Args:
            figure: Plotly figure to export
            file_path: Path to save the file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine file type from extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.html':
            figure.write_html(file_path)
        elif ext == '.json':
            with open(file_path, 'w') as f:
                json.dump(figure.to_dict(), f)
        elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.svg', '.pdf']:
            figure.write_image(file_path)
        else:
            logger.warning(f"Unsupported file extension '{ext}', defaulting to HTML")
            figure.write_html(f"{os.path.splitext(file_path)[0]}.html")
        
        logger.info(f"Visualization exported to {file_path}") 