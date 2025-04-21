#!/usr/bin/env python
"""
Minimal Standalone Test for Visualization Concepts

This script provides a completely standalone test for visualization concepts
without relying on any imports from the framework. It defines all necessary 
classes and functions within the file itself.
"""

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# ============ DATA MODELS ============

@dataclass
class StepVisualizationData:
    """Data for visualizing a reasoning step."""
    step_number: int
    step_type: str
    description: str
    content: str
    timestamp: str  # ISO format
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    substeps: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    context_ids: List[str] = field(default_factory=list)

@dataclass
class ContextSourceData:
    """Data for a context source used in visualization."""
    context_id: str
    source_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0
    timestamp: Optional[str] = None

@dataclass
class TraceVisualizationData:
    """Combined data for visualizing a reasoning trace."""
    trace_id: str
    title: str
    steps: List[StepVisualizationData]
    context_sources: List[ContextSourceData]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

# ============ MOCK VISUALIZERS ============

class StepTransitionVisualizer:
    """Mock class for visualizing transitions between steps."""
    
    def generate_transition_data(self, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate transition data from trace data."""
        steps = trace_data.steps
        transitions = []
        
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            
            transition = {
                "from_step": current_step.step_number,
                "to_step": next_step.step_number,
                "from_type": current_step.step_type,
                "to_type": next_step.step_type,
                "duration_ms": next_step.timestamp - current_step.timestamp 
                    if isinstance(current_step.timestamp, (int, float)) and 
                       isinstance(next_step.timestamp, (int, float))
                    else 0
            }
            transitions.append(transition)
        
        return {
            "transitions": transitions,
            "step_count": len(steps),
            "unique_step_types": list(set(step.step_type for step in steps)),
            "total_transitions": len(transitions)
        }
    
    def generate_step_flow_chart(self, transition_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a flow chart visualization from transition data."""
        print(f"Generating flow chart with {transition_data['total_transitions']} transitions")
        
        # In a real implementation, this would create a Plotly figure
        # Here we just return a dictionary with some mock data
        return {
            "chart_type": "flow_chart",
            "step_types": transition_data.get("unique_step_types", []),
            "node_count": transition_data.get("step_count", 0),
            "edge_count": transition_data.get("total_transitions", 0)
        }

class StepDetailVisualizer:
    """Mock class for visualizing details of steps."""
    
    def generate_step_details(self, step: StepVisualizationData) -> Dict[str, Any]:
        """Generate detailed visualization data for a step."""
        # Count words in content
        word_count = len(step.content.split())
        
        # Get context usage
        context_count = len(step.context_ids)
        
        return {
            "step_number": step.step_number,
            "step_type": step.step_type,
            "description": step.description,
            "timestamp": step.timestamp,
            "duration_ms": step.duration_ms,
            "word_count": word_count,
            "context_count": context_count,
            "has_substeps": len(step.substeps) > 0,
            "metrics": step.metrics
        }
    
    def compare_steps(self, steps: List[StepVisualizationData]) -> Dict[str, Any]:
        """Compare multiple steps and generate comparison data."""
        if not steps:
            return {"error": "No steps provided for comparison"}
        
        step_numbers = [step.step_number for step in steps]
        step_types = [step.step_type for step in steps]
        
        # Get common context IDs
        common_context_ids = set(steps[0].context_ids)
        for step in steps[1:]:
            common_context_ids &= set(step.context_ids)
        
        # Get unique context IDs for each step
        unique_contexts = {}
        for step in steps:
            unique_contexts[step.step_number] = set(step.context_ids) - common_context_ids
        
        return {
            "step_numbers": step_numbers,
            "step_types": step_types,
            "common_context_count": len(common_context_ids),
            "common_context_ids": list(common_context_ids),
            "unique_contexts": {k: list(v) for k, v in unique_contexts.items()},
            "comparison_timestamp": datetime.now().isoformat()
        }

class ContextHighlightingService:
    """Service for highlighting context references in text."""
    
    def generate_highlights(self, text: str, context_sources: List[ContextSourceData]) -> Dict[str, Any]:
        """Generate context highlighting data for text."""
        # Simple mock implementation that checks if context appears in text
        highlights = []
        
        for source in context_sources:
            # In a real implementation, this would use NLP to find matches
            # Here we just do a simple content substring check
            if source.content and len(source.content) > 10:
                sample = source.content[:10]  # Use first 10 chars as a sample
                
                # Find all occurrences
                start_idx = 0
                while True:
                    idx = text.find(sample, start_idx)
                    if idx == -1:
                        break
                        
                    # Create a highlight
                    highlights.append({
                        "start_idx": idx,
                        "end_idx": idx + len(sample),
                        "context_id": source.context_id,
                        "source_type": source.source_type,
                        "importance": source.importance
                    })
                    
                    start_idx = idx + 1
        
        return {
            "text": text,
            "highlight_count": len(highlights),
            "highlights": highlights,
            "source_types": list(set(h["source_type"] for h in highlights))
        }
    
    def merge_highlights(self, highlight_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple highlight data objects."""
        all_highlights = []
        all_source_types = set()
        
        for data in highlight_data_list:
            all_highlights.extend(data.get("highlights", []))
            all_source_types.update(data.get("source_types", []))
        
        return {
            "highlight_count": len(all_highlights),
            "highlights": all_highlights,
            "source_types": list(all_source_types)
        }

class ContextHeatmapGenerator:
    """Generator for context usage heatmaps."""
    
    def generate_context_heatmap(self, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate context usage heatmap data."""
        steps = trace_data.steps
        contexts = trace_data.context_sources
        
        # Create a matrix of context usage: steps x contexts
        heatmap_data = []
        step_labels = []
        context_labels = []
        
        # Prepare context labels
        for ctx in contexts:
            label = f"{ctx.source_type}: {ctx.context_id}"
            context_labels.append(label)
        
        # For each step, calculate context usage
        for step in steps:
            step_label = f"Step {step.step_number}: {step.step_type}"
            step_labels.append(step_label)
            
            # Get context usage for this step
            step_contexts = set(step.context_ids)
            
            # Create a row for this step
            row = []
            for ctx in contexts:
                # 1 if context used, 0 if not
                usage = 1 if ctx.context_id in step_contexts else 0
                row.append(usage)
            
            heatmap_data.append(row)
        
        return {
            "heatmap_data": heatmap_data,
            "step_labels": step_labels,
            "context_labels": context_labels,
            "max_value": 1,
            "min_value": 0
        }
    
    def generate_context_usage_summary(self, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate a summary of context usage across steps."""
        contexts = trace_data.context_sources
        steps = trace_data.steps
        
        # Calculate usage statistics for each context
        context_usage = {}
        for ctx in contexts:
            # Count steps using this context
            steps_using = sum(1 for step in steps if ctx.context_id in step.context_ids)
            usage_percentage = steps_using / len(steps) * 100 if steps else 0
            
            context_usage[ctx.context_id] = {
                "context_id": ctx.context_id,
                "source_type": ctx.source_type,
                "steps_using": steps_using,
                "usage_percentage": usage_percentage,
                "importance": ctx.importance
            }
        
        return {
            "context_count": len(contexts),
            "step_count": len(steps),
            "context_usage": context_usage,
            "most_used": max(context_usage.values(), key=lambda x: x["steps_using"], default=None),
            "least_used": min(context_usage.values(), key=lambda x: x["steps_using"], default=None)
        }

class SourceAttributionVisualizer:
    """Visualizer for source attribution in reasoning steps."""
    
    def generate_source_attribution(self, step: StepVisualizationData, 
                                 contexts: List[ContextSourceData]) -> Dict[str, Any]:
        """Generate source attribution visualization data for a step."""
        # Find context sources used in this step
        step_context_ids = set(step.context_ids)
        used_contexts = [ctx for ctx in contexts if ctx.context_id in step_context_ids]
        
        # For each used context, create attribution data
        attributions = []
        for ctx in used_contexts:
            attribution = {
                "context_id": ctx.context_id,
                "source_type": ctx.source_type,
                "source": ctx.metadata.get("source", "unknown"),
                "importance": ctx.importance
            }
            attributions.append(attribution)
        
        # Sort by importance
        attributions.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "step_number": step.step_number,
            "step_type": step.step_type,
            "attribution_count": len(attributions),
            "attributions": attributions,
            "primary_sources": [a["source"] for a in attributions[:3]] if attributions else []
        }

class ContextUsageAnalytics:
    """Analytics for context usage patterns."""
    
    def analyze_context_usage(self, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """Analyze context usage patterns across steps."""
        steps = trace_data.steps
        contexts = trace_data.context_sources
        
        # Calculate context usage by step type
        usage_by_step_type = {}
        for step in steps:
            step_type = step.step_type
            if step_type not in usage_by_step_type:
                usage_by_step_type[step_type] = {"count": 0, "contexts": set()}
            
            usage_by_step_type[step_type]["count"] += 1
            usage_by_step_type[step_type]["contexts"].update(step.context_ids)
        
        # Convert sets to lists for JSON serialization
        for step_type in usage_by_step_type:
            usage_by_step_type[step_type]["contexts"] = list(usage_by_step_type[step_type]["contexts"])
            usage_by_step_type[step_type]["context_count"] = len(usage_by_step_type[step_type]["contexts"])
        
        # Calculate usage timeline
        usage_timeline = []
        for i, step in enumerate(steps):
            step_contexts = set(step.context_ids)
            
            # Compare with previous step if not the first
            context_change = {}
            if i > 0:
                prev_contexts = set(steps[i-1].context_ids)
                added = step_contexts - prev_contexts
                removed = prev_contexts - step_contexts
                retained = step_contexts & prev_contexts
                context_change = {
                    "added": list(added),
                    "removed": list(removed),
                    "retained": list(retained)
                }
            
            timeline_entry = {
                "step_number": step.step_number,
                "step_type": step.step_type,
                "context_count": len(step_contexts),
                "contexts": list(step_contexts),
                "context_change": context_change
            }
            usage_timeline.append(timeline_entry)
        
        return {
            "usage_by_step_type": usage_by_step_type,
            "usage_timeline": usage_timeline,
            "total_steps": len(steps),
            "total_contexts": len(contexts)
        }

class ContextVisualizer:
    """Mock class for context visualization."""
    
    def __init__(self):
        """Initialize the context visualizer with component services."""
        self.highlighting_service = ContextHighlightingService()
        self.heatmap_generator = ContextHeatmapGenerator()
        self.source_attribution = SourceAttributionVisualizer()
        self.usage_analytics = ContextUsageAnalytics()
    
    def generate_context_visualization(self, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate comprehensive context visualization data."""
        # Generate context heatmap
        heatmap_data = self.heatmap_generator.generate_context_heatmap(trace_data)
        
        # Generate usage analytics
        usage_data = self.usage_analytics.analyze_context_usage(trace_data)
        
        # Generate context usage summary
        usage_summary = self.heatmap_generator.generate_context_usage_summary(trace_data)
        
        # Generate source attribution for each step
        attributions = {}
        for step in trace_data.steps:
            attribution_data = self.source_attribution.generate_source_attribution(
                step, trace_data.context_sources
            )
            attributions[step.step_number] = attribution_data
        
        # Generate highlight data for the last step (as an example)
        if trace_data.steps:
            last_step = trace_data.steps[-1]
            highlight_data = self.highlighting_service.generate_highlights(
                last_step.content, trace_data.context_sources
            )
        else:
            highlight_data = {"highlight_count": 0, "highlights": []}
        
        return {
            "trace_id": trace_data.trace_id,
            "context_heatmap": heatmap_data,
            "usage_analytics": usage_data,
            "usage_summary": usage_summary,
            "source_attributions": attributions,
            "highlights_example": highlight_data,
            "context_count": len(trace_data.context_sources)
        }

class ProgressVisualizer:
    """Mock class for visualizing progress through a reasoning trace."""
    
    def generate_progress_data(self, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate progress visualization data from trace data."""
        steps = trace_data.steps
        
        # Calculate progress metrics
        total_steps = len(steps)
        step_types = {}
        for step in steps:
            step_types[step.step_type] = step_types.get(step.step_type, 0) + 1
        
        # Check if complete based on having a conclusion step
        has_conclusion = any(step.step_type.lower() in ["conclusion", "summary"] for step in steps)
        
        return {
            "total_steps": total_steps,
            "step_type_distribution": step_types,
            "is_complete": has_conclusion,
            "duration_ms": self._calculate_total_duration(steps),
            "last_update": steps[-1].timestamp if steps else None
        }
    
    def _calculate_total_duration(self, steps: List[StepVisualizationData]) -> int:
        """Calculate the total duration across all steps."""
        return sum(step.duration_ms for step in steps)

class StepsVisualizer:
    """Mock class for visualizing reasoning steps."""
    
    def __init__(self):
        """Initialize the steps visualizer with component visualizers."""
        self.transition_visualizer = StepTransitionVisualizer()
        self.detail_visualizer = StepDetailVisualizer()
        self.progress_visualizer = ProgressVisualizer()
    
    def generate_visualization_data(self, trace_data: TraceVisualizationData) -> Dict[str, Any]:
        """Generate comprehensive visualization data from trace data."""
        # Generate transition data
        transition_data = self.transition_visualizer.generate_transition_data(trace_data)
        
        # Generate flow chart
        flow_chart = self.transition_visualizer.generate_step_flow_chart(transition_data)
        
        # Generate progress data
        progress_data = self.progress_visualizer.generate_progress_data(trace_data)
        
        # Generate step details
        step_details = {}
        for step in trace_data.steps:
            step_details[step.step_number] = self.detail_visualizer.generate_step_details(step)
        
        return {
            "trace_id": trace_data.trace_id,
            "title": trace_data.title,
            "steps": step_details,
            "flow_chart": flow_chart,
            "transitions": transition_data,
            "progress": progress_data,
            "context_count": len(trace_data.context_sources),
            "metadata": trace_data.metadata
        }
    
    def compare_steps(self, steps: List[StepVisualizationData]) -> Dict[str, Any]:
        """Compare multiple steps and generate comparison data."""
        return self.detail_visualizer.compare_steps(steps)
    
    def export_to_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """Export visualization data to a JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

# ============ TEST DATA GENERATION ============

def create_sample_data() -> TraceVisualizationData:
    """Create sample data for testing."""
    # Create some context sources
    context_sources = [
        ContextSourceData(
            context_id=f"ctx_{i}",
            source_type=["document", "knowledge_base", "user_input"][i % 3],
            content=f"This is sample content for context {i}. It contains information relevant to the reasoning process.",
            metadata={"importance": 0.7 + (i * 0.1) % 0.3, "source": f"source_{i}"},
            importance=0.7 + (i * 0.1) % 0.3,
            timestamp=datetime.now().isoformat()
        )
        for i in range(5)
    ]
    
    # Create some steps
    steps = []
    step_types = ["planning", "research", "analysis", "synthesis", "conclusion"]
    
    for i in range(5):
        # Assign context IDs - each step uses some contexts
        context_ids = [source.context_id for source in context_sources if i % 2 == 0 or source.importance > 0.8]
        
        # Add references to context content in the step content
        content_samples = []
        for ctx in context_sources:
            if ctx.context_id in context_ids:
                content_samples.append(ctx.content[:10])  # Add first 10 chars of context
        
        # Create content with context references
        content = f"This is the content for step {i+1}. It performs {step_types[i]} with the available information."
        if content_samples:
            content += f" Using context: {' and '.join(content_samples)}"
        
        steps.append(
            StepVisualizationData(
                step_number=i+1,
                step_type=step_types[i],
                description=f"Step {i+1}: {step_types[i].capitalize()}",
                content=content,
                timestamp=datetime.now().isoformat(),
                duration_ms=100 + i * 50,
                metadata={"importance": 0.8 + (i * 0.05) % 0.2},
                substeps=[],
                metrics={"relevance": 0.7 + (i * 0.05), "coherence": 0.8 - (i * 0.02) % 0.1},
                context_ids=context_ids
            )
        )
    
    # Create the trace data
    trace_data = TraceVisualizationData(
        trace_id="trace_123456",
        title="Sample Reasoning Trace",
        steps=steps,
        context_sources=context_sources,
        metadata={"source": "test_generator", "version": "1.0"}
    )
    
    return trace_data

# ============ MAIN TEST FUNCTION ============

def test_visualizer():
    """Test the visualizer components."""
    # Create sample data
    trace_data = create_sample_data()
    print(f"Created sample trace with {len(trace_data.steps)} steps and {len(trace_data.context_sources)} context sources")
    
    # Initialize visualizers
    steps_visualizer = StepsVisualizer()
    context_visualizer = ContextVisualizer()
    
    # Generate steps visualization data
    steps_vis_data = steps_visualizer.generate_visualization_data(trace_data)
    
    # Print summary information about steps
    print(f"Generated steps visualization data with ID: {steps_vis_data['trace_id']}")
    print(f"Number of steps: {len(steps_vis_data['steps'])}")
    print(f"Step types: {steps_vis_data['flow_chart']['step_types']}")
    print(f"Total transitions: {steps_vis_data['transitions']['total_transitions']}")
    print(f"Progress complete: {steps_vis_data['progress']['is_complete']}")
    
    # Test step comparison
    step1 = trace_data.steps[0]
    step_last = trace_data.steps[-1]
    comparison = steps_visualizer.compare_steps([step1, step_last])
    print("\nComparing first and last steps:")
    print(f"Step numbers compared: {comparison['step_numbers']}")
    print(f"Step types compared: {comparison['step_types']}")
    print(f"Common context count: {comparison['common_context_count']}")
    
    # Generate context visualization data
    print("\nGenerating context visualization...")
    context_vis_data = context_visualizer.generate_context_visualization(trace_data)
    
    # Print summary information about context
    print("\nContext Visualization:")
    print(f"Total contexts: {context_vis_data['context_count']}")
    print(f"Heatmap dimensions: {len(context_vis_data['context_heatmap']['step_labels'])} steps × {len(context_vis_data['context_heatmap']['context_labels'])} contexts")
    print(f"Usage analytics by step type: {list(context_vis_data['usage_analytics']['usage_by_step_type'].keys())}")
    
    # Get highlight example
    highlight_example = context_vis_data['highlights_example']
    print(f"Highlights found in example: {highlight_example['highlight_count']}")
    if highlight_example['highlight_count'] > 0:
        print(f"Source types highlighted: {highlight_example['source_types']}")
    
    # Export to JSON
    output_dir = Path(__file__).parent / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export both visualizations
    steps_output_file = output_dir / "steps_visualization.json"
    context_output_file = output_dir / "context_visualization.json"
    
    print("\nExporting visualization data to JSON files...")
    success1 = steps_visualizer.export_to_json(steps_vis_data, str(steps_output_file))
    success2 = steps_visualizer.export_to_json(context_vis_data, str(context_output_file))
    
    if success1 and success2:
        print(f"Visualization data exported to files:")
        print(f"- {steps_output_file}")
        print(f"- {context_output_file}")
    
    print("\nTest completed successfully!")

# Run the test if executed directly
if __name__ == "__main__":
    print("Starting visualization test...")
    try:
        test_visualizer()
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
    print("Test script completed execution") 