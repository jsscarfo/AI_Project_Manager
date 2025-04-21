#!/usr/bin/env python
"""
Example usage of the Reasoning Trace Visualization Framework.

This script demonstrates how to use the visualization components
to analyze and visualize a reasoning trace.
"""

import json
import os
import sys
from datetime import datetime
import random
from pprint import pprint

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

# Import visualization components
from beeai_framework.visualization.core.trace_data_model import (
    TraceVisualizationData,
    StepVisualizationData,
    ContextSourceVisualizationData,
    MetricsVisualizationData,
    VisualizationMetadata
)

from beeai_framework.visualization.components.steps_visualizer import StepsVisualizer
from beeai_framework.visualization.components.context_visualizer import ContextVisualizer
from beeai_framework.visualization.components.knowledge_graph import KnowledgeGraphVisualizer
from beeai_framework.visualization.components.metrics_visualizer import MetricsVisualizer


def create_sample_trace():
    """Create a sample trace for demonstration purposes."""
    # Create metadata
    metadata = VisualizationMetadata(
        title="Test Trace",
        description="Test trace for visualization testing",
        creator="test_suite"
    )
    
    # Create sample steps
    steps = []
    base_timestamp = datetime.now().timestamp()
    
    step_types = ["question", "analysis", "research", "synthesis", "conclusion"]
    
    for i in range(1, 8):
        # Select a step type
        step_type = step_types[min(i-1, len(step_types)-1)]
        
        # Create sample content
        content = f"This is step {i} of the reasoning process. "
        
        if step_type == "question":
            content += "The question being addressed is: How do transformer models handle long-range dependencies in text?"
        elif step_type == "analysis":
            content += "Analyzing the key components: transformers use self-attention mechanisms to capture dependencies. Self-attention allows the model to weigh the importance of different words in the input sequence."
        elif step_type == "research":
            content += "Research indicates that attention mechanisms compute a weighted sum of values, where the weights are determined by a compatibility function between the query and the corresponding key."
        elif step_type == "synthesis":
            content += "Synthesizing the information: Transformers handle long-range dependencies through attention mechanisms that directly model relationships between all words in a sequence, regardless of their distance."
        else:  # conclusion
            content += "In conclusion, transformer models excel at handling long-range dependencies because they don't rely on sequential processing like RNNs, but instead use attention to directly model relationships between all tokens."
        
        # Generate context references
        context_refs = []
        context_count = random.randint(0, 5)
        
        for j in range(context_count):
            source_id = f"source_{random.randint(1, 5)}"
            context_refs.append({
                "context_id": f"ctx_{i}_{j}",
                "source": source_id,
                "relevance_score": random.uniform(0.3, 0.9),
                "usage_type": random.choice(["reference", "quotation", "background"])
            })
        
        # Create the step
        step = StepVisualizationData(
            step_id=f"step_{i}",
            step_number=i,
            title=f"Step {i}: {step_type.capitalize()}",
            content=content,
            step_type=step_type,
            timestamp=base_timestamp + (i - 1) * 60 + random.uniform(0, 30),
            duration=random.uniform(1.0, 8.0),
            requires_next_step=i < 7,
            metrics={"complexity": random.uniform(0.1, 0.9), "confidence": random.uniform(0.5, 0.95)},
            context_references=context_refs,
            annotations={"important": i == 4}  # Mark a step as particularly important
        )
        steps.append(step)
    
    # Create sample context sources
    context_sources = []
    source_types = ["document", "code", "comment", "documentation", "web"]
    
    for i in range(1, 6):
        source_type = source_types[(i-1) % len(source_types)]
        
        # Generate sample steps referenced
        steps_referenced = []
        for step_num in range(1, 8):
            if random.random() > 0.5:
                steps_referenced.append(step_num)
        
        # Create the source
        source = ContextSourceVisualizationData(
            source_id=f"source_{i}",
            name=f"Source {i}: {source_type.capitalize()} Resource",
            source_type=source_type,
            usage_count=len(steps_referenced),
            relevance_scores=[random.uniform(0.3, 0.9) for _ in range(len(steps_referenced))],
            steps_referenced=steps_referenced,
            metadata={"url": f"https://example.com/resource_{i}" if source_type == "web" else None}
        )
        context_sources.append(source)
    
    # Create the complete trace data
    trace = TraceVisualizationData(
        trace_id="sample_trace_001",
        task="Analyze how transformer models handle long-range dependencies",
        metadata=metadata,
        steps=steps,
        context_sources=context_sources,
        final_result="Transformer models handle long-range dependencies through self-attention mechanisms that directly model relationships between all tokens in a sequence, regardless of their position or distance from each other. This is a significant improvement over RNNs, which struggle with long-range dependencies due to their sequential processing nature."
    )
    
    return trace


def visualize_steps(trace):
    """Visualize reasoning steps."""
    print("\n=== Step Visualization ===")
    
    # Create steps visualizer
    visualizer = StepsVisualizer()
    
    # Generate step visualization data
    step_data = visualizer.generate_visualization_data(trace)
    
    # Print step transitions overview
    print(f"\nStep Transitions:")
    transitions = step_data["transitions"]
    print(f"  Total transitions: {transitions['total_transitions']}")
    print(f"  Type changes: {transitions['type_changes']}")
    print(f"  Average transition time: {transitions['avg_transition_time']:.2f}s")
    
    # Print step type distribution
    print(f"\nStep Type Distribution:")
    distribution = step_data["type_distribution"]
    for i, step_type in enumerate(distribution["step_types"]):
        print(f"  {step_type}: {distribution['counts'][i]} steps ({distribution['durations'][i]:.2f}s total)")
    
    # Print details of a specific step
    print(f"\nDetails of Step 3:")
    step3 = next((step for step in step_data["steps"] if step["step_number"] == 3), None)
    if step3:
        print(f"  Type: {step3['step_type']}")
        print(f"  Duration: {step3['duration']:.2f}s")
        print(f"  Context sources: {len(step3['context_sources'])}")
        print(f"  Content preview: {step3['content'][:100]}...")


def visualize_context(trace):
    """Visualize context sources."""
    print("\n=== Context Visualization ===")
    
    # Create context visualizer
    visualizer = ContextVisualizer()
    
    # Generate context visualization data
    context_data = visualizer.generate_visualization_data(trace)
    
    # Print source overview
    print(f"\nContext Sources Overview:")
    print(f"  Total sources: {context_data['source_count']}")
    
    # Print details about sources
    print(f"\nContext Sources:")
    for source in context_data["sources"][:3]:  # Show first 3 sources
        print(f"  {source['name']} ({source['source_type']})")
        print(f"    Usage count: {source['usage_count']}")
        print(f"    Average relevance: {source['average_relevance']:.2f}")
    
    # Print influence by type
    print(f"\nInfluence by Source Type:")
    for type_info in context_data["influence"]["influence_by_type"]:
        print(f"  {type_info['type']}: {type_info['total_usage']} references in {len(type_info['sources'])} sources")


def visualize_knowledge_graph(trace):
    """Visualize knowledge graph."""
    print("\n=== Knowledge Graph Visualization ===")
    
    # Create knowledge graph visualizer
    visualizer = KnowledgeGraphVisualizer()
    
    # Generate knowledge graph visualization data
    graph_data = visualizer.generate_visualization_data(trace)
    
    # Print graph overview
    print(f"\nKnowledge Graph Overview:")
    print(f"  Nodes: {len(graph_data['layout']['nodes'])}")
    print(f"  Edges: {len(graph_data['layout']['edges'])}")
    print(f"  Communities: {graph_data['communities']['community_count']}")
    
    # Print top concepts
    print(f"\nTop Concepts (by centrality):")
    for concept in graph_data["centrality"]["top_concepts"][:3]:  # Show top 3
        print(f"  {concept['label']} (type: {concept['type']})")
        print(f"    Degree centrality: {concept['degree_centrality']:.3f}")
    
    # Print persistent concepts
    print(f"\nPersistent Concepts:")
    for concept in graph_data["concept_flow"]["persistent_concepts"][:3]:  # Show top 3
        print(f"  {concept['concept']} (type: {concept['type']})")
        print(f"    First appearance: Step {concept['first_step']}")
        print(f"    Last appearance: Step {concept['last_step']}")
        print(f"    Persistence: {concept['persistence']:.2f}")


def visualize_metrics(trace):
    """Visualize metrics."""
    print("\n=== Metrics Visualization ===")
    
    # Create metrics visualizer
    visualizer = MetricsVisualizer()
    
    # Generate metrics visualization data
    metrics_data = visualizer.generate_visualization_data(trace)
    
    # Print summary metrics
    print(f"\nSummary Metrics:")
    summary = metrics_data["summary"]
    print(f"  Step count: {summary['step_count']}")
    print(f"  Total time: {summary['total_time']:.2f}s")
    print(f"  Avg time per step: {summary['avg_time_per_step']:.2f}s")
    print(f"  Context references: {summary['context_references']}")
    print(f"  Quality score: {summary['quality_score']:.1f}/100")
    
    # Print quality radar overview
    print(f"\nQuality Assessment:")
    quality = metrics_data["radar_charts"]["quality"]["scores_object"]
    for dimension, score in quality.items():
        print(f"  {dimension}: {score:.1f}/100")
    
    # Print comparison with baseline
    print(f"\nComparison with Baseline:")
    assessment = metrics_data["comparison"]["overall_assessment"]
    print(f"  Score: {assessment['score']:.1f}")
    print(f"  Assessment: {assessment['assessment']}")
    print(f"  Improvements: {assessment['improvements']}")
    print(f"  Regressions: {assessment['regressions']}")


def save_visualization_data(trace, output_dir="./output"):
    """Save visualization data to JSON files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizers
    steps_viz = StepsVisualizer()
    context_viz = ContextVisualizer()
    graph_viz = KnowledgeGraphVisualizer()
    metrics_viz = MetricsVisualizer()
    
    # Generate and save steps data
    with open(os.path.join(output_dir, "steps_visualization.json"), "w") as f:
        f.write(steps_viz.to_json(trace, indent=2))
    
    # Generate and save context data
    with open(os.path.join(output_dir, "context_visualization.json"), "w") as f:
        f.write(context_viz.to_json(trace, indent=2))
    
    # Generate and save knowledge graph data
    with open(os.path.join(output_dir, "knowledge_graph_visualization.json"), "w") as f:
        f.write(graph_viz.to_json(trace, indent=2))
    
    # Generate and save metrics data
    with open(os.path.join(output_dir, "metrics_visualization.json"), "w") as f:
        f.write(metrics_viz.to_json(trace, indent=2))
    
    # Save complete trace data
    with open(os.path.join(output_dir, "trace_data.json"), "w") as f:
        f.write(trace.to_json(indent=2))
    
    print(f"\nVisualization data saved to {output_dir}/")


def main():
    """Main function to run the example."""
    print("=== Reasoning Trace Visualization Example ===")
    
    # Create a sample trace
    trace = create_sample_trace()
    print(f"Created sample trace with ID: {trace.trace_id}")
    print(f"Task: {trace.task}")
    print(f"Steps: {len(trace.steps)}")
    print(f"Context sources: {len(trace.context_sources)}")
    
    # Visualize steps
    visualize_steps(trace)
    
    # Visualize context
    visualize_context(trace)
    
    # Visualize knowledge graph
    visualize_knowledge_graph(trace)
    
    # Visualize metrics
    visualize_metrics(trace)
    
    # Save visualization data
    save_visualization_data(trace)


if __name__ == "__main__":
    main() 