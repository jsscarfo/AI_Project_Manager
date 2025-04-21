# Reasoning Trace Visualization System

## Overview

The Visualization System provides a comprehensive set of tools for visualizing and analyzing reasoning traces in the BeeAI Framework. It allows users to explore step-by-step reasoning processes, context usage, knowledge graphs, and quality metrics through interactive visualizations.

## Architecture

The visualization system follows a modular architecture with the following components:

1. **VisualizationService**: The central service that integrates all visualization components and provides a unified interface.
2. **BaseVisualizer**: Abstract base class for all visualization components, providing common functionality.
3. **Specialized Visualizers**:
   - ReasoningTraceVisualizer: For visualizing complete reasoning traces
   - StepsVisualizer: For visualizing reasoning steps and transitions
   - ContextVisualizer: For visualizing context source usage and relevance
   - KnowledgeGraphVisualizer: For visualizing concept relationships
   - MetricsVisualizer: For visualizing quality metrics

## Key Features

### Unified Service Interface

The `VisualizationService` acts as the main entry point for all visualization functionality, providing:

- Consistent configuration across all visualization components
- Unified methods for different visualization types
- Export capabilities for various file formats
- Trace caching for efficient reuse
- Optional integrations with other framework components

### Interactive Visualizations

The system generates interactive visualizations including:

- Reasoning step exploration with context highlighting
- Knowledge graph visualization with concept relationships
- Context relevance visualization with source attribution
- Step transition flow charts
- Timeline views of context evolution
- Quality metrics dashboards

### Data Import/Export

The system supports:

- Exporting visualizations to HTML, JSON, and image formats
- Exporting reasoning traces to JSON for sharing and persistence
- Importing traces from JSON for analysis

## Usage

### Basic Usage Example

```python
from beeai_framework.visualization import VisualizationService, ReasoningTrace

# Initialize the service
viz_service = VisualizationService(
    output_dir="./visualization_outputs",
    default_height=700,
    default_width=900
)

# Create or load a reasoning trace
trace = ReasoningTrace(...)

# Generate visualizations
viz_service.visualize_reasoning_trace(trace, export_path="reasoning_trace.html")
viz_service.visualize_knowledge_graph(trace, export_path="knowledge_graph.html")
viz_service.visualize_context_relevance(trace, export_path="context_relevance.html")
viz_service.visualize_step_transitions(trace, export_path="step_transitions.html")

# Compute and visualize quality metrics
metrics = viz_service.compute_quality_metrics(trace)
viz_service.visualize_quality_metrics(metrics, export_path="quality_metrics.html")

# Export trace for sharing
json_path = viz_service.export_trace_to_json(trace, file_path="reasoning_trace.json")
```

### Integration with Other Components

The visualization system can be integrated with other framework components:

```python
from beeai_framework.vector.sequential_thinking_integration import SequentialKnowledgeIntegration
from beeai_framework.vector.knowledge_retrieval import StepContextManager
from beeai_framework.visualization import VisualizationService

# Create integration components
knowledge_integration = SequentialKnowledgeIntegration(...)
context_manager = StepContextManager(...)

# Initialize visualization service with integrations
viz_service = VisualizationService(
    context_manager=context_manager,
    knowledge_integration=knowledge_integration
)

# Use visualization service with enhanced capabilities
```

## Components

### VisualizationService

The main service class that orchestrates all visualization components and provides a unified interface.

Key methods:
- `visualize_reasoning_trace`: Create comprehensive visualization for a reasoning trace
- `visualize_knowledge_graph`: Create knowledge graph visualization from a reasoning trace
- `visualize_context_relevance`: Create context relevance visualization for a reasoning trace
- `visualize_context_evolution`: Create timeline visualization of context evolution
- `visualize_step_transitions`: Create visualization of step transitions
- `compute_quality_metrics`: Compute reasoning quality metrics for a trace
- `visualize_quality_metrics`: Create visualization of reasoning quality metrics
- `export_trace_to_json`: Export reasoning trace to JSON file
- `import_trace_from_json`: Import reasoning trace from JSON file

### BaseVisualizer

Abstract base class for all visualization components, providing common functionality.

Key features:
- Consistent theme and color scheme management
- Default layout properties for visualizations
- Figure export capabilities
- Configurable sizing

### ReasoningTraceVisualizer

Visualizer for exploring complete reasoning traces.

Key visualizations:
- Step exploration with context highlighting
- Context relevance visualization
- Knowledge graph visualization
- Context evolution timeline

### StepsVisualizer

Visualizer for reasoning steps and transitions.

Key visualizations:
- Step transition flow charts
- Step details visualization
- Progress tracking

### ContextVisualizer

Visualizer for context source usage and relevance.

Key visualizations:
- Text highlighting based on relevance
- Source attribution visualization
- Context filtering options
- Context influence analysis

### KnowledgeGraphVisualizer

Visualizer for concept relationships extracted from reasoning.

Key features:
- Concept extraction from reasoning text
- Relationship inference
- Graph layout algorithms
- Centrality and community analysis
- Concept flow visualization

### MetricsVisualizer

Visualizer for reasoning quality metrics.

Key features:
- Quality metrics dashboard
- Comparative metrics visualization
- Drill-down capabilities for detailed metrics

## Future Enhancements

Planned enhancements for the visualization system include:

1. Web-based interactive dashboard for trace exploration
2. Real-time visualization of ongoing reasoning processes
3. Collaborative annotation and sharing of reasoning traces
4. Additional visualization types for specialized reasoning patterns
5. Integration with more BeeAI Framework components 