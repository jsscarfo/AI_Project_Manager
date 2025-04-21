# BeeAI Framework Visualization Toolkit

The BeeAI Framework Visualization Toolkit provides advanced visualization and evaluation tools for analyzing reasoning quality in the Sequential Thinking with Knowledge Retrieval system.

## Components

### Reasoning Trace Visualization

The `ReasoningTraceVisualizer` helps visualize the complete reasoning process, from initial problem definition to final solution, with insights into each step and the use of context knowledge:

```python
from beeai_framework.visualization import ReasoningTraceVisualizer, ReasoningTrace

visualizer = ReasoningTraceVisualizer()
trace = ReasoningTrace(...)  # Your reasoning trace

# Create visualizations
step_viz = visualizer.create_step_visualization(trace)
context_viz = visualizer.create_context_relevance_visualization(trace)
graph_viz = visualizer.create_knowledge_graph_visualization(trace)
timeline_viz = visualizer.create_context_evolution_timeline(trace)

# Export visualizations
step_viz.write_html("steps.html")
```

### Reasoning Quality Metrics

The `ReasoningQualityMetrics` component evaluates the quality of reasoning traces through various metrics:

```python
from beeai_framework.visualization import ReasoningQualityMetrics

metrics = ReasoningQualityMetrics()
results = metrics.calculate_all_metrics(trace)

# Access specific metrics
coherence = results["step_metrics"]["coherence"]
goal_alignment = results["trace_metrics"]["goal_alignment"]
```

### Context Usage Analytics

The `ContextUsageAnalytics` component analyzes how effectively context knowledge is used during reasoning:

```python
from beeai_framework.visualization import ContextUsageAnalytics

analytics = ContextUsageAnalytics()
usage_stats = analytics.analyze_all_metrics(trace)

# Create visualizations
token_viz = analytics.create_token_usage_chart(trace)
source_viz = analytics.create_knowledge_source_chart(trace)
density_viz = analytics.create_information_density_chart(trace)
```

### A/B Testing Framework

The `ABTestingFramework` enables comparative evaluation of different reasoning strategies:

```python
from beeai_framework.visualization import ABTestingFramework, TestCase, TestStrategy

framework = ABTestingFramework()

# Add test cases and strategies
case = TestCase(case_id="case_1", description="Test case", task="Solve problem X")
strategy = TestStrategy(strategy_id="strategy_1", name="Baseline", config={"similarity_threshold": 0.7})

framework.add_test_case(case)
framework.add_strategy(strategy)

# Run tests (with your retrieval service)
result = await framework.run_test("case_1", "strategy_1", retrieval_service, run_func)

# Analyze results
analysis = framework.analyze_results()
comp_viz = framework.create_comparative_visualization(analysis)
```

### Evaluation Dashboard

The `EvaluationDashboard` provides an interactive interface for exploring all visualizations:

```python
from beeai_framework.visualization import EvaluationDashboard, DashboardConfig

config = DashboardConfig(title="Reasoning Evaluation Dashboard", port=8050)
dashboard = EvaluationDashboard(
    config=config,
    trace_visualizer=trace_visualizer,
    quality_metrics=metrics,
    context_analytics=analytics
)

# Add traces to dashboard
dashboard.add_trace(trace, compute_metrics=True)

# Launch dashboard
dashboard.run_server()
```

## Additional Visualizers

The toolkit also includes:

- `MetricsVisualizer`: Standard ML evaluation metrics visualizations
- `DimensionReductionVisualizer`: Visualize high-dimensional data in 2D/3D
- `CalibrationVisualizer`: Analyze model calibration and reliability
- `QualityMetricsVisualizer`: Visualize quality metrics across multiple runs

## Installation

The visualization toolkit is included in the BeeAI Framework package:

```bash
pip install beeai-framework
```

## Example Usage

See the example script at `python/examples/visualization/reasoning_evaluation_demo.py` for a complete demonstration of the visualization toolkit.

## Requirements

- Python 3.8+
- plotly
- pandas
- numpy
- networkx
- scikit-learn
- dash (for interactive dashboard) 