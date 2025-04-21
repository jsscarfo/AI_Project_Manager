#!/usr/bin/env python
"""
Simple Test Runner for Visualization Components.

This script provides a direct import path that avoids the circular import 
issues in the main framework by directly importing specific components.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add the parent directory to path to allow local imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import directly to avoid circular imports
from beeai_framework.visualization.examples.simple_steps_test import (
    StepsVisualizer, 
    TraceVisualizationData,
    StepVisualizationData,
    ContextSourceData,
    create_sample_data,
    test_visualizer
)

def main():
    """Run all tests."""
    print("\n===== Running Step Visualizer Tests =====")
    trace_data = create_sample_data()
    print(f"Created trace with {len(trace_data.steps)} steps and {len(trace_data.context_sources)} context sources")
    
    visualizer = StepsVisualizer()
    vis_data = visualizer.generate_visualization_data(trace_data)
    
    print(f"Visualization data generated for trace: {vis_data['trace_id']}")
    print(f"Number of steps: {len(vis_data['steps'])}")
    print(f"Step types: {vis_data['flow_chart']['step_types']}")
    print(f"Transitions: {vis_data['transitions']['total_transitions']}")
    print(f"Progress complete: {vis_data['progress']['is_complete']}")
    
    # Compare steps visualization
    step1 = trace_data.steps[0]
    step_last = trace_data.steps[-1]
    comparison = visualizer.compare_steps([step1, step_last])
    print("\nComparing first and last steps:")
    print(f"Step numbers compared: {comparison['step_numbers']}")
    print(f"Step types compared: {comparison['step_types']}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 