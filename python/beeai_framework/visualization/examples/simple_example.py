#!/usr/bin/env python
"""
Simple example to demonstrate the visualization data format.

This script creates a mock visualization dataset without
requiring the actual visualization components to be properly installed.
"""

import json
import os
from datetime import datetime
import random

# Sample data structures
class StepData:
    def __init__(self, step_number, step_type, content, timestamp=None, duration=None):
        self.step_number = step_number
        self.step_type = step_type
        self.content = content
        self.timestamp = timestamp or datetime.now().timestamp()
        self.duration = duration or random.uniform(1.0, 5.0)
        self.context_references = []
    
    def to_dict(self):
        return {
            "step_number": self.step_number,
            "step_type": self.step_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "context_references": self.context_references
        }

class SourceData:
    def __init__(self, source_id, name, source_type):
        self.source_id = source_id
        self.name = name
        self.source_type = source_type
        self.usage_count = 0
        self.steps_referenced = []
    
    def to_dict(self):
        return {
            "source_id": self.source_id,
            "name": self.name,
            "source_type": self.source_type,
            "usage_count": self.usage_count,
            "steps_referenced": self.steps_referenced
        }

def create_sample_metrics():
    """Create sample metrics visualization data."""
    # Sample steps
    steps = []
    for i in range(1, 8):
        step_type = ["analysis", "synthesis", "research", "conclusion"][i % 4]
        step = StepData(
            step_number=i,
            step_type=step_type,
            content=f"This is step {i} with type {step_type}",
            duration=random.uniform(1.0, 8.0)
        )
        steps.append(step)
    
    # Sample sources
    sources = []
    for i in range(1, 5):
        source_type = ["document", "code", "web", "comment"][i % 4]
        source = SourceData(
            source_id=f"source_{i}",
            name=f"Source {i}: {source_type.capitalize()}",
            source_type=source_type
        )
        
        # Randomly assign to steps
        for step in steps:
            if random.random() > 0.5:
                source.steps_referenced.append(step.step_number)
                step.context_references.append({
                    "source": source.source_id,
                    "relevance_score": random.uniform(0.3, 0.9)
                })
                source.usage_count += 1
        
        sources.append(source)
    
    # Create time series data
    time_series = {
        "step_numbers": [step.step_number for step in steps],
        "durations": [step.duration for step in steps],
        "cumulative_time": []
    }
    
    # Calculate cumulative time
    cumulative = 0
    for duration in time_series["durations"]:
        cumulative += duration
        time_series["cumulative_time"].append(cumulative)
    
    # Create radar chart data
    radar_data = {
        "dimensions": ["efficiency", "relevance", "completeness", "coherence"],
        "scores": [random.uniform(50, 100) for _ in range(4)]
    }
    
    # Create heatmap data
    heatmap_data = {
        "x_labels": [source.source_id for source in sources],
        "y_labels": ["analysis", "synthesis", "research", "conclusion"],
        "data": [
            [random.uniform(0, 5) for _ in range(len(sources))]
            for _ in range(4)
        ]
    }
    
    # Combine into metrics visualization
    metrics_data = {
        "trace_id": "sample_trace_001",
        "summary": {
            "step_count": len(steps),
            "total_time": sum(time_series["durations"]),
            "avg_time_per_step": sum(time_series["durations"]) / len(steps),
            "context_references": sum(source.usage_count for source in sources)
        },
        "time_series": time_series,
        "radar_charts": {
            "quality": radar_data
        },
        "heat_maps": {
            "context_usage": heatmap_data
        }
    }
    
    return metrics_data

def create_output_directories():
    """Create output directories if they don't exist."""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    """Main function to run the example."""
    print("=== Simple Metrics Visualization Example ===")
    
    # Create metrics data
    metrics_data = create_sample_metrics()
    
    # Print summary
    print("\nMetrics Summary:")
    for key, value in metrics_data["summary"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Create output directory
    output_dir = create_output_directories()
    
    # Save metrics data to file
    output_file = os.path.join(output_dir, "metrics_example.json")
    with open(output_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\nSaved metrics data to {output_file}")
    
    print("\nThis is a simplified example that demonstrates the data format.")
    print("To see the full visualization system in action, the complete")
    print("visualization module needs to be properly installed.")

if __name__ == "__main__":
    main() 