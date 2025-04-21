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
Visualization Service Example

This module demonstrates how to use the VisualizationService to create
various visualizations of reasoning traces.
"""

import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import random
from uuid import uuid4

# BeeAI Framework imports
from beeai_framework.visualization import (
    VisualizationService,
    ReasoningTrace,
    ReasoningStep
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_reasoning_trace() -> ReasoningTrace:
    """Create a sample reasoning trace for visualization."""
    
    # Create trace
    trace_id = str(uuid4())
    task = "Analyze the environmental impact of electric vehicles"
    
    trace = ReasoningTrace(
        trace_id=trace_id,
        task=task,
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now(),
        overall_metrics={
            "steps_count": 4,
            "context_sources_used": 6,
            "execution_time_seconds": 300,
            "concepts_identified": 12,
            "reasoning_score": 0.85
        }
    )
    
    # Step 1: Problem Framing
    step1 = ReasoningStep(
        step_number=1,
        step_type="problem_framing",
        content=(
            "To analyze the environmental impact of electric vehicles (EVs), I need to "
            "consider multiple factors:\n\n"
            "1. Manufacturing emissions and resource use\n"
            "2. Energy source for charging (grid mix)\n"
            "3. Lifetime emissions compared to internal combustion engines\n"
            "4. Battery production and recycling\n"
            "5. Infrastructure requirements\n\n"
            "I'll begin by examining manufacturing impacts."
        ),
        timestamp=datetime.now() - timedelta(minutes=5),
        context_items=[
            {
                "id": "src_001",
                "source_type": "research_paper",
                "title": "Life-cycle Assessment of Electric Vehicles",
                "relevance_score": 0.92,
                "content_snippet": "Manufacturing of EVs typically produces more emissions than conventional vehicles..."
            },
            {
                "id": "src_002",
                "source_type": "dataset",
                "title": "Global EV Production Statistics 2022",
                "relevance_score": 0.78,
                "content_snippet": "EV production increased by 43% in 2022, with China accounting for 56% of global production."
            }
        ],
        metrics={
            "clarity_score": 0.88,
            "context_utilization": 0.76,
            "execution_time_seconds": 62
        },
        key_concepts=[
            {"concept": "Electric Vehicle", "importance": 0.95},
            {"concept": "Manufacturing Emissions", "importance": 0.87},
            {"concept": "Battery Production", "importance": 0.83}
        ],
        next_step_suggestions=[
            "Analyze manufacturing emissions",
            "Examine energy sources",
            "Compare to conventional vehicles"
        ]
    )
    
    # Step 2: Manufacturing Analysis
    step2 = ReasoningStep(
        step_number=2,
        step_type="analysis",
        content=(
            "Manufacturing of electric vehicles produces more upfront emissions than conventional vehicles, "
            "primarily due to battery production. Key findings:\n\n"
            "- EV battery production requires mining of lithium, cobalt, and nickel, with significant "
            "environmental impacts in mining regions\n"
            "- The carbon footprint of manufacturing an average EV battery (75kWh) is approximately "
            "7-8 metric tons of CO2\n"
            "- Overall manufacturing emissions for EVs are 15-68% higher than conventional vehicles "
            "depending on battery size and manufacturing efficiency\n"
            "- Recent advancements in battery chemistry and production methods have reduced these "
            "impacts by approximately 20% since 2018\n\n"
            "These higher upfront emissions must be offset during the vehicle's operational lifetime."
        ),
        timestamp=datetime.now() - timedelta(minutes=4),
        context_items=[
            {
                "id": "src_003",
                "source_type": "research_paper",
                "title": "Environmental Impact of Lithium-Ion Batteries",
                "relevance_score": 0.95,
                "content_snippet": "The carbon footprint of manufacturing an average EV battery (75kWh) is approximately 7-8 metric tons of CO2..."
            },
            {
                "id": "src_004",
                "source_type": "industry_report",
                "title": "Advancements in EV Battery Manufacturing 2023",
                "relevance_score": 0.89,
                "content_snippet": "Recent advancements in battery chemistry and production methods have reduced environmental impacts by approximately 20% since 2018."
            }
        ],
        metrics={
            "clarity_score": 0.92,
            "context_utilization": 0.85,
            "execution_time_seconds": 78
        },
        key_concepts=[
            {"concept": "Battery Production", "importance": 0.94},
            {"concept": "Lithium Mining", "importance": 0.82},
            {"concept": "Carbon Footprint", "importance": 0.88},
            {"concept": "Manufacturing Emissions", "importance": 0.90}
        ],
        next_step_suggestions=[
            "Analyze operational emissions",
            "Examine battery recycling",
            "Calculate lifetime emissions"
        ]
    )
    
    # Step 3: Operational Emissions
    step3 = ReasoningStep(
        step_number=3,
        step_type="analysis",
        content=(
            "Operational emissions of EVs depend primarily on the electricity source used for charging. Analysis shows:\n\n"
            "- In regions with clean energy grids (>75% renewable/nuclear), EVs produce 70-80% lower lifecycle "
            "emissions than conventional vehicles\n"
            "- In regions with coal-dominated grids, the benefit is much lower, sometimes only 10-30% reduction\n"
            "- Global average shows EVs produce approximately 50% less lifetime emissions than conventional vehicles\n"
            "- Grid decarbonization trends are improving this calculus yearly\n"
            "- Vehicle efficiency also plays a key role: modern EVs use approximately 0.2-0.3 kWh/mile\n\n"
            "The operational phase is where EVs make up for higher manufacturing emissions."
        ),
        timestamp=datetime.now() - timedelta(minutes=2),
        context_items=[
            {
                "id": "src_005",
                "source_type": "dataset",
                "title": "Global Grid Carbon Intensity 2023",
                "relevance_score": 0.93,
                "content_snippet": "Regional grid carbon intensity varies from 20 g CO2eq/kWh in Norway to 623 g CO2eq/kWh in coal-dependent regions."
            },
            {
                "id": "src_006",
                "source_type": "research_paper",
                "title": "Comparative Analysis of Vehicle Lifecycle Emissions",
                "relevance_score": 0.91,
                "content_snippet": "Global average shows EVs produce approximately 50% less lifetime emissions than conventional vehicles when accounting for current grid mix."
            }
        ],
        metrics={
            "clarity_score": 0.89,
            "context_utilization": 0.90,
            "execution_time_seconds": 85
        },
        key_concepts=[
            {"concept": "Operational Emissions", "importance": 0.92},
            {"concept": "Grid Carbon Intensity", "importance": 0.94},
            {"concept": "Renewable Energy", "importance": 0.85},
            {"concept": "Vehicle Efficiency", "importance": 0.81}
        ],
        next_step_suggestions=[
            "Analyze end-of-life impacts",
            "Calculate total lifecycle comparison",
            "Examine regional variations"
        ]
    )
    
    # Step 4: Conclusion
    step4 = ReasoningStep(
        step_number=4,
        step_type="conclusion",
        content=(
            "Comprehensive analysis of electric vehicles' environmental impact shows:\n\n"
            "1. Manufacturing: EVs have 15-68% higher emissions in production phase, primarily from battery manufacturing\n"
            "2. Operation: EVs produce 50-80% lower emissions during operation, depending on grid cleanliness\n"
            "3. End-of-life: Battery recycling technology is improving but still developing; current recycling "
            "rates recover 50-95% of key materials\n"
            "4. Lifecycle Analysis: EVs typically break even on emissions compared to conventional vehicles after "
            "20,000-40,000 miles, depending on regional grid mix\n"
            "5. Overall: EVs reduce lifecycle GHG emissions by approximately 30-70% in most scenarios\n\n"
            "The environmental benefits of EVs are clear in most contexts but can be maximized through:\n"
            "- Cleaner electricity grids\n"
            "- Improved battery manufacturing\n"
            "- Better recycling infrastructure\n"
            "- Longer vehicle lifespans\n\n"
            "As grid decarbonization continues globally, the benefits of EVs will increase."
        ),
        timestamp=datetime.now(),
        context_items=[
            {
                "id": "src_003",
                "source_type": "research_paper",
                "title": "Environmental Impact of Lithium-Ion Batteries",
                "relevance_score": 0.88,
                "content_snippet": "Battery recycling technology is improving but still developing; current recycling rates recover 50-95% of key materials."
            },
            {
                "id": "src_006",
                "source_type": "research_paper",
                "title": "Comparative Analysis of Vehicle Lifecycle Emissions",
                "relevance_score": 0.96,
                "content_snippet": "EVs typically break even on emissions compared to conventional vehicles after 20,000-40,000 miles, depending on regional grid mix."
            }
        ],
        metrics={
            "clarity_score": 0.94,
            "context_utilization": 0.87,
            "execution_time_seconds": 75
        },
        key_concepts=[
            {"concept": "Lifecycle Analysis", "importance": 0.96},
            {"concept": "Emission Reduction", "importance": 0.93},
            {"concept": "Battery Recycling", "importance": 0.85},
            {"concept": "Grid Decarbonization", "importance": 0.90},
            {"concept": "Environmental Impact", "importance": 0.92}
        ],
        next_step_suggestions=[]
    )
    
    # Add steps to trace
    trace.add_step(step1)
    trace.add_step(step2)
    trace.add_step(step3)
    trace.add_step(step4)
    
    return trace


def main():
    """Run visualization service example."""
    
    # Create output directory
    output_dir = Path("./visualization_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualization service
    viz_service = VisualizationService(
        output_dir=output_dir,
        default_height=700,
        default_width=900
    )
    
    # Create sample reasoning trace
    trace = create_sample_reasoning_trace()
    
    # Generate and export various visualizations
    logger.info("Generating reasoning trace visualization...")
    viz_service.visualize_reasoning_trace(
        trace=trace,
        export_path=str(output_dir / "reasoning_trace.html")
    )
    
    logger.info("Generating knowledge graph visualization...")
    viz_service.visualize_knowledge_graph(
        trace=trace,
        export_path=str(output_dir / "knowledge_graph.html")
    )
    
    logger.info("Generating context relevance visualization...")
    viz_service.visualize_context_relevance(
        trace=trace,
        selected_step=3,
        export_path=str(output_dir / "context_relevance.html")
    )
    
    logger.info("Generating context evolution visualization...")
    viz_service.visualize_context_evolution(
        trace=trace,
        export_path=str(output_dir / "context_evolution.html")
    )
    
    logger.info("Generating step transition visualization...")
    viz_service.visualize_step_transitions(
        trace=trace,
        export_path=str(output_dir / "step_transitions.html")
    )
    
    # Compute quality metrics
    logger.info("Computing quality metrics...")
    metrics = viz_service.compute_quality_metrics(trace)
    
    # Visualize quality metrics
    logger.info("Generating quality metrics visualization...")
    viz_service.visualize_quality_metrics(
        metrics=metrics,
        export_path=str(output_dir / "quality_metrics.html")
    )
    
    # Export trace to JSON
    logger.info("Exporting trace to JSON...")
    json_path = viz_service.export_trace_to_json(
        trace=trace,
        file_path=str(output_dir / "reasoning_trace.json")
    )
    
    logger.info(f"All visualizations and data saved to {output_dir}")
    logger.info(f"To reuse the trace data, you can import it from {json_path}")


if __name__ == "__main__":
    main() 