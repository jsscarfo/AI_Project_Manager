#!/usr/bin/env python3
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

import asyncio
import time
import random
import logging
from typing import Dict, Any, List

from beeai_framework.workflows.parallel.controller import (
    ParallelExecutionController, ResourceConstraint, TaskPriority
)
from beeai_framework.utils.cancellation import AbortController
from beeai_framework.visualization.parallel_execution import ParallelExecutionVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def simulate_api_call(duration: float, signal=None):
    """Simulate an API call with a given duration"""
    logger.info(f"Making API call that will take {duration:.2f} seconds")
    start = time.time()
    
    # Break up the sleep into smaller chunks to check for abort signal
    chunk_size = 0.1
    chunks = int(duration / chunk_size)
    
    for _ in range(chunks):
        # Check if aborted
        if signal and signal.aborted:
            logger.info("API call aborted")
            return {"status": "aborted"}
        
        await asyncio.sleep(chunk_size)
    
    # Sleep any remaining time
    remaining = duration - (chunks * chunk_size)
    if remaining > 0:
        await asyncio.sleep(remaining)
    
    elapsed = time.time() - start
    logger.info(f"API call completed after {elapsed:.2f} seconds")
    return {"status": "success", "duration": elapsed}


async def process_data(data: Dict[str, Any], signal=None):
    """Process data from an API call"""
    logger.info(f"Processing data: {data}")
    
    # Simulate processing time
    process_time = random.uniform(0.5, 2.0)
    await asyncio.sleep(process_time)
    
    logger.info(f"Data processing completed in {process_time:.2f} seconds")
    return {"processed_data": data, "processing_time": process_time}


async def analyze_results(results: List[Dict[str, Any]], signal=None):
    """Analyze multiple results"""
    logger.info(f"Analyzing {len(results)} results")
    
    # Simulate analysis time
    analysis_time = random.uniform(1.0, 3.0)
    await asyncio.sleep(analysis_time)
    
    # Extract some metrics
    total_time = sum(result.get("processing_time", 0) for result in results)
    
    logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
    return {
        "num_results": len(results),
        "total_processing_time": total_time,
        "analysis_time": analysis_time
    }


async def generate_report(analysis_result: Dict[str, Any], signal=None):
    """Generate a report from analysis results"""
    logger.info(f"Generating report from analysis: {analysis_result}")
    
    # Simulate report generation time
    report_time = random.uniform(0.5, 1.5)
    await asyncio.sleep(report_time)
    
    logger.info(f"Report generation completed in {report_time:.2f} seconds")
    return {
        "report": {
            "title": "Data Processing Report",
            "summary": f"Processed {analysis_result.get('num_results', 0)} results",
            "total_time": analysis_result.get('total_processing_time', 0),
            "analysis_time": analysis_result.get('analysis_time', 0),
            "report_generation_time": report_time
        }
    }


async def run_workflow_example():
    """Execute a parallel workflow example"""
    logger.info("Starting parallel workflow example")
    
    # Create parallel execution controller
    controller = ParallelExecutionController(max_concurrent_tasks=5)
    
    # Set up visualization
    visualizer = ParallelExecutionVisualizer(controller)
    
    # Register resources
    controller.register_resource("api_calls", 3.0)  # Max 3 concurrent API calls
    controller.register_resource("cpu", 5.0)  # 5 CPU units available
    
    # Register concurrency groups
    controller.register_concurrency_group("analysis", 2)  # Max 2 concurrent analysis tasks
    
    try:
        # Create API call tasks
        api_task_ids = []
        for i in range(5):
            duration = random.uniform(1.0, 3.0)
            task_id = controller.add_task(
                id=f"api_call_{i}",
                name=f"API Call {i}",
                function=simulate_api_call,
                args=(duration,),
                priority=TaskPriority.HIGH,
                resource_constraints=[
                    ResourceConstraint(name="api_calls", amount=1.0)
                ],
                concurrency_group="api_calls",
                max_retries=2,
                timeout=10.0
            )
            api_task_ids.append(task_id)
        
        # Create data processing tasks that depend on API calls
        processing_task_ids = []
        for i, api_task_id in enumerate(api_task_ids):
            task_id = controller.add_task(
                id=f"process_{i}",
                name=f"Process Data {i}",
                function=process_data,
                kwargs={"data": f"data_from_api_{i}"},
                dependencies=[api_task_id],
                priority=TaskPriority.MEDIUM,
                resource_constraints=[
                    ResourceConstraint(name="cpu", amount=1.0)
                ]
            )
            processing_task_ids.append(task_id)
        
        # Create analysis tasks that combine processing results
        analysis_1_id = controller.add_task(
            id="analyze_group_1",
            name="Analyze Group 1",
            function=analyze_results,
            kwargs={"results": "results_placeholder"},  # Will be replaced with actual results
            dependencies=processing_task_ids[:2],
            priority=TaskPriority.MEDIUM,
            resource_constraints=[
                ResourceConstraint(name="cpu", amount=2.0)
            ],
            concurrency_group="analysis"
        )
        
        analysis_2_id = controller.add_task(
            id="analyze_group_2",
            name="Analyze Group 2",
            function=analyze_results,
            kwargs={"results": "results_placeholder"},  # Will be replaced with actual results
            dependencies=processing_task_ids[2:],
            priority=TaskPriority.MEDIUM,
            resource_constraints=[
                ResourceConstraint(name="cpu", amount=2.0)
            ],
            concurrency_group="analysis"
        )
        
        # Create final report task
        report_id = controller.add_task(
            id="generate_report",
            name="Generate Final Report",
            function=generate_report,
            kwargs={"analysis_result": "analysis_placeholder"},  # Will be replaced with actual results
            dependencies=[analysis_1_id, analysis_2_id],
            priority=TaskPriority.LOW,
            resource_constraints=[
                ResourceConstraint(name="cpu", amount=1.0)
            ]
        )
        
        # Start the controller
        await controller.start()
        
        # Execute tasks with a timeout of 30 seconds
        abort_controller = AbortController()
        asyncio.get_event_loop().call_later(30, lambda: abort_controller.abort("Timeout exceeded"))
        
        # Wait for all tasks to complete
        results = await controller.execute(signal=abort_controller.signal)
        
        # Display results
        logger.info("Workflow completed")
        logger.info(f"Results: {results}")
        
        # Create visualizations
        visualizer.visualize_all("parallel_workflow_example")
        
        return results
        
    finally:
        # Always stop the controller
        await controller.stop()


async def run_parallel_orchestrator_example():
    """Example using the ParallelOrchestrator with agents"""
    from beeai_framework.workflows.parallel.orchestrator import (
        ParallelOrchestrator, ParallelExecutionConfig
    )
    from beeai_framework.workflows.agents.agent_base import (
        AgentRole, AgentTask, AgentResult, BaseWorkflowAgent
    )
    
    # Define a simple agent class
    class SimpleAgent(BaseWorkflowAgent):
        async def process_task(self, task: AgentTask) -> AgentResult:
            logger.info(f"Agent {self.agent_id} processing task: {task.name}")
            
            # Simulate processing time
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=True,
                result={"message": f"Task {task.name} completed by agent {self.agent_id}"},
                metadata={"execution_time": time.time()}
            )
        
        async def receive_message(self, message):
            logger.info(f"Agent {self.agent_id} received message: {message}")
        
        async def clone(self):
            return SimpleAgent(agent_id=f"{self.agent_id}_clone", role=self.role)
    
    logger.info("Starting parallel orchestrator example")
    
    # Configure parallel execution
    parallel_config = ParallelExecutionConfig(
        enabled=True,
        max_concurrent_tasks=5,
        resource_limits={
            "compute": 5.0,
            "llm_api": 3.0
        },
        concurrency_groups={
            "role_thinker": 2,
            "role_executor": 3
        }
    )
    
    # Create orchestrator with parallel execution
    orchestrator = ParallelOrchestrator(
        workflow_id="parallel_agent_workflow",
        parallel_config=parallel_config
    )
    
    try:
        # Register agents
        thinker_agent = SimpleAgent(agent_id="thinker_1", role=AgentRole.THINKER)
        executor_agent = SimpleAgent(agent_id="executor_1", role=AgentRole.EXECUTOR)
        validator_agent = SimpleAgent(agent_id="validator_1", role=AgentRole.VALIDATOR)
        
        await orchestrator.register_agent(thinker_agent, "thinker_1", AgentRole.THINKER)
        await orchestrator.register_agent(executor_agent, "executor_1", AgentRole.EXECUTOR)
        await orchestrator.register_agent(validator_agent, "validator_1", AgentRole.VALIDATOR)
        
        # Create tasks with dependencies
        task1_id = await orchestrator.add_task(
            name="Plan Work",
            description="Create a plan for the workflow",
            role=AgentRole.THINKER
        )
        
        task2_id = await orchestrator.add_task(
            name="Execute Step 1",
            description="Execute the first step of the plan",
            role=AgentRole.EXECUTOR,
            dependencies=[task1_id]
        )
        
        task3_id = await orchestrator.add_task(
            name="Execute Step 2",
            description="Execute the second step of the plan",
            role=AgentRole.EXECUTOR,
            dependencies=[task1_id]
        )
        
        task4_id = await orchestrator.add_task(
            name="Validate Results",
            description="Validate the execution results",
            role=AgentRole.VALIDATOR,
            dependencies=[task2_id, task3_id]
        )
        
        # Run orchestrator with timeout
        abort_controller = AbortController()
        asyncio.get_event_loop().call_later(30, lambda: abort_controller.abort("Timeout exceeded"))
        
        # Execute workflow
        result = await orchestrator.run(signal=abort_controller.signal)
        
        logger.info("Orchestrator workflow completed")
        logger.info(f"Final state: {result}")
        
        return result
    
    finally:
        # No explicit cleanup needed - handled in run method
        pass


async def error_handling_example():
    """Example of error handling and retries in parallel execution"""
    logger.info("Starting error handling example")
    
    # Create controller with retries
    controller = ParallelExecutionController(max_concurrent_tasks=3)
    
    # Define a task that occasionally fails
    async def flaky_task(task_id, failure_rate=0.5, signal=None):
        logger.info(f"Running flaky task {task_id}")
        
        # Simulate processing
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Randomly fail based on failure rate
        if random.random() < failure_rate:
            logger.error(f"Task {task_id} failed!")
            raise RuntimeError(f"Random failure in task {task_id}")
        
        logger.info(f"Task {task_id} succeeded")
        return {"task_id": task_id, "status": "success"}
    
    try:
        # Add tasks with retries
        for i in range(5):
            # Higher failure rate for some tasks
            failure_rate = 0.8 if i % 2 == 0 else 0.3
            
            controller.add_task(
                id=f"flaky_{i}",
                name=f"Flaky Task {i}",
                function=flaky_task,
                args=(f"flaky_{i}", failure_rate),
                max_retries=3  # Will retry up to 3 times
            )
        
        # Add dependent task
        controller.add_task(
            id="final_task",
            name="Final Task",
            function=lambda results, signal=None: {"message": "All tasks completed", "results": results},
            kwargs={"results": "placeholder"},
            dependencies=[f"flaky_{i}" for i in range(5)]
        )
        
        # Execute with visualization
        visualizer = ParallelExecutionVisualizer(controller)
        await controller.start()
        
        results = await controller.execute()
        
        # Generate visualizations showing retries
        visualizer.visualize_all("error_handling_example")
        
        logger.info("Error handling example completed")
        logger.info(f"Results: {results}")
        logger.info(f"Failed tasks: {controller.get_failed_tasks()}")
        
        return results
    
    finally:
        await controller.stop()


async def resource_management_example():
    """Example demonstrating resource management capabilities"""
    logger.info("Starting resource management example")
    
    # Create controller with resource limits
    controller = ParallelExecutionController(max_concurrent_tasks=10)
    
    # Register resources
    controller.register_resource("memory", 10.0)  # 10 GB memory
    controller.register_resource("cpu", 4.0)      # 4 CPU cores
    controller.register_resource("gpu", 1.0)      # 1 GPU
    
    # Register concurrency groups
    controller.register_concurrency_group("light_tasks", 5)
    controller.register_concurrency_group("heavy_tasks", 2)
    
    # Define task that uses specific resources
    async def resource_task(name, memory, cpu, gpu=0.0, duration=1.0, signal=None):
        logger.info(f"Task {name} starting (mem={memory}GB, cpu={cpu}, gpu={gpu})")
        await asyncio.sleep(duration)
        logger.info(f"Task {name} completed after {duration} seconds")
        return {
            "name": name,
            "resources_used": {
                "memory": memory,
                "cpu": cpu,
                "gpu": gpu
            },
            "duration": duration
        }
    
    try:
        # Add CPU+memory intensive tasks
        cpu_task_ids = []
        for i in range(8):
            task_id = controller.add_task(
                id=f"cpu_task_{i}",
                name=f"CPU Task {i}",
                function=resource_task,
                args=(f"cpu_task_{i}", 1.0, 1.0, 0.0, random.uniform(1.0, 3.0)),
                resource_constraints=[
                    ResourceConstraint(name="memory", amount=1.0),
                    ResourceConstraint(name="cpu", amount=1.0)
                ],
                concurrency_group="light_tasks"
            )
            cpu_task_ids.append(task_id)
        
        # Add GPU tasks
        gpu_task_ids = []
        for i in range(3):
            task_id = controller.add_task(
                id=f"gpu_task_{i}",
                name=f"GPU Task {i}",
                function=resource_task,
                args=(f"gpu_task_{i}", 4.0, 2.0, 1.0, random.uniform(2.0, 4.0)),
                resource_constraints=[
                    ResourceConstraint(name="memory", amount=4.0),
                    ResourceConstraint(name="cpu", amount=2.0),
                    ResourceConstraint(name="gpu", amount=1.0)
                ],
                concurrency_group="heavy_tasks",
                priority=TaskPriority.HIGH  # GPU tasks get priority
            )
            gpu_task_ids.append(task_id)
        
        # Execute with visualization
        visualizer = ParallelExecutionVisualizer(controller)
        await controller.start()
        
        results = await controller.execute()
        
        # Generate visualizations showing resource utilization
        visualizer.visualize_all("resource_management_example")
        
        logger.info("Resource management example completed")
        logger.info(f"Results: {results}")
        
        return results
    
    finally:
        await controller.stop()


if __name__ == "__main__":
    """Run all examples"""
    asyncio.run(run_workflow_example())
    asyncio.run(run_parallel_orchestrator_example())
    asyncio.run(error_handling_example())
    asyncio.run(resource_management_example()) 