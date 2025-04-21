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
import pytest
import time
from typing import Dict, Any, List

from beeai_framework.workflows.parallel.orchestrator import (
    ParallelOrchestrator, 
    ParallelExecutionConfig,
    ParallelTaskMapping
)
from beeai_framework.workflows.agents.agent_base import (
    AgentRole, AgentTask, AgentResult, BaseWorkflowAgent
)
from beeai_framework.utils.cancellation import AbortController


class SimpleTestAgent(BaseWorkflowAgent):
    """Simple agent implementation for testing"""
    
    async def process_task(self, task: AgentTask) -> AgentResult:
        await asyncio.sleep(0.2)  # Simulate processing
        
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            success=True,
            result={"message": f"Task {task.name} completed by agent {self.agent_id}"},
            metadata={"execution_time": time.time()}
        )
    
    async def receive_message(self, message):
        pass
    
    async def clone(self):
        return SimpleTestAgent(agent_id=f"{self.agent_id}_clone", role=self.role)


class FailingTestAgent(BaseWorkflowAgent):
    """Agent that fails on task processing"""
    
    async def process_task(self, task: AgentTask) -> AgentResult:
        await asyncio.sleep(0.2)  # Simulate processing
        
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            success=False,
            error="Task processing failed intentionally",
            metadata={"execution_time": time.time()}
        )
    
    async def receive_message(self, message):
        pass
    
    async def clone(self):
        return FailingTestAgent(agent_id=f"{self.agent_id}_clone", role=self.role)


@pytest.mark.asyncio
async def test_parallel_orchestrator_basic():
    """Test basic parallel orchestrator functionality"""
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
        workflow_id="test_parallel_workflow",
        parallel_config=parallel_config
    )
    
    try:
        # Register agents
        thinker_agent = SimpleTestAgent(agent_id="thinker_1", role=AgentRole.THINKER)
        executor_agent = SimpleTestAgent(agent_id="executor_1", role=AgentRole.EXECUTOR)
        validator_agent = SimpleTestAgent(agent_id="validator_1", role=AgentRole.VALIDATOR)
        
        await orchestrator.register_agent(thinker_agent, "thinker_1", AgentRole.THINKER)
        await orchestrator.register_agent(executor_agent, "executor_1", AgentRole.EXECUTOR)
        await orchestrator.register_agent(validator_agent, "validator_1", AgentRole.VALIDATOR)
        
        # Create independent tasks (no dependencies)
        task_ids = []
        for i in range(5):
            task_id = await orchestrator.add_task(
                name=f"Task {i}",
                description=f"Test task {i}",
                role=AgentRole.EXECUTOR
            )
            task_ids.append(task_id)
        
        # Run orchestrator with timeout
        abort_controller = AbortController()
        asyncio.get_event_loop().call_later(5, lambda: abort_controller.abort("Test timeout"))
        
        # Execute workflow
        start_time = time.time()
        result = await orchestrator.run(signal=abort_controller.signal)
        elapsed = time.time() - start_time
        
        # Check that all tasks completed
        assert len(result.completed_tasks) == 5
        
        # Parallel execution should be faster than sequential for independent tasks
        # Each task takes 0.2s, so with 5 tasks sequentially it would take ~1s
        # With parallel execution, it should be much faster
        assert elapsed < 0.8, "Parallel execution wasn't faster than sequential would be"
        
        # All tasks should have results
        for task_id in task_ids:
            task_result = orchestrator.get_task_result(task_id)
            assert task_result is not None
            assert task_result.success
            
        assert orchestrator.state.workflow_completed
        assert orchestrator.state.workflow_success
        
    finally:
        # No explicit cleanup needed - handled in run method
        pass


@pytest.mark.asyncio
async def test_parallel_orchestrator_dependencies():
    """Test parallel orchestrator with task dependencies"""
    # Configure parallel execution
    parallel_config = ParallelExecutionConfig(
        enabled=True,
        max_concurrent_tasks=5
    )
    
    # Create orchestrator with parallel execution
    orchestrator = ParallelOrchestrator(
        workflow_id="test_dependencies_workflow",
        parallel_config=parallel_config
    )
    
    try:
        # Register agents
        thinker_agent = SimpleTestAgent(agent_id="thinker_dep", role=AgentRole.THINKER)
        executor_agent = SimpleTestAgent(agent_id="executor_dep", role=AgentRole.EXECUTOR)
        
        await orchestrator.register_agent(thinker_agent, "thinker_dep", AgentRole.THINKER)
        await orchestrator.register_agent(executor_agent, "executor_dep", AgentRole.EXECUTOR)
        
        # Create tasks with dependencies
        task1_id = await orchestrator.add_task(
            name="Plan Work",
            description="Create a plan for the workflow",
            role=AgentRole.THINKER
        )
        
        # These tasks depend on task1
        dependent_task_ids = []
        for i in range(3):
            task_id = await orchestrator.add_task(
                name=f"Execute Step {i}",
                description=f"Execute step {i} of the plan",
                role=AgentRole.EXECUTOR,
                dependencies=[task1_id]
            )
            dependent_task_ids.append(task_id)
        
        # Final task depends on all executor tasks
        final_task_id = await orchestrator.add_task(
            name="Final Task",
            description="Final task that depends on all executor tasks",
            role=AgentRole.THINKER,
            dependencies=dependent_task_ids
        )
        
        # Execute workflow
        start_time = time.time()
        result = await orchestrator.run()
        elapsed = time.time() - start_time
        
        # Check that all tasks completed
        assert len(result.completed_tasks) == 5
        
        # Verify execution order via completion timestamps
        task1_entry = orchestrator.state.tasks[task1_id]
        final_task_entry = orchestrator.state.tasks[final_task_id]
        
        # Final task should complete after task1
        assert task1_entry.completed_at < final_task_entry.completed_at
        
        # Dependent tasks should complete after task1 but before final task
        for dep_id in dependent_task_ids:
            dep_entry = orchestrator.state.tasks[dep_id]
            assert task1_entry.completed_at < dep_entry.completed_at
            assert dep_entry.completed_at < final_task_entry.completed_at
        
        assert orchestrator.state.workflow_completed
        assert orchestrator.state.workflow_success
        
    finally:
        # No explicit cleanup needed
        pass


@pytest.mark.asyncio
async def test_parallel_orchestrator_failing_task():
    """Test parallel orchestrator with a failing task"""
    # Configure parallel execution
    parallel_config = ParallelExecutionConfig(
        enabled=True,
        max_concurrent_tasks=5
    )
    
    # Create orchestrator with parallel execution
    orchestrator = ParallelOrchestrator(
        workflow_id="test_failure_workflow",
        parallel_config=parallel_config
    )
    
    try:
        # Register agents - one normal, one failing
        success_agent = SimpleTestAgent(agent_id="success_agent", role=AgentRole.THINKER)
        failing_agent = FailingTestAgent(agent_id="failing_agent", role=AgentRole.EXECUTOR)
        
        await orchestrator.register_agent(success_agent, "success_agent", AgentRole.THINKER)
        await orchestrator.register_agent(failing_agent, "failing_agent", AgentRole.EXECUTOR)
        
        # Create tasks - one for each agent
        success_task_id = await orchestrator.add_task(
            name="Success Task",
            description="Task that will succeed",
            role=AgentRole.THINKER
        )
        
        fail_task_id = await orchestrator.add_task(
            name="Fail Task",
            description="Task that will fail",
            role=AgentRole.EXECUTOR
        )
        
        # Task that depends on both tasks - should be blocked by failing task
        dependent_task_id = await orchestrator.add_task(
            name="Dependent Task",
            description="Task that depends on both success and fail tasks",
            role=AgentRole.THINKER,
            dependencies=[success_task_id, fail_task_id]
        )
        
        # Execute workflow
        result = await orchestrator.run()
        
        # Check task statuses
        assert success_task_id in result.completed_tasks
        assert fail_task_id in result.failed_tasks
        
        # Dependent task should not be in completed tasks
        assert dependent_task_id not in result.completed_tasks
        
        # Get task entries
        success_task_entry = orchestrator.state.tasks[success_task_id]
        fail_task_entry = orchestrator.state.tasks[fail_task_id]
        dependent_task_entry = orchestrator.state.tasks[dependent_task_id]
        
        # Check statuses
        assert success_task_entry.status == "completed"
        assert fail_task_entry.status == "failed"
        assert dependent_task_entry.status == "blocked"  # Should be blocked due to dependency failure
        
        # Workflow should be completed but not successful
        assert orchestrator.state.workflow_completed
        assert not orchestrator.state.workflow_success
        
    finally:
        # No explicit cleanup needed
        pass


@pytest.mark.asyncio
async def test_parallel_orchestrator_disable_parallel():
    """Test disabling parallel execution at runtime"""
    # Start with parallel enabled
    parallel_config = ParallelExecutionConfig(
        enabled=True,
        max_concurrent_tasks=5
    )
    
    # Create orchestrator with parallel execution
    orchestrator = ParallelOrchestrator(
        workflow_id="test_disable_parallel",
        parallel_config=parallel_config
    )
    
    try:
        # Register agent
        agent = SimpleTestAgent(agent_id="test_agent", role=AgentRole.EXECUTOR)
        await orchestrator.register_agent(agent, "test_agent", AgentRole.EXECUTOR)
        
        # Add several independent tasks
        task_ids = []
        for i in range(5):
            task_id = await orchestrator.add_task(
                name=f"Parallel Task {i}",
                description=f"Task {i} for parallel execution test",
                role=AgentRole.EXECUTOR
            )
            task_ids.append(task_id)
        
        # Run with parallel execution enabled
        start_time = time.time()
        await orchestrator.run()
        parallel_time = time.time() - start_time
        
        # Reset orchestrator state and disable parallel execution
        orchestrator = ParallelOrchestrator(
            workflow_id="test_disable_parallel_2",
            parallel_config=ParallelExecutionConfig(enabled=False)
        )
        
        # Register agent again
        agent = SimpleTestAgent(agent_id="test_agent", role=AgentRole.EXECUTOR)
        await orchestrator.register_agent(agent, "test_agent", AgentRole.EXECUTOR)
        
        # Add several independent tasks again
        task_ids = []
        for i in range(5):
            task_id = await orchestrator.add_task(
                name=f"Sequential Task {i}",
                description=f"Task {i} for sequential execution test",
                role=AgentRole.EXECUTOR
            )
            task_ids.append(task_id)
        
        # Run with parallel execution disabled
        start_time = time.time()
        await orchestrator.run()
        sequential_time = time.time() - start_time
        
        # Parallel should be faster than sequential
        assert parallel_time < sequential_time, "Parallel execution wasn't faster than sequential"
        
    finally:
        # No explicit cleanup needed
        pass


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 