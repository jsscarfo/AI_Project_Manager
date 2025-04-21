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

from beeai_framework.workflows.parallel.controller import (
    ParallelExecutionController,
    ResourceConstraint,
    TaskPriority,
    TaskState,
    ParallelTask
)
from beeai_framework.utils.cancellation import AbortController


# Helper functions for testing
async def async_task(result=None, delay=0.1, signal=None):
    """Simple async task that returns a result after a delay"""
    await asyncio.sleep(delay)
    return result


async def async_task_with_signal_check(result=None, delay=0.5, signal=None):
    """Task that checks for abort signal during execution"""
    start_time = time.time()
    elapsed = 0
    
    while elapsed < delay:
        if signal and signal.aborted:
            return {"status": "aborted", "elapsed": elapsed}
        
        await asyncio.sleep(0.05)
        elapsed = time.time() - start_time
    
    return {"status": "completed", "result": result, "elapsed": elapsed}


async def failing_task(delay=0.1, signal=None):
    """Task that always fails"""
    await asyncio.sleep(delay)
    raise ValueError("Task failed intentionally")


@pytest.mark.asyncio
async def test_basic_execution():
    """Test basic execution of tasks"""
    controller = ParallelExecutionController()
    
    try:
        # Add two independent tasks
        task1_id = controller.add_task(
            id="task1",
            name="Task 1",
            function=async_task,
            kwargs={"result": "result1", "delay": 0.1}
        )
        
        task2_id = controller.add_task(
            id="task2",
            name="Task 2",
            function=async_task,
            kwargs={"result": "result2", "delay": 0.1}
        )
        
        # Execute tasks
        await controller.start()
        results = await controller.execute()
        
        # Check results
        assert len(results) == 2
        assert results[task1_id] == "result1"
        assert results[task2_id] == "result2"
        
        # Check task states
        assert controller.get_task(task1_id).state == TaskState.COMPLETED
        assert controller.get_task(task2_id).state == TaskState.COMPLETED
        
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_task_dependencies():
    """Test task dependencies are respected"""
    controller = ParallelExecutionController()
    
    try:
        # Create tasks with dependencies
        task1_id = controller.add_task(
            id="dep_task1",
            name="Dependency Task 1",
            function=async_task,
            kwargs={"result": "dep_result1", "delay": 0.1}
        )
        
        task2_id = controller.add_task(
            id="dep_task2",
            name="Dependency Task 2",
            function=async_task,
            kwargs={"result": "dep_result2", "delay": 0.1},
            dependencies=[task1_id]  # Depends on task1
        )
        
        task3_id = controller.add_task(
            id="dep_task3",
            name="Dependency Task 3",
            function=async_task,
            kwargs={"result": "dep_result3", "delay": 0.1},
            dependencies=[task2_id]  # Depends on task2
        )
        
        # Start execution
        start_time = time.time()
        await controller.start()
        results = await controller.execute()
        elapsed = time.time() - start_time
        
        # Check results
        assert len(results) == 3
        assert results[task1_id] == "dep_result1"
        assert results[task2_id] == "dep_result2"
        assert results[task3_id] == "dep_result3"
        
        # Ensure execution time respects dependencies (should be at least 0.3s)
        assert elapsed >= 0.3, "Execution completed too quickly, dependencies may not have been respected"
        
        # Check execution order via completion timestamps
        assert controller.get_task(task1_id).completed_at <= controller.get_task(task2_id).completed_at
        assert controller.get_task(task2_id).completed_at <= controller.get_task(task3_id).completed_at
        
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_parallel_execution():
    """Test tasks execute in parallel when possible"""
    controller = ParallelExecutionController(max_concurrent_tasks=10)
    
    try:
        # Create multiple independent tasks with a longer delay
        task_ids = []
        for i in range(5):
            task_id = controller.add_task(
                id=f"parallel_task_{i}",
                name=f"Parallel Task {i}",
                function=async_task,
                kwargs={"result": f"result_{i}", "delay": 0.3}
            )
            task_ids.append(task_id)
        
        # Start execution and measure time
        start_time = time.time()
        await controller.start()
        results = await controller.execute()
        elapsed = time.time() - start_time
        
        # Check results
        assert len(results) == 5
        for i, task_id in enumerate(task_ids):
            assert results[task_id] == f"result_{i}"
        
        # Ensure execution time is close to the delay (not 5x the delay)
        # Allow some buffer for test execution overhead
        assert elapsed < 0.7, "Execution took too long, tasks may not have run in parallel"
        
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_resource_constraints():
    """Test resource constraints limit concurrent execution"""
    controller = ParallelExecutionController(max_concurrent_tasks=10)
    
    # Register resource with limited capacity
    controller.register_resource("test_resource", 2.0)  # Only 2 units available
    
    try:
        # Create 5 tasks that each require 1 unit of the resource
        task_ids = []
        for i in range(5):
            task_id = controller.add_task(
                id=f"resource_task_{i}",
                name=f"Resource Task {i}",
                function=async_task_with_signal_check,
                kwargs={"result": f"resource_result_{i}", "delay": 0.3},
                resource_constraints=[
                    ResourceConstraint(name="test_resource", amount=1.0)
                ]
            )
            task_ids.append(task_id)
        
        # Start execution and measure time
        start_time = time.time()
        await controller.start()
        results = await controller.execute()
        elapsed = time.time() - start_time
        
        # Check results
        assert len(results) == 5
        for i, task_id in enumerate(task_ids):
            assert results[task_id]["result"] == f"resource_result_{i}"
        
        # Ensure execution time is consistent with resource constraints
        # With 5 tasks needing 1 unit each, and only 2 units available,
        # we should need at least 3 time periods (0.3s each) to complete all tasks
        assert elapsed >= 0.6, "Execution completed too quickly, resource constraints may not be working"
        
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_concurrency_groups():
    """Test concurrency groups limit tasks within a group"""
    controller = ParallelExecutionController(max_concurrent_tasks=10)
    
    # Register concurrency group with limit
    controller.register_concurrency_group("test_group", 2)  # Max 2 concurrent tasks
    
    try:
        # Create 5 tasks in the same concurrency group
        task_ids = []
        for i in range(5):
            task_id = controller.add_task(
                id=f"group_task_{i}",
                name=f"Group Task {i}",
                function=async_task_with_signal_check,
                kwargs={"result": f"group_result_{i}", "delay": 0.3},
                concurrency_group="test_group"
            )
            task_ids.append(task_id)
        
        # Start execution and measure time
        start_time = time.time()
        await controller.start()
        results = await controller.execute()
        elapsed = time.time() - start_time
        
        # Check results
        assert len(results) == 5
        for i, task_id in enumerate(task_ids):
            assert results[task_id]["result"] == f"group_result_{i}"
        
        # Ensure execution time is consistent with concurrency group limits
        # With 5 tasks and a limit of 2 concurrent tasks in the group,
        # we should need at least 3 time periods to complete all tasks
        assert elapsed >= 0.6, "Execution completed too quickly, concurrency group limits may not be working"
        
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_task_priority():
    """Test task priority affects execution order"""
    controller = ParallelExecutionController(max_concurrent_tasks=1)  # Force sequential execution
    
    try:
        # Add tasks with different priorities
        low_priority_task = controller.add_task(
            id="low_priority",
            name="Low Priority Task",
            function=async_task,
            kwargs={"result": "low_result", "delay": 0.1},
            priority=TaskPriority.LOW
        )
        
        medium_priority_task = controller.add_task(
            id="medium_priority",
            name="Medium Priority Task",
            function=async_task,
            kwargs={"result": "medium_result", "delay": 0.1},
            priority=TaskPriority.MEDIUM
        )
        
        high_priority_task = controller.add_task(
            id="high_priority",
            name="High Priority Task",
            function=async_task,
            kwargs={"result": "high_result", "delay": 0.1},
            priority=TaskPriority.HIGH
        )
        
        # Execute tasks
        await controller.start()
        results = await controller.execute()
        
        # Check that all tasks executed
        assert len(results) == 3
        
        # Check execution order via start timestamps
        high_task = controller.get_task(high_priority_task)
        medium_task = controller.get_task(medium_priority_task)
        low_task = controller.get_task(low_priority_task)
        
        assert high_task.started_at <= medium_task.started_at
        assert medium_task.started_at <= low_task.started_at
        
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_abort_signal():
    """Test that abort signal properly cancels execution"""
    controller = ParallelExecutionController()
    
    try:
        # Add tasks with longer delays
        for i in range(10):
            controller.add_task(
                id=f"abort_task_{i}",
                name=f"Abort Test Task {i}",
                function=async_task_with_signal_check,
                kwargs={"result": f"abort_result_{i}", "delay": 2.0}  # Long delay
            )
        
        # Create abort controller and set to abort after 0.5s
        abort_controller = AbortController()
        asyncio.get_event_loop().call_later(0.5, lambda: abort_controller.abort("Test abort"))
        
        # Start execution
        start_time = time.time()
        await controller.start()
        results = await controller.execute(signal=abort_controller.signal)
        elapsed = time.time() - start_time
        
        # Execution should abort quickly
        assert elapsed < 1.0, "Abort signal was not respected"
        
        # Some tasks might complete, some might abort mid-execution
        # Let's check if at least one task was aborted
        aborted_or_pending = False
        for task in controller.tasks.values():
            if task.state != TaskState.COMPLETED:
                aborted_or_pending = True
                break
        
        assert aborted_or_pending, "All tasks completed despite abort signal"
        
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_task_retries():
    """Test that failed tasks are retried as configured"""
    controller = ParallelExecutionController()
    
    # Track retry attempts
    retry_counter = 0
    
    async def retried_task(signal=None):
        nonlocal retry_counter
        retry_counter += 1
        
        # Fail on first two attempts, succeed on third
        if retry_counter <= 2:
            raise ValueError(f"Intentional failure, attempt {retry_counter}")
        
        return {"status": "success", "attempts": retry_counter}
    
    try:
        # Add task with retries
        task_id = controller.add_task(
            id="retry_task",
            name="Retry Test Task",
            function=retried_task,
            max_retries=3  # Allow up to 3 retry attempts
        )
        
        # Execute
        await controller.start()
        results = await controller.execute()
        
        # Check results
        assert task_id in results
        assert results[task_id]["status"] == "success"
        assert results[task_id]["attempts"] == 3
        assert controller.get_task(task_id).retry_count == 2  # 2 retries after initial attempt
        
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_task_timeout():
    """Test that tasks respect timeout settings"""
    controller = ParallelExecutionController()
    
    async def slow_task(signal=None):
        """Task that takes longer than its timeout"""
        await asyncio.sleep(2.0)  # Long delay
        return "This should never be returned"
    
    try:
        # Add task with short timeout
        task_id = controller.add_task(
            id="timeout_task",
            name="Timeout Test Task",
            function=slow_task,
            timeout=0.5  # Short timeout
        )
        
        # Execute
        await controller.start()
        results = await controller.execute()
        
        # Task should have failed due to timeout
        assert task_id not in results  # Timed out tasks shouldn't have results
        
        task = controller.get_task(task_id)
        assert task.state == TaskState.FAILED
        assert "timed out" in task.error.lower()
        
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_get_all_results_and_errors():
    """Test retrieving all results and errors"""
    controller = ParallelExecutionController()
    
    try:
        # Add a mix of successful and failing tasks
        success_ids = []
        for i in range(3):
            task_id = controller.add_task(
                id=f"success_task_{i}",
                name=f"Success Task {i}",
                function=async_task,
                kwargs={"result": f"success_result_{i}", "delay": 0.1}
            )
            success_ids.append(task_id)
        
        fail_ids = []
        for i in range(2):
            task_id = controller.add_task(
                id=f"fail_task_{i}",
                name=f"Fail Task {i}",
                function=failing_task,
                kwargs={"delay": 0.1},
                max_retries=0  # No retries
            )
            fail_ids.append(task_id)
        
        # Execute
        await controller.start()
        await controller.execute()
        
        # Check all results
        all_results = controller.get_all_results()
        assert len(all_results) == 3
        for i, task_id in enumerate(success_ids):
            assert all_results[task_id] == f"success_result_{i}"
        
        # Check failed tasks
        failed_tasks = controller.get_failed_tasks()
        assert len(failed_tasks) == 2
        for task_id in fail_ids:
            assert task_id in failed_tasks
            assert "intentionally" in failed_tasks[task_id].lower()
        
    finally:
        await controller.stop()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 