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
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, TypeVar, Generic
from pydantic import BaseModel, Field
from functools import cached_property

from beeai_framework.emitter.emitter import Emitter
from beeai_framework.context import Run, RunContext
from beeai_framework.utils.strings import to_safe_word
from beeai_framework.utils.asynchronous import ensure_async
from beeai_framework.utils.cancellation import AbortSignal, AbortController
from beeai_framework.errors import FrameworkError

T = TypeVar('T')
R = TypeVar('R')

logger = logging.getLogger(__name__)


class TaskPriority(int, Enum):
    """Priority levels for parallel tasks"""
    HIGH = 0
    MEDIUM = 1
    LOW = 2


class TaskState(str, Enum):
    """States a task can be in during its lifecycle"""
    PENDING = "pending"  # Task is created but not ready to run (dependencies not met)
    READY = "ready"      # Task is ready to run (all dependencies met)
    RUNNING = "running"  # Task is currently running
    COMPLETED = "completed"  # Task completed successfully
    FAILED = "failed"    # Task failed
    CANCELLED = "cancelled"  # Task was cancelled


class ResourceConstraint(BaseModel):
    """Resource constraint for a task"""
    name: str
    amount: float = 1.0


class ParallelTask(BaseModel, Generic[T, R]):
    """Task that can be executed in parallel"""
    id: str
    name: str
    function: Callable[..., R]
    args: Tuple = Field(default_factory=tuple)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    state: TaskState = TaskState.PENDING
    result: Optional[R] = None
    error: Optional[str] = None
    resource_constraints: List[ResourceConstraint] = Field(default_factory=list)
    concurrency_group: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParallelTaskEvent(BaseModel, Generic[T, R]):
    """Event emitted by the parallel execution controller"""
    task_id: str
    event_type: str
    task: ParallelTask[T, R]
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResourceManager(BaseModel):
    """Manages resources for parallel execution"""
    resources: Dict[str, float] = Field(default_factory=dict)
    allocations: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    def register_resource(self, name: str, capacity: float) -> None:
        """Register a resource with its capacity"""
        self.resources[name] = capacity
        self.allocations[name] = {}

    def can_allocate(self, task_id: str, constraints: List[ResourceConstraint]) -> bool:
        """Check if resources can be allocated for a task"""
        for constraint in constraints:
            if constraint.name not in self.resources:
                return False
            
            # Calculate current usage
            current_usage = sum(self.allocations.get(constraint.name, {}).values())
            
            # Check if there's enough capacity available
            if current_usage + constraint.amount > self.resources[constraint.name]:
                return False
        
        return True

    def allocate(self, task_id: str, constraints: List[ResourceConstraint]) -> bool:
        """Allocate resources for a task"""
        if not self.can_allocate(task_id, constraints):
            return False
        
        for constraint in constraints:
            if constraint.name not in self.allocations:
                self.allocations[constraint.name] = {}
            
            self.allocations[constraint.name][task_id] = constraint.amount
        
        return True

    def release(self, task_id: str) -> None:
        """Release resources allocated to a task"""
        for resource_name, allocations in self.allocations.items():
            if task_id in allocations:
                del allocations[task_id]


class ConcurrencyController(BaseModel):
    """Controls concurrency limits for different groups of tasks"""
    limits: Dict[str, int] = Field(default_factory=dict)
    current: Dict[str, int] = Field(default_factory=dict)
    
    def register_group(self, group_name: str, limit: int) -> None:
        """Register a concurrency group with its limit"""
        self.limits[group_name] = limit
        if group_name not in self.current:
            self.current[group_name] = 0
    
    def can_execute(self, group_name: Optional[str]) -> bool:
        """Check if a task from the given group can be executed"""
        if group_name is None:
            return True
        
        if group_name not in self.limits:
            return True
        
        return self.current.get(group_name, 0) < self.limits[group_name]
    
    def increment(self, group_name: Optional[str]) -> None:
        """Increment the counter for a concurrency group"""
        if group_name is None:
            return
        
        if group_name not in self.current:
            self.current[group_name] = 0
        
        self.current[group_name] += 1
    
    def decrement(self, group_name: Optional[str]) -> None:
        """Decrement the counter for a concurrency group"""
        if group_name is None or group_name not in self.current:
            return
        
        self.current[group_name] = max(0, self.current[group_name] - 1)


class ParallelExecutionController(Generic[T, R]):
    """
    Controller for parallel task execution with dependency resolution.
    
    Features:
    - Executes independent tasks concurrently
    - Manages task dependencies
    - Handles resource constraints
    - Supports concurrency limits for groups of tasks
    - Task prioritization
    - Timeouts and retries
    """
    
    def __init__(
        self, 
        max_concurrent_tasks: int = 10,
        controller_id: Optional[str] = None
    ):
        self.controller_id = controller_id or f"parallel_controller_{id(self)}"
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, ParallelTask[T, R]] = {}
        self.ready_tasks: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Set[str] = set()
        self.resource_manager = ResourceManager()
        self.concurrency_controller = ConcurrencyController()
        self._is_running = False
        self._workers_running = False
        self._task_event_callbacks: Dict[str, List[Callable]] = {}

    @cached_property
    def emitter(self) -> Emitter:
        """Get the emitter for this controller."""
        return self._create_emitter()
    
    def _create_emitter(self) -> Emitter:
        """Create an emitter for this controller."""
        return Emitter.root().child(
            namespace=["workflow", "parallel", to_safe_word(self.controller_id)],
            creator=self,
            events={
                "task_scheduled": ParallelTaskEvent,
                "task_ready": ParallelTaskEvent,
                "task_started": ParallelTaskEvent,
                "task_completed": ParallelTaskEvent,
                "task_failed": ParallelTaskEvent,
                "task_cancelled": ParallelTaskEvent,
                "controller_started": ParallelTaskEvent,
                "controller_stopped": ParallelTaskEvent,
                "execution_completed": ParallelTaskEvent,
            },
        )

    def register_resource(self, name: str, capacity: float) -> None:
        """Register a resource with its capacity"""
        self.resource_manager.register_resource(name, capacity)
    
    def register_concurrency_group(self, group_name: str, limit: int) -> None:
        """Register a concurrency group with its limit"""
        self.concurrency_controller.register_group(group_name, limit)
    
    def add_task(
        self,
        id: str,
        function: Callable[..., R],
        name: Optional[str] = None,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        dependencies: List[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        resource_constraints: List[ResourceConstraint] = None,
        concurrency_group: Optional[str] = None,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a task to the execution queue"""
        if id in self.tasks:
            raise ValueError(f"Task with ID {id} already exists")
        
        task = ParallelTask[T, R](
            id=id,
            name=name or id,
            function=function,
            args=args,
            kwargs=kwargs or {},
            dependencies=dependencies or [],
            priority=priority,
            resource_constraints=resource_constraints or [],
            concurrency_group=concurrency_group,
            max_retries=max_retries,
            timeout=timeout,
            metadata=metadata or {}
        )
        
        self.tasks[id] = task
        
        # Emit task scheduled event
        asyncio.create_task(self.emitter.emit(
            "task_scheduled", 
            ParallelTaskEvent[T, R](task_id=id, event_type="task_scheduled", task=task)
        ))
        
        # Check if the task is ready to run
        if not task.dependencies:
            self._mark_task_ready(task)
        
        return id
    
    def on_task_event(self, event_type: str, callback: Callable) -> None:
        """Register a callback for task events"""
        if event_type not in self._task_event_callbacks:
            self._task_event_callbacks[event_type] = []
        
        self._task_event_callbacks[event_type].append(callback)
    
    async def _emit_task_event(self, event_type: str, task: ParallelTask[T, R]) -> None:
        """Emit a task event and call registered callbacks"""
        event = ParallelTaskEvent[T, R](task_id=task.id, event_type=event_type, task=task)
        
        # Emit event through the emitter
        await self.emitter.emit(event_type, event)
        
        # Call registered callbacks
        for callback in self._task_event_callbacks.get(event_type, []):
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in task event callback: {e}")
    
    def _mark_task_ready(self, task: ParallelTask[T, R]) -> None:
        """Mark a task as ready to run"""
        if task.state != TaskState.PENDING:
            return
        
        task.state = TaskState.READY
        
        # Add to the ready queue with priority
        self.ready_tasks.put_nowait((task.priority.value, task.id))
        
        # Emit task ready event
        asyncio.create_task(self._emit_task_event("task_ready", task))
    
    async def _update_dependencies(self) -> None:
        """Update task dependencies and mark ready tasks"""
        for task_id, task in self.tasks.items():
            if task.state != TaskState.PENDING:
                continue
            
            # Check if all dependencies are complete
            dependencies_complete = True
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    logger.warning(f"Task {task_id} depends on non-existent task {dep_id}")
                    continue
                
                dep_task = self.tasks[dep_id]
                if dep_task.state != TaskState.COMPLETED:
                    dependencies_complete = False
                    break
            
            if dependencies_complete:
                self._mark_task_ready(task)
    
    async def _execute_task(self, task: ParallelTask[T, R]) -> None:
        """Execute a single task with timeout and retry handling"""
        if task.state != TaskState.READY:
            return
        
        # Mark task as running
        task.state = TaskState.RUNNING
        task.started_at = time.time()
        self.running_tasks.add(task.id)
        
        # Allocate resources
        self.resource_manager.allocate(task.id, task.resource_constraints)
        
        # Increment concurrency group counter
        self.concurrency_controller.increment(task.concurrency_group)
        
        # Emit task started event
        await self._emit_task_event("task_started", task)
        
        try:
            # Create abort controller for timeout
            controller = AbortController()
            signal = controller.signal
            
            # Set timeout if specified
            if task.timeout:
                asyncio.get_event_loop().call_later(
                    task.timeout,
                    lambda: controller.abort(f"Task {task.id} timed out after {task.timeout} seconds")
                )
            
            # Execute the task function
            async_fn = ensure_async(task.function)
            result = await async_fn(*task.args, **task.kwargs, signal=signal)
            
            # Store result and mark as completed
            task.result = result
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()
            
            # Emit task completed event
            await self._emit_task_event("task_completed", task)
            
        except Exception as e:
            logger.exception(f"Error executing task {task.id}: {e}")
            task.error = str(e)
            
            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.state = TaskState.READY
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
                
                # Add back to the queue with the same priority
                await self.ready_tasks.put((task.priority.value, task.id))
            else:
                task.state = TaskState.FAILED
                await self._emit_task_event("task_failed", task)
        
        finally:
            # Release resources
            self.resource_manager.release(task.id)
            
            # Decrement concurrency group counter
            self.concurrency_controller.decrement(task.concurrency_group)
            
            # Remove from running tasks
            self.running_tasks.discard(task.id)
            
            # Update dependencies for other tasks
            await self._update_dependencies()
    
    async def _worker(self) -> None:
        """Worker coroutine that processes tasks from the queue"""
        while self._workers_running:
            try:
                # Get the next ready task
                priority, task_id = await asyncio.wait_for(self.ready_tasks.get(), timeout=0.1)
                
                # Check if task still exists and is ready
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                if task.state != TaskState.READY:
                    continue
                
                # Check resource constraints
                if not self.resource_manager.can_allocate(task.id, task.resource_constraints):
                    # Put back in the queue
                    await self.ready_tasks.put((priority, task_id))
                    await asyncio.sleep(0.1)
                    continue
                
                # Check concurrency group limits
                if not self.concurrency_controller.can_execute(task.concurrency_group):
                    # Put back in the queue
                    await self.ready_tasks.put((priority, task_id))
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute the task
                asyncio.create_task(self._execute_task(task))
                
            except asyncio.TimeoutError:
                # No tasks available, continue polling
                pass
            except Exception as e:
                logger.exception(f"Error in worker: {e}")
                await asyncio.sleep(1)  # Avoid tight loop on errors
    
    async def start(self) -> None:
        """Start the parallel execution controller"""
        if self._is_running:
            return
        
        self._is_running = True
        self._workers_running = True
        
        # Emit controller started event
        dummy_task = ParallelTask[T, R](id="controller", name="controller")
        await self.emitter.emit(
            "controller_started", 
            ParallelTaskEvent[T, R](task_id="controller", event_type="controller_started", task=dummy_task)
        )
        
        # Initialize workers
        self.workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.max_concurrent_tasks)
        ]
    
    async def stop(self) -> None:
        """Stop the parallel execution controller"""
        if not self._is_running:
            return
        
        self._workers_running = False
        
        # Wait for all workers to finish
        if hasattr(self, 'workers'):
            for worker in self.workers:
                worker.cancel()
            
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self._is_running = False
        
        # Emit controller stopped event
        dummy_task = ParallelTask[T, R](id="controller", name="controller")
        await self.emitter.emit(
            "controller_stopped", 
            ParallelTaskEvent[T, R](task_id="controller", event_type="controller_stopped", task=dummy_task)
        )
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it's not already running or completed"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.state in (TaskState.COMPLETED, TaskState.FAILED):
            return False
        
        if task.state == TaskState.RUNNING:
            # Can't cancel running tasks directly
            # Could implement forced cancellation with signals in the future
            return False
        
        # Mark as cancelled
        task.state = TaskState.CANCELLED
        
        # Emit task cancelled event
        await self._emit_task_event("task_cancelled", task)
        
        return True
    
    async def execute(self, signal: Optional[AbortSignal] = None) -> Dict[str, Any]:
        """
        Execute all tasks respecting dependencies and return results
        
        Returns a dictionary mapping task IDs to their results
        """
        results: Dict[str, Any] = {}
        
        try:
            # Start the controller
            await self.start()
            
            # Process tasks until all are complete or signal is aborted
            while True:
                # Check if aborted
                if signal and signal.aborted:
                    break
                
                # Check if all tasks are complete
                pending_tasks = 0
                running_tasks = 0
                failed_tasks = 0
                
                for task in self.tasks.values():
                    if task.state in (TaskState.PENDING, TaskState.READY):
                        pending_tasks += 1
                    elif task.state == TaskState.RUNNING:
                        running_tasks += 1
                    elif task.state == TaskState.FAILED:
                        failed_tasks += 1
                
                if pending_tasks == 0 and running_tasks == 0:
                    # All tasks are completed, failed, or cancelled
                    break
                
                # Wait a bit before checking again
                await asyncio.sleep(0.1)
            
            # Collect results from completed tasks
            for task_id, task in self.tasks.items():
                if task.state == TaskState.COMPLETED:
                    results[task_id] = task.result
            
            # Emit execution completed event
            dummy_task = ParallelTask[T, R](id="controller", name="controller")
            await self.emitter.emit(
                "execution_completed", 
                ParallelTaskEvent[T, R](task_id="controller", event_type="execution_completed", task=dummy_task)
            )
            
            return results
            
        finally:
            # Stop the controller
            await self.stop()
    
    def run(self, signal: Optional[AbortSignal] = None) -> Run[Dict[str, Any]]:
        """Run the parallel execution controller as a RunContext"""
        
        async def handler(context: RunContext) -> Dict[str, Any]:
            try:
                return await self.execute(signal=signal)
            except Exception as e:
                raise FrameworkError.ensure(e)
        
        return RunContext.enter(
            self,
            handler,
            signal=signal,
            run_params={"controller_id": self.controller_id},
        )
    
    def get_task(self, task_id: str) -> Optional[ParallelTask[T, R]]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[R]:
        """Get the result of a completed task"""
        task = self.get_task(task_id)
        if task and task.state == TaskState.COMPLETED:
            return task.result
        return None
    
    def get_all_results(self) -> Dict[str, R]:
        """Get results for all completed tasks"""
        return {
            task_id: task.result
            for task_id, task in self.tasks.items()
            if task.state == TaskState.COMPLETED
        }
    
    def get_task_error(self, task_id: str) -> Optional[str]:
        """Get the error message of a failed task"""
        task = self.get_task(task_id)
        if task and task.state == TaskState.FAILED:
            return task.error
        return None
    
    def get_failed_tasks(self) -> Dict[str, str]:
        """Get all failed tasks with their error messages"""
        return {
            task_id: task.error or "Unknown error"
            for task_id, task in self.tasks.items()
            if task.state == TaskState.FAILED
        } 