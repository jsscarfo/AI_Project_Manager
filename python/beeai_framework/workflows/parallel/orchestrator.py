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
import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Type, Callable
from pydantic import BaseModel, Field
from functools import cached_property

from beeai_framework.emitter.emitter import Emitter
from beeai_framework.context import Run, RunContext
from beeai_framework.utils.strings import to_safe_word
from beeai_framework.utils.cancellation import AbortSignal, AbortController
from beeai_framework.workflows.agents.agent_base import (
    AgentTask, AgentResult, AgentMessage, AgentInterface, 
    AgentRole, BaseWorkflowAgent
)
from beeai_framework.workflows.comms.protocol import CommunicationProtocol, MessageType
from beeai_framework.workflows.agents.orchestrator import (
    Orchestrator, AgentStatus, TaskStatus, 
    AgentRegistration, TaskEntry, OrchestratorState
)
from beeai_framework.workflows.parallel.controller import (
    ParallelExecutionController, ResourceConstraint, 
    TaskPriority, TaskState, ParallelTask
)

logger = logging.getLogger(__name__)


class ParallelExecutionConfig(BaseModel):
    """Configuration for parallel execution in the orchestrator"""
    enabled: bool = True
    max_concurrent_tasks: int = 10
    resource_limits: Dict[str, float] = Field(default_factory=dict)
    concurrency_groups: Dict[str, int] = Field(default_factory=dict)
    default_task_timeout: Optional[float] = None
    default_max_retries: int = 3


class ParallelTaskMapping(BaseModel):
    """Mapping between agent tasks and parallel tasks"""
    agent_task_id: str
    parallel_task_id: str
    role: AgentRole


class ParallelOrchestrator(Orchestrator):
    """
    Enhanced agent workflow orchestrator that supports parallel execution.
    
    This orchestrator extends the base Orchestrator with:
    1. Parallel task execution capabilities
    2. Advanced resource management
    3. Concurrency control for different task groups
    4. Enhanced monitoring and visualization
    """
    
    def __init__(
        self, 
        workflow_id: str = None,
        parallel_config: Optional[ParallelExecutionConfig] = None
    ):
        super().__init__(workflow_id=workflow_id)
        
        # Set up parallel execution configuration
        self.parallel_config = parallel_config or ParallelExecutionConfig()
        
        # Create parallel execution controller if enabled
        if self.parallel_config.enabled:
            self.parallel_controller = ParallelExecutionController(
                max_concurrent_tasks=self.parallel_config.max_concurrent_tasks,
                controller_id=f"parallel_{self.workflow_id}"
            )
            
            # Register resources
            for resource_name, capacity in self.parallel_config.resource_limits.items():
                self.parallel_controller.register_resource(resource_name, capacity)
            
            # Register concurrency groups
            for group_name, limit in self.parallel_config.concurrency_groups.items():
                self.parallel_controller.register_concurrency_group(group_name, limit)
            
            # Register for controller events
            self._register_controller_event_handlers()
        else:
            self.parallel_controller = None
        
        # Task mappings between agent tasks and parallel tasks
        self.task_mappings: Dict[str, ParallelTaskMapping] = {}
        
    def _create_emitter(self) -> Emitter:
        """Create an emitter for this orchestrator with additional events."""
        emitter = super()._create_emitter()
        
        # Add parallel-specific events to the emitter
        return Emitter.root().child(
            namespace=["workflow", "orchestrator", "parallel", to_safe_word(self.workflow_id)],
            creator=self,
            events={
                **emitter.events,
                "parallel_task_created": ParallelTaskMapping,
                "parallel_task_started": ParallelTaskMapping,
                "parallel_task_completed": ParallelTaskMapping,
                "parallel_task_failed": ParallelTaskMapping,
                "parallel_execution_started": OrchestratorState,
                "parallel_execution_completed": OrchestratorState,
            },
        )
    
    def _register_controller_event_handlers(self) -> None:
        """Register event handlers for parallel controller events"""
        if not self.parallel_controller:
            return
        
        # Register for task events
        self.parallel_controller.on_task_event("task_completed", self._handle_parallel_task_completed)
        self.parallel_controller.on_task_event("task_failed", self._handle_parallel_task_failed)
    
    async def _handle_parallel_task_completed(self, event: Any) -> None:
        """Handle completion of a parallel task"""
        parallel_task_id = event.task_id
        
        # Find the corresponding agent task
        agent_task_id = None
        for mapping in self.task_mappings.values():
            if mapping.parallel_task_id == parallel_task_id:
                agent_task_id = mapping.agent_task_id
                break
        
        if not agent_task_id or agent_task_id not in self.state.tasks:
            logger.warning(f"Received completion for unknown parallel task: {parallel_task_id}")
            return
        
        # Get the task result
        result = self.parallel_controller.get_task_result(parallel_task_id)
        
        # Update the agent task
        task_entry = self.state.tasks[agent_task_id]
        task_entry.status = TaskStatus.COMPLETED
        task_entry.completed_at = asyncio.get_event_loop().time()
        task_entry.result = AgentResult(
            task_id=agent_task_id,
            agent_id=task_entry.assigned_agent_id or "parallel_controller",
            success=True,
            result=result,
            metadata={"parallel_execution": True}
        )
        
        # Move to completed tasks
        if agent_task_id not in self.state.completed_tasks:
            self.state.completed_tasks.append(agent_task_id)
        
        # Update any assigned agent status
        if task_entry.assigned_agent_id and task_entry.assigned_agent_id in self.state.agents:
            agent = self.state.agents[task_entry.assigned_agent_id]
            agent.status = AgentStatus.IDLE
            agent.current_task_id = None
        
        # Emit event
        await self.emitter.emit(
            "parallel_task_completed",
            ParallelTaskMapping(
                agent_task_id=agent_task_id,
                parallel_task_id=parallel_task_id,
                role=task_entry.task.role
            )
        )
        
        # Update dependencies
        await self._update_task_dependencies()
        
        # Check workflow completion
        await self._check_workflow_completion()
    
    async def _handle_parallel_task_failed(self, event: Any) -> None:
        """Handle failure of a parallel task"""
        parallel_task_id = event.task_id
        
        # Find the corresponding agent task
        agent_task_id = None
        for mapping in self.task_mappings.values():
            if mapping.parallel_task_id == parallel_task_id:
                agent_task_id = mapping.agent_task_id
                break
        
        if not agent_task_id or agent_task_id not in self.state.tasks:
            logger.warning(f"Received failure for unknown parallel task: {parallel_task_id}")
            return
        
        # Get the error message
        error = self.parallel_controller.get_task_error(parallel_task_id)
        
        # Update the agent task
        task_entry = self.state.tasks[agent_task_id]
        task_entry.status = TaskStatus.FAILED
        task_entry.completed_at = asyncio.get_event_loop().time()
        task_entry.result = AgentResult(
            task_id=agent_task_id,
            agent_id=task_entry.assigned_agent_id or "parallel_controller",
            success=False,
            error=error,
            metadata={"parallel_execution": True}
        )
        
        # Move to failed tasks
        if agent_task_id not in self.state.failed_tasks:
            self.state.failed_tasks.append(agent_task_id)
        
        # Update any assigned agent status
        if task_entry.assigned_agent_id and task_entry.assigned_agent_id in self.state.agents:
            agent = self.state.agents[task_entry.assigned_agent_id]
            agent.status = AgentStatus.IDLE
            agent.current_task_id = None
        
        # Emit event
        await self.emitter.emit(
            "parallel_task_failed",
            ParallelTaskMapping(
                agent_task_id=agent_task_id,
                parallel_task_id=parallel_task_id,
                role=task_entry.task.role
            )
        )
        
        # Check workflow completion
        await self._check_workflow_completion()
    
    async def _create_parallel_task(self, task_entry: TaskEntry) -> Optional[str]:
        """Create a parallel task from an agent task"""
        if not self.parallel_controller:
            return None
        
        agent_task = task_entry.task
        
        # Define the function that will execute the agent task
        async def execute_agent_task(*args, **kwargs) -> Any:
            signal = kwargs.pop("signal", None)
            
            # Find an available agent
            agent_id = await self._find_available_agent(agent_task.role)
            
            if agent_id:
                # If we found an agent, use it
                task_entry.assigned_agent_id = agent_id
                agent = self.state.agents[agent_id]
                agent.status = AgentStatus.BUSY
                agent.current_task_id = agent_task.id
                
                try:
                    # Process the task with the agent
                    return await agent.agent.process_task(agent_task)
                finally:
                    # Reset agent status
                    agent.status = AgentStatus.IDLE
                    agent.current_task_id = None
            else:
                # No agent available, fail the task
                raise RuntimeError(f"No agent available for role {agent_task.role}")
        
        # Create resource constraints based on the agent role
        resource_constraints = []
        
        # Add LLM API resource constraint for thinker agents
        if agent_task.role == AgentRole.THINKER:
            resource_constraints.append(ResourceConstraint(name="llm_api", amount=1.0))
        
        # Add compute resource constraint
        resource_constraints.append(ResourceConstraint(name="compute", amount=0.5))
        
        # Determine concurrency group based on agent role
        concurrency_group = f"role_{agent_task.role.value}"
        
        # Get dependencies as parallel task IDs
        dependencies = []
        for dep_id in agent_task.dependencies:
            if dep_id in self.task_mappings:
                dependencies.append(self.task_mappings[dep_id].parallel_task_id)
            else:
                logger.warning(f"Dependency {dep_id} not found in task mappings")
        
        # Create the parallel task
        parallel_task_id = self.parallel_controller.add_task(
            id=f"p_{agent_task.id}",
            name=agent_task.name,
            function=execute_agent_task,
            args=(),
            kwargs={},
            dependencies=dependencies,
            priority=(
                TaskPriority.HIGH if agent_task.role == AgentRole.ORCHESTRATOR 
                else TaskPriority.MEDIUM
            ),
            resource_constraints=resource_constraints,
            concurrency_group=concurrency_group,
            max_retries=self.parallel_config.default_max_retries,
            timeout=self.parallel_config.default_task_timeout,
            metadata=agent_task.metadata
        )
        
        # Store the mapping
        mapping = ParallelTaskMapping(
            agent_task_id=agent_task.id,
            parallel_task_id=parallel_task_id,
            role=agent_task.role
        )
        self.task_mappings[agent_task.id] = mapping
        
        # Emit event
        await self.emitter.emit(
            "parallel_task_created",
            mapping
        )
        
        return parallel_task_id
    
    async def add_task(
        self, 
        name: str, 
        description: str, 
        role: AgentRole, 
        context: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        dependencies: List[str] = None,
        max_retries: int = 3
    ) -> str:
        """Add a task to the orchestration workflow with parallel execution support"""
        # Add task to the orchestrator
        task_id = await super().add_task(
            name=name,
            description=description,
            role=role,
            context=context,
            metadata=metadata,
            dependencies=dependencies,
            max_retries=max_retries
        )
        
        # If parallel execution is enabled and task dependencies are met
        if self.parallel_controller and task_id in self.state.tasks:
            task_entry = self.state.tasks[task_id]
            
            # If task is ready (all dependencies complete), create a parallel task
            if task_entry.dependencies_complete:
                await self._create_parallel_task(task_entry)
        
        return task_id
    
    async def _update_task_dependencies(self) -> None:
        """Update task dependencies and create parallel tasks for ready tasks"""
        # Update dependencies using the base method
        await super()._update_task_dependencies()
        
        # If parallel execution is enabled, create parallel tasks for newly ready tasks
        if self.parallel_controller:
            for task_id in self.state.task_queue:
                if task_id not in self.task_mappings and task_id in self.state.tasks:
                    task_entry = self.state.tasks[task_id]
                    if task_entry.dependencies_complete and task_entry.status == TaskStatus.PENDING:
                        await self._create_parallel_task(task_entry)
    
    async def _process_tasks(self) -> None:
        """Process tasks in the queue"""
        if self.parallel_controller and self.parallel_config.enabled:
            # With parallel execution, we don't need to do sequential processing
            # The parallel controller handles task execution
            return
        else:
            # Fall back to base implementation for sequential processing
            await super()._process_tasks()
    
    async def run(self, signal: AbortSignal = None) -> Run[OrchestratorState]:
        """Run the workflow orchestration with parallel execution support"""
        async def handler(context: RunContext) -> OrchestratorState:
            try:
                # Start parallel controller if enabled
                if self.parallel_controller and self.parallel_config.enabled:
                    await self.parallel_controller.start()
                    
                    await self.emitter.emit(
                        "parallel_execution_started",
                        self.state.model_copy()
                    )
                
                # Initialize task dependencies
                await self._update_task_dependencies()
                
                # Process tasks until completion or signal abort
                while not self.state.workflow_completed:
                    if signal and signal.aborted:
                        break
                    
                    # If parallel execution is enabled
                    if self.parallel_controller and self.parallel_config.enabled:
                        # Check workflow completion
                        await self._check_workflow_completion()
                    else:
                        # Process tasks sequentially
                        await self._process_tasks()
                    
                    # Wait a bit before processing again
                    await asyncio.sleep(0.1)
                
                # Final check of workflow completion
                await self._check_workflow_completion()
                
                if self.parallel_controller and self.parallel_config.enabled:
                    await self.emitter.emit(
                        "parallel_execution_completed",
                        self.state.model_copy()
                    )
                
                return self.state
                
            finally:
                # Stop parallel controller if enabled
                if self.parallel_controller and self.parallel_config.enabled:
                    await self.parallel_controller.stop()
        
        return RunContext.enter(
            self,
            handler,
            signal=signal,
            run_params={"workflow_id": self.workflow_id},
        )
    
    async def execute_all_parallel(self, signal: AbortSignal = None) -> Dict[str, Any]:
        """
        Execute all tasks in parallel mode and return results
        
        This method is an alternative to run() that returns just the task results
        """
        if not self.parallel_controller or not self.parallel_config.enabled:
            raise RuntimeError("Parallel execution is not enabled")
        
        try:
            # Start the controller
            await self.parallel_controller.start()
            
            # Execute all parallel tasks
            parallel_results = await self.parallel_controller.execute(signal=signal)
            
            # Map parallel results to agent task results
            agent_results: Dict[str, Any] = {}
            for agent_task_id, mapping in self.task_mappings.items():
                if mapping.parallel_task_id in parallel_results:
                    agent_results[agent_task_id] = parallel_results[mapping.parallel_task_id]
            
            return agent_results
            
        finally:
            # Stop the controller
            await self.parallel_controller.stop()
    
    def set_parallel_execution(self, enabled: bool) -> None:
        """Enable or disable parallel execution"""
        self.parallel_config.enabled = enabled 