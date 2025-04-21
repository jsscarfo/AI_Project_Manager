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
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from functools import cached_property
import logging

from beeai_framework.emitter.emitter import Emitter
from beeai_framework.context import Run, RunContext
from beeai_framework.utils.strings import to_safe_word
from beeai_framework.workflows.agents.agent_base import (
    AgentTask, AgentResult, AgentMessage, AgentInterface, 
    AgentRole, BaseWorkflowAgent, AgentFactory
)
from beeai_framework.workflows.comms.protocol import CommunicationProtocol, MessageType
from beeai_framework.utils import AbortSignal
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status of an agent in the orchestration workflow."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class TaskStatus(str, Enum):
    """Status of a task in the orchestration workflow."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting for dependencies


class AgentRegistration(BaseModel):
    """Registration entry for an agent in the orchestrator."""
    agent_id: str
    role: AgentRole
    agent: AgentInterface
    status: AgentStatus = AgentStatus.IDLE
    capabilities: List[str] = Field(default_factory=list)
    current_task_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskEntry(BaseModel):
    """Task entry in the orchestrator's task queue."""
    task: AgentTask
    status: TaskStatus
    assigned_agent_id: Optional[str] = None
    result: Optional[AgentResult] = None
    dependencies_complete: bool = False
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


class OrchestratorState(BaseModel):
    """State of the orchestrator workflow."""
    workflow_id: str
    agents: Dict[str, AgentRegistration] = Field(default_factory=dict)
    tasks: Dict[str, TaskEntry] = Field(default_factory=dict)
    task_queue: List[str] = Field(default_factory=list)  # List of task IDs in order
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    workflow_completed: bool = False
    workflow_success: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorEvent(BaseModel):
    """Event emitted by the orchestrator."""
    workflow_id: str
    event_type: str
    data: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Orchestrator:
    """
    Agent workflow orchestrator that coordinates multiple agents.
    
    The orchestrator is responsible for:
    1. Task assignment to appropriate agents
    2. Task dependency management
    3. Overall workflow state management
    4. Communication between agents
    """
    
    def __init__(self, workflow_id: str = None):
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.state = OrchestratorState(workflow_id=self.workflow_id)
        self.comms = CommunicationProtocol()
        
        # Register for messages
        asyncio.create_task(self._register_message_handlers())
    
    @cached_property
    def emitter(self) -> Emitter:
        """Get the emitter for this orchestrator."""
        return self._create_emitter()
    
    def _create_emitter(self) -> Emitter:
        """Create an emitter for this orchestrator."""
        return Emitter.root().child(
            namespace=["workflow", "orchestrator", to_safe_word(self.workflow_id)],
            creator=self,
            events={
                "agent_registered": OrchestratorEvent,
                "agent_unregistered": OrchestratorEvent,
                "task_created": OrchestratorEvent,
                "task_assigned": OrchestratorEvent,
                "task_started": OrchestratorEvent,
                "task_completed": OrchestratorEvent,
                "task_failed": OrchestratorEvent,
                "workflow_completed": OrchestratorEvent,
                "workflow_failed": OrchestratorEvent,
            },
        )
    
    async def _register_message_handlers(self) -> None:
        """Register message handlers with the communication protocol."""
        # Handle task results
        await self.comms.subscribe(
            MessageType.TASK_RESULT, 
            self._handle_task_result_message
        )
        
        # Handle status updates
        await self.comms.subscribe(
            MessageType.STATUS_UPDATE,
            self._handle_status_update_message
        )
        
        # Handle error messages
        await self.comms.subscribe(
            MessageType.ERROR,
            self._handle_error_message
        )
    
    async def _handle_task_result_message(self, message: AgentMessage) -> None:
        """Handle a task result message from an agent."""
        if not isinstance(message.content, dict) or "task_id" not in message.content:
            logger.warning(f"Received invalid task result message: {message}")
            return
        
        task_id = message.content["task_id"]
        if task_id not in self.state.tasks:
            logger.warning(f"Received result for unknown task: {task_id}")
            return
        
        task_entry = self.state.tasks[task_id]
        if task_entry.assigned_agent_id != message.sender_id:
            logger.warning(
                f"Received result for task {task_id} from {message.sender_id}, "
                f"but task is assigned to {task_entry.assigned_agent_id}"
            )
            return
        
        # Update agent status
        if message.sender_id in self.state.agents:
            agent_reg = self.state.agents[message.sender_id]
            agent_reg.status = AgentStatus.IDLE
            agent_reg.current_task_id = None
        
        # Update task status
        success = message.content.get("success", False)
        if success:
            task_entry.status = TaskStatus.COMPLETED
            task_entry.completed_at = asyncio.get_event_loop().time()
            task_entry.result = AgentResult(
                task_id=task_id,
                agent_id=message.sender_id,
                success=True,
                result=message.content.get("result"),
                metadata=message.content.get("metadata", {})
            )
            
            # Move task to completed list
            if task_id in self.state.task_queue:
                self.state.task_queue.remove(task_id)
            if task_id not in self.state.completed_tasks:
                self.state.completed_tasks.append(task_id)
            
            # Emit task completed event
            await self.emitter.emit(
                "task_completed",
                OrchestratorEvent(
                    workflow_id=self.workflow_id,
                    event_type="task_completed",
                    data=task_entry,
                )
            )
        else:
            task_entry.retry_count += 1
            if task_entry.retry_count >= task_entry.max_retries:
                task_entry.status = TaskStatus.FAILED
                task_entry.completed_at = asyncio.get_event_loop().time()
                task_entry.result = AgentResult(
                    task_id=task_id,
                    agent_id=message.sender_id,
                    success=False,
                    error=message.content.get("error", "Unknown error"),
                    metadata=message.content.get("metadata", {})
                )
                
                # Move task to failed list
                if task_id in self.state.task_queue:
                    self.state.task_queue.remove(task_id)
                if task_id not in self.state.failed_tasks:
                    self.state.failed_tasks.append(task_id)
                
                # Emit task failed event
                await self.emitter.emit(
                    "task_failed",
                    OrchestratorEvent(
                        workflow_id=self.workflow_id,
                        event_type="task_failed",
                        data=task_entry,
                    )
                )
            else:
                # Reset task for retry
                task_entry.status = TaskStatus.PENDING
                task_entry.assigned_agent_id = None
                
                # Put back in queue
                if task_id not in self.state.task_queue:
                    self.state.task_queue.append(task_id)
        
        # Check if workflow is complete
        await self._check_workflow_completion()
    
    async def _handle_status_update_message(self, message: AgentMessage) -> None:
        """Handle a status update message from an agent."""
        if not isinstance(message.content, dict) or "status" not in message.content:
            logger.warning(f"Received invalid status update message: {message}")
            return
        
        agent_id = message.sender_id
        if agent_id not in self.state.agents:
            logger.warning(f"Received status update from unknown agent: {agent_id}")
            return
        
        # Update agent status
        status = message.content["status"]
        if status in [s.value for s in AgentStatus]:
            self.state.agents[agent_id].status = AgentStatus(status)
    
    async def _handle_error_message(self, message: AgentMessage) -> None:
        """Handle an error message from an agent."""
        agent_id = message.sender_id
        if agent_id not in self.state.agents:
            logger.warning(f"Received error from unknown agent: {agent_id}")
            return
        
        # Update agent status
        self.state.agents[agent_id].status = AgentStatus.ERROR
        
        # If agent had an assigned task, mark it as failed
        current_task_id = self.state.agents[agent_id].current_task_id
        if current_task_id and current_task_id in self.state.tasks:
            task_entry = self.state.tasks[current_task_id]
            task_entry.status = TaskStatus.FAILED
            task_entry.completed_at = asyncio.get_event_loop().time()
            task_entry.result = AgentResult(
                task_id=current_task_id,
                agent_id=agent_id,
                success=False,
                error=message.content.get("error", "Agent error"),
                metadata=message.content.get("metadata", {})
            )
            
            # Move task to failed list
            if current_task_id in self.state.task_queue:
                self.state.task_queue.remove(current_task_id)
            if current_task_id not in self.state.failed_tasks:
                self.state.failed_tasks.append(current_task_id)
            
            # Emit task failed event
            await self.emitter.emit(
                "task_failed",
                OrchestratorEvent(
                    workflow_id=self.workflow_id,
                    event_type="task_failed",
                    data=task_entry,
                )
            )
            
            # Clear agent's current task
            self.state.agents[agent_id].current_task_id = None
    
    async def register_agent(self, agent: AgentInterface, agent_id: str, role: AgentRole, capabilities: List[str] = None) -> None:
        """Register an agent with the orchestrator."""
        if not agent_id:
            agent_id = str(uuid.uuid4())
        
        # Create agent registration
        registration = AgentRegistration(
            agent_id=agent_id,
            role=role,
            agent=agent,
            status=AgentStatus.IDLE,
            capabilities=capabilities or [],
        )
        
        # Store in state
        self.state.agents[agent_id] = registration
        
        # Register with communication protocol
        self.comms.register_agent(agent_id, agent)
        
        # Process any pending messages
        await self.comms.process_pending_messages(agent_id)
        
        # Emit agent registered event
        await self.emitter.emit(
            "agent_registered",
            OrchestratorEvent(
                workflow_id=self.workflow_id,
                event_type="agent_registered",
                data=registration,
            )
        )
        
        logger.info(f"Agent {agent_id} registered with role {role}")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator."""
        if agent_id not in self.state.agents:
            logger.warning(f"Agent {agent_id} not registered")
            return
        
        # Get registration
        registration = self.state.agents[agent_id]
        
        # Unregister from communication protocol
        self.comms.unregister_agent(agent_id)
        
        # Remove from state
        del self.state.agents[agent_id]
        
        # Emit agent unregistered event
        await self.emitter.emit(
            "agent_unregistered",
            OrchestratorEvent(
                workflow_id=self.workflow_id,
                event_type="agent_unregistered",
                data=registration,
            )
        )
        
        logger.info(f"Agent {agent_id} unregistered")
    
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
        """
        Add a task to the orchestrator's workflow.
        
        Returns:
            str: The task ID.
        """
        task_id = str(uuid.uuid4())
        
        # Create task
        task = AgentTask(
            id=task_id,
            name=name,
            description=description,
            role=role,
            context=context or {},
            metadata=metadata or {},
            dependencies=dependencies or [],
        )
        
        # Create task entry
        entry = TaskEntry(
            task=task,
            status=TaskStatus.PENDING,
            created_at=asyncio.get_event_loop().time(),
            max_retries=max_retries,
        )
        
        # Check if dependencies are complete
        if not task.dependencies:
            entry.dependencies_complete = True
        else:
            entry.dependencies_complete = all(
                dep_id in self.state.completed_tasks
                for dep_id in task.dependencies
            )
            
            if not entry.dependencies_complete:
                entry.status = TaskStatus.BLOCKED
        
        # Store in state
        self.state.tasks[task_id] = entry
        
        # Add to queue if not blocked
        if entry.status != TaskStatus.BLOCKED:
            self.state.task_queue.append(task_id)
        
        # Emit task created event
        await self.emitter.emit(
            "task_created",
            OrchestratorEvent(
                workflow_id=self.workflow_id,
                event_type="task_created",
                data=entry,
            )
        )
        
        logger.info(f"Task {task_id} added to workflow")
        
        # Trigger task processing
        asyncio.create_task(self._process_tasks())
        
        return task_id
    
    async def _process_tasks(self) -> None:
        """Process pending tasks in the queue."""
        # Process dependencies first
        await self._update_task_dependencies()
        
        # Assign tasks to available agents
        for task_id in list(self.state.task_queue):
            task_entry = self.state.tasks[task_id]
            
            # Skip if task is not pending or dependencies are not complete
            if task_entry.status != TaskStatus.PENDING or not task_entry.dependencies_complete:
                continue
            
            # Find available agent
            agent_id = await self._find_available_agent(task_entry.task.role)
            if not agent_id:
                # No available agent for this role
                continue
            
            # Assign task to agent
            await self._assign_task(task_id, agent_id)
    
    async def _update_task_dependencies(self) -> None:
        """Update the dependency status of tasks."""
        for task_id, task_entry in self.state.tasks.items():
            if task_entry.status == TaskStatus.BLOCKED:
                # Check if dependencies are now complete
                dependencies_complete = all(
                    dep_id in self.state.completed_tasks
                    for dep_id in task_entry.task.dependencies
                )
                
                if dependencies_complete:
                    task_entry.dependencies_complete = True
                    task_entry.status = TaskStatus.PENDING
                    
                    # Add to queue if not already
                    if task_id not in self.state.task_queue:
                        self.state.task_queue.append(task_id)
    
    async def _find_available_agent(self, role: AgentRole) -> Optional[str]:
        """Find an available agent for the given role."""
        for agent_id, registration in self.state.agents.items():
            if (
                registration.role == role
                and registration.status == AgentStatus.IDLE
                and registration.current_task_id is None
            ):
                return agent_id
        return None
    
    async def _assign_task(self, task_id: str, agent_id: str) -> None:
        """Assign a task to an agent."""
        task_entry = self.state.tasks[task_id]
        agent_reg = self.state.agents[agent_id]
        
        # Update task status
        task_entry.status = TaskStatus.IN_PROGRESS
        task_entry.assigned_agent_id = agent_id
        task_entry.started_at = asyncio.get_event_loop().time()
        
        # Update agent status
        agent_reg.status = AgentStatus.BUSY
        agent_reg.current_task_id = task_id
        
        # Emit task assigned event
        await self.emitter.emit(
            "task_assigned",
            OrchestratorEvent(
                workflow_id=self.workflow_id,
                event_type="task_assigned",
                data={
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "task_entry": task_entry,
                },
            )
        )
        
        # Create task assignment message
        message = AgentMessage(
            sender_id="orchestrator",
            receiver_id=agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            content={
                "task": task_entry.task.model_dump(),
            },
        )
        
        # Send message to agent
        await self.comms.send_message(message)
        
        # Start task processing on agent
        asyncio.create_task(self._process_agent_task(agent_id, task_id))
    
    async def _process_agent_task(self, agent_id: str, task_id: str) -> None:
        """Process a task on an agent and handle the result."""
        if agent_id not in self.state.agents or task_id not in self.state.tasks:
            logger.warning(f"Agent {agent_id} or task {task_id} not found")
            return
        
        agent_reg = self.state.agents[agent_id]
        task_entry = self.state.tasks[task_id]
        
        try:
            # Process task
            result = await agent_reg.agent.process_task(task_entry.task)
            
            # Send result message
            message = AgentMessage(
                sender_id=agent_id,
                receiver_id="orchestrator",
                message_type=MessageType.TASK_RESULT,
                content={
                    "task_id": task_id,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "metadata": result.metadata,
                },
            )
            
            await self.comms.send_message(message)
        except Exception as e:
            # Send error message
            message = AgentMessage(
                sender_id=agent_id,
                receiver_id="orchestrator",
                message_type=MessageType.ERROR,
                content={
                    "task_id": task_id,
                    "error": str(e),
                },
            )
            
            await self.comms.send_message(message)
    
    async def _check_workflow_completion(self) -> None:
        """Check if the workflow is complete."""
        # Workflow is complete if queue is empty and no agents are busy
        if not self.state.task_queue and not any(
            reg.status == AgentStatus.BUSY for reg in self.state.agents.values()
        ):
            # Check if any tasks failed
            workflow_success = not self.state.failed_tasks
            
            # Update state
            self.state.workflow_completed = True
            self.state.workflow_success = workflow_success
            
            # Emit completion event
            if workflow_success:
                await self.emitter.emit(
                    "workflow_completed",
                    OrchestratorEvent(
                        workflow_id=self.workflow_id,
                        event_type="workflow_completed",
                        data={
                            "completed_tasks": self.state.completed_tasks,
                            "results": {
                                task_id: self.state.tasks[task_id].result
                                for task_id in self.state.completed_tasks
                            },
                        },
                    )
                )
            else:
                await self.emitter.emit(
                    "workflow_failed",
                    OrchestratorEvent(
                        workflow_id=self.workflow_id,
                        event_type="workflow_failed",
                        data={
                            "completed_tasks": self.state.completed_tasks,
                            "failed_tasks": self.state.failed_tasks,
                            "results": {
                                task_id: self.state.tasks[task_id].result
                                for task_id in self.state.completed_tasks
                            },
                            "failures": {
                                task_id: self.state.tasks[task_id].result
                                for task_id in self.state.failed_tasks
                            },
                        },
                    )
                )
            
            logger.info(
                f"Workflow {self.workflow_id} completed with success={workflow_success}. "
                f"Completed: {len(self.state.completed_tasks)}, Failed: {len(self.state.failed_tasks)}"
            )
    
    async def run(self, signal: AbortSignal = None) -> Run[OrchestratorState]:
        """
        Run the orchestrator workflow.
        
        Returns:
            Run: A run context with the final state.
        """
        
        async def handler(context: RunContext) -> OrchestratorState:
            try:
                # Process tasks until workflow completes
                while not self.state.workflow_completed:
                    # Check for abort signal
                    if signal and signal.aborted:
                        break
                    
                    # Process tasks
                    await self._process_tasks()
                    
                    # Check completion
                    await self._check_workflow_completion()
                    
                    # Wait a bit to avoid tight loop
                    await asyncio.sleep(0.1)
                
                return self.state
            except Exception as e:
                logger.error(f"Error in orchestrator workflow: {e}")
                raise
            finally:
                # Cleanup
                self.emitter.destroy()
        
        return RunContext.enter(
            self,
            handler,
            signal=signal,
            run_params={"workflow_id": self.workflow_id},
        )
    
    def get_task_result(self, task_id: str) -> Optional[AgentResult]:
        """Get the result of a task."""
        if task_id not in self.state.tasks:
            return None
        
        return self.state.tasks[task_id].result 