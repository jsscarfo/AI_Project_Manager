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

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from beeai_framework.workflows.agents.agent_base import (
    AgentTask, AgentResult, AgentMessage, AgentRole, BaseWorkflowAgent
)
from beeai_framework.workflows.agents.orchestrator import (
    Orchestrator, AgentStatus, TaskStatus
)
from beeai_framework.workflows.comms.protocol import (
    CommunicationProtocol, MessageType
)


class MockAgent(BaseWorkflowAgent):
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str, role: AgentRole):
        super().__init__(agent_id=agent_id, role=role)
        self.process_task_mock = AsyncMock()
        self.receive_message_mock = AsyncMock()
        self.clone_mock = AsyncMock()
    
    async def process_task(self, task: AgentTask) -> AgentResult:
        """Process a task."""
        return await self.process_task_mock(task)
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Handle communication from other agents."""
        await self.receive_message_mock(message)
    
    async def clone(self) -> "MockAgent":
        """Create a clone of this agent."""
        return await self.clone_mock()


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MockAgent(agent_id="test_agent", role=AgentRole.THINKER)
    
    # Configure mock return value
    agent.process_task_mock.return_value = AgentResult(
        task_id="test_task",
        agent_id="test_agent",
        success=True,
        result="Test result",
    )
    
    return agent


@pytest.fixture
def orchestrator():
    """Create an orchestrator for testing."""
    return Orchestrator(workflow_id="test_workflow")


@pytest.mark.asyncio
async def test_agent_task_processing(mock_agent):
    """Test that an agent can process a task."""
    # Create a task
    task = AgentTask(
        id="test_task",
        name="Test Task",
        description="A test task",
        role=AgentRole.THINKER,
    )
    
    # Process the task
    result = await mock_agent.process_task(task)
    
    # Verify the result
    assert result.task_id == "test_task"
    assert result.agent_id == "test_agent"
    assert result.success is True
    assert result.result == "Test result"
    
    # Verify the mock was called
    mock_agent.process_task_mock.assert_called_once_with(task)


@pytest.mark.asyncio
async def test_agent_message_receiving(mock_agent):
    """Test that an agent can receive a message."""
    # Create a message
    message = AgentMessage(
        sender_id="orchestrator",
        receiver_id="test_agent",
        message_type=MessageType.TASK_ASSIGNMENT,
        content={"task": {"id": "test_task"}},
    )
    
    # Receive the message
    await mock_agent.receive_message(message)
    
    # Verify the mock was called
    mock_agent.receive_message_mock.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_communication_protocol():
    """Test the communication protocol."""
    # Create protocol
    protocol = CommunicationProtocol()
    
    # Create mock agents
    agent1 = MockAgent(agent_id="agent1", role=AgentRole.THINKER)
    agent2 = MockAgent(agent_id="agent2", role=AgentRole.RESEARCHER)
    
    # Register agents
    protocol.register_agent("agent1", agent1)
    protocol.register_agent("agent2", agent2)
    
    # Create a message
    message = AgentMessage(
        sender_id="agent1",
        receiver_id="agent2",
        message_type=MessageType.QUERY,
        content={"query": "test"},
    )
    
    # Create a callback for subscribing
    callback = AsyncMock()
    
    # Subscribe to messages
    await protocol.subscribe(MessageType.QUERY, callback)
    
    # Send the message
    message_id = await protocol.send_message(message)
    
    # Verify the message was delivered
    agent2.receive_message_mock.assert_called_once()
    
    # Verify the subscriber was called
    callback.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_registration(orchestrator, mock_agent):
    """Test agent registration with the orchestrator."""
    # Register agent
    await orchestrator.register_agent(mock_agent, "test_agent", AgentRole.THINKER)
    
    # Verify agent was registered
    assert "test_agent" in orchestrator.state.agents
    assert orchestrator.state.agents["test_agent"].role == AgentRole.THINKER
    assert orchestrator.state.agents["test_agent"].status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_orchestrator_task_creation(orchestrator):
    """Test task creation in the orchestrator."""
    # Add a task
    task_id = await orchestrator.add_task(
        name="Test Task",
        description="A test task",
        role=AgentRole.THINKER,
    )
    
    # Verify task was created
    assert task_id in orchestrator.state.tasks
    assert orchestrator.state.tasks[task_id].task.name == "Test Task"
    assert orchestrator.state.tasks[task_id].status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_orchestrator_task_assignment(orchestrator, mock_agent):
    """Test task assignment in the orchestrator."""
    # Configure mock to return a successful result
    mock_agent.process_task_mock.return_value = AgentResult(
        task_id="test_task",
        agent_id="test_agent",
        success=True,
        result="Task completed successfully",
    )
    
    # Register agent
    await orchestrator.register_agent(mock_agent, "test_agent", AgentRole.THINKER)
    
    # Add a task
    task_id = await orchestrator.add_task(
        name="Test Task",
        description="A test task",
        role=AgentRole.THINKER,
    )
    
    # Process tasks (this will assign and run the task)
    await orchestrator._process_tasks()
    
    # Wait for task processing to complete
    await asyncio.sleep(0.1)
    
    # Verify task was assigned to the agent
    assert orchestrator.state.tasks[task_id].assigned_agent_id == "test_agent"
    
    # Verify agent was called to process the task
    mock_agent.process_task_mock.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_task_dependencies(orchestrator, mock_agent):
    """Test task dependencies in the orchestrator."""
    # Register agent
    await orchestrator.register_agent(mock_agent, "test_agent", AgentRole.THINKER)
    
    # Add first task
    task1_id = await orchestrator.add_task(
        name="Task 1",
        description="First task",
        role=AgentRole.THINKER,
    )
    
    # Add second task with dependency on first
    task2_id = await orchestrator.add_task(
        name="Task 2",
        description="Second task, depends on first",
        role=AgentRole.THINKER,
        dependencies=[task1_id],
    )
    
    # Verify second task is blocked
    assert orchestrator.state.tasks[task2_id].status == TaskStatus.BLOCKED
    
    # Process first task
    await orchestrator._process_tasks()
    
    # Simulate task1 completion
    orchestrator.state.tasks[task1_id].status = TaskStatus.COMPLETED
    orchestrator.state.completed_tasks.append(task1_id)
    if task1_id in orchestrator.state.task_queue:
        orchestrator.state.task_queue.remove(task1_id)
    
    # Update dependencies
    await orchestrator._update_task_dependencies()
    
    # Verify second task is now pending
    assert orchestrator.state.tasks[task2_id].status == TaskStatus.PENDING
    assert orchestrator.state.tasks[task2_id].dependencies_complete is True


@pytest.mark.asyncio
async def test_orchestrator_workflow_completion(orchestrator, mock_agent):
    """Test workflow completion in the orchestrator."""
    # Register agent
    await orchestrator.register_agent(mock_agent, "test_agent", AgentRole.THINKER)
    
    # Add a task
    task_id = await orchestrator.add_task(
        name="Test Task",
        description="A test task",
        role=AgentRole.THINKER,
    )
    
    # Process tasks
    await orchestrator._process_tasks()
    
    # Simulate task completion
    orchestrator.state.tasks[task_id].status = TaskStatus.COMPLETED
    orchestrator.state.completed_tasks.append(task_id)
    orchestrator.state.task_queue = []
    
    # Check workflow completion
    await orchestrator._check_workflow_completion()
    
    # Verify workflow is completed
    assert orchestrator.state.workflow_completed is True
    assert orchestrator.state.workflow_success is True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main(["-xvs", __file__]) 