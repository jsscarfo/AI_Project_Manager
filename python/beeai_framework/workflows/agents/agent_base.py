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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar, Union
from enum import Enum
from pydantic import BaseModel, Field

from beeai_framework.agents.base import BaseAgent
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.context import Run


class AgentRole(str, Enum):
    """Predefined roles for agents in the workflow system."""
    ORCHESTRATOR = "orchestrator"
    THINKER = "thinker"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    CUSTOM = "custom"


class AgentTask(BaseModel):
    """Task for agents to process."""
    id: str
    name: str
    description: str
    role: AgentRole
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)  # IDs of tasks this task depends on


class AgentResult(BaseModel):
    """Result of an agent's task processing."""
    task_id: str
    agent_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Message for inter-agent communication."""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentInterface(Protocol):
    """Base protocol for all agents in the workflow system."""
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> AgentResult:
        """Process a task and return results."""
        ...
    
    @abstractmethod
    async def receive_message(self, message: AgentMessage) -> None:
        """Handle communication from other agents."""
        ...


class BaseWorkflowAgent(ABC):
    """Base implementation for workflow agents with common functionality."""
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self._memory: Optional[BaseMemory] = None
    
    @property
    def memory(self) -> Optional[BaseMemory]:
        """Get the agent's memory."""
        return self._memory
    
    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        """Set the agent's memory."""
        self._memory = memory
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> AgentResult:
        """Process a task and return results."""
        pass
    
    @abstractmethod
    async def receive_message(self, message: AgentMessage) -> None:
        """Handle communication from other agents."""
        pass
    
    @abstractmethod
    async def clone(self) -> "BaseWorkflowAgent":
        """Create a clone of this agent with the same configuration."""
        pass


T = TypeVar("T", bound=BaseWorkflowAgent)


class AgentFactory:
    """Factory for creating agents of different roles."""
    
    @staticmethod
    async def create_agent(agent_type: Type[T], agent_id: str, role: AgentRole, **kwargs) -> T:
        """Create an agent of the specified type."""
        agent = agent_type(agent_id=agent_id, role=role, **kwargs)
        return agent 