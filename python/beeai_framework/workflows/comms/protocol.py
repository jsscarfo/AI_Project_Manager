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
from typing import Any, Dict, List, Optional, Protocol, Set, Callable, Awaitable
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from functools import cached_property

from beeai_framework.emitter.emitter import Emitter
from beeai_framework.workflows.agents.agent_base import AgentMessage, AgentInterface


class MessageType(str, Enum):
    """Types of messages that can be exchanged between agents."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    QUERY = "query"
    RESPONSE = "response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    KNOWLEDGE_SHARE = "knowledge_share"
    ORCHESTRATION = "orchestration"


class MessageState(BaseModel):
    """State tracking for messages in the communication system."""
    message_id: str
    sender_id: str
    receiver_id: str
    acknowledged: bool = False
    processed: bool = False
    response_id: Optional[str] = None
    timestamp: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CommunicationEvent(BaseModel):
    """Event emitted on message activities."""
    message: AgentMessage
    state: MessageState


class CommunicationProtocol:
    """
    Protocol implementation for agent communication.
    
    Handles routing messages between agents, tracking message states,
    and providing a pub/sub mechanism for message passing.
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentInterface] = {}
        self._message_states: Dict[str, MessageState] = {}
        self._subscribers: Dict[str, List[Callable[[AgentMessage], Awaitable[None]]]] = {}
        self._pending_messages: Dict[str, List[AgentMessage]] = {}
    
    @cached_property
    def emitter(self) -> Emitter:
        """Get the emitter for this communication protocol."""
        return self._create_emitter()
    
    def _create_emitter(self) -> Emitter:
        """Create an emitter for this communication protocol."""
        return Emitter.root().child(
            namespace=["workflow", "communication"],
            creator=self,
            events={
                "message_sent": CommunicationEvent,
                "message_received": CommunicationEvent,
                "message_processed": CommunicationEvent,
                "message_error": CommunicationEvent,
            },
        )
    
    def register_agent(self, agent_id: str, agent: AgentInterface) -> None:
        """Register an agent with the communication protocol."""
        self._agents[agent_id] = agent
        self._pending_messages[agent_id] = []
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the communication protocol."""
        if agent_id in self._agents:
            del self._agents[agent_id]
        if agent_id in self._pending_messages:
            del self._pending_messages[agent_id]
    
    async def send_message(self, message: AgentMessage) -> str:
        """
        Send a message from one agent to another.
        
        Returns:
            str: The message ID.
        """
        message_id = str(uuid.uuid4())
        
        # Create message state
        state = MessageState(
            message_id=message_id,
            sender_id=message.sender_id,
            receiver_id=message.receiver_id,
            timestamp=asyncio.get_event_loop().time(),
        )
        self._message_states[message_id] = state
        
        # Emit message sent event
        await self.emitter.emit("message_sent", CommunicationEvent(message=message, state=state))
        
        # Queue message if agent is not available or deliver immediately
        if message.receiver_id in self._agents:
            await self._deliver_message(message, state)
        else:
            if message.receiver_id not in self._pending_messages:
                self._pending_messages[message.receiver_id] = []
            self._pending_messages[message.receiver_id].append(message)
        
        # Notify subscribers
        if message.message_type in self._subscribers:
            for subscriber in self._subscribers[message.message_type]:
                try:
                    await subscriber(message)
                except Exception as e:
                    # Log the exception but continue
                    print(f"Error in message subscriber: {e}")
        
        return message_id
    
    async def _deliver_message(self, message: AgentMessage, state: MessageState) -> None:
        """Deliver a message to its recipient."""
        if message.receiver_id not in self._agents:
            # Cannot deliver, agent not registered
            return
        
        try:
            # Mark as acknowledged
            state.acknowledged = True
            
            # Emit message received event
            await self.emitter.emit("message_received", CommunicationEvent(message=message, state=state))
            
            # Deliver to agent
            await self._agents[message.receiver_id].receive_message(message)
            
            # Mark as processed
            state.processed = True
            
            # Emit message processed event
            await self.emitter.emit("message_processed", CommunicationEvent(message=message, state=state))
        except Exception as e:
            # Emit error event
            await self.emitter.emit(
                "message_error", 
                CommunicationEvent(
                    message=message, 
                    state=state.model_copy(update={"metadata": {"error": str(e)}})
                )
            )
    
    async def subscribe(self, message_type: str, callback: Callable[[AgentMessage], Awaitable[None]]) -> None:
        """Subscribe to messages of a specific type."""
        if message_type not in self._subscribers:
            self._subscribers[message_type] = []
        self._subscribers[message_type].append(callback)
    
    async def unsubscribe(self, message_type: str, callback: Callable[[AgentMessage], Awaitable[None]]) -> None:
        """Unsubscribe from messages of a specific type."""
        if message_type in self._subscribers:
            self._subscribers[message_type] = [cb for cb in self._subscribers[message_type] if cb != callback]
    
    async def process_pending_messages(self, agent_id: str) -> int:
        """
        Process any pending messages for an agent.
        
        Returns:
            int: Number of messages processed.
        """
        if agent_id not in self._pending_messages or agent_id not in self._agents:
            return 0
        
        count = 0
        messages = self._pending_messages[agent_id].copy()
        self._pending_messages[agent_id] = []
        
        for message in messages:
            state = self._message_states.get(getattr(message, "id", None) or str(uuid.uuid4()))
            if not state:
                # Create a state for this message
                state = MessageState(
                    message_id=str(uuid.uuid4()),
                    sender_id=message.sender_id,
                    receiver_id=message.receiver_id,
                    timestamp=asyncio.get_event_loop().time(),
                )
                self._message_states[state.message_id] = state
            
            await self._deliver_message(message, state)
            count += 1
        
        return count
    
    def get_message_state(self, message_id: str) -> Optional[MessageState]:
        """Get the state of a specific message."""
        return self._message_states.get(message_id)
    
    def list_pending_agent_ids(self) -> Set[str]:
        """List all agent IDs that have pending messages."""
        return {
            agent_id for agent_id, messages in self._pending_messages.items() 
            if messages and agent_id not in self._agents
        } 