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
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from beeai_framework.agents.base import BaseAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.middleware.base import Middleware, MiddlewareChain, MiddlewareContext
from beeai_framework.workflows.agents.agent_base import (
    AgentTask, AgentResult, AgentMessage, AgentRole, BaseWorkflowAgent
)


class SequentialThinkingMiddleware(Middleware):
    """Middleware that implements sequential thinking for agent reasoning."""
    
    async def process(self, context: MiddlewareContext) -> MiddlewareContext:
        """Process the request through sequential thinking steps."""
        # Extract the prompt from the context
        prompt = context.request.get("prompt", "")
        if not prompt:
            return context
        
        # Add sequential thinking instructions
        sequential_instructions = [
            "1. First, break down the problem into smaller steps",
            "2. Consider relevant facts and information",
            "3. Analyze different approaches to the problem",
            "4. Evaluate pros and cons of each approach",
            "5. Decide on the best approach and explain your reasoning",
            "6. Present a clear conclusion"
        ]
        
        enhanced_prompt = f"{prompt}\n\nUse the following sequential thinking process:\n" + "\n".join(sequential_instructions)
        
        # Update the context with the enhanced prompt
        context.request["prompt"] = enhanced_prompt
        
        return context


class ThinkerAgentConfig(BaseModel):
    """Configuration for the thinker agent."""
    llm: ChatModel
    system_prompt: str = "You are a thoughtful AI assistant that breaks down complex problems through sequential thinking."
    middleware: List[Middleware] = Field(default_factory=list)


class ThinkerAgent(BaseWorkflowAgent):
    """
    Agent specializing in sequential thinking and complex reasoning.
    
    This agent applies structured thinking processes to solve complex problems,
    breaking them down into manageable steps and analyzing from different angles.
    """
    
    def __init__(self, agent_id: str, role: AgentRole = AgentRole.THINKER, config: ThinkerAgentConfig = None):
        """Initialize the thinker agent."""
        super().__init__(agent_id=agent_id, role=role)
        
        if config is None:
            raise ValueError("ThinkerAgent requires a configuration with an LLM")
        
        self.config = config
        self._memory = UnconstrainedMemory()
        
        # Build middleware chain
        self.middleware_chain = MiddlewareChain()
        
        # Always include sequential thinking middleware
        has_sequential_thinking = any(
            isinstance(middleware, SequentialThinkingMiddleware) 
            for middleware in self.config.middleware
        )
        
        if not has_sequential_thinking:
            self.middleware_chain.add_middleware(SequentialThinkingMiddleware())
        
        # Add other middleware
        for middleware in self.config.middleware:
            self.middleware_chain.add_middleware(middleware)
    
    async def process_task(self, task: AgentTask) -> AgentResult:
        """
        Process a task using sequential thinking.
        
        Args:
            task: The task to process.
        
        Returns:
            AgentResult: The result of the task processing.
        """
        try:
            # Clear memory for new task
            self._memory = UnconstrainedMemory()
            
            # Add system message
            await self._memory.add(SystemMessage(self.config.system_prompt))
            
            # Create task prompt
            task_prompt = f"""
            Task: {task.name}
            
            Description: {task.description}
            
            Context: {task.context.get('context', '')}
            
            Please analyze this problem thoroughly.
            """
            
            # Process through middleware
            middleware_context = MiddlewareContext(
                request={"prompt": task_prompt, "task": task},
                response=None,
            )
            
            processed_context = await self.middleware_chain.process_request_context(middleware_context)
            
            # Get processed prompt
            processed_prompt = processed_context.request.get("prompt", task_prompt)
            
            # Add user message
            await self._memory.add(UserMessage(processed_prompt))
            
            # Generate response using LLM
            response = await self.config.llm.generate(self._memory.messages)
            
            # Add assistant message
            if isinstance(response, AssistantMessage):
                await self._memory.add(response)
            else:
                await self._memory.add(AssistantMessage(str(response)))
            
            # Extract relevant content from the response
            result_content = self._memory.messages[-1].text if self._memory.messages else ""
            
            # Return the result
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=True,
                result=result_content,
                metadata={
                    "role": self.role,
                    "task_name": task.name,
                }
            )
        except Exception as e:
            # Return error result
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=False,
                error=str(e),
                metadata={
                    "role": self.role,
                    "task_name": task.name,
                }
            )
    
    async def receive_message(self, message: AgentMessage) -> None:
        """
        Handle communication from other agents.
        
        Args:
            message: The message to process.
        """
        # For this specialized agent, we just log the message reception
        # In a real implementation, we might use this for agent collaboration
        print(f"ThinkerAgent {self.agent_id} received message: {message.message_type} from {message.sender_id}")
    
    async def clone(self) -> "ThinkerAgent":
        """Create a clone of this agent with the same configuration."""
        return ThinkerAgent(
            agent_id=f"{self.agent_id}_clone",
            role=self.role,
            config=self.config.model_copy(),
        ) 