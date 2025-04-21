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
from typing import Dict, Any, List

from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.backend.openai import OpenAIChatModel
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.workflows.agents.agent_base import AgentRole
from beeai_framework.workflows.agents.orchestrator import Orchestrator
from beeai_framework.workflows.agents.specialized.thinker_agent import ThinkerAgent, ThinkerAgentConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_thinker_agent(agent_id: str, api_key: str) -> ThinkerAgent:
    """Create a thinker agent with the OpenAI chat model."""
    # Create OpenAI chat model
    llm = OpenAIChatModel(
        model="gpt-4",
        api_key=api_key,
    )
    
    # Create agent config
    config = ThinkerAgentConfig(
        llm=llm,
        system_prompt=(
            "You are a thoughtful AI assistant specializing in breaking down complex problems. "
            "Analyze issues systematically and consider multiple perspectives before reaching conclusions."
        ),
    )
    
    # Create agent
    agent = ThinkerAgent(
        agent_id=agent_id,
        role=AgentRole.THINKER,
        config=config,
    )
    
    return agent


async def create_executor_agent(agent_id: str, api_key: str) -> ToolCallingAgent:
    """Create an executor agent using the tool calling agent."""
    # Create OpenAI chat model
    llm = OpenAIChatModel(
        model="gpt-4",
        api_key=api_key,
    )
    
    # Create agent
    agent = ToolCallingAgent(
        llm=llm,
        memory=UnconstrainedMemory(),
        templates={
            "system": lambda t: t.update(
                defaults={
                    "instructions": (
                        "You are an executor agent that takes plans and executes them step by step. "
                        "You should follow instructions precisely and report on your progress."
                    ),
                    "role": "executor",
                }
            )
        },
    )
    
    return agent


async def setup_workflow(api_key: str) -> Orchestrator:
    """Set up a multi-agent workflow with orchestrator."""
    # Create orchestrator
    orchestrator = Orchestrator(workflow_id="example_workflow")
    
    # Create agents
    thinker = await create_thinker_agent("thinker_1", api_key)
    executor = await create_executor_agent("executor_1", api_key)
    
    # Register agents with orchestrator
    await orchestrator.register_agent(thinker, "thinker_1", AgentRole.THINKER)
    await orchestrator.register_agent(executor, "executor_1", AgentRole.EXECUTOR)
    
    return orchestrator


async def run_multi_step_workflow(orchestrator: Orchestrator) -> Dict[str, Any]:
    """Run a multi-step workflow with the orchestrator."""
    # Step 1: Create a planning task for the thinker
    plan_task_id = await orchestrator.add_task(
        name="Create a plan for data analysis",
        description=(
            "Create a step-by-step plan for analyzing a dataset of customer reviews. "
            "The plan should include data cleaning, sentiment analysis, and visualization steps."
        ),
        role=AgentRole.THINKER,
        context={
            "context": "We have a dataset of 10,000 customer reviews for an e-commerce website.",
        },
        metadata={
            "priority": "high",
        },
    )
    
    # Step 2: Create an execution task for the executor agent that depends on the plan
    execution_task_id = await orchestrator.add_task(
        name="Execute data analysis plan",
        description="Execute the data analysis plan created by the thinker agent.",
        role=AgentRole.EXECUTOR,
        dependencies=[plan_task_id],
        context={
            "context": "Use the plan from the previous task to analyze the customer reviews dataset.",
        },
    )
    
    # Run the orchestrator workflow
    run = await orchestrator.run()
    
    # Get results
    plan_result = orchestrator.get_task_result(plan_task_id)
    execution_result = orchestrator.get_task_result(execution_task_id)
    
    # Return the final results
    return {
        "workflow_id": orchestrator.workflow_id,
        "success": orchestrator.state.workflow_success,
        "plan": plan_result.result if plan_result else None,
        "execution": execution_result.result if execution_result else None,
    }


async def main():
    """Run the example."""
    try:
        # Replace with your actual API key or use environment variables
        api_key = "YOUR_OPENAI_API_KEY"
        
        # Set up workflow
        orchestrator = await setup_workflow(api_key)
        
        # Run workflow
        results = await run_multi_step_workflow(orchestrator)
        
        # Print results
        logger.info(f"Workflow completed with success: {results['success']}")
        logger.info(f"Plan: {results['plan']}")
        logger.info(f"Execution: {results['execution']}")
        
    except Exception as e:
        logger.error(f"Error in workflow: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 