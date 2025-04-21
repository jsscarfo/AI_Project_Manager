# Multi-Agent Workflow System

This document provides an overview of the multi-agent workflow orchestration system implemented in the BeeAI Framework V5.

## Overview

The Multi-Agent Workflow System enables coordinated execution of complex workflows across multiple specialized agents. The system provides structured communication protocols, task management, and state tracking to facilitate seamless collaboration between agents with different capabilities.

## Core Components

### Agent Base Components

The foundation of the multi-agent system is defined in `agent_base.py`, which includes:

- **AgentInterface Protocol**: Defines the required methods that all workflow agents must implement.
- **AgentTask**: Data model representing tasks assigned to agents.
- **AgentResult**: Data model representing the results of agent task execution.
- **AgentMessage**: Data model for structured communication between agents.
- **BaseWorkflowAgent**: Abstract base class implementing common agent functionality and the AgentInterface protocol.

### Communication Protocol

The `protocol.py` module implements the communication infrastructure:

- Message passing system with async delivery
- Subscription mechanisms for topic-based communication
- Structured message types for different communication patterns
- State tracking for monitoring message delivery and processing

### Orchestrator

The `orchestrator.py` module implements the workflow orchestration system:

- Task dependency management and resolution
- Dynamic task assignment based on agent capabilities
- Workflow state tracking and persistence
- Event emission for monitoring workflow progress
- Error handling and recovery strategies

### Specialized Agents

The system includes specialized agent implementations:

- **Sequential Thinker Agent**: Implements step-by-step reasoning capabilities with middleware support.
- Integration with existing middleware components for enhanced functionality.

## Integration with BeeAI Framework

The Multi-Agent Workflow System integrates with several key components of the BeeAI Framework:

1. **Middleware Framework**: Agents can utilize middleware chains for request processing.
2. **Vector Memory System**: Agents can access and store information in the vector memory system.
3. **LLM Provider System**: Agents can leverage different LLM providers for task execution.

## Usage Example

```python
from beeai_framework.workflows.agents.orchestrator import WorkflowOrchestrator
from beeai_framework.workflows.agents.specialized.thinker_agent import SequentialThinkerAgent
from beeai_framework.workflows.agent_base import AgentTask

# Create specialized agents
thinker_agent = SequentialThinkerAgent()

# Create orchestrator
orchestrator = WorkflowOrchestrator(agents=[thinker_agent])

# Define tasks with dependencies
task1 = AgentTask(
    id="task1",
    agent_type="sequential_thinker",
    data={"prompt": "Analyze the requirements for this project"},
    dependencies=[]
)

task2 = AgentTask(
    id="task2",
    agent_type="sequential_thinker",
    data={"prompt": "Design the architecture based on the requirements analysis"},
    dependencies=["task1"]
)

# Add tasks to orchestrator
orchestrator.add_task(task1)
orchestrator.add_task(task2)

# Execute workflow
async def run_workflow():
    results = await orchestrator.execute_workflow()
    return results

# Process results
results = asyncio.run(run_workflow())
```

## Performance Considerations

- The system uses asynchronous processing to maximize throughput.
- Message passing is optimized to minimize overhead in agent communication.
- Task dependencies are resolved efficiently to maximize parallel execution.

## Testing

The implementation includes comprehensive unit tests for:
- Agent task processing
- Message handling
- Communication protocol
- Orchestrator functionality
- Workflow execution with dependencies

## Future Enhancements

- **Distributed Execution**: Support for distributed workflow execution across multiple machines.
- **Dynamic Workflow Modification**: Support for modifying workflow structure during execution.
- **Advanced Monitoring**: Enhanced monitoring and visualization of workflow execution.
- **Performance Optimization**: Further optimization for complex multi-agent scenarios.

## Integration with FastMCP

The Multi-Agent Workflow System is designed to integrate with FastMCP:
- Expose workflow management through MCP endpoints
- Enable remote agent communication via MCP
- Provide MCP-based monitoring and control interfaces

## Conclusion

The Multi-Agent Workflow System provides a flexible and extensible framework for orchestrating complex workflows across multiple specialized agents. It integrates seamlessly with the existing BeeAI Framework components and can be easily extended to support additional agent types and workflow patterns. 