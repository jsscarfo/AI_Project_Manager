# BeeAI FastMCP Integration

This module integrates BeeAI with the Model Context Protocol (MCP) to provide a standardized interface for LLM tools and enable multi-agent communication.

## Overview

The MCP integration allows BeeAI components (middleware, workflows, functions) to be exposed as MCP tools that can be consumed by LLM agents. It also enables BeeAI agents to communicate with other MCP-compatible systems.

## Features

- **MCP Server**: Expose BeeAI components as MCP tools
- **MCP Client**: Connect to MCP servers and use their tools
- **Adapters**: Use remote MCP tools as BeeAI middleware components or workflow steps
- **Streaming Support**: Stream responses from long-running operations

## Installation

Install the MCP integration with pip:

```bash
pip install beeai-framework[mcp]
```

This will install the required dependencies:
- `fastmcp`: The FastMCP library
- `modelcontextprotocol`: The MCP protocol implementation

## Usage

### Creating an MCP Server

```python
from beeai_framework.mcp import BeeAIMCPServer
from your_modules import YourMiddleware, YourWorkflow

# Create the server
server = BeeAIMCPServer(name="YourAppName")

# Register middleware components
middleware = YourMiddleware()
server.register_middleware(middleware)

# Register workflows
workflow = YourWorkflow()
server.register_workflow(workflow)

# Register functions
def your_function(param1: str, param2: int = 42) -> dict:
    # Function implementation
    return {"result": f"{param1}: {param2}"}

server.register_function(your_function, description="Function description")

# Start the server
server.start(host="127.0.0.1", port=5000)
```

### Using the MCP Client

```python
import asyncio
from beeai_framework.mcp import BeeAIMCPClient

async def main():
    # Connect to an MCP server
    async with BeeAIMCPClient("http://127.0.0.1:5000") as client:
        # List available tools
        tools = client.get_tool_names()
        print(f"Available tools: {tools}")
        
        # Call a tool
        result = await client.call_tool("tool_name", param1="value1", param2="value2")
        print(f"Result: {result}")
        
        # Call a tool with streaming
        async for chunk in client.call_tool_streaming("streaming_tool", query="search query"):
            print(f"Chunk: {chunk}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Using MCP Adapters

```python
from beeai_framework.mcp import BeeAIMCPClient, MCPMiddlewareAdapter, MCPWorkflowAdapter
from beeai_framework.middleware import MiddlewareChain, MiddlewareContext
from your_models import YourRequestModel, YourResponseModel, YourWorkflowState

async def setup_middleware():
    # Connect to an MCP server
    client = BeeAIMCPClient("http://127.0.0.1:5000")
    await client.connect()
    
    # Create a middleware adapter
    adapter = MCPMiddlewareAdapter(
        client=client,
        tool_name="remote_tool",
        request_type=YourRequestModel,
        response_type=YourResponseModel
    )
    
    # Add it to a middleware chain
    chain = MiddlewareChain()
    chain.add_middleware(adapter)
    
    # Use it like any other middleware
    request = YourRequestModel(...)
    response = await chain.process_request(request)
    
    # Create a workflow adapter
    workflow_adapter = MCPWorkflowAdapter(
        client=client,
        tool_name="remote_workflow",
        schema=YourWorkflowState
    )
    
    # Run the workflow
    state = YourWorkflowState(...)
    result = await workflow_adapter.run(state)
```

## Advanced Usage

### Adding an MCP Tool to an Existing Workflow

```python
from beeai_framework.mcp import BeeAIMCPClient, MCPStepAdapter
from your_workflow import YourWorkflow

async def add_mcp_step_to_workflow():
    # Connect to an MCP server
    client = BeeAIMCPClient("http://127.0.0.1:5000")
    await client.connect()
    
    # Create a workflow
    workflow = YourWorkflow()
    
    # Create a step handler that calls an MCP tool
    mcp_step = MCPStepAdapter.create_step(client, "remote_tool")
    
    # Add it to your workflow
    workflow.add_step("mcp_step", mcp_step)
```

## Demo

A demo script is included that shows how to use the MCP integration with example components:

```bash
python -m beeai_framework.mcp.demo
```

Run with `--help` to see available options.

## Testing

Unit tests are available in the `tests/mcp` directory. Run them with pytest:

```bash
pytest V5/python/tests/mcp/
``` 