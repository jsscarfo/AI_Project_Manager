# FastMCP Integration

This document provides an overview of the FastMCP integration implemented in the BeeAI Framework V5.

## Overview

The FastMCP integration enables BeeAI Framework components to be exposed as standardized MCP (Model Control Protocol) tools and facilitates communication between different AI systems. This implementation creates a bidirectional bridge that allows:

1. **Exposing BeeAI components** as MCP-compatible tools for external consumers
2. **Consuming external MCP tools** within the BeeAI Framework
3. **Standardized communication** between multi-agent systems

The integration provides a unified approach to tool discovery, invocation, and response handling across different AI systems.

## Core Components

### BeeAIMCPServer

The `BeeAIMCPServer` is responsible for exposing BeeAI components as MCP tools:

- **Dynamic Registration**: Automatically converts BeeAI components (middleware, workflows, functions) into MCP tools
- **Type Conversion**: Handles conversion between BeeAI types and MCP-compatible types
- **Error Handling**: Provides standardized error reporting and handling
- **Resource Management**: Manages resource endpoints for various BeeAI components
- **Streaming Support**: Enables streaming responses for long-running operations

### BeeAIMCPClient

The `BeeAIMCPClient` facilitates communication with MCP servers:

- **Connection Management**: Handles connection establishment and maintenance
- **Connection Pooling**: Optimizes connection reuse for improved performance
- **Tool Discovery**: Supports discovery of available tools on MCP servers
- **Request Handling**: Manages tool invocation requests and response processing
- **Context Management**: Provides clean connection handling with context managers

### Adapters

Adapters enable seamless integration between BeeAI components and MCP:

- **Middleware Adapters**: Allow MCP tools to be used as BeeAI middleware components
- **Workflow Adapters**: Enable MCP tools to be integrated into BeeAI workflows
- **Function Adapters**: Convert standard Python functions to MCP-compatible tools

## Integration with BeeAI Framework

The FastMCP integration connects with several key BeeAI Framework components:

1. **Middleware System**: Exposes middleware as MCP tools and allows MCP tools to be used as middleware
2. **Multi-Agent Workflow System**: Enables workflows and agents to communicate via MCP
3. **LLM Provider System**: Allows LLM providers to be accessed through MCP
4. **Vector Memory System**: Exposes vector memory operations as MCP tools

## Usage Examples

### Setting up an MCP Server

```python
from beeai_framework.mcp import BeeAIMCPServer
from beeai_framework.middleware import ContextEnhancementMiddleware
from beeai_framework.workflows import WorkflowOrchestrator

# Create server
server = BeeAIMCPServer(host="0.0.0.0", port=8000)

# Register middleware components
server.register_middleware(ContextEnhancementMiddleware())

# Register workflow components
orchestrator = WorkflowOrchestrator()
server.register_workflow(orchestrator)

# Register a standard function
@server.register_function
def process_data(data: dict) -> dict:
    # Process data
    return {"processed": True, "result": data}

# Start the server
server.start()
```

### Using the MCP Client

```python
from beeai_framework.mcp import BeeAIMCPClient

# Connect to an MCP server
with BeeAIMCPClient(url="http://localhost:8000") as client:
    # List available tools
    tools = client.list_tools()
    
    # Call a tool
    result = client.call_tool(
        tool_name="context_enhancement",
        parameters={"text": "This is some text to enhance"}
    )
    
    # Use streaming for long-running operations
    for chunk in client.call_tool_streaming(
        tool_name="run_workflow",
        parameters={"workflow_id": "analysis_workflow", "input": "Analyze this data"}
    ):
        print(f"Received chunk: {chunk}")
```

### Using MCP Tools as Middleware

```python
from beeai_framework.mcp.adapters import MCPToolMiddleware
from beeai_framework.middleware import MiddlewareChain

# Create middleware from an MCP tool
mcp_middleware = MCPToolMiddleware(
    tool_name="external_enhancement",
    server_url="http://external-server:8000"
)

# Add to middleware chain
chain = MiddlewareChain([
    # ... other middleware
    mcp_middleware,
    # ... other middleware
])

# Use normally
result = chain.process({"text": "Process this text"})
```

## Implementation Details

### Components

The FastMCP integration is implemented in `V5/python/beeai_framework/mcp/` and includes:

- `server.py`: BeeAIMCPServer implementation
- `client.py`: BeeAIMCPClient implementation
- `adapters/`: Integration adapters for different component types
- `models.py`: Pydantic models for type safety
- `utils.py`: Utility functions for type conversion and error handling
- `README.md`: Documentation and usage examples

### Type Conversion

The integration handles automatic conversion between:

- BeeAI middleware input/output and MCP tool parameters/results
- Workflow tasks and MCP tool invocations
- Python types and JSON-serializable types

### Error Handling

Comprehensive error handling includes:

- Connection errors with detailed diagnostics
- Tool invocation errors with standardized reporting
- Type conversion errors with helpful messages
- Graceful degradation for non-critical failures

## Performance Considerations

- Connection pooling optimizes performance for multiple tool calls
- Response streaming minimizes memory usage for large responses
- Type conversion is optimized for minimal overhead
- Server implements efficient request handling patterns

## Security Considerations

- Authentication support for secure tool access
- Input validation to prevent injection attacks
- Rate limiting to prevent abuse
- Access control for sensitive operations

## Future Enhancements

- **Enhanced Discovery**: Semantic discovery of relevant tools
- **Federated Tool Registries**: Discover tools across multiple servers
- **Advanced Authentication**: OAuth and other authentication methods
- **Tool Composition**: Compose multiple tools into higher-level tools
- **Cross-language Support**: Improved interoperability with non-Python systems

## Conclusion

The FastMCP integration provides a standardized and flexible approach to exposing and consuming AI capabilities across systems. By implementing the Model Control Protocol, the BeeAI Framework can now seamlessly interact with other AI systems, creating opportunities for more powerful multi-agent workflows and shared capabilities. 