# BeeAI External MCP Integration

This module provides a secure, robust external access layer that exposes selected BeeAI functionality to external systems through the Model Context Protocol (MCP). It enables external tools, agents, and systems to leverage BeeAI capabilities through standardized MCP endpoints.

## Overview

The BeeAI External MCP integration consists of:

1. **External MCP Server**: A secure server that exposes selected BeeAI capabilities with authentication, authorization, and rate limiting.
2. **Tool Registration System**: A configuration-driven tool registration system that defines which tools are exposed externally.
3. **Client SDK**: A Python client SDK for easy integration with the external MCP server.

## Installation

To use the external MCP integration, you need to install the appropriate dependencies:

### Server Dependencies

```bash
pip install beeai-framework[mcp-external]
```

### Client Dependencies

```bash
pip install beeai-framework[mcp-client]
```

### All Dependencies (for development)

```bash
pip install beeai-framework[full]
```

## Using the External MCP Server

### Basic Setup

```python
from beeai_framework.mcp.external_server import BeeAIExternalMCPServer
from beeai_framework.middleware.base import Middleware, MiddlewareContext

# Create middleware components to expose
class GreetingMiddleware(Middleware):
    # ... middleware implementation ...
    
# Create the external MCP server
server = BeeAIExternalMCPServer(
    name="BeeAI-External",
    secret_key="your-secret-key",  # Use a secure secret key in production
    access_token_expire_minutes=30,
    allow_origins=["https://your-app.example.com"]  # CORS configuration
)

# Register middleware components with the BeeAI MCP system
greeting_middleware = GreetingMiddleware()
server.register_middleware(greeting_middleware)

# Expose selected middleware as external tools
server.register_external_tool(
    name="greeting",
    description="Generate a personalized greeting message",
    source_tool="middleware_greeting",  # Must match the registered middleware name
    required_roles=["user"],
    metadata={"category": "demo"}
)

# Add users
server.add_user(
    username="admin",
    password="secure-password",  # Use secure passwords in production
    email="admin@example.com",
    roles=["admin", "user"]
)

# Create API keys for programmatic access
api_key = server.create_api_key(
    name="Client API Key",
    user_id="admin",  # Must be an existing user
    expiry_days=30,  # Set to None for no expiry
    rate_limit=100   # Requests per minute
)
print(f"API Key: {api_key.key}")

# Start the server
server.start(host="0.0.0.0", port=8000)
```

### Using the Tool Registry

For more control over tool registration, you can use the `ExternalToolsRegistry` class:

```python
from beeai_framework.mcp.external_tools import ExternalToolsRegistry, ExternalToolConfig

# Create a custom registry
registry = ExternalToolsRegistry()

# Register tools
registry.register(ExternalToolConfig(
    name="sequential_thinking_solve",
    description="Solve a problem using sequential thinking",
    source_tool="middleware_sequential_thinking",
    version="1.0.0",
    required_roles=["user"],
    metadata={"category": "reasoning"}
))

# Load tools from a JSON file
registry.load_from_file("tools_config.json")

# Create the server with the custom registry
server = BeeAIExternalMCPServer(
    name="BeeAI-External",
    tool_registry=registry
)
```

## Using the Client SDK

### Basic Usage

```python
import asyncio
from beeai_framework.mcp.external_client import BeeAIExternalMCPClient

async def main():
    # Create the client
    client = BeeAIExternalMCPClient(
        url="https://beeai-mcp.example.com",
        api_key="your-api-key"
        # Or use username/password for OAuth:
        # username="admin",
        # password="secure-password"
    )
    
    # Connect to the server
    await client.connect()
    
    try:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool['name'] for tool in tools]}")
        
        # Call a tool directly
        result = await client.call_tool(
            "greeting", 
            name="World", 
            message="Hello"
        )
        print(result)
    finally:
        # Disconnect when done
        await client.disconnect()

# Run the async function
asyncio.run(main())
```

### Using Domain-Specific Clients

The client SDK provides convenience classes for different BeeAI capabilities:

```python
async def use_domain_clients(client):
    # Use sequential thinking
    result = await client.sequential_thinking.solve(
        problem="Design a user authentication system",
        steps=5,
        context={"domain": "web_security"}
    )
    
    # Use vector memory
    docs = [
        {"text": "BeeAI documentation page 1", "metadata": {"type": "docs"}},
        {"text": "BeeAI documentation page 2", "metadata": {"type": "docs"}}
    ]
    
    # Store documents
    await client.vector_memory.store(
        collection="documentation",
        documents=docs
    )
    
    # Search for documents
    results = await client.vector_memory.search(
        collection="documentation",
        query="How to use BeeAI?",
        limit=5
    )
    
    # Create visualizations
    dashboard = await client.visualization.create_metrics_dashboard(
        data={"metrics": [...]},
        metrics=["accuracy", "latency"],
        title="Performance Dashboard",
        format="html"
    )
```

## Authentication

The External MCP server supports two authentication methods:

1. **API Key Authentication**: Fast and simple, ideal for programmatic access.
2. **OAuth 2.0**: Username/password authentication with JWT tokens, suitable for user-facing applications.

### API Key Authentication

```python
# Server setup
api_key = server.create_api_key(name="App Key", user_id="admin")

# Client usage
client = BeeAIExternalMCPClient(url="...", api_key=api_key.key)
```

### OAuth Authentication

```python
# Server setup - users are created as shown earlier

# Client usage
client = BeeAIExternalMCPClient(
    url="...", 
    username="admin", 
    password="secure-password"
)
# Token is automatically obtained during connect()
```

## Security Considerations

1. Always use HTTPS in production.
2. Use strong, randomly generated secret keys.
3. Set appropriate CORS restrictions.
4. Use the principle of least privilege when assigning roles.
5. Use API key expiration for temporary access.
6. Store user credentials securely (in a database, not in memory for production use).

## Example Applications

See the examples directory for complete examples:

- `external_mcp_server.py`: Example server setup
- `external_mcp_client.py`: Example client usage

## Using with Docker

To containerize your external MCP server:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "server.py"]
```

## Tool Configuration Format

Tools can be configured in a JSON file with the following format:

```json
{
  "clear_existing": false,
  "tools": [
    {
      "name": "sequential_thinking_solve",
      "description": "Solve problems using sequential thinking",
      "source_tool": "middleware_sequential_thinking",
      "version": "1.0.0",
      "required_roles": ["user"],
      "enabled": true,
      "metadata": {
        "category": "reasoning"
      }
    },
    {
      "name": "vector_memory_search",
      "description": "Search for documents in vector memory",
      "source_tool": "middleware_vector_search",
      "version": "1.0.0",
      "required_roles": ["user"],
      "enabled": true,
      "metadata": {
        "category": "vector_memory"
      }
    }
  ]
}
``` 