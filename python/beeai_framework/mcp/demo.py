#!/usr/bin/env python
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

"""
BeeAI MCP Demo Script

This script demonstrates how to use the MCP integration by:
1. Setting up a BeeAI MCP server with middleware and workflow components
2. Connecting a client to the server
3. Using the client to execute tools on the server
"""

import argparse
import asyncio
import json
import sys
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from beeai_framework.logger import Logger
from beeai_framework.middleware.base import Middleware, MiddlewareContext, MiddlewareConfig
from beeai_framework.workflows.workflow import Workflow
from beeai_framework.mcp.server import BeeAIMCPServer
from beeai_framework.mcp.client import BeeAIMCPClient

logger = Logger(__name__)


# Define a simple example middleware component
class EchoRequest(BaseModel):
    """Request model for the Echo middleware."""
    message: str = Field(..., description="Message to echo")
    prefix: Optional[str] = Field(None, description="Prefix to add to the echoed message")


class EchoResponse(BaseModel):
    """Response model for the Echo middleware."""
    original_message: str = Field(..., description="Original message")
    echoed_message: str = Field(..., description="Echoed message")
    timestamp: str = Field(..., description="Timestamp when the message was echoed")


class EchoMiddleware(Middleware[EchoRequest, EchoResponse]):
    """Simple middleware that echoes back the received message."""
    
    def __init__(self, config: Optional[MiddlewareConfig] = None):
        super().__init__(config)
    
    async def process(self, context: MiddlewareContext[EchoRequest, EchoResponse]) -> MiddlewareContext[EchoRequest, EchoResponse]:
        from datetime import datetime
        request = context.request
        prefix = request.prefix or "Echo"
        
        response = EchoResponse(
            original_message=request.message,
            echoed_message=f"{prefix}: {request.message}",
            timestamp=datetime.now().isoformat()
        )
        
        return context.set_response(response)


# Define a simple example workflow
class CalculationState(BaseModel):
    """State model for the Calculation workflow."""
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")
    operation: str = Field(..., description="Operation to perform: add, subtract, multiply, divide")
    result: Optional[int] = None
    error: Optional[str] = None


class CalculationWorkflow(Workflow[CalculationState, str]):
    """Simple workflow that performs a calculation."""
    
    def __init__(self):
        super().__init__(CalculationState, name="Calculator")
        
        # Add steps
        self.add_step("validate", self._validate)
        self.add_step("calculate", self._calculate)
        
        # Set the starting step
        self.set_start("validate")
    
    async def _validate(self, state: CalculationState) -> str:
        """Validate the input data."""
        if state.operation not in ["add", "subtract", "multiply", "divide"]:
            state.error = f"Invalid operation: {state.operation}"
            return Workflow.END
            
        if state.operation == "divide" and state.b == 0:
            state.error = "Cannot divide by zero"
            return Workflow.END
            
        return "calculate"
    
    async def _calculate(self, state: CalculationState) -> str:
        """Perform the calculation."""
        if state.operation == "add":
            state.result = state.a + state.b
        elif state.operation == "subtract":
            state.result = state.a - state.b
        elif state.operation == "multiply":
            state.result = state.a * state.b
        elif state.operation == "divide":
            state.result = state.a // state.b
            
        return Workflow.END


# Helper function to run the server
async def run_server(host: str = "127.0.0.1", port: int = 5000):
    """Run the MCP server with example components."""
    # Create and initialize the server
    server = BeeAIMCPServer(name="BeeAI-Demo")
    
    # Create and register the middleware
    echo_middleware = EchoMiddleware()
    server.register_middleware(echo_middleware)
    
    # Create and register the workflow
    calc_workflow = CalculationWorkflow()
    server.register_workflow(calc_workflow)
    
    # Register a simple function
    def greet(name: str, title: Optional[str] = None) -> Dict[str, str]:
        """Greet a person with a custom message."""
        if title:
            return {"greeting": f"Hello, {title} {name}!"}
        else:
            return {"greeting": f"Hello, {name}!"}
            
    server.register_function(greet, description="Greet a person with their name")
    
    # Start the server
    logger.info(f"Starting MCP server on {host}:{port}")
    server.start(host=host, port=port)
    
    # Keep the server running until interrupted
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        await server.stop()


# Helper function to run a client example
async def run_client(server_url: str):
    """Run the MCP client example."""
    # Create and initialize the client
    async with BeeAIMCPClient(server_url) as client:
        logger.info(f"Connected to MCP server: {server_url}")
        
        # List available tools
        tool_names = client.get_tool_names()
        logger.info(f"Available tools: {tool_names}")
        
        # Call the echo middleware
        echo_result = await client.call_tool(
            "middleware_EchoMiddleware",
            message="Hello, MCP!",
            prefix="BeeAI"
        )
        logger.info(f"Echo result: {echo_result}")
        
        # Call the calculation workflow
        calc_result = await client.call_tool(
            "workflow_Calculator",
            a=10,
            b=5,
            operation="multiply"
        )
        logger.info(f"Calculation result: {calc_result}")
        
        # Call the greet function
        greet_result = await client.call_tool(
            "function_greet",
            name="Claude",
            title="Dr."
        )
        logger.info(f"Greeting result: {greet_result}")


async def main():
    """Main function for the demo script."""
    parser = argparse.ArgumentParser(description="BeeAI MCP Demo")
    parser.add_argument(
        "--mode",
        choices=["server", "client", "both"],
        default="both",
        help="Mode to run: server, client, or both"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind the server to (default: 5000)"
    )
    
    args = parser.parse_args()
    server_url = f"http://{args.host}:{args.port}"
    
    if args.mode == "server":
        await run_server(args.host, args.port)
    elif args.mode == "client":
        await run_client(server_url)
    elif args.mode == "both":
        # Run server in a separate task
        server_task = asyncio.create_task(run_server(args.host, args.port))
        
        # Wait a moment for the server to start
        await asyncio.sleep(2)
        
        try:
            # Run client
            await run_client(server_url)
        finally:
            # Stop the server
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")
        sys.exit(0) 