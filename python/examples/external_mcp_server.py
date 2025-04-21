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
External MCP Server Example

This example demonstrates how to set up and run the BeeAI External MCP Server
with authentication, tool registration, and tool access.
"""

import os
import sys
import asyncio
import argparse
from typing import List, Dict, Any, Optional

# Import BeeAI and MCP modules
from beeai_framework.logger import Logger
from beeai_framework.mcp.external_server import BeeAIExternalMCPServer
from beeai_framework.mcp.external_tools import ExternalToolsRegistry
from beeai_framework.middleware.base import Middleware, MiddlewareContext
from beeai_framework.middleware.base import MiddlewareConfig

# Setup logging
logger = Logger("external_mcp_example")

# Define a simple middleware component for testing
class GreetingMiddleware(Middleware[Dict[str, Any], Dict[str, Any]]):
    """Simple greeting middleware for testing."""
    
    def __init__(self, config: Optional[MiddlewareConfig] = None):
        """Initialize the middleware."""
        super().__init__(config)
        
    async def process(self, context: MiddlewareContext[Dict[str, Any], Dict[str, Any]]) -> MiddlewareContext[Dict[str, Any], Dict[str, Any]]:
        """Process the request and generate a greeting."""
        request = context.request
        name = request.get("name", "Guest")
        message = request.get("message", "Hello")
        
        response = {
            "greeting": f"{message}, {name}!",
            "timestamp": self.emitter.timestamp()
        }
        
        context.set_response(response)
        return context
    
    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "greeting"


class EchoMiddleware(Middleware[Dict[str, Any], Dict[str, Any]]):
    """Simple echo middleware for testing."""
    
    def __init__(self, config: Optional[MiddlewareConfig] = None):
        """Initialize the middleware."""
        super().__init__(config)
        
    async def process(self, context: MiddlewareContext[Dict[str, Any], Dict[str, Any]]) -> MiddlewareContext[Dict[str, Any], Dict[str, Any]]:
        """Process the request and echo it back."""
        request = context.request
        
        response = {
            "echo": request,
            "processed_by": "echo_middleware",
            "timestamp": self.emitter.timestamp()
        }
        
        context.set_response(response)
        return context
    
    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "echo"


async def run_server(host: str, port: int, secret_key: str, allow_admin_creation: bool) -> None:
    """
    Set up and run the external MCP server.
    
    Args:
        host: Hostname to listen on
        port: Port to listen on
        secret_key: Secret key for JWT tokens
        allow_admin_creation: Whether to create an admin user
    """
    # Create middleware components
    greeting_middleware = GreetingMiddleware()
    echo_middleware = EchoMiddleware()
    
    # Create the external MCP server
    server = BeeAIExternalMCPServer(
        name="BeeAI-Example-External",
        secret_key=secret_key,
        access_token_expire_minutes=60,
        allow_origins=["*"]
    )
    
    # Register middleware components
    server.register_middleware(greeting_middleware)
    server.register_middleware(echo_middleware)
    
    # Only register certain middleware components for external access
    server.register_external_tool(
        name="greeting",
        description="Generate a personalized greeting message",
        source_tool="middleware_greeting",
        required_roles=["user"],
        metadata={
            "category": "demo",
            "examples": [
                {"name": "World", "message": "Hello"},
                {"name": "BeeAI", "message": "Welcome to"}
            ]
        }
    )
    
    server.register_external_tool(
        name="echo",
        description="Echo back the input with additional metadata",
        source_tool="middleware_echo",
        required_roles=["user"],
        metadata={"category": "demo"}
    )
    
    # Create admin and test users if allowed
    if allow_admin_creation:
        try:
            admin_user = server.add_user(
                username="admin",
                password="adminpassword",
                email="admin@example.com",
                roles=["admin", "user"]
            )
            logger.info(f"Created admin user: {admin_user.username}")
            
            test_user = server.add_user(
                username="testuser",
                password="testpassword",
                email="test@example.com",
                roles=["user"]
            )
            logger.info(f"Created test user: {test_user.username}")
            
            # Create API key for test user
            api_key = server.create_api_key(
                name="Test API Key",
                user_id="testuser",
                expiry_days=30,
                rate_limit=120
            )
            logger.info(f"Created API key for test user: {api_key.key}")
            
        except ValueError as e:
            logger.warning(f"Could not create users: {str(e)}")
    
    # Start the server
    logger.info(f"Starting external MCP server on {host}:{port}")
    await server.start_async(host=host, port=port)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BeeAI External MCP Server Example")
    parser.add_argument("--host", default="127.0.0.1", help="Hostname to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--secret-key", default=None, help="Secret key for JWT tokens")
    parser.add_argument("--create-admin", action="store_true", help="Create admin and test users")
    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Use provided secret key or generate a random one
    secret_key = args.secret_key or os.urandom(24).hex()
    
    try:
        await run_server(args.host, args.port, secret_key, args.create_admin)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 