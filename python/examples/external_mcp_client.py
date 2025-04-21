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
External MCP Client Example

This example demonstrates how to use the BeeAI External MCP Client to
access BeeAI functionality from an external application.
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import List, Dict, Any, Optional

# Import BeeAI MCP client
from beeai_framework.mcp.external_client import BeeAIExternalMCPClient


async def demo_greeting(client: BeeAIExternalMCPClient) -> None:
    """
    Demonstrate using the greeting middleware.
    
    Args:
        client: BeeAI external MCP client
    """
    print("\n=== Testing Greeting Tool ===")
    
    # Call the greeting tool
    result = await client.call_tool("greeting", name="World", message="Hello")
    print(f"Greeting result: {result}")
    
    # Call with different parameters
    result = await client.call_tool(
        "greeting", 
        name="BeeAI Framework", 
        message="Welcome to"
    )
    print(f"Custom greeting: {result}")


async def demo_echo(client: BeeAIExternalMCPClient) -> None:
    """
    Demonstrate using the echo middleware.
    
    Args:
        client: BeeAI external MCP client
    """
    print("\n=== Testing Echo Tool ===")
    
    # Call the echo tool with simple data
    data = {"message": "This is a test", "count": 42}
    result = await client.call_tool("echo", **data)
    print(f"Echo result: {result}")
    
    # Call with nested data
    nested_data = {
        "user": {
            "name": "Test User",
            "id": 12345
        },
        "options": {
            "verbose": True,
            "timeout": 30
        },
        "tags": ["test", "demo", "mcp"]
    }
    result = await client.call_tool("echo", **nested_data)
    print(f"Echo with nested data: {result}")


async def demo_sequential_thinking(client: BeeAIExternalMCPClient) -> None:
    """
    Demonstrate using the sequential thinking capabilities.
    
    Args:
        client: BeeAI external MCP client
    """
    print("\n=== Testing Sequential Thinking ===")
    
    try:
        # Use the domain-specific client
        result = await client.sequential_thinking.solve(
            problem="Design a user authentication system with two-factor authentication",
            steps=3,
            context={
                "domain": "web_security", 
                "requirements": ["secure", "user-friendly"]
            }
        )
        
        print("Sequential thinking result:")
        if "reasoning_steps" in result:
            for step in result["reasoning_steps"]:
                print(f"Step {step['number']}: {step['title']}")
                print(f"  {step['content'][:100]}...")
        else:
            print(result)
            
    except Exception as e:
        print(f"Sequential thinking demo error: {str(e)}")
        print("Note: This demo requires the sequential_thinking_solve tool to be available.")


async def demo_vector_memory(client: BeeAIExternalMCPClient) -> None:
    """
    Demonstrate using the vector memory capabilities.
    
    Args:
        client: BeeAI external MCP client
    """
    print("\n=== Testing Vector Memory ===")
    
    try:
        # Store documents
        documents = [
            {
                "text": "BeeAI is a framework for building AI applications with advanced context management.", 
                "metadata": {"type": "framework", "topic": "ai"}
            },
            {
                "text": "The MCP protocol allows tools to be exposed between different AI systems.",
                "metadata": {"type": "protocol", "topic": "integration"}
            },
            {
                "text": "Sequential thinking enables LLMs to solve complex problems step by step.",
                "metadata": {"type": "technique", "topic": "reasoning"}
            }
        ]
        
        store_result = await client.vector_memory.store(
            collection="demo_collection",
            documents=documents
        )
        
        print(f"Store result: {store_result}")
        
        # Search for documents
        search_result = await client.vector_memory.search(
            collection="demo_collection",
            query="How does MCP help with integration?",
            limit=2
        )
        
        print("Search results:")
        for i, result in enumerate(search_result):
            print(f"  {i+1}. Score: {result.get('score', 'N/A')}")
            print(f"     Text: {result.get('text', 'N/A')}")
            
    except Exception as e:
        print(f"Vector memory demo error: {str(e)}")
        print("Note: This demo requires vector memory tools to be available.")


async def run_client(
    url: str,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Run the example client.
    
    Args:
        url: URL of the BeeAI External MCP server
        api_key: Optional API key for authentication
        username: Optional username for OAuth authentication
        password: Optional password for OAuth authentication
        verbose: Whether to enable verbose logging
    """
    # Setup logging level
    logging_level = logging.DEBUG if verbose else logging.INFO
    
    # Create the client
    client = BeeAIExternalMCPClient(
        url=url,
        api_key=api_key,
        username=username,
        password=password,
        logging_level=logging_level
    )
    
    try:
        # Connect to the server
        await client.connect()
        
        # Get available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool['name'] for tool in tools]}")
        
        # Run demos based on available tools
        if any(tool["name"] == "greeting" for tool in tools):
            await demo_greeting(client)
            
        if any(tool["name"] == "echo" for tool in tools):
            await demo_echo(client)
            
        if any(tool["name"] == "sequential_thinking_solve" for tool in tools):
            await demo_sequential_thinking(client)
            
        if any(tool["name"] == "vector_memory_store" for tool in tools) and \
           any(tool["name"] == "vector_memory_search" for tool in tools):
            await demo_vector_memory(client)
            
    finally:
        # Disconnect
        await client.disconnect()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BeeAI External MCP Client Example")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="URL of the BeeAI External MCP server")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--username", help="Username for OAuth authentication")
    parser.add_argument("--password", help="Password for OAuth authentication")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Validate authentication parameters
    if args.api_key and (args.username or args.password):
        print("Warning: Both API key and username/password provided. API key will be used.")
    
    if (args.username and not args.password) or (args.password and not args.username):
        print("Error: Both username and password must be provided for OAuth authentication.")
        return 1
    
    try:
        await run_client(
            url=args.url,
            api_key=args.api_key,
            username=args.username,
            password=args.password,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("Client stopped by user")
    except Exception as e:
        print(f"Error running client: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 