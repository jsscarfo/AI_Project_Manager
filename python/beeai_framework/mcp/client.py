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
BeeAI FastMCP Client Module

This module provides a client for interacting with MCP servers, enabling
communication between agents and accessing remote tools through the Model
Context Protocol.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Callable, AsyncIterator

try:
    from mcp import ClientSession
    from mcp.types import Tool as MCPTool
    from mcp.types import CallToolResult
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [mcp] not found.\nRun 'pip install beeai-framework[mcp]' to install."
    ) from e

from pydantic import BaseModel, Field

from beeai_framework.logger import Logger
from beeai_framework.emitter import Emitter
from beeai_framework.tools.mcp.mcp import MCPTool as MCPToolWrapper
from beeai_framework.tools.tool import Tool
from beeai_framework.context import RunContext

logger = Logger(__name__)

T = TypeVar('T', bound=BaseModel)


class BeeAIMCPClient:
    """Client for interacting with MCP servers from BeeAI agents."""
    
    def __init__(self, server_url: str):
        """
        Initialize the BeeAI MCP client.
        
        Args:
            server_url: The URL of the MCP server to connect to
        """
        self.server_url = server_url
        self.client_session: Optional[ClientSession] = None
        self.tools: List[MCPToolWrapper] = []
        self.emitter = Emitter.root().child(
            namespace=["mcp", "client"],
            creator=self,
        )
        
        logger.info(f"Initialized BeeAI MCP client for server: {server_url}")
        
    async def connect(self) -> None:
        """Connect to the MCP server and initialize the client session."""
        logger.info(f"Connecting to MCP server at {self.server_url}")
        self.client_session = await ClientSession.connect(self.server_url)
        
        # Discover available tools
        tools_result = await self.client_session.list_tools()
        self.tools = [MCPToolWrapper(self.client_session, tool) for tool in tools_result.tools]
        
        logger.info(f"Connected to MCP server, discovered {len(self.tools)} tools")
        
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.client_session:
            await self.client_session.close()
            self.client_session = None
            self.tools = []
            logger.info("Disconnected from MCP server")
    
    async def call_tool(self, tool_name: str, **arguments) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            **arguments: Arguments to pass to the tool
            
        Returns:
            The result of the tool execution
        """
        if not self.client_session:
            raise ValueError("Not connected to MCP server")
            
        logger.debug(f"Calling tool {tool_name} with arguments: {arguments}")
        result: CallToolResult = await self.client_session.call_tool(
            name=tool_name,
            arguments=arguments
        )
        
        logger.debug(f"Tool result: {result}")
        return result.content
    
    async def call_tool_streaming(self, tool_name: str, **arguments) -> AsyncIterator[str]:
        """
        Call a tool on the MCP server with streaming response.
        
        Args:
            tool_name: Name of the tool to call
            **arguments: Arguments to pass to the tool
            
        Returns:
            An async iterator for the streaming result
        """
        if not self.client_session:
            raise ValueError("Not connected to MCP server")
            
        logger.debug(f"Calling tool {tool_name} with streaming and arguments: {arguments}")
        
        async for chunk in self.client_session.call_tool_streaming(
            name=tool_name,
            arguments=arguments
        ):
            logger.trace(f"Tool streaming chunk: {chunk}")
            yield chunk.content
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool wrapper by name.
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            Tool wrapper or None if not found
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def get_tool_names(self) -> List[str]:
        """
        Get a list of available tool names.
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self.tools]
    
    def get_tools(self) -> List[Tool]:
        """
        Get all available tools.
        
        Returns:
            List of tool wrappers
        """
        return self.tools
    
    async def __aenter__(self) -> "BeeAIMCPClient":
        """Connect when used as a context manager."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Disconnect when exiting context manager."""
        await self.disconnect() 