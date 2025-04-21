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
BeeAI MCP Middleware Adapter Module

This module provides adapters for converting between BeeAI middleware components
and MCP tools.
"""

from typing import Any, Dict, List, Optional, TypeVar, Type, cast

try:
    from mcp.types import Tool as MCPTool
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [mcp] not found.\nRun 'pip install beeai-framework[mcp]' to install."
    ) from e

from pydantic import BaseModel, create_model

from beeai_framework.logger import Logger
from beeai_framework.middleware.base import Middleware, MiddlewareContext, MiddlewareConfig
from beeai_framework.mcp.client import BeeAIMCPClient
from beeai_framework.utils.models import get_schema

logger = Logger(__name__)

T = TypeVar('T', bound=BaseModel)
R = TypeVar('R')


class MCPMiddlewareAdapter(Middleware[T, R]):
    """
    Middleware adapter that connects to an MCP server and delegates processing to an MCP tool.
    This allows using remote MCP tools as BeeAI middleware components.
    """
    
    def __init__(self, 
                 client: BeeAIMCPClient, 
                 tool_name: str, 
                 request_type: Type[T],
                 response_type: Type[R],
                 config: Optional[MiddlewareConfig] = None):
        """
        Initialize the MCP middleware adapter.
        
        Args:
            client: MCP client to use for communicating with the server
            tool_name: Name of the MCP tool to call
            request_type: Type of request this middleware accepts
            response_type: Type of response this middleware produces
            config: Optional middleware configuration
        """
        super().__init__(config)
        self.client = client
        self.tool_name = tool_name
        self.request_type = request_type
        self.response_type = response_type
        
        logger.info(f"Initialized MCP middleware adapter for tool: {tool_name}")
        
    async def process(self, context: MiddlewareContext[T, R]) -> MiddlewareContext[T, R]:
        """
        Process the request by calling the MCP tool.
        
        Args:
            context: The middleware context containing the request
            
        Returns:
            Updated context with the response from the MCP tool
        """
        logger.debug(f"Processing request through MCP middleware adapter for tool: {self.tool_name}")
        
        # Convert request to dict for the MCP call
        request_data = {}
        if hasattr(context.request, "model_dump"):
            request_data = context.request.model_dump()
        elif hasattr(context.request, "dict"):
            request_data = context.request.dict()
        elif isinstance(context.request, dict):
            request_data = context.request
        else:
            request_data = {"data": context.request}
        
        try:
            # Call the MCP tool
            result = await self.client.call_tool(self.tool_name, **request_data)
            
            # Parse the response
            if isinstance(result, dict) and issubclass(self.response_type, BaseModel):
                response = self.response_type(**result)
            else:
                # Try to convert non-dict results to the expected type
                if hasattr(self.response_type, "validate"):
                    response = self.response_type.validate(result)
                elif hasattr(self.response_type, "parse_obj"):
                    response = self.response_type.parse_obj(result)
                elif hasattr(self.response_type, "model_validate"):
                    response = self.response_type.model_validate(result)
                else:
                    # Basic case: hope the response type can be constructed from the result
                    response = self.response_type(result)
            
            # Update context with the response
            context.set_response(cast(R, response))
            
            logger.debug(f"MCP middleware adapter received response from tool: {self.tool_name}")
            
        except Exception as e:
            logger.error(f"Error in MCP middleware adapter: {str(e)}")
            context.metadata["error"] = str(e)
            
        return context
    
    @property
    def name(self) -> str:
        """Get the name of this middleware component."""
        return f"mcp_adapter_{self.tool_name}"


class MCPStreamingMiddlewareAdapter(Middleware[T, List[str]]):
    """
    Middleware adapter for streaming responses from MCP tools.
    This is useful for long-running operations where partial results are important.
    """
    
    def __init__(self, 
                 client: BeeAIMCPClient, 
                 tool_name: str, 
                 request_type: Type[T],
                 config: Optional[MiddlewareConfig] = None):
        """
        Initialize the streaming MCP middleware adapter.
        
        Args:
            client: MCP client to use for communicating with the server
            tool_name: Name of the MCP tool to call
            request_type: Type of request this middleware accepts
            config: Optional middleware configuration
        """
        super().__init__(config)
        self.client = client
        self.tool_name = tool_name
        self.request_type = request_type
        
        logger.info(f"Initialized streaming MCP middleware adapter for tool: {tool_name}")
        
    async def process(self, context: MiddlewareContext[T, List[str]]) -> MiddlewareContext[T, List[str]]:
        """
        Process the request by calling the MCP tool with streaming.
        
        Args:
            context: The middleware context containing the request
            
        Returns:
            Updated context with all streaming chunks collected as a list
        """
        logger.debug(f"Processing request through streaming MCP middleware adapter for tool: {self.tool_name}")
        
        # Convert request to dict for the MCP call
        request_data = {}
        if hasattr(context.request, "model_dump"):
            request_data = context.request.model_dump()
        elif hasattr(context.request, "dict"):
            request_data = context.request.dict()
        elif isinstance(context.request, dict):
            request_data = context.request
        else:
            request_data = {"data": context.request}
        
        try:
            # Call the MCP tool with streaming
            chunks: List[str] = []
            async for chunk in self.client.call_tool_streaming(self.tool_name, **request_data):
                chunks.append(chunk)
                # Emit event for each chunk received
                self.emitter.emit("chunk", {"chunk": chunk, "index": len(chunks) - 1})
            
            # Update context with all received chunks
            context.set_response(chunks)
            
            logger.debug(f"Streaming MCP middleware adapter received {len(chunks)} chunks from tool: {self.tool_name}")
            
        except Exception as e:
            logger.error(f"Error in streaming MCP middleware adapter: {str(e)}")
            context.metadata["error"] = str(e)
            
        return context
    
    @property
    def name(self) -> str:
        """Get the name of this middleware component."""
        return f"mcp_streaming_adapter_{self.tool_name}" 