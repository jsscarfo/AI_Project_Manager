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
BeeAI FastMCP Server Module

This module provides the MCP server implementation for BeeAI, which allows
exposing middleware components and workflows as tools through the Model Context Protocol.
"""

import inspect
import json
from typing import Any, Dict, List, Optional, Callable, Type, Union, TypeVar, cast

try:
    from mcp.server import Server as MCPServer
    from mcp.types import Tool as MCPTool
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [mcp] not found.\nRun 'pip install beeai-framework[mcp]' to install."
    ) from e

from pydantic import BaseModel, create_model, Field

from beeai_framework.logger import Logger
from beeai_framework.middleware.base import Middleware, MiddlewareContext
from beeai_framework.workflows.workflow import Workflow
from beeai_framework.utils.models import get_schema

logger = Logger(__name__)

T = TypeVar('T', bound=BaseModel)
R = TypeVar('R')


class BeeAIMCPServer:
    """MCP server implementation for BeeAI."""
    
    def __init__(self, name: str = "BeeAI"):
        """
        Initialize the BeeAI MCP server.
        
        Args:
            name: Server name for identification
        """
        self.name = name
        self.mcp_server = MCPServer(name=name)
        self.registered_tools: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized BeeAI MCP server: {name}")

    def register_middleware(self, middleware: Middleware) -> None:
        """
        Register middleware components as MCP tools.
        
        Args:
            middleware: The middleware component to register
        """
        middleware_name = middleware.name
        tool_name = f"middleware_{middleware_name}"
        
        # Get the input schema from the middleware's process method
        sig = inspect.signature(middleware.process)
        context_param = next(iter(sig.parameters.values()))
        context_type = context_param.annotation
        
        # Extract the request type from the MiddlewareContext
        if hasattr(context_type, "__args__") and len(context_type.__args__) >= 1:
            request_type = context_type.__args__[0]
            response_type = context_type.__args__[1] if len(context_type.__args__) > 1 else Any
        else:
            logger.warning(f"Could not determine request type for middleware {middleware_name}")
            request_type = BaseModel
            response_type = Any
        
        # Create input schema
        input_schema = get_schema(request_type)
        
        # Define the tool handler
        async def tool_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
            # Create a request instance
            if isinstance(request_type, type) and issubclass(request_type, BaseModel):
                request = request_type(**arguments)
            else:
                request = arguments
                
            # Create middleware context and process it
            context = MiddlewareContext[Any, Any](request=request)
            result_context = await middleware.process(context)
            
            if result_context.response_generated and result_context.response is not None:
                if hasattr(result_context.response, "model_dump"):
                    return result_context.response.model_dump()
                elif hasattr(result_context.response, "dict"):
                    return result_context.response.dict()
                else:
                    return {"result": result_context.response}
            else:
                return {"error": "No response generated", "metadata": result_context.metadata}
        
        # Register the tool with MCP server
        self.mcp_server.register_tool(
            name=tool_name,
            description=f"Execute the {middleware_name} middleware component",
            schema=input_schema,
            handler=tool_handler
        )
        
        self.registered_tools[tool_name] = {
            "type": "middleware",
            "name": middleware_name,
            "instance": middleware
        }
        
        logger.info(f"Registered middleware as MCP tool: {tool_name}")
            
    def register_workflow(self, workflow: Workflow) -> None:
        """
        Register workflow components as MCP tools.
        
        Args:
            workflow: The workflow to register
        """
        workflow_name = workflow.name
        tool_name = f"workflow_{workflow_name}"
        
        # Get the workflow's input schema
        schema_type = workflow.schema
        input_schema = get_schema(schema_type)
        
        # Define the tool handler
        async def tool_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
            # Create a state instance from the arguments
            state = schema_type(**arguments)
            
            # Run the workflow
            run = workflow.run(state)
            result = await run
            
            # Return the workflow result
            if result.result and hasattr(result.result, "model_dump"):
                return result.result.model_dump()
            elif result.result and hasattr(result.result, "dict"):
                return result.result.dict()
            else:
                return {"steps": [s.model_dump() for s in result.steps], "state": result.state.model_dump()}
        
        # Register the tool with MCP server
        self.mcp_server.register_tool(
            name=tool_name,
            description=f"Execute the {workflow_name} workflow",
            schema=input_schema,
            handler=tool_handler
        )
        
        self.registered_tools[tool_name] = {
            "type": "workflow",
            "name": workflow_name,
            "instance": workflow
        }
        
        logger.info(f"Registered workflow as MCP tool: {tool_name}")
    
    def register_function(self, 
                         func: Callable, 
                         name: Optional[str] = None, 
                         description: Optional[str] = None,
                         input_model: Optional[Type[BaseModel]] = None) -> None:
        """
        Register a Python function as an MCP tool.
        
        Args:
            func: The function to register
            name: Custom name for the tool (defaults to function name)
            description: Description of what the tool does
            input_model: Optional Pydantic model defining the input schema
        """
        func_name = name or func.__name__
        tool_name = f"function_{func_name}"
        
        # Get or create input schema
        if input_model:
            input_schema = get_schema(input_model)
        else:
            # Create input model based on function signature
            sig = inspect.signature(func)
            fields = {}
            
            for param_name, param in sig.parameters.items():
                if param.name == 'self' or param.name == 'cls':
                    continue
                    
                param_type = param.annotation if param.annotation is not inspect.Parameter.empty else Any
                default = param.default if param.default is not inspect.Parameter.empty else ...
                
                fields[param_name] = (param_type, default)
            
            dynamic_model = create_model(f"{func_name}Input", **fields)
            input_schema = get_schema(dynamic_model)
        
        # Define the tool handler
        async def tool_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
            # Call the function
            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)
                
            # Process the result
            if hasattr(result, "model_dump"):
                return result.model_dump()
            elif hasattr(result, "dict"):
                return result.dict()
            elif isinstance(result, dict):
                return result
            else:
                return {"result": result}
        
        # Register the tool with MCP server
        self.mcp_server.register_tool(
            name=tool_name,
            description=description or f"Execute the {func_name} function",
            schema=input_schema,
            handler=tool_handler
        )
        
        self.registered_tools[tool_name] = {
            "type": "function",
            "name": func_name,
            "instance": func
        }
        
        logger.info(f"Registered function as MCP tool: {tool_name}")
    
    def start(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        """
        Start the MCP server.
        
        Args:
            host: The host to bind the server to
            port: The port to bind the server to
        """
        logger.info(f"Starting BeeAI MCP server on {host}:{port}")
        self.mcp_server.start(host=host, port=port)
    
    async def stop(self) -> None:
        """Stop the MCP server."""
        logger.info("Stopping BeeAI MCP server")
        await self.mcp_server.stop() 