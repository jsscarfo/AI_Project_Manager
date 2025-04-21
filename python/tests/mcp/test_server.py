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
Tests for the BeeAI MCP server implementation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from beeai_framework.middleware.base import Middleware, MiddlewareContext, MiddlewareConfig
from beeai_framework.workflows.workflow import Workflow
from beeai_framework.mcp.server import BeeAIMCPServer


# Test models and components
class TestRequest(BaseModel):
    """Test request model."""
    input: str = Field(..., description="Input string")


class TestResponse(BaseModel):
    """Test response model."""
    output: str = Field(..., description="Output string")


class TestMiddleware(Middleware[TestRequest, TestResponse]):
    """Test middleware implementation."""
    
    async def process(self, context: MiddlewareContext[TestRequest, TestResponse]) -> MiddlewareContext[TestRequest, TestResponse]:
        """Process the test request."""
        request = context.request
        response = TestResponse(output=f"Processed: {request.input}")
        return context.set_response(response)


class TestWorkflowState(BaseModel):
    """Test workflow state model."""
    data: str = Field(..., description="Input data")
    result: Optional[str] = None


class TestWorkflow(Workflow[TestWorkflowState, str]):
    """Test workflow implementation."""
    
    def __init__(self):
        super().__init__(TestWorkflowState, name="TestWorkflow")
        self.add_step("process", self._process)
        self.set_start("process")
    
    async def _process(self, state: TestWorkflowState) -> str:
        """Process the workflow state."""
        state.result = f"Workflow result: {state.data}"
        return Workflow.END


@pytest.fixture
def mock_mcp_server():
    """Create a mock MCP server."""
    with patch("beeai_framework.mcp.server.MCPServer") as mock_server_class:
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        mock_server.register_tool = AsyncMock()
        mock_server.start = MagicMock()
        mock_server.stop = AsyncMock()
        yield mock_server


@pytest.fixture
def beeai_mcp_server(mock_mcp_server):
    """Create a BeeAI MCP server with a mock MCP server."""
    return BeeAIMCPServer(name="TestServer")


def test_server_initialization(beeai_mcp_server):
    """Test server initialization."""
    assert beeai_mcp_server.name == "TestServer"
    assert beeai_mcp_server.registered_tools == {}


def test_register_middleware(beeai_mcp_server, mock_mcp_server):
    """Test registering middleware components."""
    middleware = TestMiddleware()
    beeai_mcp_server.register_middleware(middleware)
    
    # Verify that the middleware was registered
    tool_name = f"middleware_{middleware.name}"
    assert tool_name in beeai_mcp_server.registered_tools
    assert beeai_mcp_server.registered_tools[tool_name]["type"] == "middleware"
    assert beeai_mcp_server.registered_tools[tool_name]["name"] == middleware.name
    assert beeai_mcp_server.registered_tools[tool_name]["instance"] == middleware
    
    # Verify that the MCP server was called to register the tool
    mock_mcp_server.register_tool.assert_called_once()
    assert mock_mcp_server.register_tool.call_args[1]["name"] == tool_name


def test_register_workflow(beeai_mcp_server, mock_mcp_server):
    """Test registering workflow components."""
    workflow = TestWorkflow()
    beeai_mcp_server.register_workflow(workflow)
    
    # Verify that the workflow was registered
    tool_name = f"workflow_{workflow.name}"
    assert tool_name in beeai_mcp_server.registered_tools
    assert beeai_mcp_server.registered_tools[tool_name]["type"] == "workflow"
    assert beeai_mcp_server.registered_tools[tool_name]["name"] == workflow.name
    assert beeai_mcp_server.registered_tools[tool_name]["instance"] == workflow
    
    # Verify that the MCP server was called to register the tool
    mock_mcp_server.register_tool.assert_called_once()
    assert mock_mcp_server.register_tool.call_args[1]["name"] == tool_name


def test_register_function(beeai_mcp_server, mock_mcp_server):
    """Test registering function components."""
    def test_function(param1: str, param2: int = 42) -> Dict[str, Any]:
        return {"result": f"{param1}: {param2}"}
    
    beeai_mcp_server.register_function(test_function, description="Test function")
    
    # Verify that the function was registered
    tool_name = f"function_{test_function.__name__}"
    assert tool_name in beeai_mcp_server.registered_tools
    assert beeai_mcp_server.registered_tools[tool_name]["type"] == "function"
    assert beeai_mcp_server.registered_tools[tool_name]["name"] == test_function.__name__
    assert beeai_mcp_server.registered_tools[tool_name]["instance"] == test_function
    
    # Verify that the MCP server was called to register the tool
    mock_mcp_server.register_tool.assert_called_once()
    assert mock_mcp_server.register_tool.call_args[1]["name"] == tool_name
    assert mock_mcp_server.register_tool.call_args[1]["description"] == "Test function"


def test_server_start_stop(beeai_mcp_server, mock_mcp_server):
    """Test starting and stopping the server."""
    # Test start
    beeai_mcp_server.start(host="localhost", port=8000)
    mock_mcp_server.start.assert_called_once_with(host="localhost", port=8000)
    
    # Test stop
    asyncio.run(beeai_mcp_server.stop())
    mock_mcp_server.stop.assert_called_once() 