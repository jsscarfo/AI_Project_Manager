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
Tests for the BeeAI MCP adapters implementations.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional, cast

from pydantic import BaseModel, Field

from beeai_framework.middleware.base import MiddlewareContext
from beeai_framework.mcp.client import BeeAIMCPClient
from beeai_framework.mcp.adapters.middleware import MCPMiddlewareAdapter, MCPStreamingMiddlewareAdapter
from beeai_framework.mcp.adapters.workflow import MCPWorkflowAdapter, MCPStepAdapter
from beeai_framework.workflows.workflow import Workflow


# Test models
class TestRequest(BaseModel):
    """Test request model."""
    query: str = Field(..., description="Query string")
    limit: int = Field(10, description="Limit for results")


class TestResponse(BaseModel):
    """Test response model."""
    results: List[str] = Field(default_factory=list, description="Results list")
    status: str = Field("ok", description="Status of response")


class TestWorkflowState(BaseModel):
    """Test workflow state model."""
    input: str = Field(..., description="Input string")
    output: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client."""
    mock_client = MagicMock(spec=BeeAIMCPClient)
    mock_client.call_tool = AsyncMock()
    
    # Set up a standard response for the middleware test
    mock_client.call_tool.return_value = {
        "results": ["result1", "result2", "result3"],
        "status": "success"
    }
    
    # Set up a streaming response for the streaming middleware test
    async def mock_streaming(*args, **kwargs):
        yield "Chunk 1"
        yield "Chunk 2"
        yield "Chunk 3"
    
    mock_client.call_tool_streaming = mock_streaming
    
    return mock_client


@pytest.mark.asyncio
async def test_mcp_middleware_adapter(mock_mcp_client):
    """Test the MCP middleware adapter."""
    # Create the adapter
    adapter = MCPMiddlewareAdapter(
        client=mock_mcp_client,
        tool_name="search_tool",
        request_type=TestRequest,
        response_type=TestResponse
    )
    
    # Create a test request and context
    request = TestRequest(query="test query", limit=5)
    context = MiddlewareContext[TestRequest, TestResponse](request=request)
    
    # Process the request
    result_context = await adapter.process(context)
    
    # Verify the request was properly handled
    mock_mcp_client.call_tool.assert_called_once_with(
        "search_tool",
        query="test query",
        limit=5
    )
    
    # Verify the response was properly set
    assert result_context.response_generated
    assert result_context.response is not None
    assert result_context.response.results == ["result1", "result2", "result3"]
    assert result_context.response.status == "success"
    
    # Verify the adapter name
    assert adapter.name == "mcp_adapter_search_tool"


@pytest.mark.asyncio
async def test_mcp_middleware_adapter_error_handling(mock_mcp_client):
    """Test error handling in the MCP middleware adapter."""
    # Configure the mock to raise an exception
    mock_mcp_client.call_tool.side_effect = ValueError("Test error")
    
    # Create the adapter
    adapter = MCPMiddlewareAdapter(
        client=mock_mcp_client,
        tool_name="search_tool",
        request_type=TestRequest,
        response_type=TestResponse
    )
    
    # Create a test request and context
    request = TestRequest(query="test query", limit=5)
    context = MiddlewareContext[TestRequest, TestResponse](request=request)
    
    # Process the request
    result_context = await adapter.process(context)
    
    # Verify the error was properly handled
    assert not result_context.response_generated
    assert "error" in result_context.metadata
    assert result_context.metadata["error"] == "Test error"


@pytest.mark.asyncio
async def test_mcp_streaming_middleware_adapter(mock_mcp_client):
    """Test the MCP streaming middleware adapter."""
    # Create the adapter
    adapter = MCPStreamingMiddlewareAdapter(
        client=mock_mcp_client,
        tool_name="stream_tool",
        request_type=TestRequest
    )
    
    # Create a test request and context
    request = TestRequest(query="test query", limit=5)
    context = MiddlewareContext[TestRequest, List[str]](request=request)
    
    # Create a mock emitter for the adapter
    mock_emitter = MagicMock()
    adapter.emitter = mock_emitter
    
    # Process the request
    result_context = await adapter.process(context)
    
    # Verify the response was properly set
    assert result_context.response_generated
    assert result_context.response is not None
    assert len(result_context.response) == 3
    assert result_context.response == ["Chunk 1", "Chunk 2", "Chunk 3"]
    
    # Verify events were emitted for each chunk
    assert mock_emitter.emit.call_count == 3
    mock_emitter.emit.assert_any_call("chunk", {"chunk": "Chunk 1", "index": 0})
    mock_emitter.emit.assert_any_call("chunk", {"chunk": "Chunk 2", "index": 1})
    mock_emitter.emit.assert_any_call("chunk", {"chunk": "Chunk 3", "index": 2})


@pytest.mark.asyncio
async def test_mcp_workflow_adapter(mock_mcp_client):
    """Test the MCP workflow adapter."""
    # Set up a response for the workflow test
    mock_mcp_client.call_tool.return_value = {
        "output": "Processed: test input",
        "metadata": {"processing_time": 0.5}
    }
    
    # Create the adapter
    adapter = MCPWorkflowAdapter(
        client=mock_mcp_client,
        tool_name="process_tool",
        schema=TestWorkflowState
    )
    
    # Create a test state
    state = TestWorkflowState(input="test input")
    
    # Run the workflow
    run = adapter.run(state)
    result = await run
    
    # Verify the MCP tool was called
    mock_mcp_client.call_tool.assert_called_once_with(
        "process_tool",
        input="test input"
    )
    
    # Verify the state was updated correctly
    assert result.state.output == "Processed: test input"
    assert result.state.metadata["processing_time"] == 0.5
    
    # Verify the workflow structure
    assert adapter.name == "mcp_process_tool"
    assert "execute" in adapter.steps
    assert adapter.start_step == "execute"


@pytest.mark.asyncio
async def test_mcp_workflow_adapter_error_handling(mock_mcp_client):
    """Test error handling in the MCP workflow adapter."""
    # Configure the mock to raise an exception
    mock_mcp_client.call_tool.side_effect = ValueError("Test workflow error")
    
    # Create the adapter
    adapter = MCPWorkflowAdapter(
        client=mock_mcp_client,
        tool_name="process_tool",
        schema=TestWorkflowState
    )
    
    # Create a test state
    state = TestWorkflowState(input="test input")
    
    # Run the workflow
    run = adapter.run(state)
    result = await run
    
    # Verify the error was properly handled
    assert result.state.error == "Test workflow error"


@pytest.mark.asyncio
async def test_mcp_step_adapter(mock_mcp_client):
    """Test the MCP step adapter."""
    # Set up a response for the step test
    mock_mcp_client.call_tool.return_value = {
        "output": "Processed by step: test input",
        "metadata": {"step_time": 0.3}
    }
    
    # Create a step handler
    step_handler = MCPStepAdapter.create_step(
        client=mock_mcp_client,
        tool_name="step_tool"
    )
    
    # Create a test state
    state = TestWorkflowState(input="test input")
    
    # Execute the step
    next_step = await step_handler(state)
    
    # Verify the MCP tool was called
    mock_mcp_client.call_tool.assert_called_once_with(
        "step_tool",
        input="test input"
    )
    
    # Verify the state was updated correctly
    assert state.output == "Processed by step: test input"
    assert state.metadata["step_time"] == 0.3
    
    # Verify the next step is returned
    assert next_step == Workflow.NEXT


@pytest.mark.asyncio
async def test_mcp_step_adapter_error_handling(mock_mcp_client):
    """Test error handling in the MCP step adapter."""
    # Configure the mock to raise an exception
    mock_mcp_client.call_tool.side_effect = ValueError("Test step error")
    
    # Create a step handler
    step_handler = MCPStepAdapter.create_step(
        client=mock_mcp_client,
        tool_name="step_tool"
    )
    
    # Create a test state
    state = TestWorkflowState(input="test input")
    
    # Execute the step
    next_step = await step_handler(state)
    
    # Verify the error was properly handled
    assert state.error == "Test step error"
    
    # Verify the next step is still returned (should continue the workflow)
    assert next_step == Workflow.NEXT 