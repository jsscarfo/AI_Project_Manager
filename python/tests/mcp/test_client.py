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
Tests for the BeeAI MCP client implementation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

from beeai_framework.mcp.client import BeeAIMCPClient


@pytest.fixture
def mock_client_session():
    """Create a mock MCP client session."""
    with patch("beeai_framework.mcp.client.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.connect = AsyncMock(return_value=mock_session)
        
        # Configure mock list_tools response
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Tool 1 description"
        mock_tool1.inputSchema = {"type": "object", "properties": {"input": {"type": "string"}}}
        
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Tool 2 description"
        mock_tool2.inputSchema = {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}
        
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool1, mock_tool2]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        
        # Configure mock call_tool response
        mock_call_result = MagicMock()
        mock_call_result.content = {"result": "Tool result"}
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)
        
        # Configure mock call_tool_streaming response
        async def streaming_mock(*args, **kwargs):
            chunk1 = MagicMock()
            chunk1.content = "Chunk 1"
            yield chunk1
            
            chunk2 = MagicMock()
            chunk2.content = "Chunk 2"
            yield chunk2
            
            chunk3 = MagicMock()
            chunk3.content = "Chunk 3"
            yield chunk3
            
        mock_session.call_tool_streaming = streaming_mock
        
        # Configure close method
        mock_session.close = AsyncMock()
        
        yield mock_session


@pytest.fixture
def beeai_mcp_client():
    """Create a BeeAI MCP client."""
    return BeeAIMCPClient("http://example.com/mcp")


@pytest.mark.asyncio
async def test_client_connect(beeai_mcp_client, mock_client_session):
    """Test client connection."""
    await beeai_mcp_client.connect()
    
    # Verify the client session was established
    assert beeai_mcp_client.client_session is not None
    assert len(beeai_mcp_client.tools) == 2
    
    # Check the tool names
    tool_names = beeai_mcp_client.get_tool_names()
    assert "tool1" in tool_names
    assert "tool2" in tool_names


@pytest.mark.asyncio
async def test_client_disconnect(beeai_mcp_client, mock_client_session):
    """Test client disconnection."""
    await beeai_mcp_client.connect()
    assert beeai_mcp_client.client_session is not None
    
    await beeai_mcp_client.disconnect()
    assert beeai_mcp_client.client_session is None
    assert len(beeai_mcp_client.tools) == 0
    mock_client_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_call_tool(beeai_mcp_client, mock_client_session):
    """Test calling a tool."""
    await beeai_mcp_client.connect()
    
    result = await beeai_mcp_client.call_tool("tool1", input="test")
    
    mock_client_session.call_tool.assert_called_once_with(
        name="tool1",
        arguments={"input": "test"}
    )
    assert result == {"result": "Tool result"}


@pytest.mark.asyncio
async def test_call_tool_streaming(beeai_mcp_client, mock_client_session):
    """Test calling a tool with streaming."""
    await beeai_mcp_client.connect()
    
    chunks = []
    async for chunk in beeai_mcp_client.call_tool_streaming("tool1", input="test"):
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]


@pytest.mark.asyncio
async def test_get_tool(beeai_mcp_client, mock_client_session):
    """Test getting a tool by name."""
    await beeai_mcp_client.connect()
    
    tool = beeai_mcp_client.get_tool("tool1")
    assert tool is not None
    assert tool.name == "tool1"
    
    missing_tool = beeai_mcp_client.get_tool("non_existent_tool")
    assert missing_tool is None


@pytest.mark.asyncio
async def test_get_tools(beeai_mcp_client, mock_client_session):
    """Test getting all tools."""
    await beeai_mcp_client.connect()
    
    tools = beeai_mcp_client.get_tools()
    assert len(tools) == 2
    assert tools[0].name == "tool1"
    assert tools[1].name == "tool2"


@pytest.mark.asyncio
async def test_context_manager(mock_client_session):
    """Test using the client as a context manager."""
    client = BeeAIMCPClient("http://example.com/mcp")
    
    async with client as c:
        assert c is client
        assert client.client_session is not None
        
        # Call a tool within the context
        result = await client.call_tool("tool1", input="test")
        assert result == {"result": "Tool result"}
    
    # After exiting the context, the client should be disconnected
    assert client.client_session is None
    mock_client_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling(beeai_mcp_client):
    """Test error handling when not connected."""
    with pytest.raises(ValueError, match="Not connected to MCP server"):
        await beeai_mcp_client.call_tool("tool1", input="test")
        
    with pytest.raises(ValueError, match="Not connected to MCP server"):
        async for chunk in beeai_mcp_client.call_tool_streaming("tool1", input="test"):
            pass 