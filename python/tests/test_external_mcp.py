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
External MCP Tests

Unit tests for the BeeAI External MCP server and client.
"""

import os
import asyncio
import pytest
from typing import Dict, Any, Optional, List
from unittest.mock import patch, MagicMock

from pydantic import BaseModel

from beeai_framework.middleware.base import Middleware, MiddlewareContext, MiddlewareConfig
from beeai_framework.mcp.external_server import BeeAIExternalMCPServer, ToolRegistration
from beeai_framework.mcp.external_client import BeeAIExternalMCPClient
from beeai_framework.mcp.external_tools import ExternalToolsRegistry, ExternalToolConfig


# Test middleware for external MCP testing
class TestRequest(BaseModel):
    """Test request model."""
    message: str
    value: int = 0


class TestResponse(BaseModel):
    """Test response model."""
    result: str
    processed_value: int


class TestMiddleware(Middleware[TestRequest, TestResponse]):
    """Test middleware for external MCP testing."""
    
    def __init__(self, config: Optional[MiddlewareConfig] = None):
        """Initialize the middleware."""
        super().__init__(config)
    
    async def process(self, context: MiddlewareContext[TestRequest, TestResponse]) -> MiddlewareContext[TestRequest, TestResponse]:
        """Process the request."""
        request = context.request
        response = TestResponse(
            result=f"Processed: {request.message}",
            processed_value=request.value * 2
        )
        context.set_response(response)
        return context
    
    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "test_middleware"


# Tests for ExternalToolsRegistry
class TestExternalToolsRegistry:
    """Tests for the ExternalToolsRegistry class."""
    
    def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        registry = ExternalToolsRegistry()
        
        # Register a tool
        tool_config = ExternalToolConfig(
            name="test_tool",
            description="Test tool",
            source_tool="middleware_test",
            required_roles=["user"]
        )
        registry.register(tool_config)
        
        # Get the tool
        retrieved_tool = registry.get("test_tool")
        assert retrieved_tool is not None
        assert retrieved_tool.name == "test_tool"
        assert retrieved_tool.source_tool == "middleware_test"
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ExternalToolsRegistry()
        
        # Register a tool
        tool_config = ExternalToolConfig(
            name="test_tool",
            description="Test tool",
            source_tool="middleware_test"
        )
        registry.register(tool_config)
        
        # Unregister the tool
        registry.unregister("test_tool")
        
        # Verify it's gone
        assert registry.get("test_tool") is None
    
    def test_list_tools(self):
        """Test listing all tools."""
        registry = ExternalToolsRegistry()
        
        # Register multiple tools
        registry.register(ExternalToolConfig(
            name="tool1",
            description="Tool 1",
            source_tool="middleware_1"
        ))
        
        registry.register(ExternalToolConfig(
            name="tool2",
            description="Tool 2",
            source_tool="middleware_2"
        ))
        
        # List tools
        tools = registry.list_tools()
        assert len(tools) == 2
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"tool1", "tool2"}


# Tests for BeeAIExternalMCPServer (with mocked FastAPI components)
@pytest.mark.asyncio
class TestExternalMCPServer:
    """Tests for the BeeAIExternalMCPServer class."""
    
    @pytest.fixture
    async def server(self):
        """Fixture to create a server with test middleware."""
        with patch("beeai_framework.mcp.external_server.FastAPI"):
            server = BeeAIExternalMCPServer(
                name="TestServer",
                secret_key="test_secret"
            )
            
            # Add a test middleware
            middleware = TestMiddleware()
            server.register_middleware(middleware)
            
            yield server
    
    async def test_register_external_tool(self, server):
        """Test registering an external tool."""
        # Register external tool
        server.register_external_tool(
            name="test_external",
            description="Test external tool",
            source_tool="middleware_test_middleware",
            required_roles=["user", "admin"],
            metadata={"category": "test"}
        )
        
        # Check if registered
        assert "test_external" in server.tool_registrations
        tool_reg = server.tool_registrations["test_external"]
        assert tool_reg.description == "Test external tool"
        assert tool_reg.required_roles == ["user", "admin"]
        assert tool_reg.metadata["category"] == "test"
        assert tool_reg.metadata["source_tool"] == "middleware_test_middleware"
    
    async def test_add_user(self, server):
        """Test adding a user."""
        # Add a user
        user = server.add_user(
            username="testuser",
            password="password123",
            email="test@example.com",
            roles=["user"]
        )
        
        # Check if user was added
        assert "testuser" in server.users
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        
        # Password should be hashed
        stored_user = server.users["testuser"]
        assert stored_user.hashed_password != "password123"
        assert server._verify_password("password123", stored_user.hashed_password)
    
    async def test_create_api_key(self, server):
        """Test creating an API key."""
        # Add a user first
        server.add_user(
            username="keyuser",
            password="password123",
            roles=["user"]
        )
        
        # Create API key
        api_key = server.create_api_key(
            name="Test Key",
            user_id="keyuser",
            expiry_days=30,
            rate_limit=100
        )
        
        # Check if API key was created
        assert api_key.key in server.api_keys
        assert api_key.user_id == "keyuser"
        assert api_key.name == "Test Key"
        assert api_key.rate_limit == 100
        assert api_key.expires_at is not None
    
    async def test_authenticate_api_key(self, server):
        """Test authenticating with an API key."""
        # Add a user and API key
        server.add_user(
            username="authuser",
            password="password123",
            roles=["user"]
        )
        
        api_key = server.create_api_key(
            name="Auth Key",
            user_id="authuser"
        )
        
        # Authenticate with API key
        user = server._authenticate_api_key(api_key.key)
        assert user is not None
        assert user.username == "authuser"
    
    async def test_has_access(self, server):
        """Test access control."""
        from beeai_framework.mcp.external_server import User
        
        # Create test users
        admin_user = User(username="admin", roles=["admin"])
        regular_user = User(username="regular", roles=["user"])
        multi_role_user = User(username="multi", roles=["user", "editor"])
        
        # Test access
        assert server._has_access(admin_user, ["user"])  # Admin has access to everything
        assert server._has_access(regular_user, ["user"])
        assert not server._has_access(regular_user, ["admin"])
        assert server._has_access(multi_role_user, ["editor"])
        assert server._has_access(multi_role_user, ["admin", "editor"])  # Has one of the roles
        assert not server._has_access(regular_user, ["admin", "editor"])


# Tests for BeeAIExternalMCPClient (with mocked HTTP client)
@pytest.mark.asyncio
class TestExternalMCPClient:
    """Tests for the BeeAIExternalMCPClient class."""
    
    @pytest.fixture
    async def client(self):
        """Fixture to create a client with mocked HTTP session."""
        client = BeeAIExternalMCPClient(
            url="http://test.example.com",
            api_key="test_api_key"
        )
        
        # Mock the session and responses
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()
        
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__.return_value = None
        
        mock_session.request.return_value = mock_context
        mock_session.post.return_value = mock_context
        mock_session.get.return_value = mock_context
        
        # Patch the session creation
        with patch("aiohttp.ClientSession", return_value=mock_session):
            client.session = mock_session
            yield client, mock_session
    
    async def test_call_tool(self, client):
        """Test calling a tool."""
        client_obj, mock_session = client
        
        # Setup response for tool call
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": "Tool called",
            "value": 42
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__.return_value = None
        
        mock_session.request.return_value = mock_context
        
        # Call a tool
        result = await client_obj.call_tool("test_tool", param1="value1", param2=2)
        
        # Check result
        assert result["result"] == "Tool called"
        assert result["value"] == 42
        
        # Verify request was made with correct parameters
        mock_session.request.assert_called_once()
        args, kwargs = mock_session.request.call_args
        assert args[0] == "POST"
        assert args[1] == "http://test.example.com/tools/test_tool"
        assert "X-API-Key" in kwargs["headers"]
        assert kwargs["headers"]["X-API-Key"] == "test_api_key"
    
    async def test_sequential_thinking_component(self, client):
        """Test sequential thinking component client."""
        client_obj, mock_session = client
        
        # Setup response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "reasoning_steps": [
                {"number": 1, "title": "Step 1", "content": "Step 1 content"},
                {"number": 2, "title": "Step 2", "content": "Step 2 content"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__.return_value = None
        
        mock_session.request.return_value = mock_context
        
        # Call the sequential thinking component
        result = await client_obj.sequential_thinking.solve(
            problem="Test problem",
            steps=2
        )
        
        # Check result
        assert "reasoning_steps" in result
        assert len(result["reasoning_steps"]) == 2
        
        # Verify request was made correctly
        mock_session.request.assert_called_once()
        args, kwargs = mock_session.request.call_args
        assert args[0] == "POST"
        assert args[1] == "http://test.example.com/tools/sequential_thinking_solve"
        assert kwargs["json"]["problem"] == "Test problem"
        assert kwargs["json"]["steps"] == 2
    
    async def test_vector_memory_component(self, client):
        """Test vector memory component client."""
        client_obj, mock_session = client
        
        # Setup response for store
        store_response = MagicMock()
        store_response.json.return_value = {"ids": ["doc1", "doc2"]}
        store_response.raise_for_status = MagicMock()
        
        store_context = MagicMock()
        store_context.__aenter__.return_value = store_response
        store_context.__aexit__.return_value = None
        
        # Setup response for search
        search_response = MagicMock()
        search_response.json.return_value = [
            {"id": "doc1", "score": 0.95, "text": "Document 1 text"},
            {"id": "doc2", "score": 0.85, "text": "Document 2 text"}
        ]
        search_response.raise_for_status = MagicMock()
        
        search_context = MagicMock()
        search_context.__aenter__.return_value = search_response
        search_context.__aexit__.return_value = None
        
        # Set up mock session to return different responses for different calls
        mock_session.request.side_effect = [store_context, search_context]
        
        # Test store
        store_result = await client_obj.vector_memory.store(
            collection="test_collection",
            documents=[{"text": "Doc 1"}, {"text": "Doc 2"}]
        )
        
        assert "ids" in store_result
        assert len(store_result["ids"]) == 2
        
        # Test search
        search_result = await client_obj.vector_memory.search(
            collection="test_collection",
            query="test query"
        )
        
        assert len(search_result) == 2
        assert search_result[0]["score"] == 0.95
        
        # Verify correct calls were made
        assert mock_session.request.call_count == 2
        
        # Check first call (store)
        args1, kwargs1 = mock_session.request.call_args_list[0]
        assert args1[0] == "POST"
        assert args1[1] == "http://test.example.com/tools/vector_memory_store"
        assert kwargs1["json"]["collection"] == "test_collection"
        
        # Check second call (search)
        args2, kwargs2 = mock_session.request.call_args_list[1]
        assert args2[0] == "POST"
        assert args2[1] == "http://test.example.com/tools/vector_memory_search"
        assert kwargs2["json"]["collection"] == "test_collection"
        assert kwargs2["json"]["query"] == "test query"


# Run the tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 