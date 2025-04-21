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
BeeAI External MCP Client Module

This module provides a client SDK for interacting with the BeeAI External MCP Server,
allowing external applications to access BeeAI capabilities through a simple interface.
"""

import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Callable, AsyncIterator, Tuple, Set

try:
    import aiohttp
    from pydantic import BaseModel, Field
except ModuleNotFoundError as e:
    required_modules = ["aiohttp", "pydantic"]
    raise ModuleNotFoundError(
        f"Required modules not found: {required_modules}. "
        f"Run 'pip install beeai-framework[mcp-client]' to install."
    ) from e

T = TypeVar('T', bound=BaseModel)


class BeeAIExternalMCPClient:
    """Client SDK for interacting with BeeAI External MCP Server."""
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 30.0,
        logging_level: int = logging.INFO
    ):
        """
        Initialize the BeeAI External MCP client.
        
        Args:
            url: The URL of the BeeAI External MCP server
            api_key: Optional API key for authentication
            username: Optional username for OAuth authentication
            password: Optional password for OAuth authentication
            timeout: Request timeout in seconds
            logging_level: Logging level (from logging module)
        """
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.username = username
        self.password = password
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_type: Optional[str] = None
        
        # Setup logging
        self.logger = logging.getLogger("beeai_external_mcp_client")
        self.logger.setLevel(logging_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"Initialized BeeAI External MCP client for server: {url}")
        
        # Initialize component clients
        self.sequential_thinking = SequentialThinkingClient(self)
        self.vector_memory = VectorMemoryClient(self)
        self.middleware = MiddlewareClient(self)
        self.workflows = WorkflowClient(self)
        self.visualization = VisualizationClient(self)
    
    async def connect(self) -> None:
        """
        Connect to the server and authenticate if credentials are provided.
        
        This method creates an aiohttp client session and performs OAuth authentication
        if username and password are provided.
        """
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        if self.username and self.password:
            await self._authenticate()
            
        self.logger.info("Connected to BeeAI External MCP server")
    
    async def disconnect(self) -> None:
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("Disconnected from BeeAI External MCP server")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of available tools from the server.
        
        Returns:
            List of tool metadata dictionaries
        """
        return await self._get("/tools")
    
    async def call_tool(self, tool_name: str, **arguments) -> Any:
        """
        Call a tool on the server.
        
        Args:
            tool_name: Name of the tool to call
            **arguments: Arguments to pass to the tool
            
        Returns:
            The result of the tool execution
        """
        self.logger.debug(f"Calling tool {tool_name} with arguments: {arguments}")
        return await self._post(f"/tools/{tool_name}", json=arguments)
    
    async def _authenticate(self) -> None:
        """Authenticate with the server using OAuth."""
        if not self.session:
            raise RuntimeError("Client session not initialized. Call connect() first.")
        
        if not (self.username and self.password):
            self.logger.warning("Cannot authenticate: username or password not provided")
            return
        
        self.logger.debug("Authenticating with OAuth")
        
        form_data = {
            "username": self.username,
            "password": self.password,
        }
        
        try:
            async with self.session.post(
                f"{self.url}/token",
                data=form_data,
                raise_for_status=True
            ) as response:
                token_data = await response.json()
                self.access_token = token_data["access_token"]
                self.token_type = token_data["token_type"]
                self.logger.debug("Authentication successful")
        
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            raise
    
    async def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Any:
        """
        Make a request to the server.
        
        Args:
            method: HTTP method
            path: API endpoint path
            **kwargs: Additional arguments to pass to the request
            
        Returns:
            Parsed response JSON
        """
        if not self.session:
            await self.connect()
            
        # Ensure session is available
        if not self.session:
            raise RuntimeError("Failed to create client session")
        
        headers = kwargs.pop("headers", {})
        
        # Add authentication
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.access_token:
            headers["Authorization"] = f"{self.token_type} {self.access_token}"
        
        url = f"{self.url}{path}"
        self.logger.debug(f"Making {method} request to {url}")
        
        try:
            async with self.session.request(
                method, url, headers=headers, **kwargs
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"Request failed: {str(e)}")
            
            # Try to get error details from response
            try:
                error_data = await e.response.json()
                detail = error_data.get("detail", str(e))
                raise RuntimeError(f"API error: {detail}")
            except:
                raise RuntimeError(f"API error: {str(e)}")
                
        except aiohttp.ClientError as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise RuntimeError(f"Connection error: {str(e)}")
    
    async def _get(self, path: str, **kwargs) -> Any:
        """Make a GET request."""
        return await self._request("GET", path, **kwargs)
    
    async def _post(self, path: str, **kwargs) -> Any:
        """Make a POST request."""
        return await self._request("POST", path, **kwargs)
    
    async def _put(self, path: str, **kwargs) -> Any:
        """Make a PUT request."""
        return await self._request("PUT", path, **kwargs)
    
    async def _delete(self, path: str, **kwargs) -> Any:
        """Make a DELETE request."""
        return await self._request("DELETE", path, **kwargs)
    
    async def __aenter__(self) -> "BeeAIExternalMCPClient":
        """Support for async context manager."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Support for async context manager."""
        await self.disconnect()


class SequentialThinkingClient:
    """Client component for sequential thinking capabilities."""
    
    def __init__(self, client: BeeAIExternalMCPClient):
        """
        Initialize the client component.
        
        Args:
            client: Parent client instance
        """
        self.client = client
    
    async def solve(
        self,
        problem: str,
        steps: int = 5,
        context: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        requirements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using sequential thinking.
        
        Args:
            problem: Problem description
            steps: Number of reasoning steps
            context: Additional context
            domain: Problem domain
            requirements: Problem requirements
            
        Returns:
            Solution object with reasoning steps
        """
        arguments = {
            "problem": problem,
            "steps": steps,
        }
        
        if context:
            arguments["context"] = context
        
        if domain:
            arguments["domain"] = domain
            
        if requirements:
            arguments["requirements"] = requirements
        
        return await self.client.call_tool("sequential_thinking_solve", **arguments)
    
    async def analyze(self, text: str, depth: int = 3) -> Dict[str, Any]:
        """
        Analyze text using sequential thinking.
        
        Args:
            text: Text to analyze
            depth: Analysis depth
            
        Returns:
            Analysis result
        """
        return await self.client.call_tool("sequential_thinking_analyze", text=text, depth=depth)


class VectorMemoryClient:
    """Client component for vector memory operations."""
    
    def __init__(self, client: BeeAIExternalMCPClient):
        """
        Initialize the client component.
        
        Args:
            client: Parent client instance
        """
        self.client = client
    
    async def store(
        self,
        collection: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store documents in vector memory.
        
        Args:
            collection: Collection name
            documents: List of documents to store
            
        Returns:
            Result with stored document IDs
        """
        return await self.client.call_tool(
            "vector_memory_store",
            collection=collection,
            documents=documents
        )
    
    async def search(
        self,
        collection: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search vector memory.
        
        Args:
            collection: Collection name
            query: Search query
            filters: Optional filters
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        arguments = {
            "collection": collection,
            "query": query,
            "limit": limit
        }
        
        if filters:
            arguments["filters"] = filters
        
        return await self.client.call_tool("vector_memory_search", **arguments)
    
    async def delete(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete documents from vector memory.
        
        Args:
            collection: Collection name
            ids: List of document IDs to delete
            filters: Filters to select documents for deletion
            
        Returns:
            Deletion result
        """
        arguments = {"collection": collection}
        
        if ids:
            arguments["ids"] = ids
            
        if filters:
            arguments["filters"] = filters
        
        return await self.client.call_tool("vector_memory_delete", **arguments)


class MiddlewareClient:
    """Client component for middleware operations."""
    
    def __init__(self, client: BeeAIExternalMCPClient):
        """
        Initialize the client component.
        
        Args:
            client: Parent client instance
        """
        self.client = client
    
    async def enhance_context(
        self,
        text: str,
        context_type: str = "general",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance context using middleware.
        
        Args:
            text: Input text
            context_type: Type of context enhancement
            options: Additional options
            
        Returns:
            Enhanced context
        """
        arguments = {
            "text": text,
            "context_type": context_type
        }
        
        if options:
            arguments["options"] = options
        
        return await self.client.call_tool("middleware_enhance_context", **arguments)


class WorkflowClient:
    """Client component for workflow operations."""
    
    def __init__(self, client: BeeAIExternalMCPClient):
        """
        Initialize the client component.
        
        Args:
            client: Parent client instance
        """
        self.client = client
    
    async def execute(
        self,
        workflow_name: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow_name: Name of the workflow to execute
            state: Initial workflow state
            
        Returns:
            Workflow execution result
        """
        return await self.client.call_tool(
            f"workflow_{workflow_name}",
            **state
        )


class VisualizationClient:
    """Client component for visualization services."""
    
    def __init__(self, client: BeeAIExternalMCPClient):
        """
        Initialize the client component.
        
        Args:
            client: Parent client instance
        """
        self.client = client
    
    async def create_reasoning_trace(
        self,
        steps: List[Dict[str, Any]],
        title: str,
        format: str = "html"
    ) -> Dict[str, Any]:
        """
        Create a reasoning trace visualization.
        
        Args:
            steps: Reasoning steps
            title: Visualization title
            format: Output format (html, json, svg)
            
        Returns:
            Visualization result
        """
        return await self.client.call_tool(
            "visualization_reasoning_trace",
            steps=steps,
            title=title,
            format=format
        )
    
    async def create_metrics_dashboard(
        self,
        data: Dict[str, Any],
        metrics: List[str],
        title: str,
        format: str = "html"
    ) -> Dict[str, Any]:
        """
        Create a metrics dashboard visualization.
        
        Args:
            data: Data to visualize
            metrics: List of metrics to include
            title: Dashboard title
            format: Output format
            
        Returns:
            Dashboard visualization
        """
        return await self.client.call_tool(
            "visualization_metrics_dashboard",
            data=data,
            metrics=metrics,
            title=title,
            format=format
        ) 