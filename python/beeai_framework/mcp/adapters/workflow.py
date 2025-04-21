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
BeeAI MCP Workflow Adapter Module

This module provides adapters for converting between BeeAI workflows
and MCP tools.
"""

from typing import Any, Dict, List, Optional, TypeVar, Type, Generic, cast

try:
    from mcp.types import Tool as MCPTool
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [mcp] not found.\nRun 'pip install beeai-framework[mcp]' to install."
    ) from e

from pydantic import BaseModel, create_model

from beeai_framework.logger import Logger
from beeai_framework.workflows.workflow import Workflow
from beeai_framework.workflows.types import WorkflowHandler, WorkflowRunOptions, WorkflowRun, WorkflowState
from beeai_framework.mcp.client import BeeAIMCPClient
from beeai_framework.utils.models import get_schema

logger = Logger(__name__)

T = TypeVar('T', bound=BaseModel)
K = TypeVar('K', default=str)


class MCPWorkflowAdapter(Workflow[T, K]):
    """
    Workflow adapter that connects to an MCP server and delegates execution to an MCP tool.
    This allows using remote MCP tools as workflow steps in BeeAI.
    """
    
    def __init__(self, 
                 client: BeeAIMCPClient, 
                 tool_name: str, 
                 schema: Type[T],
                 name: Optional[str] = None):
        """
        Initialize the MCP workflow adapter.
        
        Args:
            client: MCP client to use for communicating with the server
            tool_name: Name of the MCP tool to call
            schema: Pydantic model type for the workflow state
            name: Optional workflow name (defaults to the tool name)
        """
        super().__init__(schema, name=name or f"mcp_{tool_name}")
        self.client = client
        self.tool_name = tool_name
        
        # Add execution step
        self.add_step("execute", self._execute_mcp_tool)
        self.set_start("execute")
        
        logger.info(f"Initialized MCP workflow adapter for tool: {tool_name}")
        
    async def _execute_mcp_tool(self, state: T) -> K:
        """
        Execute the MCP tool with the workflow state.
        
        Args:
            state: The workflow state
            
        Returns:
            Next step identifier (always returns END since this is a single-step workflow)
        """
        logger.debug(f"Executing MCP tool as workflow step: {self.tool_name}")
        
        # Convert state to dict for the MCP call
        state_data = {}
        if hasattr(state, "model_dump"):
            state_data = state.model_dump()
        elif hasattr(state, "dict"):
            state_data = state.dict()
        elif isinstance(state, dict):
            state_data = state
        
        try:
            # Call the MCP tool
            result = await self.client.call_tool(self.tool_name, **state_data)
            
            # Update state with results if possible
            if isinstance(result, dict) and isinstance(state, BaseModel):
                # Only update fields that exist in the state model
                state_fields = state.model_fields if hasattr(state, "model_fields") else state.__fields__ if hasattr(state, "__fields__") else {}
                
                for key, value in result.items():
                    if key in state_fields:
                        setattr(state, key, value)
                        
                # Add all results to a 'results' field if it exists
                if "results" in state_fields:
                    setattr(state, "results", result)
                
                # Or add to metadata if that exists
                elif "metadata" in state_fields:
                    metadata = getattr(state, "metadata", {})
                    if not metadata:
                        metadata = {}
                    metadata["mcp_result"] = result
                    setattr(state, "metadata", metadata)
            
            logger.debug(f"MCP workflow step completed for tool: {self.tool_name}")
            
        except Exception as e:
            logger.error(f"Error in MCP workflow step: {str(e)}")
            # If state has an error field, set it
            if hasattr(state, "error") and isinstance(state, BaseModel):
                setattr(state, "error", str(e))
            # Or add to metadata if that exists
            elif hasattr(state, "metadata") and isinstance(state, BaseModel):
                metadata = getattr(state, "metadata", {})
                if not metadata:
                    metadata = {}
                metadata["error"] = str(e)
                setattr(state, "metadata", metadata)
        
        # This is a single-step workflow, so return END
        return cast(K, Workflow.END)


class MCPStepAdapter:
    """
    Adapter for using MCP tools as individual steps in a workflow.
    Unlike MCPWorkflowAdapter which creates a complete workflow,
    this creates a workflow step handler that can be added to an existing workflow.
    """
    
    @staticmethod
    def create_step(client: BeeAIMCPClient, tool_name: str) -> WorkflowHandler[T, K]:
        """
        Create a workflow step handler that calls an MCP tool.
        
        Args:
            client: MCP client to use for communicating with the server
            tool_name: Name of the MCP tool to call
            
        Returns:
            A workflow step handler function
        """
        async def step_handler(state: T) -> K:
            """
            Execute the MCP tool with the workflow state.
            
            Args:
                state: The workflow state
                
            Returns:
                Next step identifier (NEXT)
            """
            logger.debug(f"Executing MCP tool as workflow step: {tool_name}")
            
            # Convert state to dict for the MCP call
            state_data = {}
            if hasattr(state, "model_dump"):
                state_data = state.model_dump()
            elif hasattr(state, "dict"):
                state_data = state.dict()
            elif isinstance(state, dict):
                state_data = state
            
            try:
                # Call the MCP tool
                result = await client.call_tool(tool_name, **state_data)
                
                # Update state with results if possible
                if isinstance(result, dict) and isinstance(state, BaseModel):
                    # Only update fields that exist in the state model
                    state_fields = state.model_fields if hasattr(state, "model_fields") else state.__fields__ if hasattr(state, "__fields__") else {}
                    
                    for key, value in result.items():
                        if key in state_fields:
                            setattr(state, key, value)
                            
                    # Add all results to a 'results' field if it exists
                    if "results" in state_fields:
                        setattr(state, "results", result)
                    
                    # Or add to metadata if that exists
                    elif "metadata" in state_fields:
                        metadata = getattr(state, "metadata", {})
                        if not metadata:
                            metadata = {}
                        metadata["mcp_result"] = result
                        setattr(state, "metadata", metadata)
                
                logger.debug(f"MCP workflow step completed for tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Error in MCP workflow step: {str(e)}")
                # If state has an error field, set it
                if hasattr(state, "error") and isinstance(state, BaseModel):
                    setattr(state, "error", str(e))
                # Or add to metadata if that exists
                elif hasattr(state, "metadata") and isinstance(state, BaseModel):
                    metadata = getattr(state, "metadata", {})
                    if not metadata:
                        metadata = {}
                    metadata["error"] = str(e)
                    setattr(state, "metadata", metadata)
            
            # Continue to the next step in the workflow
            return cast(K, Workflow.NEXT)
        
        return step_handler 