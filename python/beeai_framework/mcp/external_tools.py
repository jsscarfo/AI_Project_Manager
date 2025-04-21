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
BeeAI External MCP Tools Registration Module

This module defines which BeeAI tools are exposed through the external MCP server,
along with their configurations and access controls.
"""

import os
import json
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field
from beeai_framework.logger import Logger

logger = Logger(__name__)


class ExternalToolConfig(BaseModel):
    """Configuration for an externally exposed BeeAI tool."""
    name: str
    description: str
    source_tool: str  # Name of the BeeAI tool to expose
    version: str = "1.0.0"
    required_roles: List[str] = ["user"]
    enabled: bool = True
    rate_limit: Optional[int] = None  # Requests per minute (None uses global default)
    metadata: Dict[str, Any] = {}


class ExternalToolsRegistry:
    """Registry for BeeAI tools exposed through the external MCP server."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the external tools registry.
        
        Args:
            config_path: Path to JSON configuration file (optional)
        """
        self.tools: Dict[str, ExternalToolConfig] = {}
        self.config_path = config_path
        
        # Load default tools
        self._register_default_tools()
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def _register_default_tools(self) -> None:
        """Register default tools that should be exposed externally."""
        # Sequential thinking tools
        self.register(ExternalToolConfig(
            name="sequential_thinking_solve",
            description="Solve a problem using the BeeAI sequential thinking approach",
            source_tool="middleware_sequential_thinking",
            version="1.0.0",
            required_roles=["user"],
            metadata={
                "category": "reasoning",
                "usage_examples": [
                    {"problem": "Design a user authentication system", "steps": 5}
                ]
            }
        ))
        
        self.register(ExternalToolConfig(
            name="sequential_thinking_analyze",
            description="Analyze text using sequential thinking",
            source_tool="middleware_sequential_analysis",
            version="1.0.0",
            required_roles=["user"],
            metadata={"category": "reasoning"}
        ))
        
        # Vector memory tools
        self.register(ExternalToolConfig(
            name="vector_memory_store",
            description="Store documents in vector memory",
            source_tool="middleware_vector_store",
            version="1.0.0",
            required_roles=["vector_admin", "admin"],
            metadata={"category": "vector_memory"}
        ))
        
        self.register(ExternalToolConfig(
            name="vector_memory_search",
            description="Search for documents in vector memory",
            source_tool="middleware_vector_search",
            version="1.0.0",
            required_roles=["user"],
            metadata={"category": "vector_memory"}
        ))
        
        self.register(ExternalToolConfig(
            name="vector_memory_delete",
            description="Delete documents from vector memory",
            source_tool="middleware_vector_delete",
            version="1.0.0",
            required_roles=["vector_admin", "admin"],
            metadata={"category": "vector_memory"}
        ))
        
        # Context enhancement tools
        self.register(ExternalToolConfig(
            name="middleware_enhance_context",
            description="Enhance text context with additional information",
            source_tool="middleware_context_enhancement",
            version="1.0.0",
            required_roles=["user"],
            metadata={"category": "context"}
        ))
        
        # Visualization tools
        self.register(ExternalToolConfig(
            name="visualization_reasoning_trace",
            description="Create a visualization of a reasoning trace",
            source_tool="middleware_reasoning_trace_visualizer",
            version="1.0.0",
            required_roles=["user"],
            metadata={"category": "visualization"}
        ))
        
        self.register(ExternalToolConfig(
            name="visualization_metrics_dashboard",
            description="Create a metrics dashboard visualization",
            source_tool="middleware_metrics_dashboard",
            version="1.0.0",
            required_roles=["user"],
            metadata={"category": "visualization"}
        ))
    
    def register(self, tool_config: ExternalToolConfig) -> None:
        """
        Register a tool for external access.
        
        Args:
            tool_config: Tool configuration
        """
        self.tools[tool_config.name] = tool_config
        logger.info(f"Registered external tool: {tool_config.name}")
    
    def unregister(self, tool_name: str) -> None:
        """
        Unregister a tool from external access.
        
        Args:
            tool_name: Name of the tool to unregister
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Unregistered external tool: {tool_name}")
    
    def get(self, tool_name: str) -> Optional[ExternalToolConfig]:
        """
        Get a tool configuration by name.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool configuration or None if not found
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[ExternalToolConfig]:
        """
        Get a list of all registered external tools.
        
        Returns:
            List of tool configurations
        """
        return list(self.tools.values())
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load tool configurations from a JSON file.
        
        Args:
            file_path: Path to JSON configuration file
        """
        try:
            with open(file_path, "r") as f:
                config_data = json.load(f)
            
            # Clear existing tools if specified
            if config_data.get("clear_existing", False):
                self.tools = {}
                logger.info("Cleared existing tool registrations")
            
            # Register tools from config
            tool_configs = config_data.get("tools", [])
            for tool_data in tool_configs:
                tool_config = ExternalToolConfig(**tool_data)
                self.register(tool_config)
            
            logger.info(f"Loaded {len(tool_configs)} tool configurations from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading tool configurations from {file_path}: {str(e)}")
            raise
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save tool configurations to a JSON file.
        
        Args:
            file_path: Path to JSON configuration file
        """
        try:
            config_data = {
                "tools": [tool.model_dump() for tool in self.tools.values()]
            }
            
            with open(file_path, "w") as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved {len(self.tools)} tool configurations to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving tool configurations to {file_path}: {str(e)}")
            raise


# Default registry instance
default_registry = ExternalToolsRegistry(
    config_path=os.environ.get("BEEAI_EXTERNAL_TOOLS_CONFIG")
) 