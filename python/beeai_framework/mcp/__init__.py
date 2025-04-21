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
BeeAI FastMCP Integration Package

This package provides the Model Context Protocol (MCP) integration for BeeAI,
allowing the framework to expose its functionality through MCP and to interact
with other MCP-compatible systems.
"""

import importlib.metadata
from typing import Dict, Any, List, Optional

from beeai_framework.logger import Logger

logger = Logger(__name__)

try:
    __version__ = importlib.metadata.version("fastmcp")
    HAS_MCP = True
except importlib.metadata.PackageNotFoundError:
    __version__ = "not-installed"
    HAS_MCP = False
    logger.warning("FastMCP not installed, MCP features will be disabled.")

from beeai_framework.mcp.server import BeeAIMCPServer
from beeai_framework.mcp.client import BeeAIMCPClient

# Import adapters for convenience
from beeai_framework.mcp.adapters.middleware import MCPMiddlewareAdapter, MCPStreamingMiddlewareAdapter
from beeai_framework.mcp.adapters.workflow import MCPWorkflowAdapter, MCPStepAdapter

# Import external MCP modules
try:
    from beeai_framework.mcp.external_server import BeeAIExternalMCPServer
    from beeai_framework.mcp.external_client import BeeAIExternalMCPClient
    HAS_EXTERNAL_MCP = True
except ImportError:
    HAS_EXTERNAL_MCP = False
    logger.warning("External MCP components not available, install with 'pip install beeai-framework[mcp-external]'")

__all__ = [
    "BeeAIMCPServer",
    "BeeAIMCPClient",
    "MCPMiddlewareAdapter",
    "MCPStreamingMiddlewareAdapter",
    "MCPWorkflowAdapter",
    "MCPStepAdapter",
    "HAS_MCP",
    "__version__",
]

# Add external MCP components to __all__ if available
if HAS_EXTERNAL_MCP:
    __all__.extend([
        "BeeAIExternalMCPServer",
        "BeeAIExternalMCPClient",
        "HAS_EXTERNAL_MCP"
    ]) 