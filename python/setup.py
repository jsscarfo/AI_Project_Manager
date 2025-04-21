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

from setuptools import setup, find_packages

# Add beeai_framework.mcp and its subpackages to the extras_require dict
# with the MCP library as a dependency
extras_require = {
    # ... existing extras ...
    
    # Add MCP integration as an optional dependency
    "mcp": [
        "fastmcp>=0.3.0",
        "modelcontextprotocol>=1.0.0"
    ],
    
    # Add external MCP server dependencies
    "mcp-external": [
        "fastmcp>=0.3.0",
        "modelcontextprotocol>=1.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "python-multipart>=0.0.6",
        "bcrypt>=4.0.1"
    ],
    
    # Add external MCP client dependencies
    "mcp-client": [
        "fastmcp>=0.3.0",
        "modelcontextprotocol>=1.0.0",
        "aiohttp>=3.8.5",
        "pydantic>=2.4.2"
    ],
    
    # Full install includes all optional dependencies
    "full": [
        # ... existing dependencies ...
        "fastmcp>=0.3.0",
        "modelcontextprotocol>=1.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "python-multipart>=0.0.6",
        "bcrypt>=4.0.1",
        "aiohttp>=3.8.5",
        "pydantic>=2.4.2"
    ]
}

setup(
    name="beeai-framework",
    version="5.0.0",
    description="BeeAI Framework V5",
    author="BeeAI Team",
    author_email="info@beeai.org",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        # ... existing requirements ...
    ],
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 