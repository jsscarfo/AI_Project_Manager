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
BeeAI External MCP Server Module

This module provides a secure external MCP server implementation for BeeAI,
allowing external applications to access BeeAI capabilities through an authenticated API.
"""

import os
import time
import uuid
import json
import inspect
import asyncio
from typing import Any, Dict, List, Optional, Callable, Type, Union, Set, Tuple
from datetime import datetime, timedelta

try:
    from fastapi import FastAPI, HTTPException, Depends, Security, status, Request, Response
    from fastapi.security import APIKeyHeader, OAuth2PasswordBearer, OAuth2PasswordRequestForm
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, EmailStr, create_model
    from jose import JWTError, jwt
    from passlib.context import CryptContext
except ModuleNotFoundError as e:
    required_modules = ["fastapi", "pydantic", "python-jose", "passlib", "uvicorn"]
    raise ModuleNotFoundError(
        f"Required modules not found: {required_modules}. "
        f"Run 'pip install beeai-framework[mcp-external]' to install."
    ) from e

from beeai_framework.logger import Logger
from beeai_framework.mcp.server import BeeAIMCPServer
from beeai_framework.mcp.external_tools import ExternalToolsRegistry, default_registry
from beeai_framework.utils.models import get_schema

logger = Logger(__name__)

# Authentication models
class Token(BaseModel):
    """OAuth token model."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """OAuth token data model."""
    username: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    """User account model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = ["user"]


class UserInDB(User):
    """User model with password hash."""
    hashed_password: str


class APIKey(BaseModel):
    """API key model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    roles: List[str] = ["user"]
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    rate_limit: int = 60  # requests per minute
    metadata: Dict[str, Any] = {}


class ToolRegistration(BaseModel):
    """Tool registration model."""
    name: str
    description: str
    version: str = "1.0.0"
    required_roles: List[str] = ["user"]
    enabled: bool = True
    metadata: Dict[str, Any] = {}


class BeeAIExternalMCPServer(BeeAIMCPServer):
    """External MCP server implementation for BeeAI with authentication."""
    
    def __init__(self, 
                 name: str = "BeeAI-External",
                 secret_key: Optional[str] = None,
                 access_token_expire_minutes: int = 30,
                 allow_origins: List[str] = None,
                 tool_registry: Optional[ExternalToolsRegistry] = None):
        """
        Initialize the BeeAI External MCP server.
        
        Args:
            name: Server name for identification
            secret_key: Secret key for JWT tokens (generated if not provided)
            access_token_expire_minutes: Token expiration time in minutes
            allow_origins: CORS allowed origins
            tool_registry: Registry of tools to expose externally
        """
        super().__init__(name=name)
        
        self.secret_key = secret_key or os.getenv("BEEAI_MCP_SECRET_KEY", str(uuid.uuid4()))
        self.access_token_expire_minutes = access_token_expire_minutes
        self.allow_origins = allow_origins or ["*"]
        self.tool_registry = tool_registry or default_registry
        
        # Setup FastAPI
        self.app = FastAPI(
            title=f"{name} API",
            description="External MCP server for BeeAI Framework",
            version="1.0.0",
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup security
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        
        # User and API key storage
        # Note: In production, these should be stored in a database
        self.users: Dict[str, UserInDB] = {}
        self.api_keys: Dict[str, APIKey] = {}
        
        # Rate limiting
        self.request_log: Dict[str, List[float]] = {}
        
        # Register API routes
        self._setup_routes()
        
        # Auto-register tools from registry
        self._register_tools_from_registry()
        
        logger.info(f"Initialized BeeAI External MCP server: {name}")
    
    def _register_tools_from_registry(self) -> None:
        """Register all tools from the tool registry."""
        for tool_config in self.tool_registry.list_tools():
            if tool_config.enabled:
                self.register_external_tool(
                    name=tool_config.name,
                    description=tool_config.description,
                    source_tool=tool_config.source_tool,
                    version=tool_config.version,
                    required_roles=tool_config.required_roles,
                    metadata=tool_config.metadata
                )
    
    def _setup_routes(self) -> None:
        """Setup API routes for authentication and tool management."""
        
        @self.app.post("/token", response_model=Token)
        async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
            user = self._authenticate_user(form_data.username, form_data.password)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
            access_token = self._create_access_token(
                data={"sub": user.username, "scopes": user.roles},
                expires_delta=access_token_expires,
            )
            
            return {"access_token": access_token, "token_type": "bearer"}
        
        @self.app.get("/tools", response_model=List[Dict[str, Any]])
        async def list_tools(current_user: User = Depends(self._get_current_user)):
            """List available MCP tools."""
            tools = []
            for tool_name, tool_reg in self.tool_registrations.items():
                # Check if user has access to this tool
                if self._has_access(current_user, tool_reg.required_roles):
                    tools.append({
                        "name": tool_name,
                        "description": tool_reg.description,
                        "version": tool_reg.version,
                        "metadata": tool_reg.metadata
                    })
            
            return tools
        
        @self.app.post("/tools/{tool_name}")
        async def call_tool(
            tool_name: str,
            request: Request,
            response: Response,
            current_user: User = Depends(self._get_current_user)
        ):
            """Call an MCP tool by name."""
            # Check if tool exists and user has access
            if tool_name not in self.tool_registrations:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            
            tool_reg = self.tool_registrations[tool_name]
            if not tool_reg.enabled:
                raise HTTPException(status_code=403, detail=f"Tool '{tool_name}' is disabled")
            
            if not self._has_access(current_user, tool_reg.required_roles):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            # Get request payload
            payload = await request.json()
            
            # Apply rate limiting
            self._check_rate_limit(current_user.username)
            
            # Call the MCP tool
            try:
                # Get the source tool name from the registry
                source_tool = self._get_source_tool_name(tool_name)
                
                if hasattr(self.mcp_server, "call_tool"):
                    # Call through server's call_tool method
                    result = await self.mcp_server.call_tool(source_tool, payload)
                elif source_tool in self.registered_tools:
                    # Call through registered handler directly
                    tool_info = self.registered_tools[source_tool]
                    handler = tool_info.get("handler")
                    if handler:
                        result = await handler(payload)
                    else:
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Tool '{source_tool}' has no handler"
                        )
                else:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Source tool '{source_tool}' not found in registered tools"
                    )
                
                # Log usage
                self._log_tool_usage(current_user.username, tool_name)
                
                return JSONResponse(content=result)
                
            except Exception as e:
                logger.error(f"Error calling tool {tool_name}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error executing tool: {str(e)}"
                )
        
        @self.app.post("/register_tool", response_model=ToolRegistration, status_code=201)
        async def register_tool(
            registration: ToolRegistration,
            current_user: User = Depends(self._get_current_admin)
        ):
            """Register a new tool (admin only)."""
            source_tool = registration.metadata.get("source_tool")
            if not source_tool:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'source_tool' in metadata"
                )
                
            if source_tool not in self.registered_tools:
                raise HTTPException(
                    status_code=404,
                    detail=f"Source tool '{source_tool}' not found in MCP server"
                )
            
            # Register the tool
            self.tool_registrations[registration.name] = registration
            return registration
        
        @self.app.post("/create_api_key", response_model=APIKey, status_code=201)
        async def create_api_key(
            name: str,
            user_id: str,
            roles: Optional[List[str]] = None,
            expiry_days: Optional[int] = None,
            rate_limit: Optional[int] = None,
            current_user: User = Depends(self._get_current_admin)
        ):
            """Create a new API key (admin only)."""
            # Validate user exists
            if user_id not in self.users:
                raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
            
            # Create API key
            api_key = APIKey(
                name=name,
                user_id=user_id,
                roles=roles or ["user"],
                rate_limit=rate_limit or 60
            )
            
            # Set expiry if provided
            if expiry_days:
                api_key.expires_at = datetime.now() + timedelta(days=expiry_days)
            
            # Store API key
            self.api_keys[api_key.key] = api_key
            
            return api_key
        
        @self.app.get("/server_info")
        async def server_info():
            """Get server information."""
            return {
                "name": self.name,
                "version": "1.0.0",
                "tools_count": len(self.tool_registrations),
                "uptime": "N/A"  # Would track server start time in production
            }
    
    def _get_source_tool_name(self, external_tool_name: str) -> str:
        """Get the source tool name for an external tool."""
        tool_config = self.tool_registry.get(external_tool_name)
        if tool_config:
            return tool_config.source_tool
        return external_tool_name  # Default to same name if not in registry
    
    def register_external_tool(self, 
                              name: str, 
                              description: str, 
                              source_tool: str,
                              version: str = "1.0.0",
                              required_roles: List[str] = None,
                              metadata: Dict[str, Any] = None) -> None:
        """
        Register a BeeAI tool for external access.
        
        Args:
            name: External tool name
            description: Human-readable description
            source_tool: Name of the BeeAI tool to expose
            version: Tool version
            required_roles: Roles allowed to access this tool
            metadata: Additional metadata
        """
        if source_tool not in self.registered_tools:
            raise ValueError(f"Source tool '{source_tool}' not found in registered tools")
        
        # Register the tool for external access
        self.tool_registrations[name] = ToolRegistration(
            name=name,
            description=description,
            version=version,
            required_roles=required_roles or ["user"],
            enabled=True,
            metadata=metadata or {"source_tool": source_tool}
        )
        
        # Update metadata to include source tool
        if "source_tool" not in self.tool_registrations[name].metadata:
            metadata = self.tool_registrations[name].metadata
            metadata["source_tool"] = source_tool
            self.tool_registrations[name].metadata = metadata
        
        logger.info(f"Registered tool for external access: {name} (source: {source_tool})")
    
    def add_user(self, username: str, password: str, email: Optional[str] = None, 
                full_name: Optional[str] = None, roles: List[str] = None) -> User:
        """
        Add a new user.
        
        Args:
            username: Unique username
            password: User password
            email: User email
            full_name: User's full name
            roles: User roles (default: ["user"])
            
        Returns:
            Created user
        """
        if username in self.users:
            raise ValueError(f"User '{username}' already exists")
        
        hashed_password = self._get_password_hash(password)
        user = UserInDB(
            username=username,
            email=email,
            full_name=full_name,
            roles=roles or ["user"],
            hashed_password=hashed_password
        )
        
        self.users[username] = user
        logger.info(f"Added user: {username}")
        
        return User(**user.model_dump(exclude={"hashed_password"}))
    
    def create_api_key(self, 
                      name: str, 
                      user_id: str, 
                      roles: List[str] = None,
                      expiry_days: Optional[int] = None,
                      rate_limit: int = 60) -> APIKey:
        """
        Create a new API key for a user.
        
        Args:
            name: Key name/description
            user_id: User ID (username)
            roles: Access roles
            expiry_days: Days until key expires (None for no expiry)
            rate_limit: Requests per minute
            
        Returns:
            Created API key
        """
        if user_id not in self.users:
            raise ValueError(f"User '{user_id}' not found")
        
        api_key = APIKey(
            name=name,
            user_id=user_id,
            roles=roles or ["user"],
            rate_limit=rate_limit
        )
        
        # Set expiry if provided
        if expiry_days:
            api_key.expires_at = datetime.now() + timedelta(days=expiry_days)
        
        # Store API key
        self.api_keys[api_key.key] = api_key
        
        logger.info(f"Created API key '{name}' for user: {user_id}")
        return api_key
    
    def _get_password_hash(self, password: str) -> str:
        """Generate a password hash."""
        return self.pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def _authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user by username and password."""
        if username not in self.users:
            return None
        
        user = self.users[username]
        if not self._verify_password(password, user.hashed_password):
            return None
        
        if user.disabled:
            return None
            
        return user
    
    def _authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate a user by API key."""
        if api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        
        # Check if expired
        if key_info.expires_at and key_info.expires_at < datetime.now():
            return None
        
        # Get user
        if key_info.user_id not in self.users:
            return None
            
        user = self.users[key_info.user_id]
        
        if user.disabled:
            return None
        
        # Create a user with the roles from the API key
        return User(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            disabled=user.disabled,
            roles=key_info.roles
        )
    
    def _create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
            
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        
        return encoded_jwt
    
    async def _get_current_user(self, 
                              api_key: str = Security(api_key_header),
                              token: str = Security(oauth2_scheme)) -> User:
        """Get current user from API key or JWT token."""
        # First try API key
        if api_key:
            user = self._authenticate_api_key(api_key)
            if user:
                return user
        
        # Then try JWT
        if token:
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
                username = payload.get("sub")
                if username is None:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                token_data = TokenData(username=username, scopes=payload.get("scopes", []))
            except JWTError:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            if token_data.username not in self.users:
                raise HTTPException(status_code=401, detail="User not found")
                
            user = self.users[token_data.username]
            
            if user.disabled:
                raise HTTPException(status_code=401, detail="User is disabled")
                
            return User(**user.model_dump(exclude={"hashed_password"}))
        
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    async def _get_current_admin(self, current_user: User = Depends(_get_current_user)) -> User:
        """Get current user and verify admin role."""
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions, admin role required"
            )
        return current_user
    
    def _has_access(self, user: User, required_roles: List[str]) -> bool:
        """Check if user has any of the required roles."""
        # Admin always has access
        if "admin" in user.roles:
            return True
            
        # Check for role intersection
        return bool(set(user.roles) & set(required_roles))
    
    def _check_rate_limit(self, user_id: str) -> None:
        """Check and enforce rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        if user_id not in self.request_log:
            self.request_log[user_id] = []
        
        # Clean up old requests
        self.request_log[user_id] = [t for t in self.request_log[user_id] if t > minute_ago]
        
        # Get API key for this user to determine rate limit
        rate_limit = 60  # Default
        for key in self.api_keys.values():
            if key.user_id == user_id:
                rate_limit = key.rate_limit
                break
        
        # Check if rate limit exceeded
        if len(self.request_log[user_id]) >= rate_limit:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded, please try again later"
            )
        
        # Log this request
        self.request_log[user_id].append(now)
    
    def _log_tool_usage(self, user_id: str, tool_name: str) -> None:
        """Log tool usage for monitoring."""
        # In a production system, this would log to a database or monitoring system
        logger.info(f"Tool usage: {tool_name} by user {user_id}")
    
    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Start the external MCP server.
        
        Args:
            host: Hostname to listen on
            port: Port to listen on
        """
        import uvicorn
        
        logger.info(f"Starting BeeAI External MCP server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

    async def start_async(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Start the external MCP server asynchronously.
        
        Args:
            host: Hostname to listen on
            port: Port to listen on
        """
        import uvicorn
        config = uvicorn.Config(self.app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self) -> None:
        """Stop the MCP server."""
        await super().stop()
        logger.info("Stopped BeeAI External MCP server") 