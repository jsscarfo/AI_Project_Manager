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
Middleware Base Module

This module defines the base abstractions for the middleware framework including:
- Abstract base classes for middleware components
- Context passing mechanisms
- Middleware chain implementation

The middleware framework is a core part of the V5 architecture that allows for
component composition and flexible request processing.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union, Callable, TypeVar, Generic, cast
import logging
from pydantic import BaseModel, Field

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class MiddlewareContext(BaseModel, Generic[T, R]):
    """
    Context object passed through middleware chain.
    
    This context carries the request, response, and any additional metadata
    needed by middleware components to process the request.
    """
    
    request: T
    response: Optional[R] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Flag to indicate if the response has been generated
    response_generated: bool = False
    
    # Flag to indicate if middleware processing should continue
    should_continue: bool = True
    
    def enhance_metadata(self, new_metadata: Dict[str, Any]) -> "MiddlewareContext[T, R]":
        """Add additional metadata to the context."""
        self.metadata.update(new_metadata)
        return self
    
    def set_response(self, response: R) -> "MiddlewareContext[T, R]":
        """Set the response and mark response as generated."""
        self.response = response
        self.response_generated = True
        return self
    
    def stop_processing(self) -> "MiddlewareContext[T, R]":
        """Stop the middleware chain processing."""
        self.should_continue = False
        return self


class MiddlewareConfig(BaseModel):
    """Configuration for middleware components."""
    
    enabled: bool = True
    priority: int = 100  # Lower values = higher priority
    name: Optional[str] = None
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.name is None:
            self.name = self.__class__.__name__


class Middleware(Generic[T, R], ABC):
    """Base abstract class for all middleware components."""
    
    def __init__(self, config: Optional[MiddlewareConfig] = None):
        """
        Initialize the middleware component.
        
        Args:
            config: Configuration options for the middleware
        """
        self.config = config or MiddlewareConfig()
        self.emitter = Emitter.root().child(
            namespace=["middleware", self.__class__.__name__.lower()],
            creator=self,
        )
        
        logger.info(f"Initialized middleware: {self.__class__.__name__}")
    
    @abstractmethod
    async def process(self, context: MiddlewareContext[T, R]) -> MiddlewareContext[T, R]:
        """
        Process the request context through this middleware.
        
        Args:
            context: The request context to process
            
        Returns:
            The (potentially modified) context
        """
        pass
    
    @property
    def name(self) -> str:
        """Get the name of this middleware component."""
        return self.config.name or self.__class__.__name__
    
    @property
    def priority(self) -> int:
        """Get the priority of this middleware component."""
        return self.config.priority
    
    @property
    def enabled(self) -> bool:
        """Check if this middleware component is enabled."""
        return self.config.enabled


class MiddlewareChain(Generic[T, R]):
    """
    Chain of middleware components that process requests sequentially.
    
    Middleware components are processed in order of priority (lowest value first).
    Processing stops when a component sets response_generated or should_continue to False,
    or when all components have been processed.
    """
    
    def __init__(self):
        """Initialize an empty middleware chain."""
        self.middlewares: List[Middleware[T, R]] = []
        self.emitter = Emitter.root().child(
            namespace=["middleware", "chain"],
            creator=self,
        )
        
        logger.info("Initialized middleware chain")
    
    def add_middleware(self, middleware: Middleware[T, R]) -> "MiddlewareChain[T, R]":
        """
        Add a middleware component to the chain.
        
        Args:
            middleware: The middleware component to add
            
        Returns:
            This chain instance for method chaining
        """
        if not middleware.enabled:
            logger.info(f"Skipping disabled middleware: {middleware.name}")
            return self
            
        self.middlewares.append(middleware)
        
        # Sort middlewares by priority (lowest value first)
        self.middlewares.sort(key=lambda m: m.priority)
        
        logger.info(f"Added middleware to chain: {middleware.name} (priority: {middleware.priority})")
        return self
    
    def remove_middleware(self, middleware_name: str) -> "MiddlewareChain[T, R]":
        """
        Remove a middleware component from the chain by name.
        
        Args:
            middleware_name: Name of the middleware to remove
            
        Returns:
            This chain instance for method chaining
        """
        original_count = len(self.middlewares)
        self.middlewares = [m for m in self.middlewares if m.name != middleware_name]
        
        if len(self.middlewares) < original_count:
            logger.info(f"Removed middleware from chain: {middleware_name}")
        else:
            logger.warning(f"Middleware not found in chain: {middleware_name}")
            
        return self
    
    def clear(self) -> "MiddlewareChain[T, R]":
        """
        Remove all middleware components from the chain.
        
        Returns:
            This chain instance for method chaining
        """
        self.middlewares = []
        logger.info("Cleared all middleware from chain")
        return self
    
    async def process_request(self, request: T) -> R:
        """
        Process a request through the middleware chain.
        
        Args:
            request: The request to process
            
        Returns:
            The response generated by the middleware chain
            
        Raises:
            FrameworkError: If no response was generated by any middleware
        """
        context = MiddlewareContext[T, R](request=request)
        
        self.emitter.emit("request", {"request": request})
        
        for middleware in self.middlewares:
            if not context.should_continue:
                logger.debug(f"Stopping middleware chain at {middleware.name} due to should_continue=False")
                break
                
            try:
                if not middleware.enabled:
                    logger.debug(f"Skipping disabled middleware: {middleware.name}")
                    continue
                    
                logger.debug(f"Processing request through middleware: {middleware.name}")
                context = await middleware.process(context)
                
                if context.response_generated:
                    logger.debug(f"Response generated by middleware: {middleware.name}")
                    break
                    
            except Exception as e:
                error = FrameworkError.ensure(e)
                self.emitter.emit("error", {"error": error, "middleware": middleware.name})
                logger.error(f"Error in middleware {middleware.name}: {str(error)}")
                raise error
        
        if not context.response_generated:
            logger.warning("No middleware generated a response")
            raise FrameworkError("No middleware generated a response")
        
        self.emitter.emit("response", {"response": context.response})
        
        return cast(R, context.response)


class MiddlewareRegistry:
    """
    Registry for middleware components.
    
    Allows dynamic discovery and configuration of middleware components.
    """
    
    def __init__(self):
        """Initialize an empty middleware registry."""
        self.registry: Dict[str, type[Middleware]] = {}
        
        logger.info("Initialized middleware registry")
    
    def register(self, middleware_class: type[Middleware]) -> None:
        """
        Register a middleware class.
        
        Args:
            middleware_class: The middleware class to register
        """
        name = middleware_class.__name__
        self.registry[name] = middleware_class
        logger.info(f"Registered middleware: {name}")
    
    def get_middleware_class(self, name: str) -> Optional[type[Middleware]]:
        """
        Get a middleware class by name.
        
        Args:
            name: The name of the middleware class
            
        Returns:
            The middleware class if found, None otherwise
        """
        return self.registry.get(name)
    
    def create_middleware(self, name: str, config: Optional[MiddlewareConfig] = None) -> Optional[Middleware]:
        """
        Create a middleware instance by name.
        
        Args:
            name: The name of the middleware class
            config: Optional configuration for the middleware
            
        Returns:
            A new middleware instance if the class is found, None otherwise
        """
        middleware_class = self.get_middleware_class(name)
        
        if middleware_class is None:
            logger.warning(f"Middleware class not found: {name}")
            return None
            
        return middleware_class(config)
    
    def list_middleware(self) -> List[str]:
        """
        List all registered middleware classes.
        
        Returns:
            List of middleware class names
        """
        return list(self.registry.keys())


# Create a global middleware registry instance
global_middleware_registry = MiddlewareRegistry()


def register_middleware(middleware_class: type[Middleware]) -> type[Middleware]:
    """
    Decorator to register a middleware class with the global registry.
    
    Args:
        middleware_class: The middleware class to register
        
    Returns:
        The middleware class (unchanged)
    """
    global_middleware_registry.register(middleware_class)
    return middleware_class 