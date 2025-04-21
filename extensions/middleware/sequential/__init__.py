#!/usr/bin/env python
"""
Sequential Thinking Middleware Module.

This package implements middleware for sequential thinking processes with context refinement,
providing a powerful way to enhance LLM capabilities with step-by-step reasoning.
"""

# Core components
from .core import (
    SequentialThought,
    SequentialThinkingProcessor,
    SEQUENTIAL_THINKING_SYSTEM_PROMPT
)

# Context refinement components
from .context_refinement import (
    ContextQualityMetrics,
    ContextItem,
    ContextRefinementProcessor
)

# Reasoning trace components
from .reasoning_trace import (
    ContextReference,
    ReasoningStep,
    ReasoningTrace,
    ReasoningTraceAnalyzer,
    ReasoningTraceStore,
    ReasoningTraceVisualizer
)

# Context templates
from .context_templates import (
    ContextTemplate,
    TemplateManager,
    PLANNING_TEMPLATE,
    CODING_TEMPLATE
)

# LLM provider adapters
from .llm_adapter import (
    LLMRequest,
    LLMResponse,
    LLMAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    LLMProviderFactory,
    LLMProviderManager
)

# Middleware integration
from .middleware import (
    SequentialThinkingRequest,
    SequentialThinkingResponse,
    StepProgressCallback,
    SequentialThinkingMiddleware
)

__all__ = [
    # Core
    'SequentialThought',
    'SequentialThinkingProcessor',
    'SEQUENTIAL_THINKING_SYSTEM_PROMPT',
    
    # Context refinement
    'ContextQualityMetrics',
    'ContextItem',
    'ContextRefinementProcessor',
    
    # Reasoning trace
    'ContextReference',
    'ReasoningStep',
    'ReasoningTrace',
    'ReasoningTraceAnalyzer',
    'ReasoningTraceStore',
    'ReasoningTraceVisualizer',
    
    # Context templates
    'ContextTemplate',
    'TemplateManager',
    'PLANNING_TEMPLATE',
    'CODING_TEMPLATE',
    
    # LLM adapters
    'LLMRequest',
    'LLMResponse',
    'LLMAdapter',
    'OpenAIAdapter',
    'AnthropicAdapter',
    'LLMProviderFactory',
    'LLMProviderManager',
    
    # Middleware
    'SequentialThinkingRequest',
    'SequentialThinkingResponse',
    'StepProgressCallback',
    'SequentialThinkingMiddleware'
] 