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
Vector Memory Module

This module provides interfaces and implementations for vector database integration,
focusing on Weaviate for Selective Contextual Retrieval functionality.
"""

# Base imports first to avoid circular dependencies
from beeai_framework.vector.base import VectorMemoryProvider, VectorMemoryProviderConfig
from beeai_framework.vector.embedding_service import EmbeddingService, EmbeddingServiceConfig
from beeai_framework.vector.weaviate_provider import WeaviateProvider, WeaviateProviderConfig

# Knowledge capture imports
from beeai_framework.vector.knowledge_capture import (
    KnowledgeCaptureProcessor, 
    KnowledgeCaptureSetting,
    KnowledgeCaptureMiddleware,
    KnowledgeCaptureMiddlewareConfig
)

# Retrieval imports
from beeai_framework.vector.knowledge_retrieval import (
    SequentialThinkingKnowledgeRetriever,
    KnowledgeRetrievalConfig,
    StepContextManager,
    RetrievedKnowledge,
    KnowledgeRetrievalResult
)

# Middleware imports - put last to avoid circular dependencies
from beeai_framework.vector.middleware import ContextualEnhancementMiddleware, ContextualEnhancementConfig

__all__ = [
    "VectorMemoryProvider",
    "VectorMemoryProviderConfig",
    "EmbeddingService",
    "EmbeddingServiceConfig",
    "WeaviateProvider",
    "WeaviateProviderConfig",
    "ContextualEnhancementMiddleware",
    "ContextualEnhancementConfig",
    "KnowledgeCaptureProcessor",
    "KnowledgeCaptureSetting",
    "KnowledgeCaptureMiddleware",
    "KnowledgeCaptureMiddlewareConfig",
    "SequentialThinkingKnowledgeRetriever",
    "KnowledgeRetrievalConfig", 
    "StepContextManager",
    "RetrievedKnowledge",
    "KnowledgeRetrievalResult"
] 