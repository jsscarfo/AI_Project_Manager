"""
Knowledge Retrieval Middleware Package.

This package provides functionality for retrieving and integrating knowledge
from vector databases into sequential thinking processes.
"""

from .core import (
    KnowledgeRetrievalProcessor,
    KnowledgeRetrievalSettings,
    KnowledgeRetrievalResult
)

from .middleware import (
    KnowledgeRetrievalMiddleware,
    KnowledgeRetrievalRequest,
    KnowledgeRetrievalResponse,
    ContextEnhancementProvider
)

__all__ = [
    'KnowledgeRetrievalProcessor',
    'KnowledgeRetrievalSettings',
    'KnowledgeRetrievalResult',
    'KnowledgeRetrievalMiddleware',
    'KnowledgeRetrievalRequest',
    'KnowledgeRetrievalResponse',
    'ContextEnhancementProvider'
] 