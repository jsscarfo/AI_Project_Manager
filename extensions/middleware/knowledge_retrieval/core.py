#!/usr/bin/env python
"""
Knowledge Retrieval Core Implementation.

This module provides the core functionality for retrieving relevant knowledge
from the vector database to enhance contextual awareness during sequential
thinking processes.
"""

import logging
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pydantic import BaseModel, Field, validator

from beeai_framework.vector.base import VectorMemoryProvider, ContextMetadata
from beeai_framework.vector.knowledge_capture import KnowledgeEntry

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeRetrievalSettings(BaseModel):
    """Settings for knowledge retrieval."""
    
    enabled: bool = Field(default=True, description="Whether knowledge retrieval is enabled")
    max_results: int = Field(default=5, description="Maximum number of knowledge entries to retrieve")
    similarity_threshold: float = Field(
        default=0.65, 
        description="Minimum similarity score to include knowledge (0-1)"
    )
    relevance_boost_factor: float = Field(
        default=1.2,
        description="Factor to boost relevance of more recent knowledge"
    )
    context_window_tokens: int = Field(
        default=2000,
        description="Maximum number of tokens to include in context window"
    )
    retrieval_strategies: List[str] = Field(
        default=["semantic", "keyword", "conceptual"],
        description="Strategies to use for knowledge retrieval"
    )


class KnowledgeRetrievalResult(BaseModel):
    """Result of knowledge retrieval operation."""
    
    entries: List[KnowledgeEntry] = Field(default_factory=list, description="Retrieved knowledge entries")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Retrieval metrics")
    query: str = Field(..., description="The query used for retrieval")
    context_str: str = Field("", description="Formatted context string for model input")
    
    @property
    def has_results(self) -> bool:
        """Whether the retrieval has any results."""
        return len(self.entries) > 0


class KnowledgeRetrievalProcessor:
    """
    Processes knowledge retrieval requests to provide contextual information.
    
    This class handles retrieval of relevant knowledge entries from vector storage
    based on the current thinking context and task requirements.
    """
    
    def __init__(
        self,
        vector_provider: VectorMemoryProvider,
        settings: Optional[KnowledgeRetrievalSettings] = None
    ):
        """
        Initialize the knowledge retrieval processor.
        
        Args:
            vector_provider: Provider for vector memory operations
            settings: Optional settings for knowledge retrieval
        """
        self.vector_provider = vector_provider
        self.settings = settings or KnowledgeRetrievalSettings()
        
    async def retrieve_knowledge(
        self,
        query: str,
        task_context: Optional[Dict[str, Any]] = None,
        previous_thoughts: Optional[List[Dict[str, Any]]] = None,
        step_number: int = 1,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> KnowledgeRetrievalResult:
        """
        Retrieve relevant knowledge based on query and context.
        
        Args:
            query: The query to use for knowledge retrieval
            task_context: Optional additional context about the task
            previous_thoughts: Optional list of previous thoughts in the reasoning chain
            step_number: Current step number in the sequential thinking process
            metadata_filter: Optional filter to apply to knowledge metadata
            
        Returns:
            KnowledgeRetrievalResult containing the retrieved entries and metrics
        """
        if not self.settings.enabled:
            logger.debug("Knowledge retrieval is disabled")
            return KnowledgeRetrievalResult(
                entries=[],
                metrics={"retrieval_time": 0, "enabled": False},
                query=query
            )
        
        start_time = time.time()
        
        # Create enhanced query by incorporating previous thoughts if available
        enhanced_query = self._enhance_query(query, previous_thoughts, step_number)
        
        # Prepare metadata filter
        final_metadata_filter = self._prepare_metadata_filter(metadata_filter, task_context)
        
        # Retrieve knowledge from vector provider
        try:
            search_results = await self.vector_provider.search(
                query=enhanced_query,
                metadata_filter=final_metadata_filter,
                limit=self.settings.max_results,
                min_score=self.settings.similarity_threshold
            )
            
            # Convert search results to knowledge entries
            entries = []
            for result in search_results:
                entry = KnowledgeEntry(
                    content=result.content,
                    metadata=result.metadata
                )
                entries.append(entry)
                
            # Create formatted context string
            context_str = self._format_context_string(entries, query)
            
            # Calculate metrics
            retrieval_time = time.time() - start_time
            sources = self._extract_sources(entries)
            categories = self._extract_categories(entries)
            
            metrics = {
                "retrieval_time": retrieval_time,
                "results_count": len(entries),
                "query_length": len(enhanced_query),
                "original_query_length": len(query),
                "sources": sources,
                "categories": categories,
                "enabled": True
            }
            
            return KnowledgeRetrievalResult(
                entries=entries,
                metrics=metrics,
                query=enhanced_query,
                context_str=context_str
            )
            
        except Exception as e:
            logger.error(f"Error during knowledge retrieval: {str(e)}")
            return KnowledgeRetrievalResult(
                entries=[],
                metrics={"error": str(e), "retrieval_time": time.time() - start_time},
                query=enhanced_query
            )
    
    def _enhance_query(
        self,
        query: str,
        previous_thoughts: Optional[List[Dict[str, Any]]],
        step_number: int
    ) -> str:
        """
        Enhance the query with context from previous thoughts.
        
        Args:
            query: Original query string
            previous_thoughts: Previous thoughts in the reasoning chain
            step_number: Current step number
            
        Returns:
            Enhanced query string
        """
        if not previous_thoughts or len(previous_thoughts) == 0:
            return query
        
        # For early steps, focus more on the original query
        if step_number <= 2:
            # Just add the most recent thought
            latest_thought = previous_thoughts[-1].get('thought', '')
            
            # Check if thought is too long, if so use a snippet
            if len(latest_thought) > 300:
                latest_thought = latest_thought[:300] + "..."
                
            return f"{query} {latest_thought}"
            
        # For middle steps, incorporate more context from previous thoughts
        elif step_number <= 5:
            # Include the last 2-3 thoughts
            relevant_thoughts = previous_thoughts[-3:] if len(previous_thoughts) >= 3 else previous_thoughts
            thoughts_text = " ".join([t.get('thought', '') for t in relevant_thoughts])
            
            # Summarize if too long
            if len(thoughts_text) > 500:
                thoughts_text = thoughts_text[:500] + "..."
                
            return f"{query} Based on these insights: {thoughts_text}"
            
        # For later steps, use more selective extraction from previous thoughts
        else:
            # Extract key sentences from previous thoughts using simple heuristics
            key_points = self._extract_key_points(previous_thoughts, max_points=5)
            key_points_text = " ".join(key_points)
            
            return f"{query} Considering these key points: {key_points_text}"
    
    def _extract_key_points(
        self,
        thoughts: List[Dict[str, Any]],
        max_points: int = 5
    ) -> List[str]:
        """
        Extract key points from previous thoughts.
        
        Args:
            thoughts: List of thought dictionaries
            max_points: Maximum number of key points to extract
            
        Returns:
            List of key point strings
        """
        key_points = []
        
        # Simple extraction based on sentence indicators
        indicators = [
            "important",
            "key",
            "crucial",
            "significant",
            "essential",
            "note",
            "remember",
            "consider",
            "therefore",
            "thus",
            "hence",
            "consequently",
            "in conclusion"
        ]
        
        for thought in thoughts:
            content = thought.get('thought', '')
            sentences = content.split('. ')
            
            for sentence in sentences:
                # Check if sentence contains any indicator words
                if any(indicator in sentence.lower() for indicator in indicators):
                    # Clean up the sentence
                    clean_sentence = sentence.strip()
                    if clean_sentence and len(clean_sentence) > 10:
                        # Add period if missing
                        if not clean_sentence.endswith('.'):
                            clean_sentence += '.'
                        key_points.append(clean_sentence)
                        
                        # Break if we've reached the maximum points
                        if len(key_points) >= max_points:
                            break
            
            # Break if we've reached the maximum points
            if len(key_points) >= max_points:
                break
                
        return key_points
    
    def _prepare_metadata_filter(
        self,
        user_filter: Optional[Dict[str, Any]],
        task_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare metadata filter for knowledge retrieval.
        
        Args:
            user_filter: User-provided metadata filter
            task_context: Context about the current task
            
        Returns:
            Final metadata filter to use for retrieval
        """
        final_filter = user_filter.copy() if user_filter else {}
        
        # Add task-specific filters if available
        if task_context:
            # Extract task type and domain if available
            task_type = task_context.get('task_type')
            domain = task_context.get('domain')
            
            # Add filters based on task type
            if task_type:
                if task_type == "code_generation":
                    # Prioritize code snippets and best practices
                    final_filter.setdefault("category", ["code_snippet", "best_practice"])
                elif task_type == "explanation":
                    # Prioritize conceptual knowledge and explanations
                    final_filter.setdefault("category", ["concept", "explanation"])
                elif task_type == "problem_solving":
                    # Prioritize approaches and patterns
                    final_filter.setdefault("category", ["best_practice", "approach"])
            
            # Add domain-specific filter if available
            if domain:
                final_filter.setdefault("domain", domain)
                
        return final_filter
    
    def _format_context_string(
        self,
        entries: List[KnowledgeEntry],
        query: str
    ) -> str:
        """
        Format knowledge entries into a context string for model consumption.
        
        Args:
            entries: List of knowledge entries
            query: Original query string
            
        Returns:
            Formatted context string
        """
        if not entries:
            return ""
        
        # Sort entries by relevance (importance score)
        sorted_entries = sorted(
            entries, 
            key=lambda e: e.metadata.get("importance", 0.0), 
            reverse=True
        )
        
        context_parts = [
            f"Relevant knowledge for: {query}\n",
            "Use the following information to inform your response:\n"
        ]
        
        for i, entry in enumerate(sorted_entries, 1):
            category = entry.metadata.get("category", "general")
            source = entry.metadata.get("source", "unknown")
            level = entry.metadata.get("level", "concept")
            
            context_part = f"{i}. [{category.upper()}] {entry.content}\n"
            context_part += f"   Source: {source} | Level: {level}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, entries: List[KnowledgeEntry]) -> Dict[str, int]:
        """
        Extract sources from knowledge entries with count.
        
        Args:
            entries: List of knowledge entries
            
        Returns:
            Dictionary mapping sources to their counts
        """
        sources = {}
        for entry in entries:
            source = entry.metadata.get("source", "unknown")
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        return sources
    
    def _extract_categories(self, entries: List[KnowledgeEntry]) -> Dict[str, int]:
        """
        Extract categories from knowledge entries with count.
        
        Args:
            entries: List of knowledge entries
            
        Returns:
            Dictionary mapping categories to their counts
        """
        categories = {}
        for entry in entries:
            category = entry.metadata.get("category", "general")
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        return categories 