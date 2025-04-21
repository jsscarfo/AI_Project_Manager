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
Knowledge Retrieval Module

This module provides functionality for retrieving knowledge from vector storage
with enhanced contextual awareness for sequential reasoning processes.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np
from dataclasses import dataclass, field

from beeai_framework.vector.base import VectorMemoryProvider, ContextResult, ContextMetadata
from beeai_framework.vector.knowledge_capture import KnowledgeEntry
from beeai_framework.middleware.base import MiddlewareConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrievedKnowledge:
    """
    A piece of knowledge retrieved from vector storage.
    
    This class represents a piece of knowledge that has been retrieved from
    vector storage, along with its metadata and similarity score.
    """
    
    content: str
    """The content of the retrieved knowledge"""
    
    metadata: Dict[str, Any]
    """Metadata associated with the knowledge"""
    
    similarity: float
    """Similarity score (0.0 to 1.0) indicating relevance"""
    
    vector_id: Optional[str] = None
    """Optional ID of the vector in storage"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "similarity": self.similarity,
            "vector_id": self.vector_id
        }


class KnowledgeRetrievalConfig(MiddlewareConfig):
    """Configuration for knowledge retrieval."""
    
    def __init__(
        self,
        enabled: bool = True,
        default_max_results: int = 5,
        default_similarity_threshold: float = 0.7,
        min_content_length: int = 20,
        content_length_limit: int = 2000,
        query_enhancement_template: str = "Retrieve knowledge that would help with: {query}",
        step_specific_prompts: Optional[Dict[str, str]] = None,
        excluded_metadata_sources: Optional[Set[str]] = None,
        relevance_boost_fields: Optional[Dict[str, float]] = None
    ):
        """
        Initialize a knowledge retrieval config.
        
        Args:
            enabled: Whether retrieval is enabled
            default_max_results: Default maximum number of results to return
            default_similarity_threshold: Default threshold for similarity scores (0.0-1.0)
            min_content_length: Minimum content length to consider as relevant
            content_length_limit: Maximum content length to return
            query_enhancement_template: Template for enhancing retrieval queries
            step_specific_prompts: Dictionary mapping step types to specific retrieval prompts
            excluded_metadata_sources: Set of metadata source values to exclude from results
            relevance_boost_fields: Dictionary mapping metadata field names to boost values
        """
        super().__init__(enabled=enabled)
        
        self.default_max_results = default_max_results
        self.default_similarity_threshold = default_similarity_threshold
        self.min_content_length = min_content_length
        self.content_length_limit = content_length_limit
        self.query_enhancement_template = query_enhancement_template
        
        # Default step-specific prompts if none provided
        self.step_specific_prompts = step_specific_prompts or {
            "planning": "Find knowledge that would help plan an approach for: {query}",
            "research": "Find factual information and context related to: {query}",
            "analysis": "Find analytical insights and patterns relevant to: {query}",
            "execution": "Find practical implementation details related to: {query}",
            "verification": "Find validation criteria and testing approaches for: {query}",
            "reflection": "Find evaluation frameworks and improvement ideas for: {query}"
        }
        
        self.excluded_metadata_sources = excluded_metadata_sources or set()
        self.relevance_boost_fields = relevance_boost_fields or {
            "importance": 1.5,
            "verified": 1.3
        }


class KnowledgeRetrievalResult(BaseModel):
    """Result of a knowledge retrieval operation."""
    
    items: List[RetrievedKnowledge] = Field(default_factory=list, description="Retrieved knowledge items")
    query: str = Field(..., description="Original query")
    enhanced_query: Optional[str] = Field(None, description="Enhanced query used for retrieval")
    step_type: Optional[str] = Field(None, description="Type of reasoning step")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Retrieval metrics")
    suggested_next_queries: List[str] = Field(default_factory=list, description="Suggested queries for next steps")


class SequentialThinkingKnowledgeRetriever:
    """
    Retriever for knowledge relevant to sequential thinking steps.
    
    This class provides methods for retrieving knowledge from vector storage
    that is relevant to specific steps in a sequential thinking process.
    """
    
    def __init__(
        self,
        vector_provider: VectorMemoryProvider,
        config: Optional[KnowledgeRetrievalConfig] = None
    ):
        """
        Initialize a sequential thinking knowledge retriever.
        
        Args:
            vector_provider: Provider for vector storage
            config: Retrieval configuration
        """
        self.vector_provider = vector_provider
        self.config = config or KnowledgeRetrievalConfig()
        
        logger.info("Initialized SequentialThinkingKnowledgeRetriever")
    
    async def retrieve_for_step(
        self,
        step_type: str,
        step_number: int,
        system_prompt: str,
        user_prompt: str,
        total_steps: int = 5,
        similarity_threshold: Optional[float] = None,
        max_results: Optional[int] = None,
        weight_multiplier: float = 1.0
    ) -> List[RetrievedKnowledge]:
        """
        Retrieve knowledge relevant to a specific sequential thinking step.
        
        Args:
            step_type: Type of step (planning, research, analysis, etc.)
            step_number: Number of the current step
            system_prompt: System prompt for the step
            user_prompt: User prompt for the step
            total_steps: Total number of steps in the sequence
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            weight_multiplier: Multiplier for relevance weights
            
        Returns:
            List of retrieved knowledge items
        """
        if not self.config.enabled:
            return []
            
        # Use defaults from config if not specified
        similarity_threshold = similarity_threshold or self.config.default_similarity_threshold
        max_results = max_results or self.config.default_max_results
        
        # Combine prompts for retrieval query
        combined_query = self._create_retrieval_query(
            step_type=step_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        # Don't proceed with empty query
        if not combined_query:
            return []
            
        try:
            # Create metadata filters
            metadata_filter = self._create_metadata_filter(step_type)
            
            # Retrieve relevant vectors
            search_results = await self.vector_provider.semantic_search(
                query=combined_query,
                limit=max_results * 2,  # Get more than we need for filtering
                metadata_filter=metadata_filter
            )
            
            # Transform and filter results
            knowledge_items = self._process_search_results(
                search_results=search_results,
                similarity_threshold=similarity_threshold,
                weight_multiplier=weight_multiplier
            )
            
            # Limit results and sort by adjusted similarity
            knowledge_items.sort(key=lambda x: x.similarity, reverse=True)
            filtered_items = knowledge_items[:max_results]
            
            logger.debug(
                f"Retrieved {len(filtered_items)} knowledge items for {step_type} "
                f"step {step_number}/{total_steps}"
            )
            
            return filtered_items
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge for step: {str(e)}")
            return []
    
    async def retrieve_for_query(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        max_results: Optional[int] = None
    ) -> List[RetrievedKnowledge]:
        """
        Retrieve knowledge relevant to a specific query.
        
        Args:
            query: Query string
            metadata_filter: Optional filter for metadata fields
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of retrieved knowledge items
        """
        if not self.config.enabled or not query:
            return []
            
        # Use defaults from config if not specified
        similarity_threshold = similarity_threshold or self.config.default_similarity_threshold
        max_results = max_results or self.config.default_max_results
        
        try:
            # Enhance the query if needed
            enhanced_query = self.config.query_enhancement_template.format(query=query)
            
            # Create combined metadata filter
            combined_filter = self._create_base_metadata_filter()
            if metadata_filter:
                combined_filter.update(metadata_filter)
            
            # Retrieve relevant vectors
            search_results = await self.vector_provider.semantic_search(
                query=enhanced_query,
                limit=max_results * 2,  # Get more than we need for filtering
                metadata_filter=combined_filter
            )
            
            # Transform and filter results
            knowledge_items = self._process_search_results(
                search_results=search_results,
                similarity_threshold=similarity_threshold
            )
            
            # Limit results and sort by similarity
            knowledge_items.sort(key=lambda x: x.similarity, reverse=True)
            filtered_items = knowledge_items[:max_results]
            
            logger.debug(f"Retrieved {len(filtered_items)} knowledge items for query")
            
            return filtered_items
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge for query: {str(e)}")
            return []
    
    def _create_retrieval_query(
        self,
        step_type: str,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Create a retrieval query for a specific step type.
        
        Args:
            step_type: Type of step (planning, research, analysis, etc.)
            system_prompt: System prompt for the step
            user_prompt: User prompt for the step
            
        Returns:
            Combined query string
        """
        # Use user prompt as the base query
        base_query = user_prompt
        
        # Use step-specific prompt template if available
        step_type_lower = step_type.lower()
        if step_type_lower in self.config.step_specific_prompts:
            prompt_template = self.config.step_specific_prompts[step_type_lower]
            return prompt_template.format(query=base_query)
        
        # Fallback to general enhancement template
        return self.config.query_enhancement_template.format(query=base_query)
    
    def _create_metadata_filter(self, step_type: str) -> Dict[str, Any]:
        """
        Create a metadata filter for a specific step type.
        
        Args:
            step_type: Type of step (planning, research, analysis, etc.)
            
        Returns:
            Metadata filter dictionary
        """
        # Start with base filter
        metadata_filter = self._create_base_metadata_filter()
        
        # Add step-specific filters if needed
        if step_type.lower() == "planning":
            # For planning, we might prioritize high-level concepts
            pass
        elif step_type.lower() == "research":
            # For research, we might prioritize factual information
            pass
        
        return metadata_filter
    
    def _create_base_metadata_filter(self) -> Dict[str, Any]:
        """
        Create a base metadata filter.
        
        Returns:
            Base metadata filter dictionary
        """
        base_filter = {}
        
        # Exclude specified metadata sources
        if self.config.excluded_metadata_sources:
            base_filter["metadata.source"] = {"$nin": list(self.config.excluded_metadata_sources)}
        
        return base_filter
    
    def _process_search_results(
        self,
        search_results: List[Tuple[str, Dict[str, Any], float]],
        similarity_threshold: float,
        weight_multiplier: float = 1.0
    ) -> List[RetrievedKnowledge]:
        """
        Process search results into retrieved knowledge items.
        
        Args:
            search_results: List of (content, metadata, similarity) tuples
            similarity_threshold: Minimum similarity threshold
            weight_multiplier: Multiplier for relevance weights
            
        Returns:
            List of retrieved knowledge items
        """
        knowledge_items = []
        
        for content, metadata, similarity in search_results:
            # Apply basic filtering
            if similarity < similarity_threshold:
                continue
                
            if len(content) < self.config.min_content_length:
                continue
                
            # Adjust similarity score based on metadata fields
            adjusted_similarity = self._adjust_similarity_score(
                similarity=similarity,
                metadata=metadata,
                weight_multiplier=weight_multiplier
            )
            
            # Extract vector ID if available
            vector_id = metadata.get("vector_id", None)
            
            # Limit content length if needed
            if len(content) > self.config.content_length_limit:
                content = content[:self.config.content_length_limit] + "..."
            
            # Create knowledge item
            knowledge_item = RetrievedKnowledge(
                content=content,
                metadata=metadata,
                similarity=adjusted_similarity,
                vector_id=vector_id
            )
            
            knowledge_items.append(knowledge_item)
        
        return knowledge_items
    
    def _adjust_similarity_score(
        self,
        similarity: float,
        metadata: Dict[str, Any],
        weight_multiplier: float = 1.0
    ) -> float:
        """
        Adjust similarity score based on metadata fields.
        
        Args:
            similarity: Original similarity score
            metadata: Metadata dictionary
            weight_multiplier: Multiplier for relevance weights
            
        Returns:
            Adjusted similarity score
        """
        adjusted_score = similarity
        
        # Apply boosts for specific metadata fields
        for field, boost in self.config.relevance_boost_fields.items():
            if field in metadata and metadata[field]:
                # Apply the boost - simple multiplicative model
                field_value = metadata[field]
                
                # Handle boolean fields
                if isinstance(field_value, bool) and field_value:
                    adjusted_score *= boost
                
                # Handle numeric fields
                elif isinstance(field_value, (int, float)):
                    # Scale the boost based on the value (assuming 0-1 range)
                    normalized_value = min(max(field_value, 0), 1)
                    adjusted_score *= (1.0 + (boost - 1.0) * normalized_value)
        
        # Apply overall weight multiplier
        adjusted_score *= weight_multiplier
        
        # Cap at 1.0
        return min(adjusted_score, 1.0)


class StepContextManager:
    """
    Manages context across sequential thinking steps.
    
    This class tracks context across steps, handles transitions between
    reasoning phases, and optimizes the context window usage.
    """
    
    def __init__(
        self,
        knowledge_retriever: SequentialThinkingKnowledgeRetriever,
        max_context_items_per_step: int = 5,
        enable_context_carryover: bool = True,
        token_budget: int = 4000
    ):
        """
        Initialize the step context manager.
        
        Args:
            knowledge_retriever: Retriever for knowledge
            max_context_items_per_step: Maximum items to include per step
            enable_context_carryover: Whether to carry context between steps
            token_budget: Maximum tokens for context
        """
        self.knowledge_retriever = knowledge_retriever
        self.max_context_items_per_step = max_context_items_per_step
        self.enable_context_carryover = enable_context_carryover
        self.token_budget = token_budget
        
        # Track context across steps
        self.current_context_items = []
        self.step_history = []
        self.context_usage = []
    
    async def get_context_for_step(
        self,
        query: str,
        step_type: str,
        step_number: int,
        explicit_concepts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get optimized context for a specific reasoning step.
        
        Args:
            query: Query for this step
            step_type: Type of reasoning step
            step_number: Position in the reasoning sequence
            explicit_concepts: Optional explicitly mentioned concepts
            
        Returns:
            Dictionary with context and metadata
        """
        # Add current query to history
        step_data = {
            "query": query,
            "step_type": step_type,
            "step_number": step_number,
            "timestamp": datetime.now().isoformat()
        }
        self.step_history.append(step_data)
        
        # Retrieve knowledge for this step
        result = await self.knowledge_retriever.retrieve_for_step(
            step_type=step_type,
            step_number=step_number,
            system_prompt=query,
            user_prompt=query,
            total_steps=5
        )
        
        # Update context items
        if self.enable_context_carryover:
            self._update_context_items(result)
        else:
            self.current_context_items = result
        
        # Format context for LLM consumption
        formatted_context = self._format_context_for_llm(step_type)
        
        # Track context usage
        self.context_usage.append({
            "step_number": step_number,
            "step_type": step_type,
            "context_items_count": len(self.current_context_items),
            "retrieval_time_ms": sum(item.similarity for item in result) * 1000
        })
        
        # Return context data
        return {
            "formatted_context": formatted_context,
            "context_items": [item.to_dict() for item in self.current_context_items],
            "retrieval_metrics": {
                "retrieval_time_ms": sum(item.similarity for item in result) * 1000,
                "result_count": len(result)
            },
            "suggested_queries": self._generate_suggested_queries(result, step_type, step_number)
        }
    
    def _update_context_items(self, new_items: List[RetrievedKnowledge]) -> None:
        """
        Update the current context items with new items.
        
        Args:
            new_items: New items to potentially include in context
        """
        # Merge current and new items, with new items taking precedence
        combined_items = []
        existing_contents = {item.content for item in self.current_context_items}
        
        # Keep high-scoring existing items
        for item in self.current_context_items:
            # Only keep items with good scores
            if item.similarity > 0.7:
                combined_items.append(item)
                
        # Add new items
        for item in new_items:
            # Skip duplicates
            if item.content in existing_contents:
                continue
                
            combined_items.append(item)
            existing_contents.add(item.content)
        
        # Sort by score and limit
        combined_items.sort(key=lambda x: x.similarity, reverse=True)
        self.current_context_items = combined_items[:self.max_context_items_per_step]
    
    def _format_context_for_llm(self, step_type: str) -> str:
        """
        Format context items for LLM consumption.
        
        Args:
            step_type: Current step type
            
        Returns:
            Formatted context string
        """
        if not self.current_context_items:
            return ""
            
        formatted_parts = ["### Relevant Context for " + step_type.replace("_", " ").title() + ":\n"]
        
        for i, item in enumerate(self.current_context_items, 1):
            formatted_parts.append(f"[{i}] {item.metadata.get('source', 'unknown')} ({item.metadata.get('level', 'unknown')}):")
            formatted_parts.append(item.content)
            formatted_parts.append("\n")
            
        formatted_parts.append(
            "Use the above context to inform your reasoning. "
            "You don't need to mention the context directly unless relevant."
        )
            
        return "\n".join(formatted_parts)
    
    def get_context_usage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about context usage.
        
        Returns:
            Dictionary with usage statistics
        """
        if not self.context_usage:
            return {"steps": 0}
            
        return {
            "steps": len(self.context_usage),
            "avg_items_per_step": sum(u["context_items_count"] for u in self.context_usage) / len(self.context_usage),
            "avg_retrieval_time_ms": sum(u["retrieval_time_ms"] for u in self.context_usage) / len(self.context_usage),
            "step_distribution": {u["step_type"]: self.context_usage.count(u["step_type"]) for u in self.context_usage}
        }
    
    def _generate_suggested_queries(
        self,
        results: List[RetrievedKnowledge],
        step_type: str,
        step_number: int
    ) -> List[str]:
        """
        Generate suggested queries for next steps based on retrieved content.
        
        Args:
            results: Current retrieval results
            step_type: Current step type
            step_number: Current step number
            
        Returns:
            List of suggested queries
        """
        # If we don't have results, return empty suggestions
        if not results:
            return []
            
        # Define query templates for different step transitions
        next_step_templates = {
            "planning": [
                "What approach should I use for {concept}?",
                "How to implement {concept}?"
            ],
            "research": [
                "What are the key considerations for {concept}?",
                "Best practices for {concept} implementation"
            ],
            "analysis": [
                "How to optimize {concept}?",
                "Implementation strategies for {concept}"
            ],
            "execution": [
                "Code example for {concept}",
                "How to test {concept} implementation?"
            ]
        }
        
        # Get templates for current step
        templates = next_step_templates.get(step_type, ["More details about {concept}"])
        
        # Extract key concepts from results (simplified version)
        concepts = set()
        for result in results[:3]:  # Use top 3 results
            # Simple concept extraction - would be more sophisticated in production
            extracted = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', result.content)
            concepts.update([c for c in extracted if len(c) > 3])  # Filter very short concepts
            
        # Generate suggestions
        suggestions = []
        for concept in list(concepts)[:2]:  # Limit to top 2 concepts
            for template in templates:
                suggestions.append(template.format(concept=concept))
                
        return suggestions[:3]  # Return top 3 suggestions 