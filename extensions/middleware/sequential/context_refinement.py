#!/usr/bin/env python
"""
Context Refinement Mechanisms for Sequential Thinking.

This module implements advanced context selection and refinement techniques
for sequential thinking processes, optimizing each step with the most
relevant contextual information.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextQualityMetrics(BaseModel):
    """Metrics for evaluating context quality."""
    relevance_score: float = Field(..., description="Relevance to current step (0-1)")
    specificity_score: float = Field(..., description="Specificity of information (0-1)")
    recency_score: float = Field(..., description="Recency of information (0-1)")
    information_density: float = Field(..., description="Information density (0-1)")
    
    def get_weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate a weighted score based on the metrics.
        
        Args:
            weights: Optional weights for each metric (default is equal weights)
            
        Returns:
            Weighted score between 0-1
        """
        default_weights = {
            'relevance_score': 0.4,
            'specificity_score': 0.3,
            'recency_score': 0.1,
            'information_density': 0.2
        }
        
        w = weights or default_weights
        
        return (
            w.get('relevance_score', 0.4) * self.relevance_score +
            w.get('specificity_score', 0.3) * self.specificity_score +
            w.get('recency_score', 0.1) * self.recency_score +
            w.get('information_density', 0.2) * self.information_density
        )


class ContextItem(BaseModel):
    """A piece of context with associated metadata and quality metrics."""
    content: str = Field(..., description="The actual context content")
    source: str = Field(..., description="Source of the context")
    level: str = Field(..., description="Hierarchical level (domain, techstack, project)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    quality_metrics: Optional[ContextQualityMetrics] = Field(None, description="Quality metrics")
    vector_id: Optional[str] = Field(None, description="ID in the vector database")
    
    def get_quality_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Get the weighted quality score for this context.
        
        Args:
            weights: Optional weighting for quality metrics
            
        Returns:
            Quality score between 0-1, or 0 if no metrics are available
        """
        if not self.quality_metrics:
            return 0.0
            
        return self.quality_metrics.get_weighted_score(weights)


class ContextRefinementProcessor:
    """
    Processor for refining context in sequential thinking.
    
    This class handles context selection, evaluation, and refinement
    to optimize the context provided at each step of a sequential
    thinking process.
    """
    
    def __init__(self, 
                 vector_provider: Any, 
                 embedding_service: Any,
                 context_window_size: int = 4000,
                 max_items_per_level: Dict[str, int] = None,
                 enable_progressive_refinement: bool = True,
                 enable_content_weighting: bool = True,
                 weaviate_client = None):
        """
        Initialize context refinement processor.
        
        Args:
            vector_provider: Provider for vector database operations
            embedding_service: Service for generating embeddings
            context_window_size: Maximum context window size in tokens
            max_items_per_level: Maximum items to retrieve per hierarchical level
            enable_progressive_refinement: Whether to progressively refine context
            enable_content_weighting: Whether to weight content based on metrics
            weaviate_client: Optional Weaviate client for direct integration
        """
        self.vector_provider = vector_provider
        self.embedding_service = embedding_service
        self.context_window_size = context_window_size
        self.max_items_per_level = max_items_per_level or {
            'domain': 3,
            'techstack': 5,
            'project': 10
        }
        self.enable_progressive_refinement = enable_progressive_refinement
        self.enable_content_weighting = enable_content_weighting
        self.weaviate_client = weaviate_client
        
        # Cache for context items to avoid repeated retrieval
        self._context_cache = {}

    async def get_enhanced_context(self,
                                 prompt: str,
                                 task_type: str,
                                 step_number: int,
                                 previous_steps: Optional[List[str]] = None,
                                 template_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get context optimized for a specific task type and step number.
        
        This method provides a higher-level interface that determines the
        appropriate step type and applies template-based optimizations.
        
        Args:
            prompt: Current prompt
            task_type: Type of task (e.g., 'planning', 'coding')
            step_number: Current step number in the sequence
            previous_steps: Previous steps in the thinking process
            template_config: Optional template configuration
            
        Returns:
            Dictionary with relevant context and metadata
        """
        # Determine step type based on step number and task type
        step_type = self._determine_step_type(step_number, task_type)
        
        # Apply template-specific settings if available
        if template_config:
            return await self._get_template_guided_context(
                prompt, 
                step_type, 
                template_config,
                previous_steps
            )
            
        # Fall back to standard context retrieval
        return await self.get_context_for_step(prompt, step_type, previous_steps)
        
    def _determine_step_type(self, step_number: int, task_type: str) -> str:
        """
        Determine step type based on step number and task type.
        
        Args:
            step_number: Current step number in the sequence
            task_type: Type of task
            
        Returns:
            Step type string
        """
        # Common step mapping for most task types
        if step_number == 1:
            return "problem_definition"
        elif step_number == 2:
            return "information_gathering"
        
        # Task-specific mappings for later steps
        if task_type == "coding":
            if step_number == 3:
                return "design_planning"
            elif step_number < 6:
                return "implementation"
            else:
                return "verification"
        elif task_type == "debugging":
            if step_number == 3:
                return "error_analysis"
            elif step_number < 6:
                return "solution_identification" 
            else:
                return "verification"
        elif task_type == "planning":
            if step_number == 3:
                return "analysis"
            elif step_number < 6:
                return "solution_design"
            else:
                return "conclusion"
        else:
            # Generic mapping for other task types
            if step_number == 3:
                return "analysis"
            elif step_number < 6:
                return "solution_formulation"
            else:
                return "conclusion"
                
    async def _get_template_guided_context(self,
                                         prompt: str,
                                         step_type: str,
                                         template_config: Dict[str, Any],
                                         previous_steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get context optimized using template settings.
        
        Args:
            prompt: Current prompt
            step_type: Type of step
            template_config: Template configuration with step-specific settings
            previous_steps: Previous steps in the thinking process
            
        Returns:
            Dictionary with relevant context and metadata
        """
        # Extract step config if available
        step_config = template_config.get("step_definitions", {}).get(step_type, {})
        
        # Default levels from template or use standard levels
        levels = step_config.get("levels") or self._get_levels_for_step(step_type)
        
        # Adjust query counts and limits based on template
        query_count = step_config.get("query_count", 2)
        context_limit = step_config.get("context_limit", 5)
        
        # Get level weights from template
        level_weights = step_config.get("level_weights") or template_config.get("level_weights")
        
        # Use previous steps based on template configuration
        use_previous_step = step_config.get("use_previous_step", True)
        prev_steps = previous_steps if use_previous_step else None
        
        # Generate queries with template guidance
        queries = self._generate_step_specific_queries(
            prompt, 
            step_type, 
            prev_steps,
            query_count=query_count
        )
        
        # Retrieve and process context items
        context_candidates = await self._retrieve_context_candidates(queries, levels)
        evaluated_context = await self._evaluate_context_quality(context_candidates, prompt, step_type)
        
        # Apply template-specific weights during selection
        refined_context = self._select_and_refine_context(
            evaluated_context, 
            levels, 
            max_per_level=context_limit,
            level_weights=level_weights
        )
        
        # Optimize for context window
        optimized_context = self._optimize_for_context_window(refined_context)
        
        return {
            'relevant_context': self._format_context_items(optimized_context),
            'context_metadata': {
                'step': step_type,
                'quality_score': self._calculate_aggregate_quality(optimized_context),
                'item_count': len(optimized_context),
                'knowledge_levels': levels,
                'template_applied': True
            }
        }
    
    def _generate_step_specific_queries(self, 
                                       prompt: str, 
                                       step: str, 
                                       previous_steps: Optional[List[str]] = None,
                                       query_count: int = 2) -> List[str]:
        """
        Generate queries tailored to the specific step.
        
        Args:
            prompt: Current prompt
            step: Step identifier
            previous_steps: Previous steps in the thinking process
            query_count: Number of queries to generate
            
        Returns:
            List of queries to run against the vector store
        """
        queries = [prompt]  # Base query is always the prompt
        
        # Add step-specific queries
        if step == "problem_definition":
            queries.append(f"key concepts in {prompt}")
            if query_count > 2:
                queries.append(f"definition of terms in {prompt}")
                
        elif step == "information_gathering":
            queries.append(f"relevant information for {prompt}")
            if previous_steps and len(previous_steps) > 0:
                # Extract key terms from first step
                key_terms = self._extract_key_terms(previous_steps[0])
                if key_terms:
                    queries.append(f"information about {', '.join(key_terms[:3])}")
            if query_count > 2:
                queries.append(f"background knowledge for {prompt}")
                
        elif step == "analysis":
            if previous_steps and len(previous_steps) > 1:
                # Use content from the information gathering step
                info_step = previous_steps[1]
                queries.append(f"analysis of {info_step[:100]}")
            queries.append(f"approaches to analyze {prompt}")
            if query_count > 2:
                queries.append(f"techniques relevant to {prompt}")
                
        elif step == "error_analysis":
            queries.append(f"common errors related to {prompt}")
            if query_count > 2:
                queries.append(f"debugging techniques for {prompt}")
                
        elif step == "design_planning":
            queries.append(f"design patterns for {prompt}")
            if query_count > 2:
                queries.append(f"architectural considerations for {prompt}")
                
        elif step == "implementation" or step == "solution_formulation":
            if previous_steps and len(previous_steps) > 2:
                # Reference the design/analysis from previous step
                queries.append(f"implementation of {previous_steps[2][:100]}")
            queries.append(f"code examples for {prompt}")
            if query_count > 2:
                queries.append(f"implementation best practices for {prompt}")
                
        elif step == "verification":
            queries.append(f"testing {prompt}")
            if query_count > 2:
                queries.append(f"validation techniques for {prompt}")
                
        elif step == "conclusion":
            queries.append(f"summary of {prompt}")
            if query_count > 2 and previous_steps:
                last_step = previous_steps[-1]
                queries.append(f"finalizing {last_step[:100]}")
        
        # Limit to the requested number of queries
        return queries[:query_count]
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text.
        
        Args:
            text: Text to extract terms from
            
        Returns:
            List of key terms
        """
        # Simple implementation - extract capitalized phrases and technical terms
        import re
        
        # Find capitalized phrases (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
        
        # Find potential technical terms (camelCase, snake_case, etc.)
        technical = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+(?:_[a-z]+)+\b', text)
        
        # Combine and remove duplicates while preserving order
        seen = set()
        key_terms = []
        
        for term in capitalized + technical:
            if term.lower() not in seen:
                seen.add(term.lower())
                key_terms.append(term)
        
        return key_terms

    async def _retrieve_context_candidates(self, 
                                        queries: List[str], 
                                        levels: List[str]) -> List[ContextItem]:
        """
        Retrieve context candidates from vector store.
        
        Args:
            queries: Queries to run against the vector store
            levels: Knowledge levels to query
            
        Returns:
            List of context items
        """
        # Generate cache key
        cache_key = f"{'-'.join(queries)}|{'-'.join(levels)}"
        
        # Check cache first
        if cache_key in self._context_cache:
            logger.debug(f"Using cached context for {cache_key}")
            return self._context_cache[cache_key]
            
        context_items = []
        
        try:
            # Process all queries
            for query in queries:
                # Process all levels for this query
                for level in levels:
                    # Get max items for this level
                    max_items = self.max_items_per_level.get(level, 5)
                    
                    try:
                        # Use direct Weaviate integration if available
                        if self.weaviate_client and hasattr(self, '_query_weaviate'):
                            items = await self._query_weaviate(query, level, max_items)
                        else:
                            # Fallback to standard vector provider
                            metadata_filter = {"level": level}
                            vector_results = await self.vector_provider.get_context(
                                query=query,
                                metadata_filter=metadata_filter,
                                limit=max_items
                            )
                            
                            # Convert to ContextItem format
                            items = [
                                ContextItem(
                                    content=item.get("content"),
                                    source=item.get("metadata", {}).get("source", "unknown"),
                                    level=level,
                                    metadata=item.get("metadata", {}),
                                    vector_id=item.get("id")
                                )
                                for item in vector_results
                            ]
                        
                        context_items.extend(items)
                    except Exception as e:
                        logger.error(f"Error retrieving context for {level}/{query}: {str(e)}")
            
            # Cache the results
            self._context_cache[cache_key] = context_items
            
            return context_items
        
        except Exception as e:
            logger.error(f"Error in context retrieval: {str(e)}")
            return []
            
    async def _query_weaviate(self, query: str, level: str, limit: int) -> List[ContextItem]:
        """
        Query Weaviate directly for context items.
        
        Args:
            query: Search query
            level: Knowledge level
            limit: Maximum number of results
            
        Returns:
            List of context items
        """
        if not self.weaviate_client:
            return []
            
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(query)
            
            # Determine class name based on level (example mapping)
            class_mapping = {
                "domain": "DomainKnowledge",
                "techstack": "TechStackKnowledge",
                "project": "ProjectKnowledge"
            }
            
            class_name = class_mapping.get(level, "Knowledge")
            
            # Build and execute query
            result = (
                self.weaviate_client.query
                .get(class_name, ["content", "source", "timestamp", "id"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(limit)
                .do()
            )
            
            # Extract results
            items = []
            if result and "data" in result and "Get" in result["data"]:
                results = result["data"]["Get"].get(class_name, [])
                
                for item in results:
                    context_item = ContextItem(
                        content=item.get("content", ""),
                        source=item.get("source", "weaviate"),
                        level=level,
                        metadata={
                            "timestamp": item.get("timestamp"),
                            "source": item.get("source", "weaviate"),
                            "level": level
                        },
                        vector_id=item.get("id")
                    )
                    items.append(context_item)
                    
            return items
        
        except Exception as e:
            logger.error(f"Error querying Weaviate: {str(e)}")
            return []

    def _select_and_refine_context(self, 
                              context_items: List[ContextItem], 
                              levels: List[str],
                              max_per_level: Optional[int] = None,
                              level_weights: Optional[Dict[str, float]] = None) -> List[ContextItem]:
        """
        Select and refine context based on quality metrics and levels.
        
        Args:
            context_items: Evaluated context items
            levels: Knowledge levels to include
            max_per_level: Optional maximum items per level (overrides default)
            level_weights: Optional weights for different levels
            
        Returns:
            List of selected and refined context items
        """
        if not context_items:
            return []
            
        # Group by level
        items_by_level = {}
        for level in levels:
            items_by_level[level] = [
                item for item in context_items 
                if item.level == level
            ]
            
        # Sort each group by quality score
        for level in levels:
            if self.enable_content_weighting and level_weights:
                # Apply level-specific weights when sorting
                items_by_level[level].sort(
                    key=lambda x: x.get_quality_score(level_weights),
                    reverse=True
                )
            else:
                # Sort by standard quality score
                items_by_level[level].sort(
                    key=lambda x: x.get_quality_score(),
                    reverse=True
                )
            
        # Select top items from each level
        selected_items = []
        for level in levels:
            if level not in items_by_level:
                continue
                
            # Determine max items for this level
            level_max = max_per_level or self.max_items_per_level.get(level, 5)
            
            # Add top items from this level
            selected_items.extend(items_by_level[level][:level_max])
            
        return selected_items

    async def get_context_for_step(self, 
                                 prompt: str, 
                                 step: str,
                                 previous_steps: Optional[List[str]] = None,
                                 current_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimized context for a specific step in sequential thinking.
        
        Args:
            prompt: Current prompt for this step
            step: Step identifier (e.g., 'problem_definition', 'analysis')
            previous_steps: Previous steps in the thinking process
            current_context: Current context from earlier refinement
            
        Returns:
            Dictionary containing relevant context and metadata
        """
        # Determine appropriate knowledge levels based on step
        levels = self._get_levels_for_step(step)
        
        # Generate queries based on the step, prompt, and previous steps
        queries = self._generate_step_specific_queries(prompt, step, previous_steps)
        
        # Retrieve initial context candidates from vector store
        context_candidates = await self._retrieve_context_candidates(queries, levels)
        
        # Evaluate context quality
        evaluated_context = await self._evaluate_context_quality(context_candidates, prompt, step)
        
        # Select and refine context based on quality metrics
        refined_context = self._select_and_refine_context(evaluated_context, levels)
        
        # Optimize for context window
        optimized_context = self._optimize_for_context_window(refined_context)
        
        # Combine with current context if available
        if current_context and current_context.get('relevant_context'):
            # Progressive refinement - combine with existing context
            return self._combine_with_current_context(optimized_context, current_context)
        
        return {
            'relevant_context': self._format_context_items(optimized_context),
            'context_metadata': {
                'step': step,
                'quality_score': self._calculate_aggregate_quality(optimized_context),
                'item_count': len(optimized_context),
                'knowledge_levels': levels
            }
        }
    
    def _get_levels_for_step(self, step: str) -> List[str]:
        """
        Determine appropriate knowledge levels based on the step.
        
        Args:
            step: The current step identifier
            
        Returns:
            List of knowledge levels to query
        """
        # Different steps need different levels of context
        if step == "problem_definition":
            return ["domain", "project"]
        elif step == "information_gathering":
            return ["domain", "techstack", "project"]
        elif step == "analysis":
            return ["techstack", "project"]
        elif step == "solution_formulation":
            return ["techstack", "project"]
        else:  # Default for conclusion or unknown steps
            return ["project"]
    
    async def _evaluate_context_quality(self, 
                                      context_items: List[ContextItem], 
                                      prompt: str, 
                                      step: str) -> List[ContextItem]:
        """
        Evaluate the quality of context items.
        
        Args:
            context_items: List of context items to evaluate
            prompt: Current prompt
            step: Current step
            
        Returns:
            Context items with quality metrics added
        """
        # Generate embedding for the prompt
        try:
            prompt_embedding = await self.embedding_service.get_embedding(prompt)
        except Exception as e:
            logger.error(f"Error generating prompt embedding: {str(e)}")
            # Fall back to simpler metrics if embedding fails
            return self._fallback_quality_evaluation(context_items, prompt)
        
        evaluated_items = []
        
        for item in context_items:
            try:
                # Get item embedding
                item_content = item.content
                item_embedding = await self.embedding_service.get_embedding(item_content)
                
                # Calculate relevance using cosine similarity
                relevance = self._cosine_similarity(prompt_embedding, item_embedding)
                
                # Calculate other metrics
                specificity = self._calculate_specificity(item_content)
                recency = self._get_recency_score(item.metadata)
                information_density = self._calculate_information_density(item_content)
                
                # Create quality metrics
                metrics = ContextQualityMetrics(
                    relevance_score=relevance,
                    specificity_score=specificity,
                    recency_score=recency,
                    information_density=information_density
                )
                
                # Update item with metrics
                item.quality_metrics = metrics
                evaluated_items.append(item)
                
            except Exception as e:
                logger.warning(f"Error evaluating context item quality: {str(e)}")
                # Still include the item, but without quality metrics
                evaluated_items.append(item)
        
        return evaluated_items
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return max(0.0, min(1.0, dot_product / (norm_a * norm_b)))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def _calculate_specificity(self, text: str) -> float:
        """
        Calculate specificity of text based on lexical features.
        
        Args:
            text: Text to analyze
            
        Returns:
            Specificity score (0-1)
        """
        # Simple heuristic based on length, technical terms, etc.
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Longer texts tend to be more specific
        length_factor = min(1.0, word_count / 200)
        
        # Check for specific indicators like code, numbers, technical terms
        has_code = '{' in text or '}' in text or '(' in text or ')' in text
        has_numbers = any(c.isdigit() for c in text)
        
        # Calculate final score
        specificity = length_factor * 0.6
        if has_code:
            specificity += 0.2
        if has_numbers:
            specificity += 0.2
            
        return min(1.0, specificity)
    
    def _get_recency_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate recency score based on metadata.
        
        Args:
            metadata: Item metadata
            
        Returns:
            Recency score (0-1)
        """
        # Check if timestamp exists
        if 'timestamp' in metadata:
            try:
                import datetime
                
                # Parse timestamp
                if isinstance(metadata['timestamp'], str):
                    from dateutil import parser
                    timestamp = parser.parse(metadata['timestamp'])
                elif isinstance(metadata['timestamp'], (int, float)):
                    timestamp = datetime.datetime.fromtimestamp(metadata['timestamp'])
                else:
                    return 0.5  # Default if timestamp format is unknown
                
                # Calculate age in days
                now = datetime.datetime.now()
                age_days = (now - timestamp).days
                
                # Recency score decreases with age
                # 0 days old = 1.0, 30 days old = 0.5, 90+ days old = 0.1
                if age_days <= 0:
                    return 1.0
                elif age_days <= 30:
                    return 1.0 - (age_days / 60)
                else:
                    return max(0.1, 0.5 - ((age_days - 30) / 120))
                
            except Exception as e:
                logger.warning(f"Error calculating recency: {str(e)}")
                return 0.5  # Default on error
        
        return 0.5  # No timestamp available
    
    def _calculate_information_density(self, text: str) -> float:
        """
        Calculate information density of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Information density score (0-1)
        """
        # Simple heuristic based on ratio of unique words to total words
        words = text.lower().split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        unique_words = len(set(words))
        
        # Calculate unique word ratio
        unique_ratio = unique_words / word_count
        
        # Adjust for very short texts
        if word_count < 10:
            return unique_ratio * 0.5
        
        # Bonus for longer texts with good unique ratio
        density = unique_ratio * 0.8
        
        # Bonus for presence of key indicators of dense information
        if ':' in text:
            density += 0.05
        if '=' in text or '->' in text:
            density += 0.05
        if any(c.isdigit() for c in text):
            density += 0.1
            
        return min(1.0, density)
    
    def _fallback_quality_evaluation(self, 
                                    context_items: List[ContextItem], 
                                    prompt: str) -> List[ContextItem]:
        """
        Fallback method for quality evaluation when embeddings aren't available.
        
        Args:
            context_items: List of context items to evaluate
            prompt: Current prompt
            
        Returns:
            Context items with basic quality metrics added
        """
        prompt_words = set(prompt.lower().split())
        
        for item in context_items:
            content = item.content.lower()
            content_words = set(content.split())
            
            # Simple word overlap as relevance
            overlap = len(prompt_words.intersection(content_words))
            relevance = min(1.0, overlap / max(1, len(prompt_words)))
            
            # Calculate other metrics
            specificity = self._calculate_specificity(item.content)
            recency = self._get_recency_score(item.metadata)
            information_density = self._calculate_information_density(item.content)
            
            # Create quality metrics
            metrics = ContextQualityMetrics(
                relevance_score=relevance,
                specificity_score=specificity,
                recency_score=recency,
                information_density=information_density
            )
            
            # Update item with metrics
            item.quality_metrics = metrics
            
        return context_items
    
    def _optimize_for_context_window(self, context_items: List[ContextItem]) -> List[ContextItem]:
        """
        Optimize context to fit within context window constraints.
        
        Args:
            context_items: Selected context items
            
        Returns:
            Optimized list of context items
        """
        # Simple word count estimation of tokens (rough approximation)
        def estimate_tokens(text: str) -> int:
            return len(text.split()) * 1.3  # Average tokens per word
        
        # If under context window, return all items
        total_tokens = sum(estimate_tokens(item.content) for item in context_items)
        if total_tokens <= self.context_window_size:
            return context_items
        
        # Otherwise, need to prioritize
        # Sort by quality score
        sorted_items = sorted(context_items, key=lambda x: x.get_quality_score(), reverse=True)
        
        # Add items until we approach context window limit
        optimized_items = []
        current_tokens = 0
        
        for item in sorted_items:
            item_tokens = estimate_tokens(item.content)
            
            # If this item would exceed our limit, skip it
            if current_tokens + item_tokens > self.context_window_size:
                # Unless it's a very high quality item, then truncate it
                quality = item.get_quality_score()
                if quality > 0.8:
                    # Truncate content to fit
                    available_tokens = self.context_window_size - current_tokens
                    words_to_keep = int(available_tokens / 1.3)
                    if words_to_keep >= 30:  # Only keep if we can include enough to be useful
                        truncated_content = " ".join(item.content.split()[:words_to_keep])
                        truncated_item = ContextItem(
                            content=truncated_content + " [truncated]",
                            source=item.source,
                            level=item.level,
                            metadata=item.metadata,
                            quality_metrics=item.quality_metrics,
                            vector_id=item.vector_id
                        )
                        optimized_items.append(truncated_item)
                        current_tokens += estimate_tokens(truncated_item.content)
                        
                continue
            
            optimized_items.append(item)
            current_tokens += item_tokens
            
            if current_tokens >= self.context_window_size:
                break
                
        return optimized_items
    
    def _format_context_items(self, context_items: List[ContextItem]) -> str:
        """
        Format context items into a string for inclusion in prompt.
        
        Args:
            context_items: Context items to format
            
        Returns:
            Formatted context string
        """
        if not context_items:
            return ""
        
        formatted = []
        
        # Group by level for better organization
        items_by_level = {}
        for item in context_items:
            level = item.level
            if level not in items_by_level:
                items_by_level[level] = []
            items_by_level[level].append(item)
        
        # Format each level
        for level, items in items_by_level.items():
            if items:
                formatted.append(f"--- {level.upper()} KNOWLEDGE ---")
                for item in items:
                    formatted.append(f"[Source: {item.source}]")
                    formatted.append(item.content)
                    formatted.append("")  # Empty line between items
        
        return "\n".join(formatted)
    
    def _calculate_aggregate_quality(self, context_items: List[ContextItem]) -> float:
        """
        Calculate aggregate quality score for the context.
        
        Args:
            context_items: Context items to evaluate
            
        Returns:
            Aggregate quality score (0-1)
        """
        if not context_items:
            return 0.0
        
        # Calculate weighted average based on content length
        total_length = sum(len(item.content) for item in context_items)
        
        if total_length == 0:
            return 0.0
        
        weighted_sum = sum(
            item.get_quality_score() * len(item.content) / total_length
            for item in context_items
            if item.quality_metrics is not None
        )
        
        return weighted_sum
    
    def _combine_with_current_context(self, 
                                     new_context: List[ContextItem], 
                                     current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine new context with current context for progressive refinement.
        
        Args:
            new_context: New context items
            current_context: Current context from previous steps
            
        Returns:
            Combined context
        """
        # Get current context string
        current_context_str = current_context.get('relevant_context', '')
        
        # Format new context
        new_context_str = self._format_context_items(new_context)
        
        # Combine them, avoiding duplication
        if not current_context_str:
            combined_context = new_context_str
        elif not new_context_str:
            combined_context = current_context_str
        else:
            # Simple combination with separator
            combined_context = f"{current_context_str}\n\n--- ADDITIONAL CONTEXT ---\n\n{new_context_str}"
        
        # Get metadata
        current_metadata = current_context.get('context_metadata', {})
        new_metadata = {
            'step': current_metadata.get('step', 'unknown'),
            'quality_score': self._calculate_aggregate_quality(new_context),
            'item_count': len(new_context) + current_metadata.get('item_count', 0),
            'knowledge_levels': current_metadata.get('knowledge_levels', [])
        }
        
        return {
            'relevant_context': combined_context,
            'context_metadata': new_metadata
        } 