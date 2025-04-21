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
Sequential Thinking - Knowledge Retrieval Integration

This module provides the integration between the Sequential Thinking
middleware and Knowledge Retrieval systems, allowing for context-aware
reasoning with knowledge from the vector store.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Set
from pydantic import BaseModel, Field
from datetime import datetime
import re

from beeai_framework.vector.knowledge_retrieval import (
    SequentialThinkingKnowledgeRetriever,
    StepContextManager,
    KnowledgeRetrievalConfig
)
from beeai_framework.vector.base import VectorMemoryProvider
from beeai_framework.vector.knowledge_capture import KnowledgeCaptureProcessor

logger = logging.getLogger(__name__)


class IntegrationConfig(BaseModel):
    """Configuration for integration between sequential thinking and knowledge retrieval."""
    
    enable_knowledge_capture: bool = Field(default=True, description="Capture knowledge from reasoning steps")
    enable_context_enhancement: bool = Field(default=True, description="Enhance prompts with retrieved context")
    enable_knowledge_retrieval: bool = Field(default=True, description="Retrieve knowledge for reasoning steps")
    max_tokens_per_step: int = Field(default=1000, description="Maximum tokens for context per step")
    apply_context_to_system_prompt: bool = Field(default=True, description="Apply context to system prompt")
    enable_feedback_collection: bool = Field(default=True, description="Collect feedback on context relevance")
    enable_step_logging: bool = Field(default=True, description="Log details of each step")
    enable_reasoning_path_optimization: bool = Field(default=True, description="Optimize reasoning path based on step results")


class SequentialKnowledgeIntegration:
    """
    Integration between Sequential Thinking and Knowledge Retrieval.
    
    This class provides the bridge between the Sequential Thinking middleware
    and the Knowledge Retrieval system, handling the flow of information between
    the reasoning process and the knowledge base.
    """
    
    def __init__(
        self,
        vector_provider: VectorMemoryProvider,
        knowledge_capture_processor: Optional[KnowledgeCaptureProcessor] = None,
        config: Optional[IntegrationConfig] = None,
        retrieval_config: Optional[KnowledgeRetrievalConfig] = None
    ):
        """
        Initialize the integration.
        
        Args:
            vector_provider: Provider for vector storage
            knowledge_capture_processor: Processor for knowledge capture
            config: Integration configuration
            retrieval_config: Knowledge retrieval configuration
        """
        self.vector_provider = vector_provider
        self.knowledge_capture_processor = knowledge_capture_processor
        self.config = config or IntegrationConfig()
        
        # Create knowledge retriever
        self.knowledge_retriever = SequentialThinkingKnowledgeRetriever(
            vector_provider=vector_provider,
            config=retrieval_config or KnowledgeRetrievalConfig()
        )
        
        # Create context manager
        self.context_manager = StepContextManager(
            knowledge_retriever=self.knowledge_retriever,
            max_context_items_per_step=5,
            enable_context_carryover=True
        )
        
        # Track feedback on context items
        self.feedback = {
            "relevant_contexts": set(),
            "irrelevant_contexts": set(),
            "step_quality_scores": []
        }
    
    async def enhance_step(
        self,
        step_info: Dict[str, Any],
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Enhance a sequential thinking step with relevant knowledge.
        
        Args:
            step_info: Information about the current step
            system_prompt: System prompt for the model
            user_prompt: User prompt for this step
            
        Returns:
            Dictionary with enhanced prompts and metadata
        """
        if not self.config.enable_knowledge_retrieval:
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context_applied": False
            }
        
        # Extract step information
        step_number = step_info.get("thought_number", 1)
        step_type = self._determine_step_type(step_number, step_info)
        
        # Get context for this step
        context_result = await self.context_manager.get_context_for_step(
            query=user_prompt,
            step_type=step_type,
            step_number=step_number
        )
        
        # Apply context to prompts
        enhanced_result = self._apply_context_to_prompts(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context_result["formatted_context"],
            step_type=step_type
        )
        
        # Prepare result
        result = {
            "system_prompt": enhanced_result["system_prompt"],
            "user_prompt": enhanced_result["user_prompt"],
            "context_applied": True,
            "context_items": context_result["context_items"],
            "context_metrics": context_result["retrieval_metrics"],
            "step_type": step_type
        }
        
        # Log step if enabled
        if self.config.enable_step_logging:
            logger.info(
                f"Enhanced step {step_number} ({step_type}) with "
                f"{len(context_result['context_items'])} context items in "
                f"{context_result['retrieval_metrics'].get('retrieval_time_ms', 0):.2f}ms"
            )
        
        return result
    
    async def process_step_result(
        self,
        step_result: Dict[str, Any],
        original_request: Dict[str, Any],
        enhanced_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process the result of a sequential thinking step.
        
        Args:
            step_result: Result from the model for this step
            original_request: Original request for this step
            enhanced_request: Enhanced request with context
            
        Returns:
            Dictionary with processed result and metadata
        """
        # Extract step data
        step_number = step_result.get("thought_number", 1)
        step_content = step_result.get("thought", "")
        step_type = enhanced_request.get("step_type", "unknown")
        
        # Initialize result dictionary
        result = {
            "step_result": step_result,
            "knowledge_captured": False,
            "feedback_collected": False,
            "next_step_suggestions": [],
            "context_quality_score": 0.0
        }
        
        # Capture knowledge if enabled
        if self.config.enable_knowledge_capture and self.knowledge_capture_processor:
            try:
                await self._capture_step_knowledge(
                    step_content=step_content,
                    step_number=step_number,
                    step_type=step_type
                )
                result["knowledge_captured"] = True
            except Exception as e:
                logger.error(f"Error capturing knowledge from step: {str(e)}")
        
        # Collect feedback on context relevance if enabled
        if self.config.enable_feedback_collection:
            try:
                feedback_metrics = self._collect_feedback_from_step(
                    step_content=step_content,
                    context_items=enhanced_request.get("context_items", [])
                )
                result["feedback_collected"] = True
                result["feedback_metrics"] = feedback_metrics
                result["context_quality_score"] = feedback_metrics.get("quality_score", 0.0)
                
                # Update feedback history for this trace
                trace_id = enhanced_request.get("trace_id")
                if trace_id:
                    self._update_trace_feedback(trace_id, step_number, feedback_metrics)
            except Exception as e:
                logger.error(f"Error collecting feedback from step: {str(e)}")
        
        # Generate next step suggestions based on the current step
        try:
            next_step_suggestions = self._generate_next_step_suggestions(
                step_content=step_content,
                step_type=step_type,
                step_number=step_number
            )
            result["next_step_suggestions"] = next_step_suggestions
        except Exception as e:
            logger.error(f"Error generating next step suggestions: {str(e)}")
        
        # Analyze key concepts that should be carried over to future steps
        try:
            key_concepts = self._extract_key_concepts(step_content)
            result["key_concepts"] = key_concepts
            
            # Store these concepts for future context enhancement
            self._update_concept_tracking(step_number, key_concepts)
        except Exception as e:
            logger.error(f"Error extracting key concepts: {str(e)}")
        
        # Analyze if the reasoning path should be adjusted based on this step
        if step_number > 1 and self.config.enable_reasoning_path_optimization:
            try:
                path_analysis = self._analyze_reasoning_path(
                    step_content=step_content,
                    step_type=step_type,
                    step_number=step_number
                )
                result["reasoning_path_analysis"] = path_analysis
            except Exception as e:
                logger.error(f"Error analyzing reasoning path: {str(e)}")
        
        # Log step processing completion
        if self.config.enable_step_logging:
            logger.info(
                f"Processed step {step_number} ({step_type}) - "
                f"Knowledge captured: {result['knowledge_captured']}, "
                f"Context quality: {result['context_quality_score']:.2f}"
            )
        
        return result
    
    def _determine_step_type(self, step_number: int, step_info: Dict[str, Any]) -> str:
        """
        Determine the type of reasoning step.
        
        Args:
            step_number: Position in the reasoning sequence
            step_info: Additional information about the step
            
        Returns:
            Step type string
        """
        # Check if step type is already provided
        if "step_type" in step_info:
            return step_info["step_type"]
        
        # Default mapping based on step number
        if step_number == 1:
            return "problem_definition"
        elif step_number == 2:
            return "information_gathering"
        elif step_number == 3:
            return "analysis"
        elif step_number == 4:
            return "solution_formulation" 
        elif step_number <= 6:
            return "implementation"
        else:
            return "verification"
    
    def _apply_context_to_prompts(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str,
        step_type: str
    ) -> Dict[str, str]:
        """
        Apply context to system and user prompts.
        
        Args:
            system_prompt: Original system prompt
            user_prompt: Original user prompt
            context: Context to apply
            step_type: Type of reasoning step
            
        Returns:
            Dictionary with enhanced prompts
        """
        if not context:
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            }
        
        if self.config.apply_context_to_system_prompt:
            # Add context to system prompt
            enhanced_system = f"{system_prompt}\n\n{context}"
            return {
                "system_prompt": enhanced_system,
                "user_prompt": user_prompt
            }
        else:
            # Add context to user prompt
            enhanced_user = f"{context}\n\n{user_prompt}"
            return {
                "system_prompt": system_prompt,
                "user_prompt": enhanced_user
            }
    
    async def _capture_step_knowledge(
        self,
        step_content: str,
        step_number: int,
        step_type: str
    ) -> None:
        """
        Capture knowledge from a reasoning step.
        
        Args:
            step_content: Content of the reasoning step
            step_number: Position in the reasoning sequence
            step_type: Type of reasoning step
        """
        if not self.knowledge_capture_processor:
            return
        
        # Skip if content is too short
        if len(step_content) < 100:
            return
            
        # Create metadata for the knowledge entry
        metadata = {
            "source": "sequential_reasoning",
            "category": "reasoning_step",
            "level": "project",  # Reasoning steps are typically project-specific
            "importance": 0.75,
            "step_number": step_number,
            "step_type": step_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store the knowledge
        await self.knowledge_capture_processor.store_knowledge_from_content(
            content=step_content,
            metadata=metadata
        )
    
    def _collect_feedback_from_step(
        self,
        step_content: str,
        context_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Collect feedback on context relevance from a step.
        
        Args:
            step_content: Content of the step
            context_items: Context items used for this step
            
        Returns:
            Dictionary with feedback metrics
        """
        if not context_items:
            return {"quality_score": 0.0, "relevance_scores": {}, "explicit_references": 0, "implicit_references": 0}
        
        # Track which context items were explicitly referenced
        explicit_references = set()
        implicit_references = set()
        
        # Calculate explicit references (mentioned by index number or direct quote)
        explicit_count = 0
        for i, item in enumerate(context_items, 1):
            # Check for index references like "[1]" or "from #1"
            item_id = item.get("vector_id") or self._get_item_id(item)
            
            # Check for reference to item number
            index_patterns = [
                f"[{i}]",
                f"#{i}",
                f"context {i}",
                f"item {i}",
                f"point {i}"
            ]
            
            if any(pattern in step_content for pattern in index_patterns):
                explicit_references.add(item_id)
                explicit_count += 1
                continue
            
            # Check for direct quotes (chunks of 5+ consecutive words)
            content = item.get("content", "")
            if content and len(content) > 20:
                # Get chunks of text (5+ words)
                chunks = self._get_content_chunks(content, min_chunk_size=5)
                
                for chunk in chunks:
                    if chunk in step_content:
                        explicit_references.add(item_id)
                        explicit_count += 1
                        break
        
        # Calculate implicit references (topic/concept overlap)
        implicit_count = 0
        context_key_phrases = {}
        
        # Extract key phrases from context items
        for item in context_items:
            item_id = item.get("vector_id") or self._get_item_id(item)
            content = item.get("content", "")
            if content:
                context_key_phrases[item_id] = self._extract_key_phrases(content)
        
        # Extract key phrases from step content
        step_key_phrases = self._extract_key_phrases(step_content, count=10)
        
        # Calculate overlap
        for item_id, phrases in context_key_phrases.items():
            # Calculate phrase overlap
            overlap = sum(1 for p in phrases if any(p.lower() in sp.lower() or sp.lower() in p.lower() 
                                                  for sp in step_key_phrases))
            
            # If there's significant overlap, count as implicit reference
            if overlap >= 1 and item_id not in explicit_references:
                implicit_references.add(item_id)
                implicit_count += 1
        
        # Calculate overall quality score
        total_items = len(context_items)
        if total_items == 0:
            quality_score = 0.0
        else:
            # Explicit references are weighted more heavily
            explicit_weight = 0.7
            implicit_weight = 0.3
            
            # Calculate weighted score
            explicit_score = explicit_count / total_items
            implicit_score = implicit_count / total_items
            quality_score = (explicit_score * explicit_weight) + (implicit_score * implicit_weight)
            
            # Scale to 0-1 range
            quality_score = min(max(quality_score, 0.0), 1.0)
        
        # Calculate per-item relevance scores
        relevance_scores = {}
        for item in context_items:
            item_id = item.get("vector_id") or self._get_item_id(item)
            if item_id in explicit_references:
                relevance_scores[item_id] = 1.0
            elif item_id in implicit_references:
                relevance_scores[item_id] = 0.5
            else:
                relevance_scores[item_id] = 0.0
        
        # Store feedback for future retrieval optimization
        self.feedback["relevant_contexts"].update(explicit_references)
        self.feedback["relevant_contexts"].update(implicit_references)
        
        # Items that weren't referenced might be irrelevant
        unused_items = set(item.get("vector_id") or self._get_item_id(item) 
                         for item in context_items) - explicit_references - implicit_references
        self.feedback["irrelevant_contexts"].update(unused_items)
        
        # Track quality score
        self.feedback["step_quality_scores"].append(quality_score)
        
        return {
            "quality_score": quality_score,
            "relevance_scores": relevance_scores,
            "explicit_references": explicit_count,
            "implicit_references": implicit_count,
            "total_items": total_items
        }
    
    def _get_content_chunks(self, text: str, min_chunk_size: int = 5) -> List[str]:
        """
        Get chunks of text for matching.
        
        Args:
            text: Text to chunk
            min_chunk_size: Minimum number of words per chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        if len(words) <= min_chunk_size:
            return [text]
            
        chunks = []
        for i in range(len(words) - min_chunk_size + 1):
            chunk = " ".join(words[i:i+min_chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def _get_item_id(self, item: Dict[str, Any]) -> str:
        """
        Get a stable ID for a context item.
        
        Args:
            item: Context item
            
        Returns:
            Stable ID string
        """
        if "vector_id" in item:
            return item["vector_id"]
            
        # Generate a content-based ID if vector_id not available
        content = item.get("content", "")
        metadata = item.get("metadata", {})
        source = metadata.get("source", "unknown")
        
        import hashlib
        hasher = hashlib.md5()
        hasher.update(f"{content[:100]}:{source}".encode("utf-8"))
        return f"item_{hasher.hexdigest()[:10]}"
    
    def _update_trace_feedback(
        self, 
        trace_id: str, 
        step_number: int, 
        feedback_metrics: Dict[str, Any]
    ) -> None:
        """
        Update feedback tracking for a specific trace.
        
        Args:
            trace_id: Trace ID
            step_number: Step number
            feedback_metrics: Feedback metrics
        """
        # This could be expanded to store trace-specific feedback
        # for more sophisticated reasoning path optimization
        pass
    
    def _extract_key_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract key concepts from text that should be carried over to future steps.
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            List of key concept dictionaries
        """
        # Extract key phrases first
        phrases = self._extract_key_phrases(text, count=5)
        
        # Convert to concept dictionaries with importance scores
        concepts = []
        for i, phrase in enumerate(phrases):
            # Higher score for earlier mentions
            importance = 1.0 - (i * 0.15)
            concepts.append({
                "concept": phrase,
                "importance": importance
            })
            
        return concepts
    
    def _update_concept_tracking(self, step_number: int, concepts: List[Dict[str, Any]]) -> None:
        """
        Update concept tracking for sequential context enhancement.
        
        Args:
            step_number: Current step number
            concepts: Key concepts extracted from the step
        """
        # This could be expanded to maintain a graph of concept relationships
        # across steps for more sophisticated reasoning
        pass
    
    def _analyze_reasoning_path(
        self,
        step_content: str,
        step_type: str,
        step_number: int
    ) -> Dict[str, Any]:
        """
        Analyze the reasoning path to optimize future steps.
        
        Args:
            step_content: Content of the current step
            step_type: Type of the current step
            step_number: Number of the current step
            
        Returns:
            Dictionary with reasoning path analysis
        """
        # Define common reasoning path adjustments
        # This is a simplified implementation, but could be expanded
        # with more sophisticated heuristics or ML-based analysis
        
        # Check for indicators of needing more context
        needs_more_context = any(phrase in step_content.lower() for phrase in [
            "need more information",
            "insufficient context",
            "more details needed",
            "unclear from the context"
        ])
        
        # Check for indicators of needing to explore alternatives
        needs_alternatives = any(phrase in step_content.lower() for phrase in [
            "alternative approach",
            "different perspective",
            "another way",
            "could also consider"
        ])
        
        # Check for indicators of needing to refocus
        needs_refocus = any(phrase in step_content.lower() for phrase in [
            "returning to the main question",
            "refocus on the original",
            "central problem",
            "primary objective"
        ])
        
        return {
            "needs_more_context": needs_more_context,
            "needs_alternatives": needs_alternatives,
            "needs_refocus": needs_refocus,
            "adjustment_suggested": needs_more_context or needs_alternatives or needs_refocus
        }
    
    def _generate_next_step_suggestions(
        self,
        step_content: str,
        step_type: str,
        step_number: int
    ) -> List[str]:
        """
        Generate suggestions for the next reasoning step.
        
        Args:
            step_content: Content of the current step
            step_type: Type of current reasoning step
            step_number: Position in the reasoning sequence
            
        Returns:
            List of suggestions for the next step
        """
        # Define suggestion templates based on current step type
        templates = {
            "problem_definition": [
                "Consider the key requirements and constraints.",
                "Identify the main challenges in this problem.",
                "Break down the problem into smaller components."
            ],
            "information_gathering": [
                "Analyze the information collected so far.",
                "Identify any missing information needed for a solution.",
                "Organize the information into relevant categories."
            ],
            "analysis": [
                "Evaluate possible solution approaches.",
                "Consider trade-offs between different approaches.",
                "Identify the most promising direction."
            ],
            "solution_formulation": [
                "Detail the implementation steps.",
                "Consider edge cases and potential issues.",
                "Define success criteria for the solution."
            ],
            "implementation": [
                "Test the implementation with examples.",
                "Verify correctness of the solution.",
                "Consider optimization opportunities."
            ],
            "verification": [
                "Summarize the solution and its justification.",
                "Verify all requirements have been addressed.",
                "Suggest improvements or extensions."
            ]
        }
        
        # Get templates for the next step
        current_index = list(templates.keys()).index(step_type) if step_type in templates else 0
        next_step_type = list(templates.keys())[min(current_index + 1, len(templates) - 1)]
        
        return templates.get(next_step_type, ["Continue with the next logical step."])
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the integration.
        
        Returns:
            Dictionary with integration statistics
        """
        # Get context usage stats
        context_stats = self.context_manager.get_context_usage_stats()
        
        # Compute feedback stats
        feedback_stats = {
            "relevant_contexts": len(self.feedback["relevant_contexts"]),
            "irrelevant_contexts": len(self.feedback["irrelevant_contexts"]),
            "relevance_ratio": 0.0
        }
        
        total_feedback = (
            len(self.feedback["relevant_contexts"]) + 
            len(self.feedback["irrelevant_contexts"])
        )
        
        if total_feedback > 0:
            feedback_stats["relevance_ratio"] = (
                len(self.feedback["relevant_contexts"]) / total_feedback
            )
        
        # Return comprehensive stats
        return {
            "context_usage": context_stats,
            "feedback": feedback_stats,
            "configuration": {
                "knowledge_capture_enabled": self.config.enable_knowledge_capture,
                "context_enhancement_enabled": self.config.enable_context_enhancement,
                "feedback_collection_enabled": self.config.enable_feedback_collection
            }
        }


class ReasoningFeedbackCollector:
    """
    Collects and processes feedback from reasoning steps.
    
    This class is responsible for collecting explicit and implicit feedback
    about the relevance and usefulness of context provided during reasoning.
    """
    
    def __init__(
        self,
        vector_provider: VectorMemoryProvider,
        enable_explicit_feedback: bool = True,
        enable_implicit_feedback: bool = True
    ):
        """
        Initialize the feedback collector.
        
        Args:
            vector_provider: Provider for vector storage
            enable_explicit_feedback: Whether to collect explicit feedback
            enable_implicit_feedback: Whether to collect implicit feedback
        """
        self.vector_provider = vector_provider
        self.enable_explicit_feedback = enable_explicit_feedback
        self.enable_implicit_feedback = enable_implicit_feedback
        
        # Feedback storage
        self.explicit_feedback = {}
        self.implicit_feedback = {}
        self.aggregate_scores = {}
    
    def collect_feedback_from_step(
        self,
        step_content: str,
        step_number: int,
        context_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Collect feedback from a reasoning step.
        
        Args:
            step_content: Content of the reasoning step
            step_number: Position in the reasoning sequence
            context_items: Context items used for this step
            
        Returns:
            Dictionary with feedback metrics
        """
        feedback_metrics = {
            "explicit_references": 0,
            "implicit_references": 0,
            "total_contexts": len(context_items),
            "reference_ratio": 0.0
        }
        
        if not context_items:
            return feedback_metrics
        
        # Collect explicit feedback if enabled
        explicit_references = 0
        if self.enable_explicit_feedback:
            explicit_references = self._extract_explicit_feedback(
                step_content, context_items
            )
            feedback_metrics["explicit_references"] = explicit_references
        
        # Collect implicit feedback if enabled
        implicit_references = 0
        if self.enable_implicit_feedback:
            implicit_references = self._extract_implicit_feedback(
                step_content, context_items
            )
            feedback_metrics["implicit_references"] = implicit_references
        
        # Calculate reference ratio
        total_references = explicit_references + implicit_references
        if context_items:
            feedback_metrics["reference_ratio"] = total_references / len(context_items)
        
        return feedback_metrics
    
    def _extract_explicit_feedback(
        self,
        step_content: str,
        context_items: List[Dict[str, Any]]
    ) -> int:
        """
        Extract explicit feedback mentions from step content.
        
        Args:
            step_content: Content of the reasoning step
            context_items: Context items used for this step
            
        Returns:
            Number of explicit references
        """
        # Look for explicit references like "Based on context [1]" or "As mentioned in item 2"
        import re
        explicit_patterns = [
            r'context\s+\[?(\d+)\]?',
            r'item\s+\[?(\d+)\]?',
            r'source\s+\[?(\d+)\]?',
            r'reference\s+\[?(\d+)\]?'
        ]
        
        references = set()
        for pattern in explicit_patterns:
            matches = re.findall(pattern, step_content, re.IGNORECASE)
            for match in matches:
                try:
                    # Convert to 0-based index
                    index = int(match) - 1
                    if 0 <= index < len(context_items):
                        item_id = self._get_item_id(context_items[index])
                        references.add(item_id)
                        
                        # Record explicit feedback
                        if item_id not in self.explicit_feedback:
                            self.explicit_feedback[item_id] = 0
                        self.explicit_feedback[item_id] += 1
                except ValueError:
                    pass
        
        return len(references)
    
    def _extract_implicit_feedback(
        self,
        step_content: str,
        context_items: List[Dict[str, Any]]
    ) -> int:
        """
        Extract implicit feedback through content similarity.
        
        Args:
            step_content: Content of the reasoning step
            context_items: Context items used for this step
            
        Returns:
            Number of implicit references
        """
        # Check for substantive content overlap
        references = set()
        for item in context_items:
            content = item.get("content", "")
            if not content:
                continue
            
            # Extract key phrases (3-5 word sequences)
            key_phrases = self._extract_key_phrases(content)
            
            # Check if any key phrases appear in the step content
            for phrase in key_phrases:
                if len(phrase) > 10 and phrase.lower() in step_content.lower():
                    item_id = self._get_item_id(item)
                    references.add(item_id)
                    
                    # Record implicit feedback
                    if item_id not in self.implicit_feedback:
                        self.implicit_feedback[item_id] = 0
                    self.implicit_feedback[item_id] += 1
                    break
        
        return len(references)
    
    def _extract_key_phrases(self, text: str, count: int = 3) -> List[str]:
        """
        Extract key phrases from text.
        
        Args:
            text: Text to extract from
            count: Number of phrases to extract
            
        Returns:
            List of key phrases
        """
        if not text:
            return []
            
        # Simple extraction based on sentence splitting and phrase detection
        # Note: In a production system, this would use more sophisticated NLP
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Get noun phrases (simplified approach)
        phrases = []
        
        # Common technical terms that might be important
        technical_patterns = [
            r'\b[A-Z][a-z]*(?:API|SDK)\b',  # APIs, SDKs
            r'\b[A-Z][a-z]+(?:\.js|\.py|\.ts)\b',  # JavaScript/Python libraries
            r'\b(?:middleware|function|class|module|system|framework)\b',  # Software concepts
            r'\b(?:vector|retrieval|knowledge|context|reasoning)\b'  # Domain-specific terms
        ]
        
        # Check for domain-specific technical terms first
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend(matches)
        
        # Look for noun phrases (capitalized words followed by 1-3 lowercase words)
        noun_phrase_pattern = r'\b[A-Z][a-z]+(?:\s+[a-z]+){1,3}\b'
        
        # Look for important phrases with markers of significance
        significance_markers = [
            "important", "critical", "essential", "key", "main", "primary",
            "crucial", "significant", "notable", "central"
        ]
        
        for sentence in sentences:
            # First check for phrases marked as significant
            for marker in significance_markers:
                if marker in sentence.lower():
                    # Extract the part after the marker
                    parts = sentence.lower().split(marker, 1)
                    if len(parts) > 1:
                        # Take the next 4-6 words after the marker
                        words = parts[1].strip().split()[:6]
                        if words:
                            phrases.append(" ".join(words))
            
            # Then look for capitalized noun phrases
            np_matches = re.findall(noun_phrase_pattern, sentence)
            if np_matches:
                phrases.extend(np_matches)
            
            # For shorter sentences, include the whole sentence if it's not too long
            if len(sentence.split()) <= 8 and len(phrases) < count:
                phrases.append(sentence)
        
        # Add specific "from the context" references
        context_references = re.findall(r'(?:according to|from|as mentioned in)(?:\s+the)?\s+context[^.]*', text, re.IGNORECASE)
        phrases.extend(context_references)
        
        # Clean and deduplicate phrases
        cleaned_phrases = []
        for phrase in phrases:
            # Clean up the phrase
            cleaned = re.sub(r'^\W+|\W+$', '', phrase).strip()
            # Ensure minimum length and not already included
            if cleaned and len(cleaned) >= 3 and cleaned not in cleaned_phrases:
                cleaned_phrases.append(cleaned)
        
        # Sort by length (prefer shorter, more precise phrases) and take the top ones
        return sorted(cleaned_phrases, key=len)[:count]
    
    def _get_item_id(self, item: Dict[str, Any]) -> str:
        """Get a unique ID for a context item."""
        # Use ID from metadata if available, otherwise use content hash
        metadata = item.get("metadata", {})
        if "id" in metadata:
            return str(metadata["id"])
        
        content = item.get("content", "")
        if content:
            return f"content_{hash(content)}"
        
        return f"item_{hash(str(item))}"
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback.
        
        Returns:
            Dictionary with feedback statistics
        """
        # Combine explicit and implicit feedback
        all_items = set(self.explicit_feedback.keys()) | set(self.implicit_feedback.keys())
        combined_scores = {}
        
        for item_id in all_items:
            explicit_score = self.explicit_feedback.get(item_id, 0)
            implicit_score = self.implicit_feedback.get(item_id, 0)
            
            # Weight explicit feedback higher
            combined_scores[item_id] = explicit_score * 2 + implicit_score
        
        # Return comprehensive stats
        return {
            "items_with_feedback": len(all_items),
            "explicit_feedback_count": len(self.explicit_feedback),
            "implicit_feedback_count": len(self.implicit_feedback),
            "top_items": sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "feedback_distribution": {
                "high": len([s for s in combined_scores.values() if s > 3]),
                "medium": len([s for s in combined_scores.values() if 1 < s <= 3]),
                "low": len([s for s in combined_scores.values() if s <= 1])
            }
        }
    
    async def apply_feedback_adjustments(self) -> Dict[str, Any]:
        """
        Apply feedback adjustments to vector store relevance.
        
        Returns:
            Dictionary with adjustment metrics
        """
        # This is a placeholder for a method that would adjust
        # the relevance of items in the vector store based on feedback
        # In a real implementation, this would interact with the 
        # vector database to adjust relevance or weights
        
        return {
            "adjustments_applied": len(self.explicit_feedback),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    
    def track_context_quality(self, trace_id: str) -> Dict[str, Any]:
        """
        Track context quality metrics across a reasoning trace.
        
        This provides insight into how well the context enhancement is working
        across multiple steps in the reasoning process.
        
        Args:
            trace_id: ID of the reasoning trace
            
        Returns:
            Dictionary with context quality metrics
        """
        if not self.feedback["step_quality_scores"]:
            return {
                "average_quality": 0.0,
                "quality_trend": "unknown",
                "step_count": 0
            }
        
        # Calculate quality metrics
        quality_scores = self.feedback["step_quality_scores"]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Analyze quality trend
        trend = "stable"
        if len(quality_scores) >= 3:
            first_half = quality_scores[:len(quality_scores)//2]
            second_half = quality_scores[len(quality_scores)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg * 1.1:
                trend = "improving"
            elif second_avg < first_avg * 0.9:
                trend = "declining"
        
        # Calculate high/low quality steps
        high_quality_steps = sum(1 for score in quality_scores if score >= 0.7)
        low_quality_steps = sum(1 for score in quality_scores if score <= 0.3)
        
        return {
            "average_quality": avg_quality,
            "quality_trend": trend,
            "step_count": len(quality_scores),
            "high_quality_steps": high_quality_steps,
            "low_quality_steps": low_quality_steps,
            "quality_scores": quality_scores
        } 