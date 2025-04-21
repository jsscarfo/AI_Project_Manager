"""
Integration layer between sequential thinking and knowledge retrieval.

This module provides components for enhancing sequential thinking steps with
relevant knowledge from vector storage.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from beeai_framework.middleware.config import MiddlewareConfig
from beeai_framework.vector.base import VectorMemoryProvider
from beeai_framework.vector.knowledge_capture import KnowledgeCaptureProcessor
from beeai_framework.vector.knowledge_retrieval import (
    KnowledgeRetrievalConfig, 
    RetrievedKnowledge,
    SequentialThinkingKnowledgeRetriever
)

logger = logging.getLogger(__name__)

class IntegrationConfig(MiddlewareConfig):
    """Configuration for the integration between knowledge retrieval and sequential thinking."""
    
    def __init__(
        self,
        enabled: bool = True,
        capture_knowledge: bool = True,
        enhance_with_knowledge: bool = True,
        system_prompt_template: str = "{original_prompt}\n\nRelevant context:\n{context}",
        user_prompt_template: str = "{original_prompt}",
        min_context_items: int = 1,
        max_context_items: int = 5,
        confidence_threshold: float = 0.70,
        step_type_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize an integration config.
        
        Args:
            enabled: Whether the integration is enabled
            capture_knowledge: Whether to capture knowledge from steps
            enhance_with_knowledge: Whether to enhance steps with knowledge
            system_prompt_template: Template for the enhanced system prompt
            user_prompt_template: Template for the enhanced user prompt
            min_context_items: Minimum number of context items to include
            max_context_items: Maximum number of context items to include
            confidence_threshold: Minimum confidence threshold for including context
            step_type_weights: Weight multipliers for different step types (e.g., planning=1.2)
        """
        super().__init__(enabled=enabled)
        
        self.capture_knowledge = capture_knowledge
        self.enhance_with_knowledge = enhance_with_knowledge
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.min_context_items = min_context_items
        self.max_context_items = max_context_items
        self.confidence_threshold = confidence_threshold
        
        # Default step type weights
        self.step_type_weights = step_type_weights or {
            "planning": 1.2,
            "research": 1.3, 
            "analysis": 1.1,
            "execution": 0.9,
            "verification": 1.0,
            "reflection": 1.1
        }
        
class SequentialKnowledgeIntegration:
    """
    Integration layer between sequential thinking and knowledge retrieval.
    
    This class provides methods for enhancing sequential thinking steps with
    relevant knowledge from vector storage, and for capturing knowledge from
    step results.
    """
    
    def __init__(
        self,
        vector_provider: VectorMemoryProvider,
        knowledge_capture_processor: Optional[KnowledgeCaptureProcessor] = None,
        config: Optional[IntegrationConfig] = None,
        retrieval_config: Optional[KnowledgeRetrievalConfig] = None
    ):
        """
        Initialize a sequential knowledge integration.
        
        Args:
            vector_provider: Provider for vector storage
            knowledge_capture_processor: Optional processor for knowledge capture
            config: Integration configuration
            retrieval_config: Knowledge retrieval configuration
        """
        self.vector_provider = vector_provider
        self.knowledge_capture_processor = knowledge_capture_processor
        self.config = config or IntegrationConfig()
        
        # Initialize the knowledge retriever
        self.retriever = SequentialThinkingKnowledgeRetriever(
            vector_provider=vector_provider,
            config=retrieval_config or KnowledgeRetrievalConfig()
        )
        
        logger.info("Initialized SequentialKnowledgeIntegration")
    
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
            system_prompt: Original system prompt
            user_prompt: Original user prompt
            
        Returns:
            Dictionary with enhanced prompts and context information
        """
        result = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context_applied": False,
            "context_items": [],
            "step_type": step_info.get("step_type"),
            "thought_number": step_info.get("thought_number", 1)
        }
        
        # Skip if enhancement is disabled or prompts are not provided
        if not self.config.enabled or not self.config.enhance_with_knowledge:
            return result
            
        if not system_prompt and not user_prompt:
            return result
        
        try:
            # Get step-specific parameters
            step_type = step_info.get("step_type", "unknown")
            step_number = step_info.get("thought_number", 1)
            total_steps = step_info.get("total_thoughts", 5)
            
            # Apply step type weight if available
            retrieval_weight = self.config.step_type_weights.get(step_type.lower(), 1.0)
            
            # Progress weighting - late steps may need less external knowledge
            progress_factor = 1.0 - (step_number / (total_steps * 2.0))  # Decreases from 1.0 to 0.5
            
            # Retrieve relevant knowledge for this step
            context_items = await self.retriever.retrieve_for_step(
                step_type=step_type,
                step_number=step_number,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                total_steps=total_steps,
                similarity_threshold=self.config.confidence_threshold * progress_factor,
                max_results=self.config.max_context_items,
                weight_multiplier=retrieval_weight
            )
            
            logger.debug(
                f"Retrieved {len(context_items)} context items for {step_type} step "
                f"{step_number}/{total_steps} with weight {retrieval_weight:.2f}"
            )
            
            # If we have enough context items, enhance the prompts
            if len(context_items) >= self.config.min_context_items:
                enhanced_prompts = self._apply_context_to_prompts(
                    context_items=context_items,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                
                result["system_prompt"] = enhanced_prompts["system_prompt"]
                result["user_prompt"] = enhanced_prompts["user_prompt"]
                result["context_applied"] = True
                result["context_items"] = [item.to_dict() for item in context_items]
            
        except Exception as e:
            logger.error(f"Error enhancing step with knowledge: {str(e)}")
            
        return result
    
    async def process_step_result(
        self,
        step_result: Dict[str, Any],
        original_request: Any,
        enhanced_request: Dict[str, Any]
    ) -> None:
        """
        Process a step result to potentially capture knowledge.
        
        Args:
            step_result: Result of a sequential thinking step
            original_request: Original request (pre-enhancement)
            enhanced_request: Enhanced request metadata
        """
        # Skip if knowledge capture is disabled
        if not self.config.enabled or not self.config.capture_knowledge:
            return
            
        # Skip if no knowledge capture processor available
        if not self.knowledge_capture_processor:
            return
            
        try:
            # Extract relevant information
            step_type = enhanced_request.get("step_type", "unknown")
            step_number = enhanced_request.get("step_number", 1)
            thought_content = step_result.get("thought", "")
            
            if not thought_content:
                return
                
            # Create metadata for the knowledge entry
            metadata = {
                "source": "sequential_thinking",
                "step_type": step_type,
                "step_number": step_number,
                "total_steps": step_result.get("total_thoughts", 5),
                "final_step": not step_result.get("next_thought_needed", True)
            }
            
            # Extract and store knowledge from the thought content
            await self.knowledge_capture_processor.store_knowledge_from_content(
                content=thought_content,
                metadata=metadata
            )
            
            logger.debug(
                f"Captured knowledge from {step_type} step {step_number} "
                f"with {len(thought_content)} characters"
            )
                
        except Exception as e:
            logger.error(f"Error capturing knowledge from step result: {str(e)}")
    
    def _apply_context_to_prompts(
        self, 
        context_items: List[RetrievedKnowledge],
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, str]:
        """
        Apply retrieved context to the system and user prompts.
        
        Args:
            context_items: Retrieved knowledge items
            system_prompt: Original system prompt
            user_prompt: Original user prompt
            
        Returns:
            Dictionary with enhanced system and user prompts
        """
        # Format the context as a string
        context_text = self._format_context_items(context_items)
        
        # Apply to system prompt
        enhanced_system_prompt = self.config.system_prompt_template.format(
            original_prompt=system_prompt,
            context=context_text
        )
        
        # Apply to user prompt
        enhanced_user_prompt = self.config.user_prompt_template.format(
            original_prompt=user_prompt,
            context=context_text
        )
        
        return {
            "system_prompt": enhanced_system_prompt,
            "user_prompt": enhanced_user_prompt
        }
    
    def _format_context_items(self, context_items: List[RetrievedKnowledge]) -> str:
        """
        Format context items into a string for inclusion in prompts.
        
        Args:
            context_items: Retrieved knowledge items
            
        Returns:
            Formatted context string
        """
        formatted_items = []
        
        for i, item in enumerate(context_items, 1):
            # Extract metadata for citation
            source = item.metadata.get("source", "knowledge base")
            
            # Format the knowledge text
            knowledge_text = item.content.strip()
            
            # Append formatted item
            formatted_items.append(f"{i}. {knowledge_text} [Source: {source}, Similarity: {item.similarity:.2f}]")
        
        return "\n\n".join(formatted_items) 