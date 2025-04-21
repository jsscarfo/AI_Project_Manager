"""
Composite Context Provider

Combines multiple context providers into a single provider interface,
aggregating results from all configured providers.
"""

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union
import logging
from pydantic import BaseModel, Field

from ..core.context_middleware import ContextProvider, ContextProviderConfig

logger = logging.getLogger(__name__)

class ProviderMapping(BaseModel):
    """Configuration for a provider within the composite provider."""
    provider: ContextProvider
    weight: float = 1.0
    enabled: bool = True

class CompositeProviderConfig(ContextProviderConfig):
    """Configuration for the Composite Context Provider."""
    batch_size: int = 5
    concurrent_providers: int = 3
    timeout_seconds: int = 10

class CompositeProvider(ContextProvider):
    """
    Context provider that combines results from multiple providers.
    
    This provider acts as an aggregator for multiple context providers,
    allowing different sources of context to be queried simultaneously
    and combining their results.
    """
    
    def __init__(
        self, 
        providers: Sequence[Union[ContextProvider, ProviderMapping]] = None,
        config: Optional[CompositeProviderConfig] = None
    ):
        """
        Initialize the Composite Provider.
        
        Args:
            providers: List of providers or provider mappings to include
            config: Configuration for the composite provider
        """
        self.config = config or CompositeProviderConfig()
        super().__init__(self.config)
        
        # Convert providers to ProviderMapping objects if needed
        self.providers = []
        if providers:
            for provider in providers:
                if isinstance(provider, ContextProvider):
                    self.providers.append(ProviderMapping(provider=provider))
                else:
                    self.providers.append(provider)
        
        logger.info(f"Initialized CompositeProvider with {len(self.providers)} providers")
    
    def add_provider(self, provider: ContextProvider, weight: float = 1.0) -> None:
        """
        Add a provider to the composite.
        
        Args:
            provider: The provider to add
            weight: Weight to give this provider's results (higher = more important)
        """
        self.providers.append(ProviderMapping(
            provider=provider,
            weight=weight
        ))
        logger.debug(f"Added provider {provider.__class__.__name__} with weight {weight}")
    
    async def initialize(self) -> None:
        """Initialize all providers."""
        if not self.providers:
            logger.warning("CompositeProvider has no providers configured")
            return
        
        # Initialize providers with concurrency limit
        semaphore = asyncio.Semaphore(self.config.concurrent_providers)
        
        async def initialize_with_semaphore(provider_mapping):
            async with semaphore:
                try:
                    await provider_mapping.provider.initialize()
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize provider {provider_mapping.provider.__class__.__name__}: {str(e)}")
                    provider_mapping.enabled = False
                    return False
        
        # Create initialization tasks for all providers
        tasks = [initialize_with_semaphore(provider) for provider in self.providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful initializations
        successful = sum(1 for r in results if r is True)
        logger.info(f"Initialized {successful}/{len(self.providers)} providers")
    
    async def get_context(self, query: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve context from all providers and combine results.
        
        Args:
            query: The query to retrieve context for
            metadata: Additional metadata for the query
            
        Returns:
            Combined list of context items from all providers
        """
        if not query:
            return []
        
        if not self.providers:
            logger.warning("No providers available for context retrieval")
            return []
        
        metadata = metadata or {}
        
        # Only use enabled providers
        active_providers = [p for p in self.providers if p.enabled]
        
        if not active_providers:
            logger.warning("No enabled providers available for context retrieval")
            return []
        
        # Create a task for each provider with timeout
        async def get_provider_context(provider_mapping):
            try:
                # Apply a timeout to each provider
                results = await asyncio.wait_for(
                    provider_mapping.provider.get_context(query, metadata),
                    timeout=self.config.timeout_seconds
                )
                
                # Apply provider weight to each result
                for result in results:
                    # Adjust the score by the provider weight
                    if "score" in result:
                        result["score"] = result["score"] * provider_mapping.weight
                    
                    # Add provider info to metadata
                    if "metadata" not in result:
                        result["metadata"] = {}
                    
                    result["metadata"]["provider"] = provider_mapping.provider.__class__.__name__
                
                return results
            except asyncio.TimeoutError:
                logger.warning(f"Provider {provider_mapping.provider.__class__.__name__} timed out")
                return []
            except Exception as e:
                logger.error(f"Error retrieving context from {provider_mapping.provider.__class__.__name__}: {str(e)}")
                return []
        
        # Run providers with concurrency limit
        semaphore = asyncio.Semaphore(self.config.concurrent_providers)
        
        async def get_with_semaphore(provider_mapping):
            async with semaphore:
                return await get_provider_context(provider_mapping)
        
        # Create retrieval tasks
        tasks = [get_with_semaphore(provider) for provider in active_providers]
        all_results = await asyncio.gather(*tasks)
        
        # Combine all results
        combined_results = []
        for results in all_results:
            combined_results.extend(results)
        
        # Sort by score
        sorted_results = sorted(combined_results, key=lambda x: x.get("score", 0), reverse=True)
        
        return sorted_results
    
    async def add_context(self, content: str, metadata: Dict[str, Any] = None) -> List[str]:
        """
        Add content to all providers that support it.
        
        Args:
            content: The content to store
            metadata: Metadata for the content
            
        Returns:
            List of IDs from each provider
        """
        if not content:
            raise ValueError("Content cannot be empty")
        
        if not self.providers:
            logger.warning("No providers available for adding context")
            return []
        
        metadata = metadata or {}
        
        # Only use enabled providers
        active_providers = [p for p in self.providers if p.enabled]
        
        if not active_providers:
            logger.warning("No enabled providers available for adding context")
            return []
        
        # Track results from each provider
        results = []
        
        # Process in batches to avoid overwhelming system
        for i in range(0, len(active_providers), self.config.batch_size):
            batch = active_providers[i:i+self.config.batch_size]
            
            # Create a task for each provider in the batch
            async def add_to_provider(provider_mapping):
                try:
                    if hasattr(provider_mapping.provider, "add_context"):
                        result = await provider_mapping.provider.add_context(content, metadata)
                        return {
                            "provider": provider_mapping.provider.__class__.__name__,
                            "id": result
                        }
                    return None
                except Exception as e:
                    logger.error(f"Error adding context to {provider_mapping.provider.__class__.__name__}: {str(e)}")
                    return None
            
            # Create tasks for this batch
            tasks = [add_to_provider(provider) for provider in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Add successful results
            for result in batch_results:
                if result:
                    results.append(result)
        
        return results 