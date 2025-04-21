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
Weaviate Provider Module

This module implements the VectorMemoryProvider interface for Weaviate,
providing vector and graph-based contextual retrieval functionality.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from pydantic import BaseModel, Field, model_validator

from beeai_framework.errors import FrameworkError
from beeai_framework.vector.base import ContextMetadata, ContextResult, VectorMemoryProvider, VectorMemoryProviderConfig
from beeai_framework.vector.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

try:
    import weaviate
    from weaviate.client import Client as WeaviateClient
    from weaviate.exceptions import WeaviateBaseError
    WEAVIATE_AVAILABLE = True
except ImportError:
    logger.warning("Weaviate Python client not installed. Install with 'pip install weaviate-client'")
    WEAVIATE_AVAILABLE = False


class WeaviateProviderConfig(VectorMemoryProviderConfig):
    """Configuration for the Weaviate provider."""
    
    # Connection settings
    host: str = Field(default="localhost", description="Weaviate host")
    port: int = Field(default=8080, description="Weaviate port")
    protocol: str = Field(default="http", description="Protocol (http or https)")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Class settings
    class_name: str = Field(default="ContextMemory", description="Name of the Weaviate class to use")
    class_description: str = Field(
        default="Contextual memory for AI agents",
        description="Description for the Weaviate class"
    )
    
    # Schema settings
    create_schema_if_missing: bool = Field(default=True, description="Create schema if it doesn't exist")
    vector_index_type: str = Field(default="hnsw", description="Vector index type")
    vector_index_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "efConstruction": 128,
            "maxConnections": 64,
        },
        description="Vector index configuration"
    )
    
    # Search settings
    search_use_hybrid: bool = Field(default=True, description="Use hybrid search (vector + keyword)")
    search_alpha: float = Field(
        default=0.5, 
        description="Hybrid search alpha (0=vector only, 1=keyword only)"
    )
    
    # Node relationship settings
    enable_references: bool = Field(default=True, description="Enable graph references between objects")
    
    # Performance settings
    batch_size: int = Field(default=50, description="Batch size for bulk operations")
    
    @model_validator(mode='after')
    def validate_config(self) -> 'WeaviateProviderConfig':
        """Validate the configuration."""
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate Python client not installed. Install with 'pip install weaviate-client'")
        
        return self


class WeaviateProvider(VectorMemoryProvider):
    """
    Weaviate implementation of the VectorMemoryProvider interface.
    
    This provider integrates with Weaviate to provide:
    - Vector-based similarity search
    - Metadata filtering
    - Graph-based relationship queries
    - Hybrid search combining vector and keyword matching
    - Hierarchical knowledge organization
    
    It supports the Selective Contextual Retrieval system by enabling
    precise, targeted retrieval of context at different levels of abstraction.
    """
    
    def __init__(self, config: WeaviateProviderConfig, embedding_service: EmbeddingService):
        """
        Initialize the Weaviate provider.
        
        Args:
            config: Configuration for the Weaviate provider
            embedding_service: Service for generating text embeddings
        """
        super().__init__(config)
        
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate Python client not installed. Install with 'pip install weaviate-client'")
            
        self.config = config
        self.embedding_service = embedding_service
        self.client: Optional[WeaviateClient] = None
        
        # Initialize connection info
        self.connection_url = f"{config.protocol}://{config.host}:{config.port}"
        self.auth_config = weaviate.auth.AuthApiKey(api_key=config.api_key) if config.api_key else None
        
    async def initialize(self) -> None:
        """
        Initialize the Weaviate provider.
        
        This establishes the connection to Weaviate and ensures the
        required schema exists.
        """
        try:
            # Connect to Weaviate
            logger.info(f"Connecting to Weaviate at {self.connection_url}")
            self.client = weaviate.Client(
                url=self.connection_url,
                auth_client_secret=self.auth_config,
                timeout_config=(self.config.timeout, self.config.timeout)
            )
            
            # Check if Weaviate is ready
            if not self.client.is_ready():
                raise FrameworkError("Weaviate is not ready. Please check your connection.")
            
            # Ensure schema exists
            if self.config.create_schema_if_missing:
                await self._ensure_schema_exists()
                
            logger.info(f"Successfully initialized Weaviate provider at {self.connection_url}")
            
        except WeaviateBaseError as e:
            logger.error(f"Error initializing Weaviate provider: {str(e)}")
            raise FrameworkError(f"Error initializing Weaviate provider: {str(e)}")
    
    async def _ensure_schema_exists(self) -> None:
        """Ensure the required schema exists in Weaviate."""
        if not self.client:
            raise FrameworkError("Weaviate client not initialized")
            
        # Check if the class already exists
        existing_schema = self.client.schema.get()
        existing_classes = [c["class"] for c in existing_schema.get("classes", [])]
        
        if self.config.class_name in existing_classes:
            logger.debug(f"Weaviate class {self.config.class_name} already exists")
            return
        
        # Define class properties
        properties = [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The context content",
                "indexInverted": True,
            },
            {
                "name": "source",
                "dataType": ["text"],
                "description": "Source of the context",
                "indexInverted": True,
            },
            {
                "name": "category",
                "dataType": ["text"],
                "description": "Category of the context",
                "indexInverted": True,
            },
            {
                "name": "level",
                "dataType": ["text"],
                "description": "Hierarchical level (domain, techstack, project)",
                "indexInverted": True,
            },
            {
                "name": "importance",
                "dataType": ["number"],
                "description": "Importance score (0-1)",
            },
            {
                "name": "timestamp",
                "dataType": ["date"],
                "description": "Timestamp when context was added",
            },
        ]
        
        # Add reference properties if enabled
        if self.config.enable_references:
            properties.append({
                "name": "relatedTo",
                "dataType": [self.config.class_name],
                "description": "Related context items",
            })
        
        # Create the class schema
        class_schema = {
            "class": self.config.class_name,
            "description": self.config.class_description,
            "vectorizer": "none",  # We provide our own vectors
            "vectorIndexType": self.config.vector_index_type,
            "vectorIndexConfig": self.config.vector_index_config,
            "properties": properties,
        }
        
        # Create the class
        try:
            self.client.schema.create_class(class_schema)
            logger.info(f"Created Weaviate class {self.config.class_name}")
        except Exception as e:
            logger.error(f"Error creating Weaviate class: {str(e)}")
            raise FrameworkError(f"Error creating Weaviate class: {str(e)}")
            
    def _prepare_metadata_filters(self, metadata_filter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare metadata filters for Weaviate queries."""
        if not metadata_filter:
            return {}
            
        # Convert metadata filter to Weaviate where filter
        where_filter = {}
        
        for key, value in metadata_filter.items():
            if key == "$or" and isinstance(value, list):
                # Handle $or operator
                operands = []
                for item in value:
                    if isinstance(item, dict):
                        operands.append({"operator": "Equal", "path": list(item.keys())[0], "valueText": list(item.values())[0]})
                where_filter["operator"] = "Or"
                where_filter["operands"] = operands
            elif key == "$and" and isinstance(value, list):
                # Handle $and operator
                operands = []
                for item in value:
                    if isinstance(item, dict):
                        operands.append({"operator": "Equal", "path": list(item.keys())[0], "valueText": list(item.values())[0]})
                where_filter["operator"] = "And"
                where_filter["operands"] = operands
            elif key == "$in" and isinstance(value, dict):
                # Handle $in operator (e.g., {"level": {"$in": ["domain", "techstack"]}})
                for field, values in value.items():
                    if isinstance(values, list):
                        operands = []
                        for val in values:
                            operands.append({"operator": "Equal", "path": [field], "valueText": val})
                        where_filter["operator"] = "Or"
                        where_filter["operands"] = operands
            else:
                # Handle simple equality
                where_filter["operator"] = "Equal"
                where_filter["path"] = [key]
                if isinstance(value, str):
                    where_filter["valueText"] = value
                elif isinstance(value, int):
                    where_filter["valueInt"] = value
                elif isinstance(value, float):
                    where_filter["valueNumber"] = value
                elif isinstance(value, bool):
                    where_filter["valueBoolean"] = value
                else:
                    # Skip complex values
                    logger.warning(f"Skipping unsupported filter value type for {key}: {type(value)}")
                    continue
        
        return where_filter
    
    async def get_context(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[ContextResult]:
        """
        Retrieve context based on a query and optional filters.
        
        Args:
            query: The query text to find similar contexts
            metadata_filter: Optional metadata-based filters
            limit: Optional limit on number of results
            
        Returns:
            List of context results ordered by relevance
        """
        if not self.client:
            raise FrameworkError("Weaviate client not initialized")
        
        limit = limit or self.config.max_results
        
        try:
            # Get query embedding
            query_vector = await self.embedding_service.get_embedding(query)
            
            # Prepare query
            query_builder = self.client.query.get(
                self.config.class_name,
                ["content", "source", "category", "level", "importance", "timestamp", "_additional {certainty}"]
            )
            
            # Add vector search
            query_builder = query_builder.with_near_vector({
                "vector": query_vector
            })
            
            # Add hybrid search if enabled
            if self.config.search_use_hybrid and query:
                query_builder = query_builder.with_hybrid(
                    query=query,
                    alpha=self.config.search_alpha,
                    properties=["content"]
                )
            
            # Add metadata filters if provided
            if metadata_filter:
                where_filter = self._prepare_metadata_filters(metadata_filter)
                if where_filter:
                    query_builder = query_builder.with_where(where_filter)
            
            # Execute query
            results = query_builder.with_limit(limit).do()
            
            # Extract and format results
            context_results = []
            
            try:
                items = results.get("data", {}).get("Get", {}).get(self.config.class_name, [])
                for item in items:
                    # Extract certainty score
                    certainty = item.get("_additional", {}).get("certainty", 0)
                    
                    # Create metadata model
                    metadata = ContextMetadata(
                        source=item.get("source", ""),
                        category=item.get("category"),
                        level=item.get("level", "project"),
                        importance=item.get("importance"),
                        timestamp=item.get("timestamp"),
                    )
                    
                    # Create context result
                    context_result = ContextResult(
                        content=item.get("content", ""),
                        metadata=metadata,
                        score=certainty,
                    )
                    
                    context_results.append(context_result)
            except Exception as e:
                logger.error(f"Error processing Weaviate results: {str(e)}")
                # Continue with any results we could parse
            
            return context_results
            
        except Exception as e:
            logger.error(f"Error retrieving context from Weaviate: {str(e)}")
            raise FrameworkError(f"Error retrieving context from Weaviate: {str(e)}")
    
    async def add_context(
        self, 
        content: str, 
        metadata: Union[Dict[str, Any], ContextMetadata],
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Add context to the vector memory.
        
        Args:
            content: The context content to add
            metadata: Metadata about this context
            embedding: Optional pre-computed embedding vector
            
        Returns:
            ID of the added context
        """
        if not self.client:
            raise FrameworkError("Weaviate client not initialized")
        
        try:
            # Convert metadata to dict if it's a model
            if isinstance(metadata, ContextMetadata):
                metadata_dict = metadata.model_dump()
            else:
                metadata_dict = metadata.copy()
                
            # Generate embedding if not provided
            if embedding is None:
                embedding = await self.embedding_service.get_embedding(content)
            
            # Add timestamp if not provided
            if "timestamp" not in metadata_dict or not metadata_dict["timestamp"]:
                metadata_dict["timestamp"] = datetime.now().isoformat()
            
            # Prepare data object
            data_object = {
                "content": content,
                **{k: v for k, v in metadata_dict.items() if k != "custom_metadata"},
            }
            
            # Generate UUID
            item_uuid = str(uuid.uuid4())
            
            # Add to Weaviate
            self.client.data_object.create(
                class_name=self.config.class_name,
                data_object=data_object,
                uuid=item_uuid,
                vector=embedding
            )
            
            return item_uuid
            
        except Exception as e:
            logger.error(f"Error adding context to Weaviate: {str(e)}")
            raise FrameworkError(f"Error adding context to Weaviate: {str(e)}")
    
    async def add_contexts_batch(
        self,
        contexts: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Add multiple contexts in a batch operation.
        
        Args:
            contexts: List of context objects with 'content', 'metadata', and optional 'embedding'
            
        Returns:
            List of IDs for the added contexts
        """
        if not self.client:
            raise FrameworkError("Weaviate client not initialized")
        
        # Collect all IDs
        item_uuids = []
        
        try:
            # Process in batches for better performance
            with self.client.batch as batch:
                batch.batch_size = self.config.batch_size
                
                # Process each content item
                for i, context_item in enumerate(contexts):
                    content = context_item.get("content", "")
                    if not content:
                        logger.warning(f"Skipping context at index {i}: missing content")
                        continue
                    
                    metadata = context_item.get("metadata", {})
                    if isinstance(metadata, ContextMetadata):
                        metadata_dict = metadata.model_dump()
                    else:
                        metadata_dict = metadata.copy()
                    
                    # Get embedding
                    embedding = context_item.get("embedding")
                    if embedding is None:
                        # We'll need to generate the embedding
                        embedding = await self.embedding_service.get_embedding(content)
                    
                    # Add timestamp if not provided
                    if "timestamp" not in metadata_dict or not metadata_dict["timestamp"]:
                        metadata_dict["timestamp"] = datetime.now().isoformat()
                    
                    # Prepare data object
                    data_object = {
                        "content": content,
                        **{k: v for k, v in metadata_dict.items() if k != "custom_metadata"},
                    }
                    
                    # Generate UUID
                    item_uuid = str(uuid.uuid4())
                    item_uuids.append(item_uuid)
                    
                    # Add to batch
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=self.config.class_name,
                        uuid=item_uuid,
                        vector=embedding
                    )
            
            return item_uuids
            
        except Exception as e:
            logger.error(f"Error batch adding contexts to Weaviate: {str(e)}")
            raise FrameworkError(f"Error batch adding contexts to Weaviate: {str(e)}")
    
    async def clear_context(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Clear context from the vector memory.
        
        Args:
            filter_metadata: Optional metadata to filter which contexts to clear
            
        Returns:
            Number of contexts removed
        """
        if not self.client:
            raise FrameworkError("Weaviate client not initialized")
        
        try:
            # If no filter, delete all objects of the class
            if not filter_metadata:
                return self.client.batch.delete_objects(
                    class_name=self.config.class_name,
                    where=None,
                    output="verbose"
                ).get("results", {}).get("successful", 0)
            
            # Convert filter to Weaviate where clause
            where_filter = self._prepare_metadata_filters(filter_metadata)
            
            # Delete matching objects
            return self.client.batch.delete_objects(
                class_name=self.config.class_name,
                where=where_filter,
                output="verbose"
            ).get("results", {}).get("successful", 0)
            
        except Exception as e:
            logger.error(f"Error clearing context from Weaviate: {str(e)}")
            raise FrameworkError(f"Error clearing context from Weaviate: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector memory.
        
        Returns:
            Dictionary of statistics including counts by level, category, etc.
        """
        if not self.client:
            raise FrameworkError("Weaviate client not initialized")
        
        try:
            stats = {
                "total_count": 0,
                "by_level": {},
                "by_category": {},
                "by_source": {},
            }
            
            # Get total count
            count_query = f"""
            {{
              Aggregate {{
                {self.config.class_name} {{
                  meta {{
                    count
                  }}
                }}
              }}
            }}
            """
            count_result = self.client.query.raw(count_query)
            stats["total_count"] = count_result.get("data", {}).get("Aggregate", {}).get(self.config.class_name, [{}])[0].get("meta", {}).get("count", 0)
            
            # Get counts by level
            level_query = f"""
            {{
              Aggregate {{
                {self.config.class_name} {{
                  level {{
                    count
                    value
                  }}
                }}
              }}
            }}
            """
            level_result = self.client.query.raw(level_query)
            for level_item in level_result.get("data", {}).get("Aggregate", {}).get(self.config.class_name, [{}])[0].get("level", []):
                stats["by_level"][level_item.get("value")] = level_item.get("count", 0)
                
            # Get counts by category (group by)
            category_query = f"""
            {{
              Aggregate {{
                {self.config.class_name} {{
                  category {{
                    count
                    value
                  }}
                }}
              }}
            }}
            """
            category_result = self.client.query.raw(category_query)
            for cat_item in category_result.get("data", {}).get("Aggregate", {}).get(self.config.class_name, [{}])[0].get("category", []):
                stats["by_category"][cat_item.get("value") or "unclassified"] = cat_item.get("count", 0)
                
            # Get counts by source
            source_query = f"""
            {{
              Aggregate {{
                {self.config.class_name} {{
                  source {{
                    count
                    value
                  }}
                }}
              }}
            }}
            """
            source_result = self.client.query.raw(source_query)
            for source_item in source_result.get("data", {}).get("Aggregate", {}).get(self.config.class_name, [{}])[0].get("source", []):
                stats["by_source"][source_item.get("value") or "unspecified"] = source_item.get("count", 0)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats from Weaviate: {str(e)}")
            raise FrameworkError(f"Error getting stats from Weaviate: {str(e)}")
            
    async def add_relationship(self, source_id: str, target_id: str, relationship_type: str = "relatedTo") -> None:
        """
        Add a relationship between two context items.
        
        Args:
            source_id: UUID of the source context
            target_id: UUID of the target context
            relationship_type: Type of relationship (default is 'relatedTo')
        """
        if not self.client or not self.config.enable_references:
            raise FrameworkError("Weaviate client not initialized or references not enabled")
            
        try:
            # Add reference
            self.client.data_object.reference.add(
                from_class_name=self.config.class_name,
                from_uuid=source_id,
                from_property_name=relationship_type,
                to_class_name=self.config.class_name,
                to_uuid=target_id
            )
        except Exception as e:
            logger.error(f"Error adding relationship in Weaviate: {str(e)}")
            raise FrameworkError(f"Error adding relationship in Weaviate: {str(e)}")
            
    async def get_related_contexts(self, context_id: str, relationship_type: str = "relatedTo") -> List[ContextResult]:
        """
        Get contexts related to a given context.
        
        Args:
            context_id: UUID of the context
            relationship_type: Type of relationship
            
        Returns:
            List of related contexts
        """
        if not self.client or not self.config.enable_references:
            raise FrameworkError("Weaviate client not initialized or references not enabled")
            
        try:
            # Query for related contexts
            query = f"""
            {{
              Get {{
                {self.config.class_name} (id: "{context_id}") {{
                  {relationship_type} {{
                    ... on {self.config.class_name} {{
                      content
                      source
                      category
                      level
                      importance
                      timestamp
                      _additional {{
                        id
                      }}
                    }}
                  }}
                }}
              }}
            }}
            """
            
            result = self.client.query.raw(query)
            related_items = result.get("data", {}).get("Get", {}).get(self.config.class_name, [{}])[0].get(relationship_type, [])
            
            context_results = []
            for item in related_items:
                metadata = ContextMetadata(
                    source=item.get("source", ""),
                    category=item.get("category"),
                    level=item.get("level", "project"),
                    importance=item.get("importance"),
                    timestamp=item.get("timestamp"),
                )
                
                context_result = ContextResult(
                    content=item.get("content", ""),
                    metadata=metadata,
                    score=1.0,  # Explicit relationship has highest score
                )
                
                context_results.append(context_result)
                
            return context_results
            
        except Exception as e:
            logger.error(f"Error getting related contexts from Weaviate: {str(e)}")
            raise FrameworkError(f"Error getting related contexts from Weaviate: {str(e)}")
    
    async def shutdown(self) -> None:
        """Properly close the Weaviate client connection."""
        if self.client:
            try:
                # There's no explicit close method in the Weaviate client,
                # but we can set it to None to help garbage collection
                self.client = None
                logger.info("Weaviate provider shut down")
            except Exception as e:
                logger.warning(f"Error during Weaviate shutdown: {str(e)}") 