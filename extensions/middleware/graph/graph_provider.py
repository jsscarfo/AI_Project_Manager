"""
Knowledge Graph Context Provider

Adapts the knowledge graph system to the ContextProvider interface for
use with the Contextual Enhancement Middleware.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, Union
import logging
from pydantic import BaseModel, Field

# Import our middleware core
from ..core.context_middleware import ContextProvider, ContextProviderConfig

logger = logging.getLogger(__name__)

class KnowledgeGraphProviderConfig(ContextProviderConfig):
    """Configuration for the Knowledge Graph Provider."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_results: int = 20
    default_depth: int = 2
    relevance_cutoff: float = 0.65
    
    # Load from environment variables if available
    def __init__(self, **data: Any):
        # Override defaults with environment variables if they exist
        if "GRAPH_URI" in os.environ:
            data["uri"] = os.environ["GRAPH_URI"]
        if "GRAPH_USERNAME" in os.environ:
            data["username"] = os.environ["GRAPH_USERNAME"]
        if "GRAPH_PASSWORD" in os.environ:
            data["password"] = os.environ["GRAPH_PASSWORD"]
        if "GRAPH_DATABASE" in os.environ:
            data["database"] = os.environ["GRAPH_DATABASE"]
        
        super().__init__(**data)

class KnowledgeGraphProvider(ContextProvider):
    """
    Context provider that uses knowledge graph for retrieval.
    
    This provider adapts the knowledge graph system to work
    with the contextual enhancement middleware.
    """
    
    def __init__(self, config: Optional[KnowledgeGraphProviderConfig] = None):
        """
        Initialize the Knowledge Graph Provider.
        
        Args:
            config: Configuration for the knowledge graph provider
        """
        self.config = config or KnowledgeGraphProviderConfig()
        super().__init__(self.config)
        
        # We'll import the knowledge graph client lazily
        self._graph_client = None
        logger.info(f"Initialized KnowledgeGraphProvider with URI: {self.config.uri}")
    
    async def initialize(self) -> None:
        """Initialize the knowledge graph client."""
        await self._get_graph_client()
        logger.info("Initialized knowledge graph client")
    
    async def _get_graph_client(self):
        """
        Get or create a knowledge graph client.
        
        Returns:
            Knowledge graph client
        """
        if self._graph_client is not None:
            return self._graph_client
        
        try:
            # First try importing from our extensions
            from ...graph.knowledge_graph import KnowledgeGraph
        except ImportError:
            try:
                # Fall back to the original implementation
                from src.graph.KnowledgeGraph import KnowledgeGraph
            except ImportError:
                # Create minimal implementation that logs warnings but doesn't break
                logger.warning("KnowledgeGraph implementation not found, using minimal implementation")
                self._graph_client = MinimalGraphClient()
                return self._graph_client
        
        # Create a new client instance
        self._graph_client = KnowledgeGraph({
            "uri": self.config.uri,
            "username": self.config.username,
            "password": self.config.password,
            "database": self.config.database
        })
        
        # Initialize the client
        await self._graph_client.initialize()
        
        return self._graph_client
    
    async def get_context(self, query: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve context relevant to the query from knowledge graph.
        
        Args:
            query: The query to retrieve context for
            metadata: Additional metadata for the query
            
        Returns:
            List of context items
        """
        if not query:
            return []
        
        metadata = metadata or {}
        depth = metadata.get("depth", self.config.default_depth)
        max_results = metadata.get("max_results", self.config.max_results)
        
        # Get the graph client
        graph_client = await self._get_graph_client()
        
        # Check if we can perform semantic search
        if hasattr(graph_client, "semanticSearch"):
            # First try with semantic search
            try:
                results = await graph_client.semanticSearch(
                    query, 
                    {"depth": depth, "maxResults": max_results}
                )
                
                return self._format_results(results, query)
            except Exception as e:
                logger.error(f"Error in semantic search: {str(e)}")
                # Fall back to keyword search
        
        # Try keyword search as fallback
        if hasattr(graph_client, "keywordSearch"):
            try:
                results = await graph_client.keywordSearch(
                    query, 
                    {"depth": depth, "maxResults": max_results}
                )
                
                return self._format_results(results, query)
            except Exception as e:
                logger.error(f"Error in keyword search: {str(e)}")
        
        # Last resort - try direct query by entity name
        try:
            results = await graph_client.getEntityByName(
                query, 
                {"depth": depth}
            )
            
            if results:
                return self._format_results([results], query)
        except Exception as e:
            logger.error(f"Error in direct entity query: {str(e)}")
        
        return []
    
    def _format_results(self, results: List[Dict], query: str) -> List[Dict[str, Any]]:
        """Format graph results for the context middleware."""
        formatted_results = []
        
        for result in results:
            # Skip results with no content
            if not result.get("content") and not result.get("properties"):
                continue
            
            # Get the content
            content = result.get("content", "")
            
            # If no direct content, build from properties
            if not content and result.get("properties"):
                props = result["properties"]
                content = f"{props.get('name', 'Unnamed Entity')}: {props.get('description', '')}"
            
            # Skip empty content
            if not content.strip():
                continue
            
            # Build metadata
            result_metadata = {
                "source": "knowledge_graph",
                "type": result.get("type", "entity"),
                "id": result.get("id", ""),
                "labels": result.get("labels", []),
            }
            
            # Copy over any original metadata
            if result.get("metadata"):
                result_metadata.update(result["metadata"])
            
            # Add to formatted results
            formatted_results.append({
                "content": content,
                "metadata": result_metadata,
                "score": result.get("score", 1.0)
            })
        
        # Filter out results below relevance cutoff
        filtered_results = [
            r for r in formatted_results 
            if r["score"] >= self.config.relevance_cutoff
        ]
        
        # Sort by relevance score
        sorted_results = sorted(filtered_results, key=lambda x: x["score"], reverse=True)
        
        return sorted_results
    
    async def add_context(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add content to knowledge graph.
        
        Args:
            content: The content to store
            metadata: Metadata for the content
            
        Returns:
            ID of the stored entity
        """
        if not content:
            raise ValueError("Content cannot be empty")
        
        metadata = metadata or {}
        
        # Get the graph client
        graph_client = await self._get_graph_client()
        
        # Prepare entity data
        entity_type = metadata.get("type", "Note")
        entity_data = {
            "content": content,
            "name": metadata.get("name", content[:50] + ("..." if len(content) > 50 else "")),
            "description": metadata.get("description", content[:150] + ("..." if len(content) > 150 else "")),
        }
        
        # Add any additional properties from metadata
        properties = metadata.get("properties", {})
        entity_data.update(properties)
        
        # Create entity in graph
        try:
            entity_id = await graph_client.createEntity(entity_type, entity_data)
            
            # Create any relationships specified in metadata
            relationships = metadata.get("relationships", [])
            for relationship in relationships:
                await graph_client.createRelationship(
                    entity_id,
                    relationship["target_id"],
                    relationship["type"],
                    relationship.get("properties", {})
                )
            
            return entity_id
        except Exception as e:
            logger.error(f"Error adding context to knowledge graph: {str(e)}")
            raise


class MinimalGraphClient:
    """Minimal implementation of graph client for when the real one isn't available."""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize the client."""
        self.initialized = True
        logger.warning("Using minimal graph client - no actual graph operations will be performed")
    
    async def semanticSearch(self, query, options=None):
        """Warn and return empty results for semantic search."""
        logger.warning(f"Semantic search called with query '{query}' but no implementation available")
        return []
    
    async def keywordSearch(self, query, options=None):
        """Warn and return empty results for keyword search."""
        logger.warning(f"Keyword search called with query '{query}' but no implementation available")
        return []
    
    async def getEntityByName(self, name, options=None):
        """Warn and return empty results for entity lookup."""
        logger.warning(f"Entity lookup called with name '{name}' but no implementation available")
        return None
    
    async def createEntity(self, entity_type, data):
        """Warn and return dummy ID for entity creation."""
        logger.warning(f"Entity creation called for type '{entity_type}' but no implementation available")
        return "dummy-id-" + str(hash(str(data)))[:8]
    
    async def createRelationship(self, source_id, target_id, rel_type, properties=None):
        """Warn for relationship creation."""
        logger.warning(f"Relationship creation called but no implementation available")
        return None 