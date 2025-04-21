# Weaviate Integration for BeeAI Framework

## Overview

This document outlines the integration of Weaviate as the vector database and knowledge graph solution for the BeeAI Framework. This implementation provides high-performance vector search, metadata filtering, and graph relationship capabilities to enhance contextual retrieval for AI interactions.

## Architecture

The Weaviate integration follows the BeeAI middleware pattern through:

1. **WeaviateProvider** - Core class implementing the `ContextProvider` interface
2. **Composite pattern** - Inclusion in the composite provider for multiple data sources
3. **Configuration system** - Settings management via environment variables and config files

```
┌─────────────────────────────────────┐
│         LLM Interaction Layer       │
└───────────────────┬─────────────────┘
                    │
┌───────────────────▼─────────────────┐
│    Contextual Enhancement Middleware │
└───────────────────┬─────────────────┘
                    │
┌───────────────────▼─────────────────┐
│        Composite Provider           │
└─┬─────────────────┬─────────────────┘
  │                 │
┌─▼─────────────┐ ┌─▼─────────────┐
│WeaviateProvider│ │GraphProvider  │
└───────────────┘ └───────────────┘
```

## Implementation

### WeaviateProvider Class

```python
class WeaviateProviderConfig(ContextProviderConfig):
    """Configuration for Weaviate Provider."""
    url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    batch_size: int = 100
    vector_dimension: int = 384
    default_collection: str = "Memory"
    timeout_seconds: float = 5.0
    
    # Load from environment variables
    def __init__(self, **data: Any):
        if "WEAVIATE_URL" in os.environ:
            data["url"] = os.environ["WEAVIATE_URL"]
        if "WEAVIATE_API_KEY" in os.environ:
            data["api_key"] = os.environ["WEAVIATE_API_KEY"]
        if "WEAVIATE_DIMENSION" in os.environ:
            data["vector_dimension"] = int(os.environ["WEAVIATE_DIMENSION"])
        
        super().__init__(**data)


class WeaviateProvider(ContextProvider):
    """Context provider using Weaviate for storage and retrieval."""
    
    def __init__(self, config: Optional[WeaviateProviderConfig] = None):
        self.config = config or WeaviateProviderConfig()
        super().__init__(self.config)
        self._client = None
        logger.info(f"Initialized WeaviateProvider with URL: {self.config.url}")
    
    async def initialize(self) -> None:
        """Initialize the Weaviate client and ensure schema exists."""
        await self._get_client()
        await self._ensure_schema()
    
    async def _get_client(self):
        """Get or create a Weaviate client."""
        if self._client is not None:
            return self._client
            
        import weaviate
        
        auth_config = weaviate.auth.AuthApiKey(self.config.api_key) if self.config.api_key else None
        
        self._client = weaviate.Client(
            url=self.config.url,
            auth_client_secret=auth_config,
            timeout_config=(self.config.timeout_seconds, self.config.timeout_seconds)
        )
        
        return self._client
    
    async def _ensure_schema(self):
        """Ensure the required schema exists in Weaviate."""
        client = await self._get_client()
        
        # Define the class schema
        class_obj = {
            "class": self.config.default_collection,
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The main content text"
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata for the content"
                },
                {
                    "name": "source",
                    "dataType": ["string"],
                    "description": "Source of the content"
                },
                {
                    "name": "category",
                    "dataType": ["string"],
                    "description": "Category of the content"
                },
                {
                    "name": "created",
                    "dataType": ["number"],
                    "description": "Creation timestamp"
                }
            ]
        }
        
        # Check if class exists, create if not
        if not client.schema.exists(self.config.default_collection):
            client.schema.create_class(class_obj)
            logger.info(f"Created schema for {self.config.default_collection}")
    
    async def get_context(self, query: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve context relevant to the query from Weaviate.
        
        Args:
            query: The query to retrieve context for
            metadata: Additional metadata for the query
            
        Returns:
            List of context items
        """
        if not query:
            return []
        
        client = await self._get_client()
        metadata = metadata or {}
        
        # Generate vector embedding for the query
        vector = await self._generate_embedding(query)
        
        # Build query filters from metadata
        filter_clause = self._build_filter(metadata)
        
        # Perform vector search
        try:
            result = (
                client.query
                .get(self.config.default_collection, ["content", "metadata", "source", "category", "created"])
                .with_near_vector({"vector": vector})
                .with_limit(metadata.get("limit", 10))
                .with_additional(["certainty", "id"])
            )
            
            if filter_clause:
                result = result.with_where(filter_clause)
                
            response = result.do()
            
            # Process and format results
            if "data" in response and "Get" in response["data"]:
                items = response["data"]["Get"][self.config.default_collection]
                formatted_results = []
                
                for item in items:
                    # Calculate score from certainty
                    score = item["_additional"]["certainty"]
                    
                    formatted_results.append({
                        "content": item["content"],
                        "metadata": {
                            "id": item["_additional"]["id"],
                            "source": item.get("source", "unknown"),
                            "category": item.get("category", ""),
                            **(item.get("metadata", {}))
                        },
                        "score": score
                    })
                
                return formatted_results
        except Exception as e:
            logger.error(f"Error searching Weaviate: {str(e)}")
            
        return []
    
    async def add_context(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add content to Weaviate.
        
        Args:
            content: The content to store
            metadata: Metadata for the content
            
        Returns:
            ID of the stored object
        """
        if not content:
            raise ValueError("Content cannot be empty")
        
        client = await self._get_client()
        metadata = metadata or {}
        
        # Generate embedding
        vector = await self._generate_embedding(content)
        
        # Extract common metadata fields
        source = metadata.pop("source", "user_input")
        category = metadata.pop("category", "general")
        created = metadata.pop("timestamp", time.time())
        
        # Create object data
        object_data = {
            "content": content,
            "source": source,
            "category": category,
            "created": created,
            "metadata": metadata  # Store remaining metadata fields
        }
        
        try:
            # Add object with vector
            result = client.data_object.create(
                data_object=object_data,
                class_name=self.config.default_collection,
                vector=vector
            )
            
            logger.info(f"Added content to Weaviate with ID: {result}")
            return result
        except Exception as e:
            logger.error(f"Error adding to Weaviate: {str(e)}")
            raise
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text using model or API."""
        # Implementation depends on how embeddings are generated in the framework
        # This is a placeholder - in a real implementation, we'd call an embedding service
        
        # Attempt to use framework embedding service if available
        try:
            from ...utils.embedding_service import get_embedding
            return await get_embedding(text)
        except ImportError:
            # Fall back to a basic embedding service
            from beeai_framework.utils.embeddings import generate_embedding
            return await generate_embedding(text)
    
    def _build_filter(self, metadata: Dict[str, Any]) -> Dict:
        """Build a Weaviate filter expression from metadata."""
        if not metadata:
            return {}
        
        filter_props = {}
        operator_map = {
            "eq": "Equal",
            "neq": "NotEqual",
            "gt": "GreaterThan",
            "gte": "GreaterThanEqual",
            "lt": "LessThan",
            "lte": "LessThanEqual",
            "like": "Like"
        }
        
        # Process simple filters
        where_filter = {"operator": "And", "operands": []}
        
        # Add source filter if specified
        if "source" in metadata:
            where_filter["operands"].append({
                "path": ["source"],
                "operator": "Equal",
                "valueString": metadata["source"]
            })
        
        # Add category filter if specified
        if "category" in metadata:
            where_filter["operands"].append({
                "path": ["category"],
                "operator": "Equal", 
                "valueString": metadata["category"]
            })
        
        # Handle explicit filter expressions
        if "filters" in metadata:
            for f in metadata["filters"]:
                if "field" in f and "operator" in f and "value" in f:
                    # Handle metadata fields with special path
                    path = f["field"].split(".")
                    if path[0] != "source" and path[0] != "category" and path[0] != "created":
                        path = ["metadata"] + path
                    
                    # Map operator
                    op = operator_map.get(f["operator"], "Equal")
                    
                    # Add filter based on value type
                    value_type = type(f["value"])
                    if value_type == str:
                        where_filter["operands"].append({
                            "path": path,
                            "operator": op,
                            "valueString": f["value"]
                        })
                    elif value_type == int or value_type == float:
                        where_filter["operands"].append({
                            "path": path,
                            "operator": op,
                            "valueNumber": f["value"]
                        })
                    elif value_type == bool:
                        where_filter["operands"].append({
                            "path": path,
                            "operator": op,
                            "valueBoolean": f["value"]
                        })
        
        # Only return filter if we have operands
        if len(where_filter["operands"]) > 0:
            return where_filter
        return {}
```

### Integration with Composite Provider

```python
from ..core.context_middleware import ContextProvider
from ..vector.weaviate_provider import WeaviateProvider
from ..graph.graph_provider import KnowledgeGraphProvider
from ..composite.composite_provider import CompositeProvider

# Create individual providers
weaviate_provider = WeaviateProvider()
graph_provider = KnowledgeGraphProvider()

# Create composite provider
composite = CompositeProvider()
composite.add_provider(weaviate_provider, weight=1.0)
composite.add_provider(graph_provider, weight=0.8)

# Initialize all providers
await composite.initialize()

# Use the composite provider for context retrieval
results = await composite.get_context("How does the system work?")
```

## Docker Deployment

For development and production, set up Weaviate using Docker:

```yaml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.30.1
    ports:
     - "8080:8080"
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

## Environment Variables

Configure the Weaviate provider using environment variables:

```
# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=
WEAVIATE_DIMENSION=384
```

## Schema Design

The primary schema for BeeAI Framework's context storage:

```
Class: Memory
  Properties:
    - content (text): The main content to be retrieved
    - metadata (object): Flexible metadata storage
    - source (string): Source identifier
    - category (string): Category classification
    - created (number): Timestamp of creation
```

## Usage Examples

### Basic Context Retrieval

```python
from beeai_framework.middleware.vector.weaviate_provider import WeaviateProvider

provider = WeaviateProvider()
await provider.initialize()

# Get context for a query
context = await provider.get_context("How do I implement a GraphQL API?")

# Add new context
context_id = await provider.add_context(
    "GraphQL is a query language for APIs and a runtime for fulfilling those queries with your existing data.",
    {
        "source": "documentation",
        "category": "api",
        "tags": ["graphql", "api", "query"]
    }
)
```

### Advanced Filtering

```python
# Search with metadata filters
context = await provider.get_context(
    "How do I implement authentication?",
    {
        "category": "security",
        "filters": [
            {"field": "created", "operator": "gt", "value": 1672531200},  # After Jan 1, 2023
            {"field": "tags", "operator": "contains", "value": "oauth"}
        ],
        "limit": 15
    }
)
```

## Performance Considerations

- **Batched Operations**: Use batch importing for large datasets
- **Index Optimization**: Optimize HNSW parameters for your specific needs
- **Query Construction**: Use efficient filters to reduce search space
- **Connection Pooling**: Maintain persistent connections to Weaviate

## Testing

Tests are provided for the Weaviate provider implementation:

- Unit tests for provider functionality
- Integration tests with actual Weaviate instance
- Performance benchmarks for tuning

## Extension Points

The current implementation can be extended in the following ways:

1. **Advanced Filtering**: Implement more complex filtering logic
2. **Cross-References**: Add support for entity cross-references
3. **Schema Evolution**: Support schema evolution as data model changes
4. **Multi-Modal Storage**: Extend to support image and other data types 