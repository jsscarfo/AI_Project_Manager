# Weaviate Integration for BeeAI Framework V5

## Overview

The Weaviate integration for BeeAI Framework V5 provides a robust vector database solution that builds upon the learnings and architectural patterns established in V4's contextual retrieval system. This integration enhances the framework's memory capabilities with Weaviate's powerful vector search, filtering, and hybrid search capabilities, along with knowledge graph functionality to enable a sophisticated selective contextual retrieval system.

## Vision: Selective Contextual Retrieval

The primary goal of this integration extends beyond traditional Retrieval-Augmented Generation (RAG) to implement a more sophisticated Selective Contextual Retrieval system. This approach:

- **Selectively compartmentalizes context** to provide LLM agents with the exact amount of information needed for their specific tasks - no more, no less
- **Organizes knowledge hierarchically** across Domain, TechStack, and Project-specific information levels
- **Integrates with sequential thinking** processes to enable step-by-step context-aware reasoning
- **Functions as transparent middleware** that enhances LLM interactions without requiring explicit tool calls
- **Automatically captures and indexes new knowledge** encountered during system operation

This approach mirrors how human experts with years of experience work - having access to information at many different levels, with the relevant context automatically available when needed without being overwhelmed by unnecessary details.

### Hierarchical Knowledge Organization

The system organizes knowledge across three primary hierarchical levels:

1. **Domain Knowledge**: Business information, marketing concepts, world information, and other general domain knowledge
2. **TechStack Knowledge**: Data sources for components, programming languages, manuals, debugging systems, technical articles, etc.
3. **Project-specific Information**: Codebase, documentation, tasks, user data, and other project-specific details

This hierarchical organization allows the system to provide the appropriate level of context for different types of tasks.

### Integration with Sequential Thinking

A key enhancement is the integration with the sequential thinking component, enabling:

- Context-aware step-by-step reasoning processes
- Progressive context refinement as the reasoning develops
- Efficient use of limited context windows by providing only the most relevant information at each step

## Current Implementation Status

The Weaviate integration currently includes:

1. **Core Implementation**:
   - `WeaviateProvider` class implementing the vector memory provider interface
   - Connection configuration and management for Weaviate instances
   - Basic schema creation and management
   - Vector search capabilities with configurable similarity thresholds
   - Support for metadata filtering in queries

2. **Middleware Integration**:
   - Integration with the `ContextualEnhancementMiddleware` architecture
   - Sequential thinking integration through the `KnowledgeRetrievalMiddleware`
   - Context-aware enhancements based on reasoning step

3. **Testing Framework**:
   - Unit tests for core Weaviate provider functionality
   - Mock implementations for testing without a live Weaviate instance

## Components Under Development

Key components that require further development include:

1. **Knowledge Capture Middleware**:
   - Automatic extraction of knowledge from LLM outputs
   - Importance and relevance scoring for new knowledge
   - Integration with the knowledge storage system

2. **Complete Hybrid Search Functionality**:
   - Enhanced hybrid search combining vector similarity with BM25 and filtering
   - Advanced query construction for precise context retrieval
   - Graph-based relationship traversal for deeper context understanding

3. **Hierarchical Knowledge Implementation**:
   - Full implementation of the three-level knowledge hierarchy
   - Task-specific knowledge level selection
   - Balanced retrieval across knowledge levels

4. **Context Quality Evaluation**:
   - Metrics for evaluating context relevance and utility
   - Feedback mechanisms for improving retrieval quality
   - Systems for prioritizing high-quality context sources

## Architecture

### Vector Memory System Evolution

Building on V4's vector memory system, the V5 Weaviate integration maintains the core concepts of contextual retrieval while enhancing the implementation with a middleware-based approach:

- **Middleware Architecture**: Weaviate integration is implemented as a middleware component, providing a clean separation of concerns and allowing for flexible integration with other components.
  
- **Automatic Contextual Retrieval**: Following the ACRS (Automatic Contextual Retrieval System) pattern established in V4, the integration automatically enhances LLM interactions by analyzing requests, generating appropriate queries, and retrieving relevant context.

- **Hybrid Retrieval Techniques**: The V5 integration supports both vector similarity search and graph-based relationship filtering, allowing for precise, selective context retrieval.

### Component Diagram

```
┌─────────────────────────────────┐
│      BeeAI Core Framework       │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Selective Contextual Retrieval │
│       Middleware System         │
│                                 │
│  ┌─────────────┐ ┌────────────┐ │
│  │ Embedding   │ │ Weaviate   │ │
│  │ Service     │ │ Provider   │ │
│  └─────────────┘ └────────────┘ │
│          ┌──────────────┐       │
│          │ Sequential   │       │
│          │ Thinking     │       │
│          │ Integration  │       │
│          └──────────────┘       │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│       Weaviate Database         │
│  (Vector + Knowledge Graph)     │
└─────────────────────────────────┘
```

## Implementation Details

### WeaviateProvider Class

The `WeaviateProvider` class implements the `VectorMemoryProvider` interface, adapting Weaviate to work within the BeeAI Framework's vector memory system.

Key components include:

1. **Configuration**:
   - Connection settings (host, port, protocol)
   - Class and collection management
   - Authentication
   - Search parameters (similarity threshold, result limits)

2. **Core Methods**:
   - `initialize()`: Establishes connection and ensures schema existence
   - `get_context(query, metadata_filter)`: Retrieves relevant context based on query and optional metadata filters
   - `add_context(content, metadata)`: Adds content to the vector database with appropriate metadata
   - `clear_context()`: Removes all stored contexts

3. **Schema Management**:
   - Automatic schema creation with configurable vector dimensions
   - Support for metadata properties
   - Cross-references and relationships
   - Knowledge hierarchy representation

4. **Advanced Features**:
   - Batch import for efficient data loading
   - Hybrid search capabilities for combining vector and graph-based search
   - Configurable similarity metrics (cosine, dot product, L2)
   - Context level filtering (Domain, TechStack, Project-specific)

### Embedding Service

The integration includes a robust `EmbeddingService` that:

- Generates vector embeddings for text using configurable embedding models
- Implements caching for efficient embedding generation
- Supports batch processing for performance optimization
- Handles both synchronous and asynchronous embedding generation

### Context Selection Logic

A critical component of the selective contextual retrieval system is the context selection logic that:

- Analyzes the current task or reasoning step to determine appropriate context levels
- Dynamically adjusts the balance between breadth and depth of context
- Evaluates context relevance and quality before inclusion
- Optimizes context for the specific LLM and task requirements

## Docker Deployment

To simplify deployment, a Docker Compose configuration is provided:

```yaml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.22.4
    ports:
      - "8080:8080"
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

## Usage Examples

### Basic Usage

```python
from vector.weaviate_provider import WeaviateProvider, WeaviateProviderConfig
from vector.embedding_service import EmbeddingService

# Initialize embedding service
def get_embedding(text):
    # Implementation of embedding generation
    pass

embedding_service = EmbeddingService(embedding_fn=get_embedding, 
                                     cache_dir="./cache",
                                     cache_ttl=86400)

# Configure Weaviate provider
config = WeaviateProviderConfig(
    host="localhost",
    port=8080,
    class_name="ContextMemory",
    dimension=1536
)

# Initialize provider
provider = WeaviateProvider(config, embedding_service)
await provider.initialize()

# Add context
await provider.add_context(
    content="This is important information about the project.",
    metadata={"source": "documentation", "category": "project_info", "level": "project"}
)

# Retrieve context
results = await provider.get_context(
    query="Tell me about the project",
    metadata_filter={"source": "documentation", "level": "project"}
)
```

### Integration with Sequential Thinking

Building on the Selective Contextual Retrieval vision, here's how integration with sequential thinking processes works:

```python
# Sequential thinking with contextual enhancement
async def enhanced_sequential_thinking(task, initial_context=None):
    # Initialize the sequence
    sequence = []
    current_context = initial_context or {}
    
    # Step 1: Define the problem and goals
    step_context = await contextual_middleware.get_context_for_step(
        task=task, 
        step="problem_definition",
        current_context=current_context
    )
    problem_definition = await llm_provider.generate(
        prompt=f"Define the problem and goals for: {task}",
        context=step_context
    )
    sequence.append(problem_definition)
    
    # Step 2: Break down into subtasks
    step_context = await contextual_middleware.get_context_for_step(
        task=task, 
        step="task_breakdown",
        previous_steps=sequence,
        current_context=current_context
    )
    subtasks = await llm_provider.generate(
        prompt=f"Break down this task into subtasks: {task}",
        context=step_context
    )
    sequence.append(subtasks)
    
    # Additional steps with relevant context at each stage...
    
    # Final output synthesis
    return sequence
```

### Hierarchical Context Selection

```python
# Example of selecting context based on hierarchical level
async def get_hierarchical_context(query, task_type):
    # Determine appropriate knowledge levels based on task type
    if task_type == "high_level_planning":
        levels = ["domain", "techstack"]
    elif task_type == "code_implementation":
        levels = ["techstack", "project"]
    elif task_type == "debugging":
        levels = ["project"]
    else:
        levels = ["domain", "techstack", "project"]  # Default to all levels
    
    # Retrieve context with level filtering
    context = await weaviate_provider.get_context(
        query=query,
        metadata_filter={"level": {"$in": levels}}
    )
    
    # Further processing and relevance scoring...
    return context
```

## Performance Considerations

### Batch Import Process

For efficient importing of large datasets, the V5 integration provides a batch import process:

```python
async def batch_import(provider, data_items):
    batch_size = 100
    for i in range(0, len(data_items), batch_size):
        batch = data_items[i:i+batch_size]
        memories = []
        
        # Generate embeddings for batch
        texts = [item["content"] for item in batch]
        embeddings = await embedding_service.get_embeddings(texts)
        
        # Create memory objects
        for j, item in enumerate(batch):
            memories.append({
                "content": item["content"],
                "embedding": embeddings[j],
                "metadata": item["metadata"]
            })
        
        # Add batch to Weaviate
        await provider.add_memories(memories)
```

### Optimization Tips

1. **Embedding Caching**: Utilize the `EmbeddingService` caching to avoid regenerating embeddings for repeated content
2. **Connection Pooling**: The Weaviate client maintains connection pooling for efficient request handling
3. **Schema Design**: Design your metadata schema to reflect hierarchical knowledge levels
4. **Context Size Management**: Implement intelligent truncation and summarization to optimize context size
5. **Query Optimization**: Use hybrid retrieval combining vector similarity with graph filtering
6. **Step-Specific Caching**: Cache context relevant to specific reasoning steps

## Integration with BeeAI Framework

The Weaviate integration follows the middleware pattern established in V5's architecture:

1. **Registration**: The Weaviate provider is registered with the vector memory middleware
2. **Configuration**: Configuration is managed through the middleware's configuration system
3. **Lifecycle Management**: The provider follows the framework's initialization and shutdown patterns
4. **Error Handling**: Errors are propagated through the middleware's error handling system
5. **Sequential Thinking Integration**: Seamless integration with the sequential thinking component

## Benefits for AI System Development

This selective contextual retrieval approach provides specific benefits for building AI systems:

1. **Focused Context**: AI agents receive precisely the context needed for their specific tasks
2. **Knowledge Transfer**: The system enables effective knowledge transfer between different components
3. **Hierarchical Understanding**: Components understand where they fit in the larger system architecture
4. **Evolving Knowledge Base**: The system continuously learns and improves as it processes more information
5. **Reduced Context Overload**: Prevents overwhelming LLMs with irrelevant information, improving output quality

## Future Development Areas

Based on the current implementation, the following areas need further development:

1. **Knowledge Capture System**:
   - Automated extraction and storage of knowledge from LLM interactions
   - Classification and organization of captured knowledge
   - Relationship detection between knowledge entities

2. **Advanced Hybrid Search**:
   - Full implementation of Weaviate's hybrid search capabilities
   - Sophisticated filter construction based on task type
   - Dynamic query refinement based on initial search results

3. **Context Quality Evaluation**:
   - Metrics for measuring context relevance and utility
   - Feedback mechanisms for improving retrieval quality
   - LLM-based context summarization and enhancement

4. **Optimization and Scaling**:
   - Parallel query processing for improved performance
   - Distributed embedding generation
   - Tiered caching strategies for frequently accessed knowledge

## Conclusion

The Weaviate integration for BeeAI Framework V5 represents a significant advancement in contextual retrieval capabilities, moving beyond traditional RAG approaches to a more sophisticated selective contextual retrieval system. By leveraging both vector similarity and knowledge graph relationships, the system can provide AI agents with precisely the information they need at each step of their reasoning process.

This approach mirrors how human experts work, with relevant knowledge automatically available when needed, without the cognitive load of processing irrelevant information. The middleware-based architecture ensures a clean separation of concerns, allowing for flexible integration with other components while maintaining a seamless developer experience.

By providing a hierarchical knowledge organization spanning Domain, TechStack, and Project-specific information, the system is uniquely positioned to support the development of advanced AI systems that can understand and operate at multiple levels of abstraction. 