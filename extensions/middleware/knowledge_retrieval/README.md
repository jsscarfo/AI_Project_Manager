# Knowledge Retrieval Middleware

This middleware provides intelligent retrieval of knowledge from vector storage to enhance
the contextual awareness of sequential thinking processes.

## Features

- **Contextual Knowledge Retrieval**: Retrieves relevant knowledge based on the current reasoning step and context
- **Adaptive Query Enhancement**: Enhances queries with context from previous reasoning steps
- **Metadata-Based Filtering**: Allows filtering knowledge based on categories, sources, and importance
- **Sequential Thinking Integration**: Seamlessly integrates with the Sequential Thinking middleware
- **Customizable Settings**: Configure retrieval parameters like similarity threshold and result count

## Usage

### Basic Usage with Sequential Thinking

```python
from beeai_framework.vector.base import VectorMemoryProvider
from extensions.middleware.sequential.middleware import SequentialThinkingMiddleware
from extensions.middleware.knowledge_retrieval import KnowledgeRetrievalMiddleware, KnowledgeRetrievalSettings

# Initialize components
vector_provider = VectorMemoryProvider(...)  # Your vector provider implementation
chat_model = ChatModel(...)  # Your chat model implementation

# Initialize knowledge retrieval middleware
knowledge_retrieval = KnowledgeRetrievalMiddleware(
    vector_provider=vector_provider,
    settings=KnowledgeRetrievalSettings(
        enabled=True,
        max_results=5,
        similarity_threshold=0.65
    )
)

# Initialize sequential thinking middleware
sequential_thinking = SequentialThinkingMiddleware(
    llm_client=chat_model,
    context_refinement_processor=None  # Will be set by integration
)

# Integrate knowledge retrieval with sequential thinking
await knowledge_retrieval.set_sequential_middleware(sequential_thinking)

# Process a task with sequential thinking and knowledge retrieval
request = SequentialThinkingRequest(
    prompt="Explain how Python decorators work",
    task_type="explanation"
)

response = await sequential_thinking.process_request(request)
```

### Standalone Usage

```python
from beeai_framework.middleware.base import MiddlewareRequest
from extensions.middleware.knowledge_retrieval import KnowledgeRetrievalMiddleware

# Initialize the middleware
middleware = KnowledgeRetrievalMiddleware(
    vector_provider=vector_provider,
    settings=KnowledgeRetrievalSettings(enabled=True)
)

# Create a request
request = MiddlewareRequest(
    prompt="Explain Python decorators",
    context={"task_type": "explanation"}
)

# Process the request
response = await middleware.process(request)

# Use the enhanced prompt
enhanced_prompt = response.prompt
```

## Configuration

The `KnowledgeRetrievalSettings` class allows customization of the knowledge retrieval behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `True` | Whether knowledge retrieval is enabled |
| `max_results` | int | `5` | Maximum number of knowledge entries to retrieve |
| `similarity_threshold` | float | `0.65` | Minimum similarity score to include knowledge (0-1) |
| `relevance_boost_factor` | float | `1.2` | Factor to boost relevance of more recent knowledge |
| `context_window_tokens` | int | `2000` | Maximum tokens to include in context window |
| `retrieval_strategies` | List[str] | `["semantic", "keyword", "conceptual"]` | Strategies for knowledge retrieval |

## Integration with Knowledge Capture

This middleware is designed to work with the Knowledge Capture Processor, which captures and stores
knowledge from agent interactions. The knowledge capture processor stores knowledge entries in the
vector database, which are then retrieved by this middleware to enhance reasoning.

For a complete workflow:

1. Use the Knowledge Capture Processor to extract and store knowledge from conversations
2. Set up the Knowledge Retrieval Middleware with the same vector provider
3. Integrate with Sequential Thinking for enhanced reasoning with contextual knowledge

## Development

### Running Tests

```bash
pytest extensions/middleware/knowledge_retrieval/tests
```

### Example

See `example.py` for a complete working example that demonstrates the middleware in action.

 