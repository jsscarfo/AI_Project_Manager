# Knowledge Capture Middleware

The `KnowledgeCaptureMiddleware` is a component designed to automatically extract and store valuable knowledge from LLM interactions. This document explains how to integrate and configure the middleware in your application.

## Overview

The Knowledge Capture Middleware intercepts LLM responses, analyzes them to extract valuable knowledge, and stores this knowledge in a vector database for future retrieval. This creates a continuous learning loop where the system becomes more knowledgeable over time based on its interactions.

Key features:
- Automatic knowledge extraction from LLM interactions
- Importance scoring and categorization of knowledge
- Efficient batch processing for storage
- Asynchronous processing to avoid impacting user experience
- Source tracking and attribution

## Integration

### Basic Setup

```python
from beeai_framework.vector import (
    WeaviateProvider, 
    WeaviateProviderConfig,
    EmbeddingService,
    KnowledgeCaptureMiddleware,
    KnowledgeCaptureMiddlewareConfig
)
from beeai_framework.middleware.base import MiddlewareChain
from beeai_framework.backend.providers import OpenAIProvider

# Initialize dependencies
vector_provider = WeaviateProvider(
    config=weaviate_config,
    embedding_service=embedding_service
)
await vector_provider.initialize()

# Configure the middleware
knowledge_capture_config = KnowledgeCaptureMiddlewareConfig(
    enabled=True,
    importance_threshold=0.5,
    batch_processing=True,
    batch_size=10
)

# Create the middleware
knowledge_capture_middleware = KnowledgeCaptureMiddleware(
    vector_provider=vector_provider,
    chat_model=openai_provider,
    config=knowledge_capture_config
)

# Add to middleware chain
middleware_chain = MiddlewareChain()
middleware_chain.add_middleware(knowledge_capture_middleware)
```

### Using with Request Processing

```python
# Process a request
context = MiddlewareContext(request=request)
context = await middleware_chain.process_request(context)

# Generate response from LLM (using your preferred method)
response_content = await generate_response(request)

# Set the response and process through middleware chain
context.set_response(response_content)
context.response_generated = True

# If your middleware chain supports post-processing
if hasattr(middleware_chain, "post_process_request"):
    context = await middleware_chain.post_process_request(context)
```

### Proper Shutdown

To ensure all pending knowledge entries are processed, call the shutdown method when your application is shutting down:

```python
await knowledge_capture_middleware.shutdown()
await vector_provider.shutdown()
```

## Configuration Options

The `KnowledgeCaptureMiddlewareConfig` provides several options for customizing behavior:

| Option | Description | Default |
| ------ | ----------- | ------- |
| `enabled` | Whether knowledge capture is enabled | `True` |
| `importance_threshold` | Minimum importance score for knowledge to be stored | `0.6` |
| `content_types` | Types of content to capture | `["code_snippet", "concept", "explanation", "best_practice"]` |
| `skip_short_responses` | Skip processing for short responses | `True` |
| `min_response_length` | Minimum words required in responses to process | `50` |
| `batch_processing` | Whether to use batch processing for storage | `False` |
| `batch_size` | Number of entries to store in a batch | `10` |
| `track_request_metadata` | Whether to track request metadata for attribution | `True` |
| `capture_conversation_context` | Whether to capture the context of the conversation | `True` |

## Knowledge Structure

Extracted knowledge is stored with the following metadata:

- `source`: Origin of the knowledge (default: "chat_conversation")
- `category`: Type of knowledge (e.g., "explanation", "code_snippet")
- `level`: Hierarchical level (e.g., "domain", "techstack", "project")
- `importance`: Score from 0 to 1 indicating importance
- `timestamp`: When the knowledge was captured

## Best Practices

1. **Set appropriate thresholds**: Adjust `importance_threshold` to balance quantity vs. quality of knowledge.
2. **Use batch processing**: For high-volume applications, enable batch processing to reduce database operations.
3. **Implement proper shutdown**: Always call `shutdown()` to ensure pending entries are processed.
4. **Monitor storage growth**: Implement cleanup policies for older or less relevant knowledge.
5. **Augment with manual knowledge**: Use `KnowledgeCaptureProcessor.store_knowledge_from_content()` to add verified knowledge directly.

## Example Use Cases

1. **Documentation generation**: Capture technical explanations from expert answers to build documentation.
2. **Code pattern libraries**: Extract coding patterns and best practices from AI-assisted development.
3. **Knowledge refinement**: Start with basic prompts and let the system learn domain-specific knowledge over time.
4. **Contextual assistance**: When paired with `ContextualEnhancementMiddleware`, creates a system that continuously improves its contextual awareness.

## Troubleshooting

- **No knowledge being captured**: Check the `importance_threshold` and `min_response_length` settings.
- **Irrelevant knowledge**: Try adjusting the extraction prompt in `KnowledgeExtractor`.
- **Performance issues**: Enable batch processing or adjust batch size for optimal performance.
- **Missing metadata**: Ensure the `track_request_metadata` option is enabled.

## Advanced Usage

### Custom Knowledge Processor

You can provide a custom `KnowledgeCaptureProcessor` with specialized extraction logic:

```python
custom_processor = KnowledgeCaptureProcessor(
    vector_provider=vector_provider,
    chat_model=chat_model,
    settings=custom_settings
)

middleware = KnowledgeCaptureMiddleware(
    vector_provider=vector_provider,
    chat_model=chat_model,
    config=config,
    knowledge_processor=custom_processor
)
```

### Adding Manual Knowledge

You can directly add verified knowledge to the system:

```python
await knowledge_processor.store_knowledge_from_content(
    content="Python generators are functions that use the yield statement to return a sequence of values.",
    metadata={
        "source": "documentation",
        "category": "explanation",
        "level": "concept",
        "importance": 0.9
    }
)
```

## Further Reading

- See the [Knowledge Retrieval documentation](knowledge_retrieval.md) for information on retrieving captured knowledge
- Refer to the [Vector Memory documentation](vector_memory.md) for details on the underlying storage system
- Check the [Middleware Framework documentation](middleware_framework.md) for general middleware concepts 