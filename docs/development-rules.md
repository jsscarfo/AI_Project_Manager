# Development Rules for Project Manager V5

This document outlines the coding standards, development practices, and conventions to be followed during the development of Project Manager V5.

## Code Structure & Organization

### File Organization

- Source code is organized by feature/domain in the following structure:
  ```
  V5/
  ├── src/
  │   ├── middleware/          # Middleware framework and implementations
  │   ├── vector/              # Vector memory system (Weaviate)
  │   ├── llm/                 # LLM provider system
  │   ├── tools/               # Tool registry and implementations
  │   ├── orchestration/       # Orchestration framework
  │   ├── mcp/                 # FastMCP server implementation
  │   └── utils/               # Utility functions
  ├── extensions/
  │   ├── middleware/          # Extension middleware implementations
  │   │   └── vector/          # Vector provider (Weaviate)
  │   ├── tools/               # Extension tools
  │   └── llm/                 # LLM provider extensions
  ├── tests/                   # Test files mirror src structure
  ├── docs/                    # Documentation
  └── examples/                # Example implementations
  ```

### Naming Conventions

- **Files**: Use descriptive, lowercase names with underscores
  - Python: `weaviate_provider.py`, `embedding_service.py`

- **Classes**: Use PascalCase
  - `WeaviateProvider`, `MiddlewareChain`

- **Abstract Base Classes**: Use PascalCase with "Base" or "ABC" suffix
  - `ProviderBase`, `MiddlewareABC`

- **Functions/Methods**: Use snake_case
  - `retrieve_context()`, `add_memory()`

- **Variables**: Use snake_case
  - `relevance_threshold`, `embedding_cache`

- **Constants**: Use UPPER_SNAKE_CASE
  - `DEFAULT_VECTOR_DIMENSION`, `MAX_BATCH_SIZE`

## Coding Standards

### Python Standards

- Follow PEP 8 style guide
- Use type hints for all function parameters and return values
- Format code with Black
- Use docstrings for all public functions, classes, and methods
- Use abstract base classes for interfaces
- Prefer async/await for asynchronous operations

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorProviderBase(ABC):
    """Base class for vector database providers."""
    
    @abstractmethod
    async def retrieve_context(
        self, 
        query: str, 
        limit: int = 5, 
        threshold: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context based on similarity to query.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of context items with content and metadata
        """
        pass
```

### FastMCP Standards

- Use FastMCP decorators for tool and resource definition
- Follow the MCP protocol specifications
- Define clear input/output types using Pydantic models
- Include comprehensive docstrings for tools and resources

```python
from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("VectorMemoryService")

class SearchParams(BaseModel):
    query: str
    limit: int = 5
    threshold: float = 0.75

@mcp.tool()
async def search_vector_db(params: SearchParams) -> List[Dict[str, Any]]:
    """
    Search the vector database for relevant content.
    
    Args:
        params: Search parameters including query, limit, and threshold
        
    Returns:
        List of matching results with content and metadata
    """
    # Implementation
    return results
```

### General Standards

- Maximum line length of 100 characters
- No function should exceed 50 lines
- No file should exceed 500 lines (refactor if approaching)
- All code must have corresponding tests
- Use clear, descriptive variable and function names
- Avoid abbreviations unless they are standard (e.g., HTTP, URL)

## Testing Standards

### Test Organization

- Test files should mirror the structure of the source code
- Name test files with `test_` prefix
- Group tests by functionality using pytest fixtures and parametrization

### Test Coverage

- Minimum 80% code coverage required
- Unit tests for all public functions and methods
- Integration tests for components that interact
- Each test file should include:
  - Happy path tests (expected use)
  - Edge case tests
  - Error handling tests

### Test Implementation

- Use pytest for all tests
- Use pytest-asyncio for async tests
- Use mocks for external dependencies
- Tests should be isolated and not depend on each other
- Avoid testing implementation details, focus on behavior

```python
# Python test example
import pytest
import asyncio

@pytest.mark.asyncio
async def test_retrieve_context():
    """Test retrieving context from Weaviate provider."""
    # Setup
    provider = WeaviateProvider(config)
    await provider.initialize()
    
    # Test
    results = await provider.retrieve_context("test query", limit=3)
    
    # Assertions
    assert len(results) <= 3
    assert all("content" in r for r in results)
    assert all("score" in r for r in results)
```

## Documentation Standards

### Code Documentation

- All public functions, classes, and methods must have docstrings
- Document parameters, return values, and exceptions
- Include examples for complex functions
- Add inline comments for non-obvious code

### Project Documentation

- Update README.md when adding new features
- Keep architecture diagrams up-to-date
- Document all configuration options
- Provide examples for major functionality
- Update CHANGELOG.md with all changes

## Development Workflow

### Git Workflow

- Feature branches should be created from `main`
- Branch naming: `feature/descriptive-name` or `fix/issue-description`
- Make frequent, small commits with clear messages
- Pull requests require code review and passing tests
- Squash merge to main after approval

### Commit Messages

- Follow conventional commits format:
  - `feat: add new feature`
  - `fix: resolve issue with X`
  - `docs: update documentation`
  - `test: add test for feature Y`
  - `refactor: improve performance of Z`

### Code Review Process

- All code must be reviewed before merging
- Reviewers should check for:
  - Correctness
  - Test coverage
  - Code quality
  - Documentation
  - Adherence to standards
- Address all review comments before merging

## Performance Considerations

- Use batching for vector operations where possible
- Cache expensive operations (e.g., embeddings)
- Use vectorized operations instead of loops where applicable
- Profile middleware chains to identify bottlenecks
- Monitor memory usage, especially for vector operations
- Use streams for large responses

## Error Handling

- Use specific exception types
- Log all errors with appropriate context
- Provide helpful error messages
- Implement graceful fallbacks where appropriate
- Return clear error responses from API endpoints

## Security Considerations

- Never hardcode credentials
- Use environment variables for sensitive configuration
- Validate all user input
- Implement proper authentication and authorization
- Follow OWASP security guidelines
- Sanitize data before logging

## Dependency Management

- All dependencies must be explicitly declared in requirements.txt or pyproject.toml
- Specify version ranges to avoid breaking changes
- Regularly update dependencies for security fixes
- Minimize the number of dependencies
- Document all third-party dependencies

## Conclusion

These development rules ensure consistency, quality, and maintainability across the Project Manager V5 codebase. All developers working on the project should follow these guidelines to create a cohesive, performant, and maintainable system. 