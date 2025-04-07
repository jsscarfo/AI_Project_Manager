# AI Project Manager and Developer Coach - Project Planning

## Project Overview
This project implements an intelligent agent that acts as a project manager and senior developer to guide the development of machine learning applications. It leverages an extended memory system, vector database for documentation, and an extensible tool system to provide code review, debugging assistance, and project structure management.

> **Important**: Always refer to `project_map.md` for the most up-to-date file structure and component descriptions before creating new files or modifying existing ones.

## Architecture
- **Storage**: In-memory for short-term storage; Vector database for long-term storage and semantic search.
- **Communication**: Message-based system with different message types (system, user, assistant, tool).
- **Processing**: Threaded message processing with tool execution and context analysis.

## Components

1. **ProjectManager**
   - Core component managing the entire system
   - Handles message processing and tool execution
   - Maintains memory and vector store

2. **Memory**
   - Short-term memory for recent interactions
   - Long-term memory for project context
   - Message history management

3. **VectorStore**
   - Document storage with semantic search capabilities
   - Embedding-based retrieval system
   - Context-aware documentation access

4. **Tool**
   - Extensible tool system
   - Built-in code review and debugging tools
   - Custom tool registration capability

5. **MLProjectAssistant**
   - High-level interface for ML projects
   - Project structure management
   - Development workflow assistance

6. **Frontend Interface**
   - React-based UI for interacting with the system
   - Code editor with syntax highlighting
   - Project visualization and management

## Environment Configuration
- `VECTOR_DB_PATH`: Path to the vector database storage.
- `LOG_LEVEL`: Logging level configuration.
- `MODEL_PATH`: Path to embedding and AI models.