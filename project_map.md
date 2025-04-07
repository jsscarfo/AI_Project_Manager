# AI Project Manager and Developer Coach - Project Map

> **IMPORTANT**: Before creating new files or functionality, always check this map to prevent duplication. Reuse existing components whenever possible to maintain code consistency and reduce redundancy.

This document serves as a comprehensive map of the entire codebase, listing all project files with their functions and usage. Always reference this document before creating new files to ensure you're reusing existing functionality.

## Project Structure Overview

```
ai_project_manager/
├── documentation/          # Project documentation
├── backend/                # Backend Python code
│   ├── core/               # Core system components
│   ├── assistants/         # Specialized assistants
│   └── utils/              # Utility functions
├── frontend/               # React+TypeScript frontend
│   ├── components/         # Reusable UI components
│   ├── services/           # API and state management
│   └── pages/              # Page components
├── tests/                  # Test scripts and test cases
└── logs/                   # Log files
```

## Core Files

### Root Directory

| File | Purpose | Key Functions | Usage |
|------|---------|---------------|-------|
| `__init__.py` | Package initialization | - | Imported when using the package |
| `config.py` | Configuration settings | - | Stores global configuration variables |
| `requirements.txt` | Dependencies list | - | Used for installing dependencies |
| `README.md` | Project documentation | - | Overview of the project |

### Backend Core Module

| File | Purpose | Key Functions | Usage |
|------|---------|---------------|-------|
| `backend/core/project_manager.py` | Core system management | `add_message()`, `execute_tool()` | Central component managing the system |
| `backend/core/memory.py` | Memory management | `add_message()`, `get_context()` | Handles short and long-term memory |
| `backend/core/vector_store.py` | Vector database | `add_document()`, `search()` | Manages document storage and retrieval |
| `backend/core/tools.py` | Tool system | `register_tool()`, `execute_tool()` | Handles tool registration and execution |

### Backend Assistants Module

| File | Purpose | Key Functions | Usage |
|------|---------|---------------|-------|
| `backend/assistants/ml_assistant.py` | ML project assistance | `start_project()`, `review_code()` | Provides ML project guidance |
| `backend/assistants/code_assistant.py` | Code assistance | `review_code()`, `debug_issue()` | Helps with code review and debugging |

### Backend Utils Module

| File | Purpose | Key Functions | Usage |
|------|---------|---------------|-------|
| `backend/utils/logging_utils.py` | Logging utilities | `setup_logging()` | Configures logging system |
| `backend/utils/embedding_utils.py` | Embedding utilities | `compute_embedding()` | Handles text embeddings |
| `backend/utils/message_utils.py` | Message utilities | `process_message()` | Processes different message types |