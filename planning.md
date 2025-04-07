# AI Project Manager and Developer Coach

## Project Overview
The AI Project Manager and Developer Coach is an intelligent agent designed to assist in the development of machine learning applications. It combines project management capabilities with code development assistance, providing a comprehensive solution for ML project development.

## Architecture

### Components

1. **Project Manager Core**
   - Manages project structure and organization
   - Handles task scheduling and tracking
   - Maintains project state and context

2. **Memory System**
   - Short-term memory for current context
   - Long-term memory for project history
   - Vector database for semantic storage

3. **Vector Store**
   - Stores project documentation
   - Enables semantic search
   - Maintains code context

4. **Tool System**
   - Extensible framework for development tasks
   - Code generation and review
   - Project structure management
   - Real-time message processing

## File Structure
ai_project_manager/
├── documentation/          # Project documentation
│   ├── core_documentation.md
│   ├── debugging_tools.md
│   ├── planning.md
│   ├── project_map.md
│   └── tasks.md
├── src/                    # React+TypeScript frontend
│   ├── App.tsx
│   ├── index.css
│   ├── main.tsx
│   └── vite-env.d.ts
├── __init__.py            # Package initialization
├── ai_project_manager.py  # Core Python implementation
├── index.html            # Frontend entry point
├── requirements.txt       # Python dependencies
├── package.json          # Frontend dependencies
└── config files:
    ├── tsconfig.json
    ├── tsconfig.app.json
    ├── tsconfig.node.json
    ├── vite.config.ts
    ├── postcss.config.js
    └── tailwind.config.js

## Database Structure
The system uses a vector database for semantic storage with the following collections:

1. **Documentation Collection**
   - Stores project documentation
   - Enables semantic search
   - Maintains version history

2. **Code Context Collection**
   - Stores code snippets and context
   - Enables code-aware responses
   - Maintains relationships between code elements

3. **Project History Collection**
   - Stores project changes
   - Maintains task history
   - Enables context-aware responses

## Style Guidelines

### Code Style
- Follow PEP8 for Python code
- Use TypeScript for frontend development
- Maintain consistent naming conventions
- Use clear, descriptive variable names
- Document all public functions and classes

### Documentation
- Maintain up-to-date README.md
- Document all API endpoints
- Include setup instructions
- Provide usage examples

### Testing
- Write unit tests for all functions
- Include integration tests
- Maintain test coverage above 80%

## Environment Configuration

### Development
- Python 3.9+
- Node.js 16+
- Required packages in requirements.txt
- Frontend dependencies in package.json

### Production
- Containerized deployment
- Environment variables for configuration
- Logging and monitoring setup

## To Be Updated
- Database structure needs to be updated with current implementation details
- Additional configuration options need to be documented
- Deployment instructions need to be finalized
- Testing strategies need to be documented

Please update these sections as the project evolves.