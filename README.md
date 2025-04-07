# AI Project Manager and Developer Coach

An intelligent agent that acts as a project manager and senior developer to guide the development of machine learning applications.

## Features

- Extended memory system with short-term and long-term storage
- Vector database for documentation and context awareness
- Extensible tool system for development tasks
- Code review and debugging assistance
- Project structure management
- Real-time message processing

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from ai_project_manager import ProjectManager, MLProjectAssistant

# Initialize the project manager
pm = ProjectManager()

# Create ML project assistant
assistant = MLProjectAssistant(pm)

# Start a new project
project_desc = """
Create a machine learning application for sentiment analysis
using transformer models with the following requirements:
- Data preprocessing pipeline
- Model training and evaluation
- API endpoint for predictions
- Monitoring and logging
"""

project_structure = assistant.start_project(project_desc)
```