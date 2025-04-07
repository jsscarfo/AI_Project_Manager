# AI Project Manager and Developer Coach

An intelligent agent that acts as a project manager and senior developer to guide the development of machine learning applications.

## Features

- Project management and organization
- Code development assistance
- Context-aware code review
- Debugging assistance
- Real-time message processing
- Vector database integration

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- Required Python packages (see requirements.txt)
- Required frontend dependencies (see package.json)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install frontend dependencies:
   ```bash
   npm install
   ```

### Development

1. Start the Python backend:
   ```bash
   python ai_project_manager.py
   ```
2. Start the frontend development server:
   ```bash
   npm run dev
   ```

## Project Structure

```
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
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.