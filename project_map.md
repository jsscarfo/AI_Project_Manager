## Project Structure Overview

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

## Core Files

### Root Directory

| File | Purpose | Key Functions | Usage |
|------|---------|---------------|-------|
| `__init__.py` | Package initialization | - | Imported when using the package |
| `ai_project_manager.py` | Core system implementation | `ProjectManager()`, `MLProjectAssistant()` | Main entry point for the AI system |
| `requirements.txt` | Dependencies list | - | Used for installing Python dependencies |
| `package.json` | Frontend dependencies | - | Used for installing frontend dependencies |
| `README.md` | Project documentation | - | Overview of the project |

### Frontend Structure

| File | Purpose | Key Functions | Usage |
|------|---------|---------------|-------|
| `src/App.tsx` | Main React component | - | Root component for the application |
| `src/main.tsx` | Frontend entry point | - | Initializes the React application |
| `index.html` | HTML entry point | - | Base HTML structure |

### Configuration Files

| File | Purpose | Key Functions | Usage |
|------|---------|---------------|-------|
| `tsconfig.json` | TypeScript configuration | - | Configures TypeScript compiler |
| `vite.config.ts` | Vite configuration | - | Configures the build tool |
| `tailwind.config.js` | Tailwind CSS configuration | - | Configures CSS framework |
| `postcss.config.js` | PostCSS configuration | - | Configures CSS processing |