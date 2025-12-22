# Q-Route Alpha - Project Memory

## Project Overview

Q-Route Alpha is a Quantum-Ready Logistics Optimizer that solves the Capacitated Vehicle Routing Problem (CVRP) using QUBO formulation and simulated annealing.

## Tech Stack

### Backend
- **Framework**: FastAPI
- **Quantum Solver**: dwave-samplers (SimulatedAnnealingSampler)
- **Core Libraries**: dimod, numpy
- **Python**: 3.11+

### Frontend
- **Framework**: React 18+ with Vite
- **Styling**: Tailwind CSS
- **Map**: Leaflet (react-leaflet)
- **HTTP Client**: Axios
- **State Management**: React Context + useState

### Infrastructure
- **Containerization**: Docker + docker-compose
- **API Communication**: REST (JSON)

## Architecture

```
QOptimiser/
├── src/q_route/          # Core Python QUBO solver library
├── backend/              # FastAPI REST API
│   └── main.py          # POST /solve endpoint
├── frontend/             # React + Vite application
│   └── src/
│       ├── components/   # UI components
│       ├── pages/        # Route pages
│       └── api/          # API client
└── docker-compose.yml    # Container orchestration
```

## Key Endpoints

- `POST /solve` - Submit CVRP problem, returns optimized route

## Design System

Based on UI/UX Specification v1.0:
- Primary Color: #0043CE (Quantum Blue)
- Success Color: #24A148 (Route Green)
- Warning Color: #FF832B (Energy Orange)
- Error Color: #DA1E28 (Constraint Red)
- Font: IBM Plex Sans

## Development Workflow

- TDD: Write tests first, confirm FAIL, then implement
- Phase-by-phase approval before proceeding

## References

- `docs/PRD.md` - Product Requirements
- `docs/SYSTEM_DESIGN.md` - QUBO Mathematical Formulation
- UI/UX Specification v1.0 - Interface Design Standards
