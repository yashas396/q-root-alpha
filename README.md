# Q-Route Alpha

[![CI](https://github.com/yashas396/q-root-alpha/actions/workflows/ci.yml/badge.svg)](https://github.com/yashas396/q-root-alpha/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Quantum-Ready Logistics Optimizer**

A full-stack logistics optimization platform that solves the Capacitated Vehicle Routing Problem (CVRP) using Quantum-Inspired Simulated Annealing with QUBO formulation. Features a React dashboard with real-time route visualization.

## Features

- **QUBO-based CVRP Solver** - Quantum-ready optimization using D-Wave's simulated annealing
- **React Dashboard** - Interactive web UI with Leaflet map visualization
- **FastAPI Backend** - REST API for optimization requests
- **Real-time Visualization** - See optimized routes on an interactive map
- **Performance Metrics** - Compare against random baseline solutions
- **Docker Ready** - One-command deployment with Docker Compose
- **CI/CD Pipeline** - GitHub Actions for testing and deployment

## Installation

```bash
# Clone the repository
cd QOptimiser

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### CLI Usage

```bash
# Solve a 5-node problem
q-route solve examples/simple_5_node.json

# With visualization
q-route solve examples/simple_5_node.json --visualize

# Custom parameters
q-route solve examples/simple_5_node.json --num-reads 2000 --num-sweeps 1500
```

### Web UI (Docker)

```bash
# Start both frontend and backend with Docker Compose
docker-compose up --build

# Access the dashboard at http://localhost:3000
# API available at http://localhost:8000
```

### Development Mode

```bash
# Terminal 1: Start backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## Python API

```python
from q_route.models import CVRPProblem, Customer
from q_route.solvers import SimulatedAnnealingSolver

# Define problem
problem = CVRPProblem(
    depot=(0, 0),
    customers=[
        Customer(id=1, x=10, y=15, demand=4),
        Customer(id=2, x=-8, y=12, demand=3),
        Customer(id=3, x=5, y=-10, demand=5),
    ],
    vehicle_capacity=20
)

# Solve
solver = SimulatedAnnealingSolver(num_reads=1000)
solution = solver.solve(problem)

print(f"Route: {solution.route}")
print(f"Distance: {solution.total_distance:.2f}")
```

## Architecture

The system transforms CVRP into a Quadratic Unconstrained Binary Optimization (QUBO) problem:

```
H_total = H_objective + A * H_visit + B * H_position
```

Where:
- `H_objective`: Minimizes total route distance
- `H_visit`: Ensures each customer visited exactly once
- `H_position`: Ensures each route position has exactly one customer

## Quantum Migration Path

```python
# Current: Local Simulation
from dwave.samplers import SimulatedAnnealingSampler

# Future: D-Wave Quantum Hardware
from dwave.system import LeapHybridSampler
# or
from dwave.system import DWaveSampler, EmbeddingComposite
```

## Cloud Deployment

Deploy to your preferred cloud platform:

### Railway (Recommended)
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### Render
- Connect GitHub repo at https://render.com
- Render will auto-detect `render.yaml` and deploy both services

### Fly.io
```bash
fly launch
fly deploy
```

See [Deployment Guide](docs/DEPLOYMENT.md) for detailed instructions.

## Testing

```bash
# Backend tests (14 tests)
cd backend && python -m pytest test_main.py -v

# Frontend tests (51 tests)
cd frontend && npm test

# Core package tests (13 tests)
python -m pytest tests/ -v
```

## Documentation

- [Product Requirements](docs/PRD.md)
- [System Design](docs/SYSTEM_DESIGN.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI, Python 3.11, D-Wave Samplers |
| **Frontend** | React 19, Vite, Tailwind CSS, Leaflet |
| **Infrastructure** | Docker, nginx, GitHub Actions |

## License

MIT License - Quantum Gandiva AI
