# Q-Route Alpha: Project Documentation

## Quantum-Optimized Logistics Routing Engine

**Version:** 1.0.0  
**Organization:** Quantum Gandiva AI  
**Last Updated:** December 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Development Phases](#2-development-phases)
3. [Algorithm Deep Dive](#3-algorithm-deep-dive)
4. [Technology Stack](#4-technology-stack)
5. [Architecture](#5-architecture)
6. [File Structure](#6-file-structure)
7. [API Reference](#7-api-reference)
8. [Deployment](#8-deployment)

---

## 1. Project Overview

### 1.1 Purpose

Q-Route Alpha is a **quantum-ready logistics optimization platform** that solves the Capacitated Vehicle Routing Problem (CVRP). The project demonstrates how quantum computing concepts can be applied to real-world logistics challenges, providing:

- **Cost Optimization**: Minimize total travel distance, reducing fuel costs and CO2 emissions
- **Quantum Readiness**: QUBO formulation compatible with D-Wave quantum annealers
- **Practical Demonstration**: Full-stack application with interactive visualization
- **Zero Cloud Costs**: Local simulation during development phases

### 1.2 Problem Statement

The CVRP is an **NP-hard** combinatorial optimization problem. Given:
- One central depot (warehouse)
- Multiple customers with known locations and demands
- Vehicles with capacity constraints

**Goal:** Find the shortest route visiting all customers exactly once, starting and ending at the depot.

### 1.3 Why This Matters

| Problem Size | Possible Routes | Classical Time |
|--------------|-----------------|----------------|
| 5 customers | 120 | < 1 microsecond |
| 10 customers | 3.6 million | ~4 milliseconds |
| 20 customers | 2.4 × 10^18 | ~76,000 years |

Classical brute-force becomes intractable beyond ~15 customers. Quantum-inspired approaches provide near-optimal solutions in practical timeframes.

---

## 2. Development Phases

### Phase 1: Architecture & Design (Completed)

**Objective:** Define requirements and mathematical foundations

**Deliverables:**
- [PRD.md](file:///d:/QGAI%20RND/QOptimiser/docs/PRD.md) - Product Requirements Document
- [SYSTEM_DESIGN.md](file:///d:/QGAI%20RND/QOptimiser/docs/SYSTEM_DESIGN.md) - Technical specifications

**Key Decisions:**
- QUBO-based approach for quantum portability
- Simulated Annealing for MVP (local, $0 cost)
- Modular architecture for solver extensibility

---

### Phase 2: Core Implementation (Completed)

**Objective:** Build the optimization engine

**Deliverables:**

| File | Purpose |
|------|---------|
| [data_gen.py](file:///d:/QGAI%20RND/QOptimiser/src/data_gen.py) | Generate CVRP problem instances |
| [solver.py](file:///d:/QGAI%20RND/QOptimiser/src/solver.py) | QUBO construction and simulated annealing |
| [app.py](file:///d:/QGAI%20RND/QOptimiser/app.py) | Orchestration and visualization |

**Components Built:**
1. **Distance Matrix Computation** - Euclidean distances between all nodes
2. **QUBO Builder** - Transform CVRP into quadratic optimization
3. **Penalty Calculator** - Calibrate constraint coefficients
4. **Solution Decoder** - Convert binary variables to route
5. **Route Validator** - Verify feasibility
6. **Matplotlib Visualizer** - Generate route plots

---

### Phase 3: Production System (Completed)

**Objective:** Full-stack application with web interface

**Backend Components:**

| File | Purpose |
|------|---------|
| [backend/main.py](file:///d:/QGAI%20RND/QOptimiser/backend/main.py) | FastAPI REST API |
| [src/q_route/](file:///d:/QGAI%20RND/QOptimiser/src/q_route/) | Core package with modular architecture |

**Frontend Components:**

| File | Purpose |
|------|---------|
| [frontend/src/App.jsx](file:///d:/QGAI%20RND/QOptimiser/frontend/src/App.jsx) | Main React application |
| [frontend/src/components/](file:///d:/QGAI%20RND/QOptimiser/frontend/src/components/) | UI components (Map, Form, Results) |

**Infrastructure:**

| File | Purpose |
|------|---------|
| [docker-compose.yml](file:///d:/QGAI%20RND/QOptimiser/docker-compose.yml) | Container orchestration |
| [.github/workflows/ci.yml](file:///d:/QGAI%20RND/QOptimiser/.github/workflows/ci.yml) | CI/CD pipeline |

---

## 3. Algorithm Deep Dive

### 3.1 Why Simulated Annealing?

We chose **Simulated Annealing (SA)** for the following reasons:

| Criterion | Simulated Annealing | Genetic Algorithms | Exact Methods |
|-----------|--------------------|--------------------|---------------|
| **Local Minima Escape** | ✓ Thermal transitions | ✓ Crossover | ✗ Can get stuck |
| **Parameter Tuning** | Minimal | Extensive | None |
| **Quantum Migration** | ✓ Direct QUBO/Ising | ✗ Requires reformulation | ✗ Not compatible |
| **Convergence Guarantee** | Probabilistic | None | ✓ Optimal |
| **Scalability** | Good | Moderate | Poor (exponential) |

**Key Insight:** SA is a classical approximation of quantum annealing dynamics, making it the ideal stepping stone to actual quantum hardware.

### 3.2 QUBO Formulation

**Quadratic Unconstrained Binary Optimization (QUBO):**

```
minimize E(x) = x^T Q x
```

Where:
- **x** = vector of binary decision variables
- **Q** = QUBO matrix encoding the problem

**Variable Encoding:**

```
x_{i,p} = 1  if customer i is visited at position p
        = 0  otherwise
```

For N customers: N² binary variables (e.g., 5 customers = 25 variables)

### 3.3 Hamiltonian Construction

The total energy function combines objective and constraints:

```
H_total = H_objective + A × H_visit + B × H_position
```

**Distance Objective (H_objective):**
- Sum of distances for consecutive visits
- Plus depot-to-first and last-to-depot

**Visit Constraint (H_visit):**
- Each customer visited exactly once
- Penalty: A × (Σ_p x_{i,p} - 1)²

**Position Constraint (H_position):**
- Each position has exactly one customer
- Penalty: B × (Σ_i x_{i,p} - 1)²

### 3.4 Penalty Coefficient Calibration

```python
D_max = max(distance_matrix)
A = B = 2.0 × D_max
```

Coefficients must be:
- **Large enough** to make violations unfavorable
- **Not too large** to avoid numerical issues

### 3.5 Simulated Annealing Process

```
1. Initialize random solution x
2. Set temperature T = T_initial

3. While T > T_final:
   a. Generate neighbor x' by flipping bits
   b. Calculate ΔE = E(x') - E(x)
   c. If ΔE < 0: Accept (better solution)
      Else: Accept with probability e^(-ΔE/T)
   d. Reduce T according to schedule

4. Return best solution found
```

**Parameters:**
- `num_reads`: Independent annealing runs (default: 1000)
- `num_sweeps`: Iterations per run (default: 1000)

---

## 4. Technology Stack

### 4.1 Backend Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Core language |
| **FastAPI** | 0.100+ | REST API framework |
| **dwave-samplers** | 1.0+ | Simulated Annealing sampler |
| **dimod** | 0.12+ | QUBO/BQM construction |
| **NumPy** | 1.24+ | Numerical operations |
| **Pydantic** | 2.0+ | Data validation |

**Why FastAPI?**
- Automatic OpenAPI documentation
- Async support for concurrent requests
- Type hints with Pydantic validation
- High performance (Starlette + Uvicorn)

**Why D-Wave Samplers?**
- Industry-standard for quantum optimization
- Seamless migration to quantum hardware
- Well-maintained, production-ready

### 4.2 Frontend Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.x | UI framework |
| **Vite** | 6.x | Build tool |
| **Tailwind CSS** | 4.x | Styling |
| **Leaflet** | 1.9+ | Map visualization |

**Why React?**
- Component-based architecture
- Large ecosystem and community
- Excellent developer experience

**Why Vite?**
- Lightning-fast HMR (Hot Module Replacement)
- Optimized production builds
- Native ES modules support

**Why Tailwind CSS?**
- Utility-first approach
- Consistent design system
- No CSS file management

**Why Leaflet?**
- Lightweight map library
- Mobile-friendly
- Extensive plugin ecosystem

### 4.3 Infrastructure

| Technology | Purpose |
|------------|---------|
| **Docker** | Containerization |
| **Docker Compose** | Multi-container orchestration |
| **nginx** | Frontend static file serving |
| **GitHub Actions** | CI/CD automation |

---

## 5. Architecture

### 5.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Q-ROUTE ALPHA                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐         ┌──────────────────────────────┐ │
│   │    FRONTEND      │         │          BACKEND             │ │
│   │  (React + Vite)  │ ──API─→ │         (FastAPI)            │ │
│   │                  │         │                              │ │
│   │  ┌────────────┐  │         │  ┌────────────────────────┐  │ │
│   │  │ RouteMap   │  │         │  │   POST /solve          │  │ │
│   │  │ (Leaflet)  │  │         │  │   GET  /health         │  │ │
│   │  └────────────┘  │         │  └────────────────────────┘  │ │
│   │                  │         │              │               │ │
│   │  ┌────────────┐  │         │              ▼               │ │
│   │  │ ProblemForm│  │         │  ┌────────────────────────┐  │ │
│   │  └────────────┘  │         │  │      CORE PACKAGE      │  │ │
│   │                  │         │  │      (q_route)         │  │ │
│   │  ┌────────────┐  │         │  │                        │  │ │
│   │  │ ResultsPane│  │         │  │  ┌──────────────────┐  │  │ │
│   │  └────────────┘  │         │  │  │  QUBOBuilder     │  │  │ │
│   │                  │         │  │  └──────────────────┘  │  │ │
│   └──────────────────┘         │  │  ┌──────────────────┐  │  │ │
│                                │  │  │  SASolver        │  │  │ │
│                                │  │  └──────────────────┘  │  │ │
│                                │  │  ┌──────────────────┐  │  │ │
│                                │  │  │  RouteValidator  │  │  │ │
│                                │  │  └──────────────────┘  │  │ │
│                                │  └────────────────────────┘  │ │
│                                └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow

```
1. User Input (Web UI)
       │
       ▼
2. API Request (POST /solve)
       │
       ▼
3. Problem Validation
       │
       ▼
4. Distance Matrix Computation
       │
       ▼
5. QUBO Construction
   ├── Objective terms (distances)
   ├── Visit constraints
   └── Position constraints
       │
       ▼
6. Simulated Annealing
   ├── 1000 independent runs
   └── Return lowest energy solution
       │
       ▼
7. Solution Decoding
   ├── Binary → Route
   └── Feasibility check
       │
       ▼
8. Response (JSON)
       │
       ▼
9. Visualization (Leaflet Map)
```

---

## 6. File Structure

```
QOptimiser/
├── app.py                    # CLI entry point with visualization
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Package configuration
├── docker-compose.yml       # Container orchestration
│
├── docs/
│   ├── PRD.md               # Product Requirements Document
│   ├── SYSTEM_DESIGN.md     # Technical specifications
│   ├── DEPLOYMENT.md        # Deployment guide
│   └── PROJECT_DOCUMENTATION.md  # This file
│
├── src/
│   ├── data_gen.py          # Problem instance generator
│   ├── solver.py            # Standalone QUBO solver
│   └── q_route/             # Core package
│       ├── __init__.py
│       ├── core/
│       │   ├── qubo_builder.py      # QUBO construction
│       │   ├── distance_matrix.py   # Distance calculations
│       │   └── penalty_calculator.py # Constraint penalties
│       ├── models/
│       │   ├── problem.py           # CVRPProblem dataclass
│       │   └── solution.py          # CVRPSolution dataclass
│       ├── solvers/
│       │   ├── base_solver.py       # Abstract solver interface
│       │   └── sa_solver.py         # Simulated Annealing solver
│       └── utils/
│           ├── visualization.py     # Route plotting
│           └── metrics.py           # Performance metrics
│
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Backend dependencies
│   ├── Dockerfile           # Backend container
│   └── test_main.py         # API tests
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── RouteMap.jsx
│   │   │   ├── ProblemForm.jsx
│   │   │   └── ResultsPanel.jsx
│   │   └── api/
│   │       └── client.js    # API client
│   ├── package.json         # Node dependencies
│   ├── vite.config.js       # Vite configuration
│   ├── Dockerfile           # Frontend container
│   └── nginx.conf           # Production server config
│
├── examples/
│   ├── simple_5_node.json   # Sample 5-customer problem
│   └── medium_10_node.json  # Sample 10-customer problem
│
├── tests/
│   └── test_solver.py       # Core package tests
│
└── .github/
    └── workflows/
        └── ci.yml           # GitHub Actions CI/CD
```

---

## 7. API Reference

### POST /solve

Solve a CVRP problem instance.

**Request Body:**

```json
{
  "depot": {"x": 0, "y": 0},
  "customers": [
    {"id": 1, "x": 10, "y": 15, "demand": 4, "name": "Customer A"},
    {"id": 2, "x": -8, "y": 12, "demand": 3, "name": "Customer B"}
  ],
  "vehicle_capacity": 20,
  "num_reads": 1000,
  "num_sweeps": 1000,
  "seed": null
}
```

**Response:**

```json
{
  "route": [0, 2, 1, 0],
  "total_distance": 45.23,
  "energy": -125.67,
  "is_feasible": true,
  "constraint_violations": [],
  "execution_time_seconds": 0.542,
  "num_reads": 1000,
  "improvement_vs_random": 23.5
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

---

## 8. Deployment

### Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Docker Deployment

```bash
docker-compose up --build
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

### Cloud Deployment

Supported platforms:
- **Railway** - `railway up`
- **Render** - Auto-detects `render.yaml`
- **Fly.io** - `fly deploy`

See [DEPLOYMENT.md](file:///d:/QGAI%20RND/QOptimiser/docs/DEPLOYMENT.md) for detailed instructions.

---

## Appendix: Quantum Migration Path

The system is designed for seamless quantum hardware migration:

```python
# Current: Local Simulation ($0 cost)
from dwave.samplers import SimulatedAnnealingSampler
sampler = SimulatedAnnealingSampler()

# Future: D-Wave Hybrid Cloud
from dwave.system import LeapHybridSampler
sampler = LeapHybridSampler()

# Future: Direct QPU Access
from dwave.system import DWaveSampler, EmbeddingComposite
sampler = EmbeddingComposite(DWaveSampler())
```

The same QUBO formulation works across all backends—only the sampler changes.

---

*Document prepared by Quantum Gandiva AI - December 2025*
