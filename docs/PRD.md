# Q-Route Alpha: Product Requirements Document

## Quantum-Optimized Logistics Routing Engine

**Version:** 1.0
**Status:** Phase 1 - Architecture
**Organization:** Quantum Gandiva AI

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [Technology Stack](#4-technology-stack)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Success Metrics](#7-success-metrics)
8. [Scope Definition](#8-scope-definition)
9. [Risk Assessment](#9-risk-assessment)

---

## 1. Executive Summary

Q-Route Alpha is a simulation-based logistics optimization engine that solves the **Capacitated Vehicle Routing Problem (CVRP)** using **Quantum-Inspired Simulated Annealing**. This prototype demonstrates the capability to transition logistics optimization from classical heuristics to quantum-ready mathematical formulations.

### Strategic Objectives

| Objective | Description |
|-----------|-------------|
| **Quantum Readiness** | Validate QUBO formulation compatibility with D-Wave architecture |
| **Cost Optimization** | Minimize total travel distance and fuel consumption |
| **Technical Validation** | Prove simulated annealing approach on logistics problems |
| **Vendor Neutrality** | Architecture compatible with D-Wave, IBM Qiskit, and classical solvers |

---

## 2. Problem Statement

### 2.1 The Combinatorial Explosion Challenge

The Capacitated Vehicle Routing Problem (CVRP) is **NP-hard** (Lenstra & Rinnooy Kan, 1981). The solution space grows **factorially**, making brute-force approaches intractable beyond trivial problem sizes.

#### Computational Complexity Growth

| Customers (n) | Possible Routes | Brute Force Time @ 10^9/sec |
|---------------|-----------------|------------------------------|
| 5 | 120 | < 1 microsecond |
| 10 | 3,628,800 | ~4 milliseconds |
| 15 | 1.3 x 10^12 | ~15 days |
| 20 | 2.4 x 10^18 | ~76,000 years |
| 25 | 1.5 x 10^25 | ~480 million years |

**Critical Insight:** Classical exhaustive search becomes computationally infeasible beyond ~15 nodes. This is precisely where quantum and quantum-inspired approaches provide value through energy landscape optimization rather than exhaustive enumeration.

### 2.2 The CVRP Definition

The Capacitated Vehicle Routing Problem is defined as finding optimal routes for a fleet of vehicles to service a set of customers, where:

- Each customer has a known demand
- Each vehicle has a maximum capacity
- All routes start and end at a central depot
- Each customer is visited exactly once
- Total demand on each route cannot exceed vehicle capacity

### 2.3 Limitations of Classical Approaches

| Approach | Limitation |
|----------|------------|
| **Greedy Algorithms** | Quick but highly suboptimal solutions |
| **Local Search (2-opt, 3-opt)** | Trapped in local minima |
| **Genetic Algorithms** | Require careful parameter tuning, no optimality guarantee |
| **Branch and Bound** | Exponential worst-case complexity |

### 2.4 Why Quantum-Inspired Optimization?

**Quantum Annealing Advantages:**
- Natural mapping to energy minimization problems
- Potential for quantum tunneling through energy barriers (on real hardware)
- Parallelism in exploring solution space

**Simulated Annealing (Our MVP Approach):**
- Classical approximation of quantum annealing dynamics
- Thermal fluctuations allow escape from local minima
- Well-understood convergence properties
- Zero quantum hardware cost for prototyping
- Seamless migration path to actual quantum hardware

---

## 3. Solution Overview

### 3.1 Core Approach

Transform the CVRP into a **Quadratic Unconstrained Binary Optimization (QUBO)** problem, then solve using simulated annealing. The QUBO formulation is directly portable to quantum annealing hardware.

### 3.2 High-Level Data Flow

```
Problem Input (JSON)
       |
       v
+------------------+
| Parse Locations  |  --> [(x1,y1), (x2,y2), ...]
| Parse Demands    |  --> [d1, d2, ...]
| Parse Capacity   |  --> Q
+------------------+
       |
       v
+------------------+
| Build Distance   |  --> D[i][j] = ||loc_i - loc_j||
| Matrix           |
+------------------+
       |
       v
+------------------+
| Compute Penalty  |  --> lambda_1, lambda_2, lambda_3
| Coefficients     |
+------------------+
       |
       v
+------------------+
| Build QUBO       |  --> Q matrix (n^2 elements)
| H = H_obj + Sum  |
+------------------+
       |
       v
+------------------+
| Simulated        |  --> SampleSet with energies
| Annealing        |
+------------------+
       |
       v
+------------------+
| Decode Solution  |  --> Route: [0, 3, 1, 4, 2, 5, 0]
| Validate         |
+------------------+
       |
       v
+------------------+
| Compute Metrics  |  --> {distance, energy, time, gap}
| Generate Report  |
+------------------+
```

---

## 4. Technology Stack

### 4.1 Runtime Environment

| Component | Specification | Rationale |
|-----------|---------------|-----------|
| Python | 3.11+ | Latest stable, performance improvements |
| OS | Windows / Linux / macOS | Cross-platform development |
| Architecture | x86_64 / ARM64 | M1/M2 Mac compatibility |

### 4.2 Core Dependencies

```toml
[project]
name = "q-route-alpha"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "dwave-ocean-sdk>=6.0.0",      # Primary D-Wave integration
    "dwave-samplers>=1.0.0",       # Simulated annealing sampler
    "dimod>=0.12.0",               # QUBO/BQM construction
    "numpy>=1.24.0",               # Numerical operations
    "matplotlib>=3.7.0",           # Visualization
    "networkx>=3.0",               # Graph operations
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

### 4.3 D-Wave SDK Components

| Package | Purpose | Version |
|---------|---------|---------|
| `dwave-ocean-sdk` | Meta-package for all D-Wave tools | >= 6.0.0 |
| `dwave-samplers` | Local classical samplers (SA, Tabu) | >= 1.0.0 |
| `dimod` | Binary quadratic model representation | >= 0.12.0 |
| `dwave-system` | QPU access (future phases) | >= 1.20.0 |

**Important Note:** `dwave-neal` is deprecated as of dwave-ocean-sdk 6.1.0. Use `dwave-samplers` instead:

```python
# Old (deprecated)
from neal import SimulatedAnnealingSampler

# New (recommended)
from dwave.samplers import SimulatedAnnealingSampler
```

### 4.4 Quantum Hardware Migration Path

| Level | Description | Implementation |
|-------|-------------|----------------|
| QRL-1 | Problem formulated as QUBO | Current phase |
| QRL-2 | Classical simulation validated | MVP target |
| QRL-3 | D-Wave cloud access configured | Phase 3 |
| QRL-4 | Hybrid solver integration | Phase 3 |
| QRL-5 | Direct QPU submission | Future |

```python
# Level 2: Local Simulation (MVP)
from dwave.samplers import SimulatedAnnealingSampler
sampler = SimulatedAnnealingSampler()

# Level 3: D-Wave Hybrid Cloud (Future)
from dwave.system import LeapHybridSampler
sampler = LeapHybridSampler()

# Level 4: Direct QPU (Future)
from dwave.system import DWaveSampler, EmbeddingComposite
sampler = EmbeddingComposite(DWaveSampler())
```

---

## 5. Functional Requirements

### 5.1 Core Requirements (P0 - Critical)

#### FR-001: Problem Instance Definition
- Accept CVRP problem instances via JSON with depot, customer locations, demands, and vehicle capacity
- Validate feasibility (total demand <= capacity)
- Support 1-15 customer nodes

**Input Format:**
```json
{
  "depot": {"x": 0, "y": 0},
  "customers": [
    {"id": 1, "x": 10, "y": 15, "demand": 5},
    {"id": 2, "x": -8, "y": 12, "demand": 3}
  ],
  "vehicle_capacity": 20
}
```

#### FR-002: QUBO Construction
- Construct mathematically correct QUBO formulation
- Distance objective correctly encoded
- All constraint penalties included
- Penalty coefficients properly scaled
- QUBO matrix is symmetric

#### FR-003: Simulated Annealing Execution
- Use `dwave-samplers` SimulatedAnnealingSampler
- Complete within 10 seconds for <= 10 nodes
- Return lowest energy sample
- Support configurable `num_reads` (default: 1000)

#### FR-004: Route Extraction
- Decode binary solution into valid route
- Route starts and ends at depot
- All customers visited exactly once
- No capacity violation
- Handle infeasible solutions gracefully

### 5.2 Extended Requirements (P1 - High)

#### FR-005: Visualization
- 2D plot showing all nodes
- Depot clearly distinguished
- Route path with directional arrows
- Customer demands labeled
- Total distance displayed

#### FR-006: Benchmark Comparison
- Generate random route baseline
- Generate nearest-neighbor heuristic distance
- Calculate improvement percentages

#### FR-007: Solution Quality Metrics
- Total route distance
- QUBO energy value
- Constraint satisfaction status
- Execution time
- Optimality gap (vs. known optimal if available)

---

## 6. Non-Functional Requirements

### NFR-001: Performance
| Metric | Target |
|--------|--------|
| 5-node solve time | < 2 seconds |
| 10-node solve time | < 10 seconds |
| Memory usage (15 nodes) | < 1 GB |

### NFR-002: Portability
- Run on Python 3.10+
- No D-Wave API key required for MVP
- Single command execution

### NFR-003: Maintainability
- Modular component architecture
- Comprehensive docstrings
- Type hints throughout codebase
- Test coverage > 80%

### NFR-004: Extensibility
- Abstract solver interface for multiple backends
- Plugin architecture for new samplers
- Configuration-driven penalty tuning

---

## 7. Success Metrics

### 7.1 Technical KPIs

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Optimality Gap** | < 5% vs OR-Tools | Benchmark comparison |
| **Execution Time (5 nodes)** | < 2 seconds | Wall clock time |
| **Execution Time (10 nodes)** | < 10 seconds | Wall clock time |
| **Solution Feasibility** | 100% | Constraint validation |
| **Test Coverage** | > 80% | pytest-cov |

### 7.2 Business KPIs

| Metric | Target | Calculation |
|--------|--------|-------------|
| **Distance Improvement** | > 15% vs random | (random - SA) / random |
| **CO2 Reduction Potential** | 10-15% | Distance x emission factor |
| **Compute Cost (MVP)** | $0.00 | No cloud API charges |
| **Quantum Portability** | 100% | Sampler swap test |
| **Demo Success Rate** | > 90% | Executive demo pass rate |

### 7.3 Validation Criteria

**Solution Quality Validation:**
- Compare against OR-Tools optimal solutions for small instances
- Verify constraint satisfaction (all customers visited once)
- Confirm capacity constraints respected

**QUBO Validation:**
- Hand-calculate QUBO for 3-node problem
- Verify matrix symmetry
- Test penalty coefficient scaling

---

## 8. Scope Definition

### 8.1 In Scope (MVP)

- Single-depot CVRP with 5-15 customer nodes
- Single vehicle with capacity constraints
- Euclidean distance metric
- Local simulation using `dwave-samplers`
- Visual route output and optimization metrics
- JSON input/output
- CLI interface

### 8.2 Out of Scope (Future Phases)

| Feature | Phase |
|---------|-------|
| Multi-depot scenarios (MDVRP) | Phase 2 |
| Time window constraints (CVRPTW) | Phase 2 |
| Real-time fleet management | Phase 3 |
| Integration with live traffic APIs | Phase 3 |
| Multi-vehicle simultaneous routing | Phase 2 |
| D-Wave Leap cloud integration | Phase 3 |
| Web-based dashboard | Phase 4 |

---

## 9. Risk Assessment

### 9.1 Risk Matrix

| Risk ID | Description | Probability | Impact | Mitigation |
|---------|-------------|-------------|--------|------------|
| R-001 | QUBO formulation error | Medium | High | Unit tests, hand calculation verification |
| R-002 | Poor solution quality | Medium | High | Parameter tuning, multiple restarts |
| R-003 | Scalability limitations | High | Medium | Hybrid decomposition approach |
| R-004 | D-Wave API deprecation | Low | Medium | Abstraction layer, vendor-neutral design |
| R-005 | Performance regression | Medium | Medium | Continuous benchmarking |

### 9.2 Technical Clarifications

| Original Claim | Validation Status | Clarification |
|----------------|-------------------|---------------|
| "Quantum Annealers excel at CVRP" | Partially Accurate | D-Wave handles ~6-7 customers directly; larger requires hybrid decomposition |
| "Simulator tunnels through peaks" | Simplified | SA uses thermal transitions, not quantum tunneling |
| "< 2 seconds for 5-10 nodes" | Achievable | Validated for simulated annealing |
| "100% Quantum Portability" | Optimistic | Sampler swap is simple; embedding/tuning differs significantly |

---

## References

1. **Feld et al.** (2019). A Hybrid Solution Method for the Capacitated Vehicle Routing Problem Using a Quantum Annealer. *Frontiers in ICT*
2. **Lucas, A.** (2014). Ising formulations of many NP problems. *Frontiers in Physics*
3. **Lenstra & Rinnooy Kan** (1981). Complexity of Vehicle Routing and Scheduling Problems. *Networks*
4. **D-Wave Systems** (2024). D-Wave Ocean SDK Documentation

---

*Document prepared for Q-Route Alpha Phase 1: Architecture Review*
