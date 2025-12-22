# Q-Route Alpha: System Design Document

## QUBO-Based CVRP Optimization Architecture

**Version:** 1.0
**Status:** Phase 1 - Architecture
**Organization:** Quantum Gandiva AI

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [QUBO Fundamentals](#2-qubo-fundamentals)
3. [Ising Model Equivalence](#3-ising-model-equivalence)
4. [CVRP to QUBO Transformation](#4-cvrp-to-qubo-transformation)
5. [Mathematical Formulation](#5-mathematical-formulation)
6. [Penalty Coefficient Calibration](#6-penalty-coefficient-calibration)
7. [Solution Decoding](#7-solution-decoding)
8. [Component Architecture](#8-component-architecture)
9. [Data Structures](#9-data-structures)
10. [Algorithm Flow](#10-algorithm-flow)

---

## 1. System Overview

### 1.1 Design Philosophy

The Q-Route Alpha system is designed around a core principle: **express combinatorial optimization as energy minimization**. By reformulating the Capacitated Vehicle Routing Problem (CVRP) as a Quadratic Unconstrained Binary Optimization (QUBO) problem, we create a unified mathematical representation that:

1. Can be solved by classical simulated annealing today
2. Is directly portable to quantum annealing hardware tomorrow
3. Provides a rigorous mathematical framework for constraint handling

### 1.2 Architecture Diagram

```
+-----------------------------------------------------------------------------+
|                         Q-ROUTE ALPHA ARCHITECTURE                           |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +-------------+     +-------------+     +-------------+     +-----------+  |
|  |   INPUT     |---->|   QUBO      |---->|  SAMPLER    |---->|  OUTPUT   |  |
|  |   LAYER     |     |   BUILDER   |     |  INTERFACE  |     |  LAYER    |  |
|  +-------------+     +-------------+     +-------------+     +-----------+  |
|        |                   |                   |                   |        |
|        v                   v                   v                   v        |
|  +-----------+       +-----------+       +-----------+       +---------+   |
|  | Location  |       | Distance  |       | Simulated |       | Route   |   |
|  | Parser    |       | Matrix    |       | Annealing |       | Decoder |   |
|  |           |       | Generator |       | Sampler   |       |         |   |
|  +-----------+       +-----------+       +-----------+       +---------+   |
|  | Demand    |       | Penalty   |       | D-Wave    |       | Route   |   |
|  | Handler   |       | Calculator|       | QPU       |       | Plotter |   |
|  |           |       |           |       | (Future)  |       |         |   |
|  +-----------+       +-----------+       +-----------+       +---------+   |
|  | Capacity  |       | QUBO      |       | Hybrid    |       | Metrics |   |
|  | Validator |       | Compiler  |       | Solver    |       | Engine  |   |
|  +-----------+       +-----------+       +-----------+       +---------+   |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

## 2. QUBO Fundamentals

### 2.1 What is QUBO?

**Quadratic Unconstrained Binary Optimization (QUBO)** is a mathematical formulation for combinatorial optimization problems. The general form is:

```
minimize E(x) = x^T Q x = sum_i(Q_ii * x_i) + sum_{i<j}(Q_ij * x_i * x_j)
```

Where:
- **x** is a vector of binary decision variables: x_i in {0, 1}
- **Q** is the QUBO matrix encoding the problem
- **E(x)** is the "energy" function to minimize

### 2.2 Key Properties

| Property | Description |
|----------|-------------|
| **Binary Variables** | All decision variables are 0 or 1 |
| **Quadratic Terms** | Interactions between pairs of variables |
| **Unconstrained** | Constraints encoded as penalty terms in objective |
| **Symmetric Matrix** | Q_ij = Q_ji (or use upper triangular) |

### 2.3 Why QUBO for Optimization?

1. **Universal Formulation**: Many NP-hard problems can be expressed as QUBO
2. **Physical Analog**: Maps directly to energy minimization in physical systems
3. **Quantum Ready**: Native representation for quantum annealers
4. **Efficient Solvers**: Specialized algorithms exploit QUBO structure

### 2.4 Binary Variable Property

A critical identity for QUBO construction:

```
For binary variables: x^2 = x

If x in {0, 1}:
  - x = 0: x^2 = 0 = x
  - x = 1: x^2 = 1 = x
```

This property simplifies quadratic constraint expansions.

---

## 3. Ising Model Equivalence

### 3.1 The Ising Hamiltonian

The Ising model from statistical physics describes interacting spins:

```
H = sum_i(h_i * sigma_i) + sum_{i<j}(J_ij * sigma_i * sigma_j)
```

Where:
- **sigma_i** in {-1, +1} are spin variables
- **h_i** are local fields (bias terms)
- **J_ij** are coupling strengths between spins

### 3.2 QUBO-Ising Transformation

The transformation between QUBO (binary) and Ising (spin) is:

```
x_i = (1 + sigma_i) / 2

Inverse:
sigma_i = 2*x_i - 1
```

| QUBO (x) | Ising (sigma) |
|----------|---------------|
| 0 | -1 |
| 1 | +1 |

### 3.3 Why This Matters

D-Wave quantum annealers **natively implement the Ising model**. The Ocean SDK handles the QUBO-to-Ising conversion automatically, but understanding this equivalence is crucial for:

- Interpreting hardware constraints
- Understanding qubit coupling topology
- Optimizing problem embedding

---

## 4. CVRP to QUBO Transformation

### 4.1 Problem Parameters

| Symbol | Description | Type |
|--------|-------------|------|
| N | Number of customers (excluding depot) | Integer |
| K | Number of vehicles (K=1 for MVP) | Integer |
| L | Maximum route length (= N for single vehicle) | Integer |
| D_ij | Distance from node i to node j | Float matrix |
| q_i | Demand at customer i | Integer |
| Q | Vehicle capacity | Integer |

### 4.2 Decision Variable Encoding

For single-vehicle CVRP (K=1), we define:

```
x_{i,p} = 1 if customer i is visited at position p
        = 0 otherwise
```

Where:
- i in {1, 2, ..., N} (customer index)
- p in {1, 2, ..., N} (position in route)

**Total Binary Variables:** N^2

| Customers | Variables | QUBO Matrix Size |
|-----------|-----------|------------------|
| 5 | 25 | 25 x 25 = 625 |
| 10 | 100 | 100 x 100 = 10,000 |
| 15 | 225 | 225 x 225 = 50,625 |

### 4.3 Variable Naming Convention

We use a flattened index for implementation:

```python
def var_index(customer: int, position: int, n_customers: int) -> int:
    """Convert (customer, position) to linear index."""
    return (customer - 1) * n_customers + (position - 1)

def var_name(customer: int, position: int) -> str:
    """Generate human-readable variable name."""
    return f"x_{customer}_{position}"
```

### 4.4 Visual Representation

For a 3-customer problem, the variable matrix looks like:

```
           Position
           1    2    3
         +----+----+----+
Cust 1   |x1,1|x1,2|x1,3|
         +----+----+----+
Cust 2   |x2,1|x2,2|x2,3|
         +----+----+----+
Cust 3   |x3,1|x3,2|x3,3|
         +----+----+----+

A valid solution has exactly one "1" in each row and each column.
```

---

## 5. Mathematical Formulation

### 5.1 Complete Hamiltonian Structure

The total energy function (Hamiltonian) combines the objective and constraints:

```
H_total = H_objective + A * H_visit + B * H_position
```

Where A and B are penalty coefficients.

### 5.2 Objective Function: Distance Minimization

**Goal:** Minimize total route distance

```
H_obj = sum_{p=1}^{N-1} sum_{i=1}^{N} sum_{j=1}^{N} D_ij * x_{i,p} * x_{j,p+1}
      + sum_{i=1}^{N} D_{0,i} * x_{i,1}      # Depot to first customer
      + sum_{i=1}^{N} D_{i,0} * x_{i,N}      # Last customer to depot
```

**Explanation:**
- First term: Cost of traveling from customer at position p to customer at position p+1
- Second term: Cost from depot (node 0) to first customer
- Third term: Cost from last customer back to depot

### 5.3 Constraint 1: Each Customer Visited Exactly Once

**Constraint:** Every customer must appear in exactly one position

```
For each customer i: sum_{p=1}^{N} x_{i,p} = 1
```

**QUBO Penalty Form:**

```
H_visit = A * sum_{i=1}^{N} (sum_{p=1}^{N} x_{i,p} - 1)^2
```

**Expanding the square:**

```
(sum_p x_{i,p} - 1)^2 = (sum_p x_{i,p})^2 - 2*(sum_p x_{i,p}) + 1

= sum_p x_{i,p}^2 + 2*sum_{p<p'} x_{i,p}*x_{i,p'} - 2*sum_p x_{i,p} + 1

Using x^2 = x:
= sum_p x_{i,p} + 2*sum_{p<p'} x_{i,p}*x_{i,p'} - 2*sum_p x_{i,p} + 1
= -sum_p x_{i,p} + 2*sum_{p<p'} x_{i,p}*x_{i,p'} + 1
```

**QUBO Contribution:**
- Linear terms: -A for each x_{i,p}
- Quadratic terms: +2A for each pair (x_{i,p}, x_{i,p'}) where p != p'
- Constant: +A*N (can be ignored for optimization)

### 5.4 Constraint 2: Each Position Has Exactly One Customer

**Constraint:** Every position must have exactly one customer

```
For each position p: sum_{i=1}^{N} x_{i,p} = 1
```

**QUBO Penalty Form:**

```
H_position = B * sum_{p=1}^{N} (sum_{i=1}^{N} x_{i,p} - 1)^2
```

**Expansion follows same pattern as H_visit:**

- Linear terms: -B for each x_{i,p}
- Quadratic terms: +2B for each pair (x_{i,p}, x_{j,p}) where i != j
- Constant: +B*N

### 5.5 Capacity Constraint

For single-vehicle MVP where total demand <= capacity, this is validated at input time. For multi-vehicle extensions:

```
H_capacity = C * max(0, sum_{i=1}^{N} q_i * (sum_{p=1}^{L} x_{i,p,k}) - Q_k)^2
```

This requires slack variables for QUBO representation (future phase).

### 5.6 Complete QUBO Matrix Construction

The QUBO matrix Q is constructed as follows:

**Diagonal Elements (Linear Terms):**

```
Q[(i,p), (i,p)] = -A - B + D_{0,i}*delta(p,1) + D_{i,0}*delta(p,N)

Where delta(a,b) = 1 if a=b, else 0
```

**Off-Diagonal Elements (Quadratic Terms):**

| Condition | Value | Source |
|-----------|-------|--------|
| Same customer i, different positions p != p' | +2A | Visit constraint |
| Same position p, different customers i != j | +2B | Position constraint |
| Consecutive positions (i,p) -> (j,p+1) | +D_ij | Distance objective |

### 5.7 Worked Example: 3 Customers

**Setup:**
- Depot at (0, 0)
- Customer 1 at (3, 4) - demand 2
- Customer 2 at (6, 0) - demand 3
- Customer 3 at (3, -4) - demand 2

**Distance Matrix:**

```
     0    1    2    3
0 [  0,   5,   6,   5 ]   # Depot
1 [  5,   0,   5,   8 ]   # Customer 1
2 [  6,   5,   0,   5 ]   # Customer 2
3 [  5,   8,   5,   0 ]   # Customer 3
```

**Variables:** x_1_1, x_1_2, x_1_3, x_2_1, x_2_2, x_2_3, x_3_1, x_3_2, x_3_3

**Optimal Solution:** Route [0, 1, 2, 3, 0] with distance = 5 + 5 + 5 + 5 = 20

---

## 6. Penalty Coefficient Calibration

### 6.1 The Balancing Problem

Penalty coefficients must be:
- **Large enough** to make constraint violations energetically unfavorable
- **Not too large** to avoid numerical issues and poor annealing dynamics

### 6.2 Theoretical Lower Bound

For a penalty to be effective:

```
lambda > max_improvement_from_violation
```

The maximum benefit from violating a constraint is bounded by the maximum distance saved, so:

```
lambda > D_max (maximum distance in problem)
```

### 6.3 Empirical Values from Literature

Based on Feld et al. (2019) and empirical testing:

| Coefficient | Formula | Typical Range |
|-------------|---------|---------------|
| A (visit) | 1.5 - 2.0 * D_max | 15 - 40 |
| B (position) | 1.5 - 2.0 * D_max | 15 - 40 |
| C (capacity) | 2.0 - 5.0 * D_max | 40 - 100 |

### 6.4 Implementation

```python
def calculate_penalties(distance_matrix: np.ndarray) -> dict:
    """
    Calculate penalty coefficients for QUBO construction.

    Args:
        distance_matrix: NxN matrix of distances between nodes

    Returns:
        Dictionary with penalty coefficients A, B, C
    """
    D_max = np.max(distance_matrix)
    D_mean = np.mean(distance_matrix[distance_matrix > 0])

    return {
        'A': 2.0 * D_max,      # Visit-once constraint
        'B': 2.0 * D_max,      # Position-once constraint
        'C': 3.0 * D_max,      # Capacity constraint (future)
    }
```

### 6.5 Tuning Guidelines

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Many constraint violations | Penalties too low | Increase A, B by 1.5x |
| Solutions stuck at high energy | Penalties too high | Decrease A, B by 0.7x |
| Slow convergence | Poor penalty balance | Make A = B |

---

## 7. Solution Decoding

### 7.1 Binary to Route Conversion

After sampling, we receive a dictionary of binary variable values:

```python
sample = {
    'x_1_1': 0, 'x_1_2': 1, 'x_1_3': 0,
    'x_2_1': 0, 'x_2_2': 0, 'x_2_3': 1,
    'x_3_1': 1, 'x_3_2': 0, 'x_3_3': 0
}
```

This encodes: Position 1 -> Customer 3, Position 2 -> Customer 1, Position 3 -> Customer 2

Route: [0, 3, 1, 2, 0]

### 7.2 Decoding Algorithm

```python
def decode_solution(sample: dict, n_customers: int) -> list[int]:
    """
    Convert binary sample to route.

    Args:
        sample: Dictionary mapping variable names to 0/1 values
        n_customers: Number of customers (excluding depot)

    Returns:
        Route as list of node IDs, starting and ending at depot (0)
    """
    route = [0]  # Start at depot

    for position in range(1, n_customers + 1):
        for customer in range(1, n_customers + 1):
            var_name = f"x_{customer}_{position}"
            if sample.get(var_name, 0) == 1:
                route.append(customer)
                break

    route.append(0)  # Return to depot
    return route
```

### 7.3 Feasibility Validation

```python
def validate_solution(route: list[int], n_customers: int) -> dict:
    """
    Validate that a decoded route is feasible.

    Returns:
        Dictionary with validation results and any violations
    """
    violations = []

    # Check starts at depot
    if route[0] != 0:
        violations.append("Route does not start at depot")

    # Check ends at depot
    if route[-1] != 0:
        violations.append("Route does not end at depot")

    # Check all customers visited
    customers_visited = set(route[1:-1])
    expected_customers = set(range(1, n_customers + 1))

    missing = expected_customers - customers_visited
    if missing:
        violations.append(f"Missing customers: {missing}")

    duplicates = [c for c in route[1:-1] if route[1:-1].count(c) > 1]
    if duplicates:
        violations.append(f"Duplicate visits: {set(duplicates)}")

    return {
        'is_feasible': len(violations) == 0,
        'violations': violations,
        'customers_visited': len(customers_visited),
        'expected_customers': n_customers
    }
```

### 7.4 Handling Infeasible Solutions

Simulated annealing may occasionally return infeasible solutions (constraint violations). Strategies:

1. **Rejection**: Discard and use next-best sample
2. **Repair**: Apply greedy repair heuristic
3. **Re-run**: Increase num_reads and re-sample

```python
def repair_route(route: list[int], n_customers: int) -> list[int]:
    """
    Attempt to repair an infeasible route using greedy insertion.
    """
    visited = set(route[1:-1])
    missing = set(range(1, n_customers + 1)) - visited

    # Remove duplicates, keeping first occurrence
    seen = set()
    repaired = [0]
    for node in route[1:-1]:
        if node not in seen:
            seen.add(node)
            repaired.append(node)

    # Insert missing customers at best positions
    for customer in missing:
        best_pos, best_cost = None, float('inf')
        for i in range(1, len(repaired) + 1):
            # Calculate insertion cost
            # ... implementation details
            pass
        repaired.insert(best_pos, customer)

    repaired.append(0)
    return repaired
```

---

## 8. Component Architecture

### 8.1 Module Structure

```
src/q_route/
|-- __init__.py
|-- core/
|   |-- __init__.py
|   |-- qubo_builder.py       # QUBO construction
|   |-- distance_matrix.py    # Distance calculations
|   +-- penalty_calculator.py # Constraint penalties
|-- solvers/
|   |-- __init__.py
|   |-- base_solver.py        # Abstract interface
|   |-- sa_solver.py          # Simulated annealing
|   +-- hybrid_solver.py      # D-Wave hybrid (future)
|-- models/
|   |-- __init__.py
|   |-- problem.py            # CVRP problem definition
|   +-- solution.py           # Solution representation
+-- utils/
    |-- __init__.py
    |-- visualization.py      # Route plotting
    |-- metrics.py            # KPI calculations
    +-- benchmarks.py         # Baseline comparisons
```

### 8.2 Class Diagram

```
+-------------------+       +-------------------+
|   CVRPProblem     |       |   CVRPSolution    |
+-------------------+       +-------------------+
| - depot           |       | - route           |
| - customers       |       | - total_distance  |
| - vehicle_capacity|       | - energy          |
+-------------------+       | - is_feasible     |
| + validate()      |       | - execution_time  |
| + to_dict()       |       +-------------------+
+-------------------+       | + to_dict()       |
         |                  +-------------------+
         |                           ^
         v                           |
+-------------------+       +-------------------+
|   QUBOBuilder     |       |   BaseSolver      |
+-------------------+       +-------------------+
| - distance_matrix |       | + solve()         |
| - penalties       |       | + get_info()      |
+-------------------+       +-------------------+
| + build_bqm()     |                ^
| + get_variable()  |                |
+-------------------+       +-------------------+
                            |    SASolver       |
                            +-------------------+
                            | - num_reads       |
                            | - num_sweeps      |
                            +-------------------+
                            | + solve()         |
                            +-------------------+
```

### 8.3 Solver Interface

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseSolver(ABC):
    """Abstract base class for CVRP solvers."""

    @abstractmethod
    def solve(self, problem: 'CVRPProblem', **kwargs) -> 'CVRPSolution':
        """
        Solve the CVRP instance.

        Args:
            problem: The CVRP problem to solve
            **kwargs: Solver-specific parameters

        Returns:
            CVRPSolution with route and metrics
        """
        pass

    @abstractmethod
    def get_solver_info(self) -> dict[str, Any]:
        """
        Return solver metadata.

        Returns:
            Dictionary with solver name, version, parameters
        """
        pass
```

---

## 9. Data Structures

### 9.1 Problem Definition

```python
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Customer:
    """Represents a customer node in the CVRP."""
    id: int
    x: float
    y: float
    demand: int
    name: str = ""

@dataclass
class CVRPProblem:
    """Complete CVRP problem instance."""
    depot: Tuple[float, float]
    customers: list[Customer]
    vehicle_capacity: int
    name: str = "unnamed"

    @property
    def n_customers(self) -> int:
        return len(self.customers)

    @property
    def total_demand(self) -> int:
        return sum(c.demand for c in self.customers)

    def validate(self) -> bool:
        """Check if problem is feasible."""
        return self.total_demand <= self.vehicle_capacity
```

### 9.2 Solution Representation

```python
@dataclass
class CVRPSolution:
    """Solution to a CVRP instance."""
    route: list[int]
    total_distance: float
    energy: float
    is_feasible: bool
    constraint_violations: list[str]
    execution_time_seconds: float
    num_reads: int
    best_sample_occurrence: int

    def to_dict(self) -> dict:
        """Serialize solution to dictionary."""
        return {
            'route': self.route,
            'total_distance': self.total_distance,
            'energy': self.energy,
            'is_feasible': self.is_feasible,
            'violations': self.constraint_violations,
            'execution_time': self.execution_time_seconds,
            'num_reads': self.num_reads,
            'sample_occurrence': self.best_sample_occurrence
        }
```

### 9.3 QUBO Representation

Using D-Wave's `dimod.BinaryQuadraticModel`:

```python
import dimod

# Create empty BQM
bqm = dimod.BinaryQuadraticModel(vartype='BINARY')

# Add linear bias (diagonal of Q)
bqm.add_variable('x_1_1', -penalty_A - penalty_B)

# Add quadratic interaction (off-diagonal of Q)
bqm.add_interaction('x_1_1', 'x_2_2', distance_1_2)

# Alternative: Build from Q matrix
Q = {
    ('x_1_1', 'x_1_1'): -10,
    ('x_1_1', 'x_2_1'): 5,
    # ...
}
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
```

---

## 10. Algorithm Flow

### 10.1 Main Execution Flow

```
START
  |
  v
[1. Load Problem]
  |-- Parse JSON input
  |-- Create CVRPProblem instance
  +-- Validate feasibility
  |
  v
[2. Build Distance Matrix]
  |-- Calculate pairwise Euclidean distances
  +-- Include depot (node 0)
  |
  v
[3. Calculate Penalties]
  |-- Compute D_max
  |-- Set A = B = 2 * D_max
  +-- (Optional) Tune based on problem size
  |
  v
[4. Construct QUBO]
  |-- Initialize BinaryQuadraticModel
  |-- Add objective terms (distances)
  |-- Add visit constraint penalties
  +-- Add position constraint penalties
  |
  v
[5. Sample with Simulated Annealing]
  |-- Create SimulatedAnnealingSampler
  |-- Run with num_reads samples
  +-- Get SampleSet with energies
  |
  v
[6. Decode Best Solution]
  |-- Extract lowest-energy sample
  |-- Convert binary to route
  +-- Validate feasibility
  |
  v
[7. Compute Metrics]
  |-- Calculate total distance
  |-- Compare to random baseline
  +-- Calculate improvement percentage
  |
  v
[8. Generate Output]
  |-- Create visualization
  |-- Build JSON report
  +-- Return CVRPSolution
  |
  v
END
```

### 10.2 QUBO Building Pseudocode

```python
def build_qubo(problem: CVRPProblem) -> dimod.BinaryQuadraticModel:
    n = problem.n_customers
    D = compute_distance_matrix(problem)
    penalties = calculate_penalties(D)
    A, B = penalties['A'], penalties['B']

    bqm = dimod.BinaryQuadraticModel(vartype='BINARY')

    # Add linear terms
    for i in range(1, n + 1):
        for p in range(1, n + 1):
            var = f"x_{i}_{p}"

            # Constraint penalties (negative = favor selection)
            linear_bias = -A - B

            # Depot connections
            if p == 1:
                linear_bias += D[0][i]  # From depot to first
            if p == n:
                linear_bias += D[i][0]  # From last to depot

            bqm.add_variable(var, linear_bias)

    # Add quadratic terms
    for i in range(1, n + 1):
        for p in range(1, n + 1):
            var_ip = f"x_{i}_{p}"

            # Visit constraint: same customer, different positions
            for p2 in range(p + 1, n + 1):
                var_ip2 = f"x_{i}_{p2}"
                bqm.add_interaction(var_ip, var_ip2, 2 * A)

            # Position constraint: same position, different customers
            for j in range(i + 1, n + 1):
                var_jp = f"x_{j}_{p}"
                bqm.add_interaction(var_ip, var_jp, 2 * B)

            # Distance objective: consecutive positions
            if p < n:
                for j in range(1, n + 1):
                    var_j_next = f"x_{j}_{p+1}"
                    bqm.add_interaction(var_ip, var_j_next, D[i][j])

    return bqm
```

### 10.3 Sampling Pseudocode

```python
from dwave.samplers import SimulatedAnnealingSampler

def solve_qubo(
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    seed: int = None
) -> dimod.SampleSet:
    """
    Solve QUBO using simulated annealing.
    """
    sampler = SimulatedAnnealingSampler()

    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed
    )

    # Return sorted by energy (lowest first)
    return sampleset
```

---

## Summary

This document provides the complete mathematical and architectural foundation for Q-Route Alpha. The QUBO formulation transforms the NP-hard CVRP into an energy minimization problem that:

1. **Today**: Can be solved by classical simulated annealing
2. **Tomorrow**: Can be submitted directly to D-Wave quantum annealers
3. **Always**: Provides a rigorous mathematical framework for constraint satisfaction

The modular architecture supports future extensions including:
- Multi-vehicle routing
- Time window constraints
- Hybrid quantum-classical solvers
- Alternative quantum backends (IBM Qiskit QAOA)

---

*System Design Document - Q-Route Alpha Phase 1: Architecture*
*Prepared for Technical Review*
