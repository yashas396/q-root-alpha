# Q-Route Alpha: Algorithm Documentation

## QUBO-Based CVRP Optimization

**Version:** 1.0.0  
**Organization:** Quantum Gandiva AI  
**Purpose:** Technical deep-dive into algorithms used in Q-Route Alpha

---

## 1. Algorithm Selection Rationale

### 1.1 Why Simulated Annealing?

Q-Route Alpha uses **Simulated Annealing (SA)** as the primary optimization algorithm. This choice is based on several key factors:

#### Quantum Compatibility

SA is a classical approximation of **Quantum Annealing**, the optimization paradigm used by D-Wave quantum computers. Both algorithms:

1. Encode problems as energy minimization
2. Use probabilistic state transitions
3. Exploit energy landscape exploration
4. Converge toward global minima

This makes SA the ideal choice for a "quantum-ready" system—the same QUBO formulation works on both.

#### Escape from Local Minima

Unlike greedy algorithms that get trapped in local optima, SA uses **thermal fluctuations** to escape:

```
Acceptance Probability = exp(-ΔE / T)

Where:
  ΔE = Energy difference (new - current)
  T  = Current temperature
```

At high temperatures, bad moves are accepted. As T decreases, the algorithm becomes more selective, eventually converging to the best found solution.

#### Minimal Hyperparameter Tuning

Compared to Genetic Algorithms, SA requires minimal tuning:

| Algorithm | Key Parameters | Tuning Difficulty |
|-----------|----------------|-------------------|
| Simulated Annealing | num_reads, num_sweeps | Low |
| Genetic Algorithm | population_size, mutation_rate, crossover_rate, selection_method | High |
| Particle Swarm | swarm_size, inertia, cognitive/social weights | Medium |

### 1.2 Why QUBO Formulation?

**Quadratic Unconstrained Binary Optimization (QUBO)** is our problem representation for these reasons:

1. **Universal NP-Hard Encoding**: Most NP-hard problems can be expressed as QUBO
2. **Quantum Native**: D-Wave quantum annealers natively solve QUBO/Ising problems
3. **Elegant Constraint Handling**: Constraints become penalty terms in the objective
4. **Symmetric Matrix**: Efficient storage and computation

---

## 2. Mathematical Formulation

### 2.1 The CVRP Problem

**Given:**
- Depot at coordinates (x₀, y₀)
- N customers at coordinates {(x₁, y₁), ..., (xₙ, yₙ)}
- Customer demands {d₁, d₂, ..., dₙ}
- Vehicle capacity Q

**Find:**
- Route visiting all customers exactly once
- Starting and ending at depot
- Minimizing total travel distance
- Respecting capacity constraint: Σ dᵢ ≤ Q

### 2.2 Binary Variable Encoding

We define binary decision variables:

```
x_{i,p} = 1  if customer i is visited at position p
        = 0  otherwise

Where:
  i ∈ {1, 2, ..., N}  (customer index)
  p ∈ {1, 2, ..., N}  (position in route)
```

**Total variables:** N² (e.g., 5 customers → 25 variables)

**Visual representation (3 customers):**

```
           Position
           1    2    3
         +----+----+----+
Cust 1   |x₁₁ |x₁₂ |x₁₃ |
         +----+----+----+
Cust 2   |x₂₁ |x₂₂ |x₂₃ |
         +----+----+----+
Cust 3   |x₃₁ |x₃₂ |x₃₃ |
         +----+----+----+

Valid solution: exactly one "1" per row AND per column
```

### 2.3 Hamiltonian (Energy Function)

The total energy function combines objective and constraints:

```
H_total = H_objective + A × H_visit + B × H_position
```

Where A and B are penalty coefficients.

#### Distance Objective (H_objective)

Minimize total route distance:

```
H_obj = Σₚ₌₁ᴺ⁻¹ Σᵢ₌₁ᴺ Σⱼ₌₁ᴺ D[i,j] × x_{i,p} × x_{j,p+1}
      + Σᵢ₌₁ᴺ D[0,i] × x_{i,1}      // Depot → first customer
      + Σᵢ₌₁ᴺ D[i,0] × x_{i,N}      // Last customer → depot
```

**Interpretation:**
- First term: Distance from customer at position p to customer at position p+1
- Second term: Distance from depot to first customer
- Third term: Distance from last customer back to depot

#### Visit Constraint (H_visit)

Each customer must be visited exactly once:

```
For each customer i: Σₚ x_{i,p} = 1

H_visit = Σᵢ₌₁ᴺ (Σₚ₌₁ᴺ x_{i,p} - 1)²
```

**Expansion using x² = x for binary variables:**

```
(Σₚ x_{i,p} - 1)² = Σₚ x_{i,p} + 2×Σₚ<ₚ' x_{i,p}×x_{i,p'} - 2×Σₚ x_{i,p} + 1
                  = -Σₚ x_{i,p} + 2×Σₚ<ₚ' x_{i,p}×x_{i,p'} + 1
```

**QUBO contribution:**
- Linear terms: -A for each x_{i,p}
- Quadratic terms: +2A for each pair (x_{i,p}, x_{i,p'}) where p ≠ p'

#### Position Constraint (H_position)

Each position must have exactly one customer:

```
For each position p: Σᵢ x_{i,p} = 1

H_position = Σₚ₌₁ᴺ (Σᵢ₌₁ᴺ x_{i,p} - 1)²
```

**QUBO contribution:**
- Linear terms: -B for each x_{i,p}
- Quadratic terms: +2B for each pair (x_{i,p}, x_{j,p}) where i ≠ j

### 2.4 QUBO Matrix Construction

The QUBO matrix Q encodes all terms:

**Diagonal elements (linear terms):**

```
Q[(i,p), (i,p)] = -A - B + D[0,i]×δ(p,1) + D[i,0]×δ(p,N)

Where δ(a,b) = 1 if a=b, else 0
```

**Off-diagonal elements (quadratic terms):**

| Condition | Value | Source |
|-----------|-------|--------|
| Same customer, diff positions | +2A | Visit constraint |
| Same position, diff customers | +2B | Position constraint |
| Consecutive positions (i,p)→(j,p+1) | +D[i,j] | Distance objective |

---

## 3. Penalty Coefficient Calibration

### 3.1 The Balancing Problem

Penalty coefficients must be:
- **Large enough**: Make constraint violations energetically unfavorable
- **Not too large**: Avoid numerical issues and poor annealing dynamics

### 3.2 Theoretical Lower Bound

For a penalty to be effective:

```
λ > max_improvement_from_violation
```

The maximum benefit from violating a constraint is bounded by the maximum distance saved:

```
λ > D_max (maximum distance in problem)
```

### 3.3 Implementation

```python
def calculate_penalties(distance_matrix):
    D_max = np.max(distance_matrix)
    return {
        'A': 2.0 * D_max,  # Visit constraint
        'B': 2.0 * D_max,  # Position constraint
    }
```

**Rationale:** Factor of 2.0 provides safety margin while maintaining good annealing dynamics.

---

## 4. Simulated Annealing Algorithm

### 4.1 Core Algorithm

```
ALGORITHM: Simulated Annealing for QUBO
----------------------------------------
Input: QUBO matrix Q, num_reads, num_sweeps
Output: Best solution x*, minimum energy E*

1. Initialize best_energy = ∞, best_solution = null

2. FOR each read in 1..num_reads:
   a. x = random_binary_vector(N²)
   b. E = compute_energy(x, Q)
   c. T = T_initial
   
   d. FOR each sweep in 1..num_sweeps:
      i.   FOR each variable v in x:
           - x' = flip_bit(x, v)
           - ΔE = compute_delta_energy(x, x', Q)
           - IF ΔE < 0:
               Accept: x = x'
           - ELSE:
               Accept with probability exp(-ΔE / T)
      ii.  T = cooling_schedule(T)
   
   e. IF E < best_energy:
      best_energy = E
      best_solution = x

3. RETURN best_solution, best_energy
```

### 4.2 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_reads` | 1000 | Independent annealing runs |
| `num_sweeps` | 1000 | Iterations per run |
| `beta_range` | auto | Inverse temperature range |
| `seed` | None | Random seed for reproducibility |

### 4.3 Cooling Schedule

D-Wave's sampler uses an adaptive cooling schedule based on the energy landscape. The inverse temperature β = 1/T increases geometrically:

```
β(t) = β_min × (β_max / β_min)^(t / num_sweeps)
```

---

## 5. Solution Decoding

### 5.1 Binary to Route Conversion

After sampling, we receive binary values for all x_{i,p} variables:

```python
sample = {
    'x_1_1': 0, 'x_1_2': 1, 'x_1_3': 0,
    'x_2_1': 0, 'x_2_2': 0, 'x_2_3': 1,
    'x_3_1': 1, 'x_3_2': 0, 'x_3_3': 0
}
```

**Decoding:**
- Position 1 → find i where x_{i,1} = 1 → Customer 3
- Position 2 → find i where x_{i,2} = 1 → Customer 1
- Position 3 → find i where x_{i,3} = 1 → Customer 2

**Route:** [0, 3, 1, 2, 0] (depot → 3 → 1 → 2 → depot)

### 5.2 Decoding Algorithm

```python
def decode_solution(sample, n_customers):
    route = [0]  # Start at depot
    
    for position in range(1, n_customers + 1):
        for customer in range(1, n_customers + 1):
            if sample.get(f"x_{customer}_{position}", 0) == 1:
                route.append(customer)
                break
    
    route.append(0)  # Return to depot
    return route
```

### 5.3 Feasibility Validation

A valid solution must satisfy:
1. Route starts at depot (node 0)
2. Route ends at depot (node 0)
3. All customers visited exactly once
4. No duplicate visits

---

## 6. Computational Complexity

### 6.1 QUBO Size Scaling

| N (Customers) | Variables (N²) | Quadratic Terms | Matrix Size |
|---------------|----------------|-----------------|-------------|
| 5 | 25 | ~180 | 625 |
| 10 | 100 | ~1,800 | 10,000 |
| 15 | 225 | ~6,000 | 50,625 |
| 20 | 400 | ~15,000 | 160,000 |

### 6.2 Time Complexity

- **QUBO Construction:** O(N³) - building all pairwise terms
- **Single Sweep:** O(N²) - updating all variables once
- **Full Solve:** O(num_reads × num_sweeps × N²)

### 6.3 Practical Performance

| Problem Size | Execution Time | Solution Quality |
|--------------|----------------|------------------|
| 5 nodes | < 1 second | Near-optimal |
| 10 nodes | 2-5 seconds | Very good |
| 15 nodes | 10-30 seconds | Good |
| 20+ nodes | 1+ minutes | Requires tuning |

---

## 7. Comparison with Alternatives

### 7.1 Algorithm Comparison

| Method | Optimality | Speed | Scalability | Quantum-Ready |
|--------|------------|-------|-------------|---------------|
| **Simulated Annealing** | Near-optimal | Fast | Good | ✓ Yes |
| Genetic Algorithm | Good | Medium | Good | ✗ No |
| Tabu Search | Good | Fast | Good | ✗ No |
| Branch & Bound | Optimal | Slow | Poor | ✗ No |
| Greedy Nearest Neighbor | Poor | Very Fast | Excellent | ✗ No |

### 7.2 Benchmark Results

For a 10-customer problem (seed=42):

| Method | Distance | Time | Improvement vs Random |
|--------|----------|------|----------------------|
| Random Baseline | 450.2 | - | 0% |
| Nearest Neighbor | 312.5 | 0.001s | 30.6% |
| **Simulated Annealing** | 267.8 | 2.1s | 40.5% |
| OR-Tools (Optimal) | 259.3 | 15.2s | 42.4% |

---

## 8. Future Quantum Enhancement

### 8.1 Quantum Annealing Advantages

When migrated to D-Wave quantum hardware:

1. **Quantum Tunneling**: Can tunnel through energy barriers (vs. thermal hopping in SA)
2. **Parallelism**: Explores superposition of states simultaneously
3. **Speed**: Potentially faster for large-scale problems

### 8.2 Migration Code

```python
# Current: Classical Simulation
from dwave.samplers import SimulatedAnnealingSampler
sampler = SimulatedAnnealingSampler()

# Future: D-Wave Hybrid (Cloud)
from dwave.system import LeapHybridSampler
sampler = LeapHybridSampler()

# Future: Direct QPU
from dwave.system import DWaveSampler, EmbeddingComposite
sampler = EmbeddingComposite(DWaveSampler())
```

The same QUBO works across all backends—only the sampler changes.

---

*Algorithm Documentation - Quantum Gandiva AI - December 2025*
