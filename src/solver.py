#!/usr/bin/env python3
"""
Q-Route Alpha - QUBO Solver Module
===================================

This module solves the Capacitated Vehicle Routing Problem (CVRP) using
Quadratic Unconstrained Binary Optimization (QUBO) and Simulated Annealing.

The implementation follows the mathematical formulation from SYSTEM_DESIGN.md:
- Transform CVRP into QUBO representation
- Solve using D-Wave's SimulatedAnnealingSampler (local, $0 cost)
- Decode binary solution back to route

Key Concepts:
-------------
1. QUBO: Express optimization as minimizing E(x) = x^T Q x
2. Binary Variables: x_{i,p} = 1 if customer i is at position p in the route
3. Constraints are encoded as penalty terms in the objective function

References:
-----------
- Feld et al. (2019): Hybrid CVRP solution using quantum annealing
- Lucas, A. (2014): Ising formulations of NP problems
- D-Wave Ocean SDK Documentation

Author: Quantum Gandiva AI
Version: 1.0.0
Phase: 2 - Core Implementation
"""

import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import dimod
from neal import SimulatedAnnealingSampler


# =============================================================================
# DISTANCE MATRIX COMPUTATION
# =============================================================================

def compute_distance_matrix(problem: Dict[str, Any]) -> np.ndarray:
    """
    Compute the Euclidean distance matrix for all nodes (depot + customers).

    The distance matrix D is an (N+1) x (N+1) symmetric matrix where:
    - D[0][j] = distance from depot to customer j
    - D[i][0] = distance from customer i to depot
    - D[i][j] = distance from customer i to customer j

    Mathematical Formula:
    --------------------
    D[i][j] = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)

    Args:
        problem (Dict[str, Any]): The CVRP problem instance containing:
            - "depot": {"x": float, "y": float}
            - "customers": List of {"id": int, "x": float, "y": float, "demand": int}

    Returns:
        np.ndarray: Distance matrix of shape (n_nodes, n_nodes) where
            n_nodes = n_customers + 1 (including depot as node 0).

    Example:
        >>> problem = {"depot": {"x": 0, "y": 0}, "customers": [{"id": 1, "x": 3, "y": 4, "demand": 1}]}
        >>> D = compute_distance_matrix(problem)
        >>> D[0][1]  # Distance from depot to customer 1
        5.0
    """
    # -------------------------------------------------------------------------
    # Extract all node coordinates (depot = index 0, customers = index 1..N)
    # -------------------------------------------------------------------------
    depot = problem["depot"]
    customers = problem["customers"]
    n_customers = len(customers)
    n_nodes = n_customers + 1  # Include depot

    # Create coordinate array: [(x_0, y_0), (x_1, y_1), ...]
    coords = [(depot["x"], depot["y"])]
    for customer in customers:
        coords.append((customer["x"], customer["y"]))
    
    coords = np.array(coords, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Compute pairwise Euclidean distances
    # -------------------------------------------------------------------------
    # Using vectorized computation for efficiency
    distance_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)

    return distance_matrix


# =============================================================================
# PENALTY COEFFICIENT CALCULATION
# =============================================================================

def calculate_penalty_coefficients(distance_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate penalty coefficients for QUBO constraint terms.

    The penalty coefficients must be large enough to make constraint violations
    energetically unfavorable, but not so large that they dominate the landscape
    and prevent effective annealing.

    Theoretical Basis (from SYSTEM_DESIGN.md Section 6):
    ----------------------------------------------------
    - Penalty must exceed maximum possible benefit from violation
    - Lower bound: lambda > D_max (maximum distance in problem)
    - Empirical range: 1.5 to 2.0 * D_max

    Args:
        distance_matrix (np.ndarray): The (N+1) x (N+1) distance matrix.

    Returns:
        Dict[str, float]: Dictionary containing:
            - "A": Penalty for visit-once constraint
            - "B": Penalty for position-once constraint
            - "D_max": Maximum distance in the problem

    Notes:
        - We set A = B for balanced constraint enforcement
        - The factor of 2.0 provides a good safety margin
    """
    # -------------------------------------------------------------------------
    # Compute maximum distance (excluding diagonal zeros)
    # -------------------------------------------------------------------------
    D_max = np.max(distance_matrix)
    
    # -------------------------------------------------------------------------
    # Set penalty coefficients
    # -------------------------------------------------------------------------
    # Using 2.0 * D_max as per literature recommendations
    # This ensures constraint violations are always energetically unfavorable
    penalty_A = 2.0 * D_max  # Visit-once constraint
    penalty_B = 2.0 * D_max  # Position-once constraint

    return {
        "A": penalty_A,
        "B": penalty_B,
        "D_max": D_max
    }


# =============================================================================
# QUBO CONSTRUCTION
# =============================================================================

def build_qubo(
    problem: Dict[str, Any],
    distance_matrix: np.ndarray,
    penalties: Dict[str, float]
) -> dimod.BinaryQuadraticModel:
    """
    Construct the QUBO (Binary Quadratic Model) for the CVRP.

    This function builds the complete Hamiltonian as described in
    SYSTEM_DESIGN.md Section 5:

        H_total = H_objective + A * H_visit + B * H_position

    Where:
    - H_objective: Minimizes total route distance
    - H_visit: Ensures each customer is visited exactly once
    - H_position: Ensures each route position has exactly one customer

    Variable Encoding:
    ------------------
    x_{i,p} = 1 if customer i is visited at position p
            = 0 otherwise

    Where:
        - i in {1, ..., N} (customer indices)
        - p in {1, ..., N} (position indices)

    Args:
        problem (Dict[str, Any]): The CVRP problem instance.
        distance_matrix (np.ndarray): Precomputed distance matrix.
        penalties (Dict[str, float]): Penalty coefficients from calculate_penalty_coefficients().

    Returns:
        dimod.BinaryQuadraticModel: The complete QUBO model ready for sampling.

    Mathematical Details:
    --------------------
    1. Distance Objective:
       Sum over consecutive positions: D[i,j] * x_{i,p} * x_{j,p+1}
       Plus depot-to-first and last-to-depot terms

    2. Visit Constraint (for each customer i):
       A * (sum_p x_{i,p} - 1)^2
       Expands to: -A*x_{i,p} + 2A*x_{i,p}*x_{i,p'} for p != p'

    3. Position Constraint (for each position p):
       B * (sum_i x_{i,p} - 1)^2
       Expands to: -B*x_{i,p} + 2B*x_{i,p}*x_{j,p} for i != j
    """
    n_customers = len(problem["customers"])
    A = penalties["A"]  # Visit penalty
    B = penalties["B"]  # Position penalty
    D = distance_matrix

    # -------------------------------------------------------------------------
    # Initialize Binary Quadratic Model (QUBO)
    # -------------------------------------------------------------------------
    bqm = dimod.BinaryQuadraticModel(vartype='BINARY')

    # -------------------------------------------------------------------------
    # Helper function: Generate variable name
    # -------------------------------------------------------------------------
    def var_name(customer: int, position: int) -> str:
        """Generate variable name x_{customer}_{position}."""
        return f"x_{customer}_{position}"

    # -------------------------------------------------------------------------
    # 1. Add Linear Terms (Diagonal of Q matrix)
    # -------------------------------------------------------------------------
    # Each variable x_{i,p} gets:
    # - (-A - B) from constraint penalty expansion
    # - D[0,i] if p=1 (depot to first customer)
    # - D[i,0] if p=N (last customer to depot)

    for i in range(1, n_customers + 1):  # Customer indices
        for p in range(1, n_customers + 1):  # Position indices
            var = var_name(i, p)
            
            # Base linear term from constraint penalties
            # (sum x - 1)^2 expands to: x^2 - 2x + 1 = x - 2x + 1 = -x + 1 (using x^2=x)
            linear_bias = -A - B
            
            # Add depot connection costs
            if p == 1:  # First position: add depot-to-customer distance
                linear_bias += D[0, i]
            if p == n_customers:  # Last position: add customer-to-depot distance
                linear_bias += D[i, 0]
            
            bqm.add_variable(var, linear_bias)

    # -------------------------------------------------------------------------
    # 2. Add Quadratic Terms for VISIT Constraint
    # -------------------------------------------------------------------------
    # For each customer i, penalize having multiple positions
    # 2A * x_{i,p} * x_{i,p'} for all p != p'

    for i in range(1, n_customers + 1):
        for p1 in range(1, n_customers + 1):
            for p2 in range(p1 + 1, n_customers + 1):
                var1 = var_name(i, p1)
                var2 = var_name(i, p2)
                # Add penalty for same customer in different positions
                bqm.add_interaction(var1, var2, 2 * A)

    # -------------------------------------------------------------------------
    # 3. Add Quadratic Terms for POSITION Constraint
    # -------------------------------------------------------------------------
    # For each position p, penalize having multiple customers
    # 2B * x_{i,p} * x_{j,p} for all i != j

    for p in range(1, n_customers + 1):
        for i1 in range(1, n_customers + 1):
            for i2 in range(i1 + 1, n_customers + 1):
                var1 = var_name(i1, p)
                var2 = var_name(i2, p)
                # Add penalty for different customers at same position
                bqm.add_interaction(var1, var2, 2 * B)

    # -------------------------------------------------------------------------
    # 4. Add Quadratic Terms for Distance Objective
    # -------------------------------------------------------------------------
    # Cost of going from customer i at position p to customer j at position p+1
    # D[i,j] * x_{i,p} * x_{j,p+1}

    for p in range(1, n_customers):  # Positions 1 to N-1
        for i in range(1, n_customers + 1):
            for j in range(1, n_customers + 1):
                if i != j:  # Different customers
                    var1 = var_name(i, p)       # Customer i at position p
                    var2 = var_name(j, p + 1)   # Customer j at position p+1
                    # Add distance cost
                    bqm.add_interaction(var1, var2, D[i, j])

    return bqm


# =============================================================================
# SOLUTION DECODING
# =============================================================================

def decode_solution(sample: Dict[str, int], n_customers: int) -> List[int]:
    """
    Convert a binary sample from the QUBO solver into a route.

    The sample contains binary values for all x_{i,p} variables.
    We decode this by finding which customer is assigned to each position.

    Decoding Algorithm:
    ------------------
    For each position p = 1 to N:
        Find customer i where x_{i,p} = 1
        Add customer i to route

    Args:
        sample (Dict[str, int]): Dictionary mapping variable names to 0/1 values.
        n_customers (int): Number of customers in the problem.

    Returns:
        List[int]: Route as list of node IDs.
            - Starts with depot (0)
            - Contains customer IDs in visit order
            - Ends with depot (0)

    Example:
        >>> sample = {"x_1_2": 1, "x_2_1": 1, "x_3_3": 1, ...}
        >>> route = decode_solution(sample, 3)
        >>> route
        [0, 2, 1, 3, 0]  # Visit customer 2 first, then 1, then 3
    """
    route = [0]  # Start at depot

    for position in range(1, n_customers + 1):
        customer_found = None
        for customer in range(1, n_customers + 1):
            var_name = f"x_{customer}_{position}"
            if sample.get(var_name, 0) == 1:
                customer_found = customer
                break
        
        if customer_found is not None:
            route.append(customer_found)
        # If no customer found for this position, the solution is infeasible

    route.append(0)  # Return to depot
    return route


def validate_solution(route: List[int], n_customers: int) -> Dict[str, Any]:
    """
    Validate that a decoded route is a feasible CVRP solution.

    Feasibility Criteria:
    --------------------
    1. Route starts at depot (node 0)
    2. Route ends at depot (node 0)
    3. All customers are visited exactly once
    4. No duplicate customer visits

    Args:
        route (List[int]): The route to validate (including depot at start/end).
        n_customers (int): Expected number of customers.

    Returns:
        Dict[str, Any]: Validation results containing:
            - "is_feasible": True if all criteria met
            - "violations": List of violation descriptions
            - "customers_visited": Number of unique customers in route

    Example:
        >>> result = validate_solution([0, 1, 2, 3, 0], 3)
        >>> result["is_feasible"]
        True
    """
    violations = []

    # -------------------------------------------------------------------------
    # Check 1: Route starts at depot
    # -------------------------------------------------------------------------
    if len(route) == 0 or route[0] != 0:
        violations.append("Route does not start at depot (node 0)")

    # -------------------------------------------------------------------------
    # Check 2: Route ends at depot
    # -------------------------------------------------------------------------
    if len(route) == 0 or route[-1] != 0:
        violations.append("Route does not end at depot (node 0)")

    # -------------------------------------------------------------------------
    # Check 3: All customers visited exactly once
    # -------------------------------------------------------------------------
    customer_portion = route[1:-1] if len(route) > 2 else []
    customers_visited = set(customer_portion)
    expected_customers = set(range(1, n_customers + 1))

    # Check for missing customers
    missing = expected_customers - customers_visited
    if missing:
        violations.append(f"Missing customers: {sorted(missing)}")

    # Check for extra/invalid customers
    extra = customers_visited - expected_customers
    if extra:
        violations.append(f"Invalid customer IDs: {sorted(extra)}")

    # -------------------------------------------------------------------------
    # Check 4: No duplicate visits
    # -------------------------------------------------------------------------
    if len(customer_portion) != len(customers_visited):
        duplicates = [c for c in customer_portion if customer_portion.count(c) > 1]
        violations.append(f"Duplicate visits: {sorted(set(duplicates))}")

    return {
        "is_feasible": len(violations) == 0,
        "violations": violations,
        "customers_visited": len(customers_visited),
        "expected_customers": n_customers
    }


def calculate_route_distance(
    route: List[int],
    distance_matrix: np.ndarray
) -> float:
    """
    Calculate the total distance of a route.

    Sums all consecutive node-to-node distances along the route.

    Args:
        route (List[int]): Ordered list of node IDs to visit.
        distance_matrix (np.ndarray): Precomputed distance matrix.

    Returns:
        float: Total route distance.

    Example:
        >>> route = [0, 1, 2, 0]  # Depot -> Customer 1 -> Customer 2 -> Depot
        >>> distance = calculate_route_distance(route, D)
    """
    total_distance = 0.0
    
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        total_distance += distance_matrix[from_node, to_node]
    
    return total_distance


# =============================================================================
# MAIN SOLVER FUNCTION
# =============================================================================

def solve_cvrp(
    problem: Dict[str, Any],
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Solve a CVRP instance using QUBO and Simulated Annealing.

    This is the main entry point for the solver. It:
    1. Builds the distance matrix
    2. Calculates penalty coefficients
    3. Constructs the QUBO
    4. Solves using SimulatedAnnealingSampler
    5. Decodes and validates the solution

    Solver Details:
    ---------------
    - Uses D-Wave's neal.SimulatedAnnealingSampler
    - Runs entirely locally (no cloud API, $0 cost)
    - Multiple reads (num_reads) to find better solutions
    - num_sweeps controls annealing thoroughness

    Args:
        problem (Dict[str, Any]): CVRP problem instance from data_gen.
        num_reads (int): Number of independent annealing runs.
            Higher = better solutions but slower. Default: 1000.
        num_sweeps (int): Number of sweeps per read.
            Higher = more thorough search. Default: 1000.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        Dict[str, Any]: Solution containing:
            - "route": List[int] - The optimized route
            - "total_distance": float - Total route distance
            - "is_feasible": bool - Whether solution satisfies constraints
            - "violations": List[str] - Any constraint violations
            - "energy": float - QUBO energy of solution
            - "num_reads": int - Number of samples taken
            - "execution_time": float - Time in seconds
            - "sample_occurrence": int - How many times best solution was found

    Example:
        >>> problem = generate_problem_instance(n_customers=5)
        >>> solution = solve_cvrp(problem)
        >>> print(solution["route"])
        [0, 3, 1, 4, 2, 5, 0]
    """
    # -------------------------------------------------------------------------
    # Record start time
    # -------------------------------------------------------------------------
    start_time = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Compute distance matrix
    # -------------------------------------------------------------------------
    n_customers = len(problem["customers"])
    distance_matrix = compute_distance_matrix(problem)

    # -------------------------------------------------------------------------
    # Step 2: Calculate penalty coefficients
    # -------------------------------------------------------------------------
    penalties = calculate_penalty_coefficients(distance_matrix)

    # -------------------------------------------------------------------------
    # Step 3: Build QUBO
    # -------------------------------------------------------------------------
    bqm = build_qubo(problem, distance_matrix, penalties)

    # -------------------------------------------------------------------------
    # Step 4: Run Simulated Annealing
    # -------------------------------------------------------------------------
    # Using D-Wave's SimulatedAnnealingSampler (local simulation, $0 cost)
    sampler = SimulatedAnnealingSampler()
    
    # Sample the QUBO
    sample_set = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed
    )

    # -------------------------------------------------------------------------
    # Step 5: Get best solution (lowest energy)
    # -------------------------------------------------------------------------
    best_sample = sample_set.first.sample
    best_energy = sample_set.first.energy
    
    # Count how many times the best solution was found
    best_occurrence = sample_set.first.num_occurrences

    # -------------------------------------------------------------------------
    # Step 6: Decode binary solution to route
    # -------------------------------------------------------------------------
    route = decode_solution(best_sample, n_customers)

    # -------------------------------------------------------------------------
    # Step 7: Validate the solution
    # -------------------------------------------------------------------------
    validation = validate_solution(route, n_customers)

    # -------------------------------------------------------------------------
    # Step 8: Calculate total distance
    # -------------------------------------------------------------------------
    total_distance = calculate_route_distance(route, distance_matrix)

    # -------------------------------------------------------------------------
    # Record execution time
    # -------------------------------------------------------------------------
    execution_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # Construct and return solution
    # -------------------------------------------------------------------------
    solution = {
        "route": route,
        "total_distance": total_distance,
        "is_feasible": validation["is_feasible"],
        "violations": validation["violations"],
        "energy": best_energy,
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "execution_time": execution_time,
        "sample_occurrence": best_occurrence,
        "penalties": penalties,
        "n_variables": len(bqm.variables),
        "n_interactions": len(bqm.quadratic)
    }

    return solution


def print_solution_summary(solution: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of a CVRP solution.

    Args:
        solution (Dict[str, Any]): Solution from solve_cvrp().
    """
    print("=" * 60)
    print("Q-ROUTE ALPHA - SOLUTION SUMMARY")
    print("=" * 60)
    print()
    
    # Route
    route_str = " -> ".join(map(str, solution["route"]))
    print(f"OPTIMAL ROUTE: {route_str}")
    print()
    
    # Metrics
    print("METRICS:")
    print("-" * 40)
    print(f"  Total Distance:    {solution['total_distance']:.2f} units")
    print(f"  QUBO Energy:       {solution['energy']:.2f}")
    print(f"  Feasible:          {'Yes' if solution['is_feasible'] else 'No'}")
    print(f"  Execution Time:    {solution['execution_time']:.3f} seconds")
    print()
    
    # Solver details
    print("SOLVER CONFIGURATION:")
    print("-" * 40)
    print(f"  Num Reads:         {solution['num_reads']}")
    print(f"  Num Sweeps:        {solution['num_sweeps']}")
    print(f"  Best Occurrence:   {solution['sample_occurrence']}")
    print(f"  QUBO Variables:    {solution['n_variables']}")
    print(f"  QUBO Interactions: {solution['n_interactions']}")
    print()
    
    # Violations (if any)
    if solution["violations"]:
        print("CONSTRAINT VIOLATIONS:")
        print("-" * 40)
        for violation in solution["violations"]:
            print(f"  - {violation}")
        print()
    
    print("=" * 60)


# =============================================================================
# Main Entry Point (for standalone testing)
# =============================================================================
if __name__ == "__main__":
    # Test the solver with a simple problem
    from data_gen import generate_problem_instance, print_problem_summary
    
    print("=" * 60)
    print("Q-ROUTE ALPHA - SOLVER TEST")
    print("=" * 60)
    print()
    
    # Generate test problem
    print("Generating test problem (5 customers)...")
    problem = generate_problem_instance(n_customers=5, seed=42)
    print_problem_summary(problem)
    
    # Solve
    print("\nSolving with Simulated Annealing...")
    print("(Using local simulation - $0 cost)")
    print()
    
    solution = solve_cvrp(problem, num_reads=1000, seed=42)
    
    # Display results
    print_solution_summary(solution)
