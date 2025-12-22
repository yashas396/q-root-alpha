#!/usr/bin/env python3
"""
Q-Route Alpha - Data Generator Module
======================================

This module generates synthetic problem instances for the Capacitated Vehicle
Routing Problem (CVRP). It creates a depot (warehouse) and a set of customer
locations with associated demands.

The generated data follows the format specified in the PRD (Section 5.1, FR-001):
- A central depot at defined coordinates
- Customer nodes with unique IDs, (x, y) positions, and integer demands
- A vehicle capacity constraint

Usage:
    from src.data_gen import generate_problem_instance
    problem = generate_problem_instance(n_customers=5, seed=42)
    print(problem)

Author: Quantum Gandiva AI
Version: 1.0.0
Phase: 2 - Core Implementation
"""

import random
from typing import Dict, List, Tuple, Any, Optional


def generate_problem_instance(
    n_customers: int = 5,
    grid_size: int = 100,
    demand_range: Tuple[int, int] = (1, 10),
    vehicle_capacity: int = 50,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a synthetic CVRP problem instance.

    Creates a depot (warehouse) at the center of the grid and randomly places
    customers throughout the grid. Each customer is assigned a random demand
    within the specified range.

    Mathematical Context:
    ---------------------
    The CVRP is defined by:
    - A depot at coordinates (x_0, y_0)
    - N customers at coordinates (x_i, y_i) for i in {1, ..., N}
    - Demands d_i for each customer i
    - Vehicle capacity Q

    The goal is to find a route visiting all customers exactly once,
    starting and ending at the depot, while respecting capacity constraints.

    Args:
        n_customers (int): Number of customer nodes to generate.
            Default is 5 as specified in Phase 2 requirements.
        grid_size (int): Size of the coordinate grid (0 to grid_size).
            Default is 100, giving a 100x100 unit area.
        demand_range (Tuple[int, int]): Min and max demand per customer.
            Default is (1, 10) ensuring non-zero demands.
        vehicle_capacity (int): Maximum capacity of the delivery vehicle.
            Default is 50, sufficient for 5 customers with max demand 10.
        seed (Optional[int]): Random seed for reproducibility.
            If None, results will vary between runs.

    Returns:
        Dict[str, Any]: A problem instance dictionary containing:
            - "depot": {"x": float, "y": float}
            - "customers": List of {"id": int, "x": float, "y": float, "demand": int}
            - "vehicle_capacity": int
            - "metadata": Additional problem information

    Raises:
        ValueError: If n_customers < 1 or if invalid parameter ranges.

    Example:
        >>> problem = generate_problem_instance(n_customers=3, seed=42)
        >>> print(problem["depot"])
        {'x': 50, 'y': 50}
        >>> print(len(problem["customers"]))
        3
    """
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    if n_customers < 1:
        raise ValueError(f"n_customers must be >= 1, got {n_customers}")
    
    if grid_size < 10:
        raise ValueError(f"grid_size must be >= 10, got {grid_size}")
    
    if demand_range[0] < 1 or demand_range[1] < demand_range[0]:
        raise ValueError(f"Invalid demand_range: {demand_range}")
    
    if vehicle_capacity < 1:
        raise ValueError(f"vehicle_capacity must be >= 1, got {vehicle_capacity}")

    # -------------------------------------------------------------------------
    # Set Random Seed (if provided)
    # -------------------------------------------------------------------------
    if seed is not None:
        random.seed(seed)

    # -------------------------------------------------------------------------
    # Generate Depot (Warehouse) - Positioned at Grid Center
    # -------------------------------------------------------------------------
    # The depot is placed at the center of the grid. This is a common
    # configuration in logistics where the warehouse is centrally located.
    depot = {
        "x": grid_size // 2,
        "y": grid_size // 2
    }

    # -------------------------------------------------------------------------
    # Generate Customer Locations
    # -------------------------------------------------------------------------
    # Each customer is assigned:
    #   - A unique ID (1 to n_customers, 0 is reserved for depot)
    #   - Random (x, y) coordinates within the grid
    #   - A random demand within the specified range
    customers: List[Dict[str, Any]] = []
    
    for customer_id in range(1, n_customers + 1):
        customer = {
            "id": customer_id,
            "x": random.randint(0, grid_size),
            "y": random.randint(0, grid_size),
            "demand": random.randint(demand_range[0], demand_range[1])
        }
        customers.append(customer)

    # -------------------------------------------------------------------------
    # Calculate Total Demand (for feasibility check)
    # -------------------------------------------------------------------------
    total_demand = sum(c["demand"] for c in customers)

    # -------------------------------------------------------------------------
    # Construct and Return Problem Instance
    # -------------------------------------------------------------------------
    problem_instance = {
        "depot": depot,
        "customers": customers,
        "vehicle_capacity": vehicle_capacity,
        "metadata": {
            "n_customers": n_customers,
            "grid_size": grid_size,
            "total_demand": total_demand,
            "is_feasible": total_demand <= vehicle_capacity,
            "generator_version": "1.0.0"
        }
    }

    return problem_instance


def print_problem_summary(problem: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of a CVRP problem instance.

    This utility function displays the problem configuration in a formatted
    manner, useful for debugging and verification.

    Args:
        problem (Dict[str, Any]): The problem instance from generate_problem_instance().

    Example:
        >>> problem = generate_problem_instance(n_customers=3, seed=42)
        >>> print_problem_summary(problem)
        ============================================
        Q-ROUTE ALPHA - CVRP PROBLEM INSTANCE
        ============================================
        ...
    """
    print("=" * 60)
    print("Q-ROUTE ALPHA - CVRP PROBLEM INSTANCE")
    print("=" * 60)
    print()
    
    # Depot Information
    depot = problem["depot"]
    print(f"DEPOT (Warehouse):")
    print(f"  Location: ({depot['x']}, {depot['y']})")
    print()
    
    # Customer Information
    print(f"CUSTOMERS ({len(problem['customers'])} total):")
    print("-" * 40)
    print(f"{'ID':<5} {'X':<8} {'Y':<8} {'Demand':<8}")
    print("-" * 40)
    
    for customer in problem["customers"]:
        print(f"{customer['id']:<5} {customer['x']:<8} {customer['y']:<8} {customer['demand']:<8}")
    
    print("-" * 40)
    print()
    
    # Capacity Information
    metadata = problem["metadata"]
    print(f"VEHICLE CAPACITY: {problem['vehicle_capacity']}")
    print(f"TOTAL DEMAND:     {metadata['total_demand']}")
    print(f"FEASIBLE:         {'Yes' if metadata['is_feasible'] else 'No'}")
    print()
    print("=" * 60)


# =============================================================================
# Main Entry Point (for standalone testing)
# =============================================================================
if __name__ == "__main__":
    # Generate a sample problem with 5 customers (as per Phase 2 spec)
    print("Generating sample CVRP problem instance...")
    print()
    
    # Using seed=42 for reproducible results during development
    problem = generate_problem_instance(n_customers=5, seed=42)
    
    # Display the problem
    print_problem_summary(problem)
    
    # Also print raw data for verification
    print("\nRAW DATA (JSON-compatible):")
    print("-" * 40)
    import json
    print(json.dumps(problem, indent=2))
