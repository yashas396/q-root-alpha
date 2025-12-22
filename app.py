#!/usr/bin/env python3
"""
Q-Route Alpha - Main Application
=================================

This is the main entry point for the Q-Route Alpha logistics optimization engine.
It orchestrates the complete workflow:

    1. Data Generation - Create a CVRP problem instance
    2. Optimization    - Solve using Simulated Annealing
    3. Visualization   - Plot the optimized route

The application demonstrates quantum-inspired optimization for routing problems.
It uses local simulation only ($0 cost, no D-Wave API required).

Usage:
------
    python app.py

Output:
-------
    - Console output with problem and solution details
    - Route visualization saved as 'route_plot.png'

Author: Quantum Gandiva AI
Version: 1.0.0
Phase: 2 - Core Implementation
Date: December 2025
"""

import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List

from data_gen import generate_problem_instance, print_problem_summary
from solver import solve_cvrp, print_solution_summary, compute_distance_matrix


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_route(
    problem: Dict[str, Any],
    solution: Dict[str, Any],
    save_path: str = "route_plot.png",
    show_plot: bool = False
) -> str:
    """
    Visualize the optimized route on a 2D map.

    Creates a plot showing:
    - Depot (warehouse) as a red square
    - Customer nodes as blue circles
    - Customer demands as labels
    - Optimized route as green arrows

    Visual Design:
    --------------
    - Dark theme for modern appearance
    - Clear distinction between node types
    - Directional arrows show route order
    - Legend for easy interpretation

    Args:
        problem (Dict[str, Any]): The CVRP problem instance.
        solution (Dict[str, Any]): The solution from solve_cvrp().
        save_path (str): Path to save the plot image.
        show_plot (bool): If True, display plot interactively.

    Returns:
        str: Path to the saved plot image.

    Example:
        >>> plot_route(problem, solution, "my_route.png")
        'my_route.png'
    """
    # -------------------------------------------------------------------------
    # Extract data
    # -------------------------------------------------------------------------
    depot = problem["depot"]
    customers = problem["customers"]
    route = solution["route"]
    total_distance = solution["total_distance"]
    is_feasible = solution["is_feasible"]

    # -------------------------------------------------------------------------
    # Create figure with dark theme
    # -------------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set background color
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # -------------------------------------------------------------------------
    # Build coordinate lookup
    # -------------------------------------------------------------------------
    # Node 0 = depot, Node i = customer i
    coords = {0: (depot["x"], depot["y"])}
    demands = {0: 0}
    
    for customer in customers:
        customer_id = customer["id"]
        coords[customer_id] = (customer["x"], customer["y"])
        demands[customer_id] = customer["demand"]

    # -------------------------------------------------------------------------
    # Plot the route (green path with arrows)
    # -------------------------------------------------------------------------
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        
        x1, y1 = coords[from_node]
        x2, y2 = coords[to_node]
        
        # Calculate arrow offset (so arrow doesn't overlap nodes)
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Shorten arrow slightly
            shrink = 3  # pixels to shrink
            dx_norm = dx / length * shrink
            dy_norm = dy / length * shrink
            
            ax.annotate(
                "",
                xy=(x2 - dx_norm, y2 - dy_norm),
                xytext=(x1 + dx_norm, y1 + dy_norm),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#00ff88",
                    lw=2.5,
                    connectionstyle="arc3,rad=0.1"
                )
            )

    # -------------------------------------------------------------------------
    # Plot depot (red square)
    # -------------------------------------------------------------------------
    ax.scatter(
        depot["x"], depot["y"],
        c='#ff4757',
        s=300,
        marker='s',
        edgecolors='white',
        linewidths=2,
        zorder=5,
        label='Depot (Warehouse)'
    )
    
    # Add depot label
    ax.annotate(
        "DEPOT",
        (depot["x"], depot["y"]),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        color='#ff4757'
    )

    # -------------------------------------------------------------------------
    # Plot customers (blue circles with demand labels)
    # -------------------------------------------------------------------------
    customer_x = [c["x"] for c in customers]
    customer_y = [c["y"] for c in customers]
    
    ax.scatter(
        customer_x, customer_y,
        c='#5dade2',
        s=200,
        marker='o',
        edgecolors='white',
        linewidths=1.5,
        zorder=5,
        label='Customers'
    )
    
    # Add customer labels (ID and demand)
    for customer in customers:
        ax.annotate(
            f"C{customer['id']}\n(d={customer['demand']})",
            (customer["x"], customer["y"]),
            xytext=(8, -15),
            textcoords='offset points',
            fontsize=9,
            color='white',
            ha='left'
        )

    # -------------------------------------------------------------------------
    # Add route order numbers
    # -------------------------------------------------------------------------
    for i, node in enumerate(route[1:-1], start=1):  # Skip depot at start/end
        x, y = coords[node]
        ax.annotate(
            str(i),
            (x, y),
            fontsize=10,
            fontweight='bold',
            color='#1a1a2e',
            ha='center',
            va='center',
            zorder=6
        )

    # -------------------------------------------------------------------------
    # Add title and labels
    # -------------------------------------------------------------------------
    status_text = "✓ FEASIBLE" if is_feasible else "✗ INFEASIBLE"
    status_color = "#00ff88" if is_feasible else "#ff4757"
    
    ax.set_title(
        f"Q-Route Alpha - Optimized Delivery Route\n"
        f"Total Distance: {total_distance:.2f} units | {status_text}",
        fontsize=14,
        fontweight='bold',
        color='white',
        pad=20
    )
    
    ax.set_xlabel("X Coordinate", fontsize=12, color='white')
    ax.set_ylabel("Y Coordinate", fontsize=12, color='white')

    # -------------------------------------------------------------------------
    # Add legend
    # -------------------------------------------------------------------------
    legend = ax.legend(
        loc='upper right',
        fontsize=10,
        facecolor='#1a1a2e',
        edgecolor='white',
        labelcolor='white'
    )

    # -------------------------------------------------------------------------
    # Add route summary text box
    # -------------------------------------------------------------------------
    route_str = " → ".join(map(str, route))
    textstr = f"Route: {route_str}\nExecution Time: {solution['execution_time']:.3f}s"
    
    props = dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='white', alpha=0.9)
    ax.text(
        0.02, 0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        fontfamily='monospace',
        color='white',
        bbox=props
    )

    # -------------------------------------------------------------------------
    # Grid and axis settings
    # -------------------------------------------------------------------------
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set axis limits with padding
    all_x = [depot["x"]] + [c["x"] for c in customers]
    all_y = [depot["y"]] + [c["y"] for c in customers]
    padding = 10
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    
    # Equal aspect ratio
    ax.set_aspect('equal')

    # -------------------------------------------------------------------------
    # Save and optionally show
    # -------------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"\n[✓] Route plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    
    return save_path


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application entry point.

    Executes the complete Q-Route Alpha workflow:
    1. Generate problem instance (1 depot, 5 customers)
    2. Solve using QUBO + Simulated Annealing
    3. Visualize the optimized route

    All processing is local - no cloud API calls, no costs.
    """
    print()
    print("=" * 70)
    print("      Q-ROUTE ALPHA - Quantum-Inspired Logistics Optimization")
    print("=" * 70)
    print()
    print("  Phase 2: Core Implementation")
    print("  Solver: Simulated Annealing (Local, $0 Cost)")
    print("  Problem: Capacitated Vehicle Routing (CVRP)")
    print()
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Generate Problem Instance
    # -------------------------------------------------------------------------
    print("STEP 1: Generating Problem Instance")
    print("-" * 50)
    print("  - 1 Depot (Warehouse)")
    print("  - 5 Customers")
    print("  - Seed=42 for reproducibility")
    print()

    problem = generate_problem_instance(
        n_customers=5,
        grid_size=100,
        demand_range=(1, 10),
        vehicle_capacity=50,
        seed=42
    )
    
    print_problem_summary(problem)
    print()

    # -------------------------------------------------------------------------
    # Step 2: Solve with Simulated Annealing
    # -------------------------------------------------------------------------
    print("STEP 2: Solving with Simulated Annealing")
    print("-" * 50)
    print("  - Using dwave-neal SimulatedAnnealingSampler")
    print("  - num_reads=1000 (sample attempts)")
    print("  - num_sweeps=1000 (annealing thoroughness)")
    print("  - Running locally (no cloud, no cost)")
    print()
    print("  Solving... ", end="", flush=True)

    solution = solve_cvrp(
        problem,
        num_reads=1000,
        num_sweeps=1000,
        seed=42
    )
    
    print(f"DONE! ({solution['execution_time']:.3f}s)")
    print()
    
    print_solution_summary(solution)
    print()

    # -------------------------------------------------------------------------
    # Step 3: Visualize Route
    # -------------------------------------------------------------------------
    print("STEP 3: Generating Route Visualization")
    print("-" * 50)
    
    plot_path = plot_route(
        problem,
        solution,
        save_path="route_plot.png",
        show_plot=False
    )
    
    print()

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("                      EXECUTION COMPLETE")
    print("=" * 70)
    print()
    print("  Summary:")
    print(f"    • Route: {' → '.join(map(str, solution['route']))}")
    print(f"    • Distance: {solution['total_distance']:.2f} units")
    print(f"    • Valid: {'Yes ✓' if solution['is_feasible'] else 'No ✗'}")
    print(f"    • Time: {solution['execution_time']:.3f} seconds")
    print(f"    • Plot: {plot_path}")
    print()
    print("  Cost: $0.00 (local simulation only)")
    print()
    print("=" * 70)

    return solution


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
