"""Visualization utilities for Q-Route Alpha."""

from typing import Optional, List, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if TYPE_CHECKING:
    from q_route.models.problem import CVRPProblem
    from q_route.models.solution import CVRPSolution


def plot_problem(
    problem: "CVRPProblem",
    ax: Optional[plt.Axes] = None,
    show_demands: bool = True,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot the CVRP problem instance (nodes only, no route).

    Args:
        problem: CVRPProblem to visualize
        ax: Optional matplotlib axes (creates new figure if None)
        show_demands: Whether to show demand values
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot depot
    ax.scatter(
        [problem.depot[0]], [problem.depot[1]],
        c='red', s=200, marker='s', zorder=5,
        label='Depot', edgecolors='black', linewidths=2
    )
    ax.annotate(
        'Depot',
        (problem.depot[0], problem.depot[1]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=10,
        fontweight='bold'
    )

    # Plot customers
    for customer in problem.customers:
        ax.scatter(
            [customer.x], [customer.y],
            c='blue', s=150, marker='o', zorder=4,
            edgecolors='black', linewidths=1
        )

        label = f"C{customer.id}"
        if show_demands:
            label += f" (d={customer.demand})"

        ax.annotate(
            label,
            (customer.x, customer.y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9
        )

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'CVRP Problem: {problem.name}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def plot_route(
    problem: "CVRPProblem",
    solution: "CVRPSolution",
    ax: Optional[plt.Axes] = None,
    show_demands: bool = True,
    show_arrows: bool = True,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the solution route on a 2D plot.

    Args:
        problem: CVRPProblem instance
        solution: CVRPSolution with route
        ax: Optional matplotlib axes
        show_demands: Whether to show demand values
        show_arrows: Whether to show directional arrows
        figsize: Figure size tuple
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    route = solution.route

    # Get all coordinates for the route
    route_x = []
    route_y = []
    for node_id in route:
        loc = problem.get_node_location(node_id)
        route_x.append(loc[0])
        route_y.append(loc[1])

    # Plot route path
    if show_arrows:
        # Plot with arrows
        for i in range(len(route) - 1):
            dx = route_x[i + 1] - route_x[i]
            dy = route_y[i + 1] - route_y[i]
            ax.annotate(
                '',
                xy=(route_x[i + 1], route_y[i + 1]),
                xytext=(route_x[i], route_y[i]),
                arrowprops=dict(
                    arrowstyle='->',
                    color='green',
                    lw=2,
                    mutation_scale=15
                ),
                zorder=2
            )
    else:
        ax.plot(route_x, route_y, 'g-', linewidth=2, zorder=2, label='Route')

    # Plot depot
    ax.scatter(
        [problem.depot[0]], [problem.depot[1]],
        c='red', s=250, marker='s', zorder=5,
        label='Depot', edgecolors='black', linewidths=2
    )
    ax.annotate(
        'DEPOT',
        (problem.depot[0], problem.depot[1]),
        textcoords="offset points",
        xytext=(12, 12),
        fontsize=11,
        fontweight='bold',
        color='red'
    )

    # Plot customers with visit order
    visit_order = 1
    for node_id in route[1:-1]:  # Skip depot at start and end
        customer = problem.get_customer_by_id(node_id)
        if customer:
            ax.scatter(
                [customer.x], [customer.y],
                c='blue', s=180, marker='o', zorder=4,
                edgecolors='black', linewidths=1.5
            )

            # Label with visit order and demand
            label = f"#{visit_order}: C{customer.id}"
            if show_demands:
                label += f"\n(d={customer.demand})"

            ax.annotate(
                label,
                (customer.x, customer.y),
                textcoords="offset points",
                xytext=(10, -15),
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )
            visit_order += 1

    # Add solution info box
    info_text = (
        f"Total Distance: {solution.total_distance:.2f}\n"
        f"Energy: {solution.energy:.2f}\n"
        f"Feasible: {'Yes' if solution.is_feasible else 'No'}\n"
        f"Time: {solution.execution_time_seconds:.3f}s"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props,
        family='monospace'
    )

    # Formatting
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(
        f'Q-Route Solution: {problem.name}\n'
        f'Route: {" → ".join(map(str, route))}',
        fontsize=12
    )
    ax.grid(True, alpha=0.3)

    # Create legend
    depot_patch = mpatches.Patch(color='red', label='Depot')
    customer_patch = mpatches.Patch(color='blue', label='Customers')
    route_patch = mpatches.Patch(color='green', label='Route')
    ax.legend(handles=[depot_patch, customer_patch, route_patch], loc='upper right')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_comparison(
    problem: "CVRPProblem",
    solutions: List["CVRPSolution"],
    labels: List[str],
    figsize: tuple = (15, 5)
) -> plt.Figure:
    """
    Plot multiple solutions side by side for comparison.

    Args:
        problem: CVRPProblem instance
        solutions: List of CVRPSolution objects
        labels: Labels for each solution
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    n = len(solutions)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, solution, label in zip(axes, solutions, labels):
        plot_route(problem, solution, ax=ax, show_demands=False, figsize=None)
        ax.set_title(f'{label}\nDistance: {solution.total_distance:.2f}')

    plt.tight_layout()
    return fig
