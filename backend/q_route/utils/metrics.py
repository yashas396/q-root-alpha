"""Metrics and benchmarking utilities for Q-Route Alpha."""

import random
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

from q_route.core.distance_matrix import compute_distance_matrix, get_route_distance

if TYPE_CHECKING:
    from q_route.models.problem import CVRPProblem


def calculate_route_distance(
    route: List[int],
    problem: "CVRPProblem"
) -> float:
    """
    Calculate total distance of a route.

    Args:
        route: List of node IDs in visit order
        problem: CVRPProblem instance

    Returns:
        Total route distance
    """
    distance_matrix = compute_distance_matrix(problem)
    return get_route_distance(route, distance_matrix)


def random_baseline(
    problem: "CVRPProblem",
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Generate random route baseline for comparison.

    Creates n_samples random routes and calculates statistics.

    Args:
        problem: CVRPProblem instance
        n_samples: Number of random routes to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - mean_distance: Average random route distance
        - std_distance: Standard deviation
        - min_distance: Best random route found
        - max_distance: Worst random route
        - best_route: The best random route found
    """
    if seed is not None:
        random.seed(seed)

    distance_matrix = compute_distance_matrix(problem)
    customer_ids = list(range(1, problem.n_customers + 1))

    distances = []
    best_route = None
    best_distance = float('inf')

    for _ in range(n_samples):
        # Generate random permutation
        shuffled = customer_ids.copy()
        random.shuffle(shuffled)

        # Create route with depot
        route = [0] + shuffled + [0]

        # Calculate distance
        dist = get_route_distance(route, distance_matrix)
        distances.append(dist)

        if dist < best_distance:
            best_distance = dist
            best_route = route

    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'best_route': best_route,
        'n_samples': n_samples,
    }


def nearest_neighbor_baseline(problem: "CVRPProblem") -> Dict[str, float]:
    """
    Generate nearest neighbor heuristic baseline.

    Greedy algorithm: always visit the nearest unvisited customer.

    Args:
        problem: CVRPProblem instance

    Returns:
        Dictionary with:
        - distance: Total route distance
        - route: The route found
    """
    distance_matrix = compute_distance_matrix(problem)
    n = problem.n_customers

    # Start at depot
    current = 0
    route = [0]
    visited = set([0])

    while len(visited) <= n:
        # Find nearest unvisited customer
        best_next = None
        best_dist = float('inf')

        for customer in range(1, n + 1):
            if customer not in visited:
                dist = distance_matrix[current][customer]
                if dist < best_dist:
                    best_dist = dist
                    best_next = customer

        if best_next is None:
            break

        route.append(best_next)
        visited.add(best_next)
        current = best_next

    # Return to depot
    route.append(0)

    total_distance = get_route_distance(route, distance_matrix)

    return {
        'distance': total_distance,
        'route': route,
    }


def calculate_improvement(
    solution_distance: float,
    baseline: Dict[str, float],
    metric: str = 'mean_distance'
) -> float:
    """
    Calculate percentage improvement over baseline.

    Args:
        solution_distance: Distance of the optimized solution
        baseline: Baseline metrics dictionary
        metric: Which baseline metric to compare against

    Returns:
        Improvement percentage (positive = better than baseline)
    """
    baseline_distance = baseline[metric]
    if baseline_distance == 0:
        return 0.0

    improvement = (baseline_distance - solution_distance) / baseline_distance * 100
    return improvement


def solution_quality_report(
    problem: "CVRPProblem",
    solution_distance: float,
    execution_time: float,
    n_random_samples: int = 1000
) -> Dict[str, any]:
    """
    Generate comprehensive quality report for a solution.

    Args:
        problem: CVRPProblem instance
        solution_distance: Distance of the solution
        execution_time: Time to find solution
        n_random_samples: Number of random samples for baseline

    Returns:
        Dictionary with quality metrics
    """
    # Generate baselines
    random_stats = random_baseline(problem, n_samples=n_random_samples)
    nn_stats = nearest_neighbor_baseline(problem)

    # Calculate improvements
    improvement_vs_random_mean = calculate_improvement(
        solution_distance, random_stats, 'mean_distance'
    )
    improvement_vs_random_best = calculate_improvement(
        solution_distance, random_stats, 'min_distance'
    )
    improvement_vs_nn = calculate_improvement(
        solution_distance, {'distance': nn_stats['distance']}, 'distance'
    )

    # Determine quality rating
    if improvement_vs_random_mean >= 20:
        quality_rating = "Excellent"
    elif improvement_vs_random_mean >= 10:
        quality_rating = "Good"
    elif improvement_vs_random_mean >= 0:
        quality_rating = "Fair"
    else:
        quality_rating = "Poor"

    return {
        'solution_distance': solution_distance,
        'execution_time': execution_time,
        'random_baseline': {
            'mean': random_stats['mean_distance'],
            'std': random_stats['std_distance'],
            'best': random_stats['min_distance'],
            'worst': random_stats['max_distance'],
        },
        'nearest_neighbor_distance': nn_stats['distance'],
        'improvement_vs_random_mean': improvement_vs_random_mean,
        'improvement_vs_random_best': improvement_vs_random_best,
        'improvement_vs_nearest_neighbor': improvement_vs_nn,
        'quality_rating': quality_rating,
    }


def format_quality_report(report: Dict[str, any]) -> str:
    """Format quality report as human-readable string."""
    lines = [
        "=" * 60,
        "Solution Quality Report",
        "=" * 60,
        "",
        f"Solution Distance: {report['solution_distance']:.2f}",
        f"Execution Time: {report['execution_time']:.3f}s",
        "",
        "Baselines:",
        f"  Random (mean):     {report['random_baseline']['mean']:.2f}",
        f"  Random (best):     {report['random_baseline']['best']:.2f}",
        f"  Nearest Neighbor:  {report['nearest_neighbor_distance']:.2f}",
        "",
        "Improvements:",
        f"  vs Random Mean:    {report['improvement_vs_random_mean']:+.1f}%",
        f"  vs Random Best:    {report['improvement_vs_random_best']:+.1f}%",
        f"  vs NN Heuristic:   {report['improvement_vs_nearest_neighbor']:+.1f}%",
        "",
        f"Quality Rating: {report['quality_rating']}",
        "=" * 60,
    ]
    return "\n".join(lines)
