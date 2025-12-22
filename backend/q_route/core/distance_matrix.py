"""Distance matrix computation for CVRP."""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from q_route.models.problem import CVRPProblem


def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        p1: (x, y) coordinates of first point
        p2: (x, y) coordinates of second point

    Returns:
        Euclidean distance between points
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_distance_matrix(problem: "CVRPProblem") -> np.ndarray:
    """
    Compute the Euclidean distance matrix for a CVRP problem.

    The matrix includes the depot as node 0, so for N customers,
    the result is an (N+1) x (N+1) matrix.

    Matrix structure:
        D[0][j] = distance from depot to customer j
        D[i][0] = distance from customer i to depot
        D[i][j] = distance from customer i to customer j

    Args:
        problem: CVRPProblem instance

    Returns:
        numpy array of shape (n_nodes, n_nodes) with distances

    Example:
        >>> problem = CVRPProblem(
        ...     depot=(0, 0),
        ...     customers=[Customer(1, 3, 4, 5), Customer(2, 6, 0, 3)],
        ...     vehicle_capacity=10
        ... )
        >>> D = compute_distance_matrix(problem)
        >>> D.shape
        (3, 3)
        >>> D[0][1]  # Depot to customer 1: sqrt(3^2 + 4^2) = 5.0
        5.0
    """
    n_nodes = problem.n_nodes  # Includes depot

    # Build list of all node locations (depot first)
    locations = [problem.depot]
    for customer in problem.customers:
        locations.append(customer.location)

    # Compute distance matrix
    distance_matrix = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                distance_matrix[i][j] = euclidean_distance(
                    locations[i], locations[j]
                )

    return distance_matrix


def get_route_distance(
    route: list, distance_matrix: np.ndarray
) -> float:
    """
    Calculate total distance of a route.

    Args:
        route: List of node IDs in visit order (should start and end at 0)
        distance_matrix: Precomputed distance matrix

    Returns:
        Total route distance

    Example:
        >>> route = [0, 1, 2, 0]  # Depot -> Cust1 -> Cust2 -> Depot
        >>> get_route_distance(route, D)
        15.0
    """
    total = 0.0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        total += distance_matrix[from_node][to_node]
    return total
