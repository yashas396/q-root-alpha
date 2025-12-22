"""Penalty coefficient calculation for QUBO construction."""

import numpy as np
from typing import Dict


def calculate_penalties(
    distance_matrix: np.ndarray,
    multiplier: float = 2.0
) -> Dict[str, float]:
    """
    Calculate penalty coefficients for QUBO constraint encoding.

    Penalties must be large enough to make constraint violations
    energetically unfavorable, but not so large as to cause numerical
    issues or poor annealing dynamics.

    Based on empirical validation from Feld et al. (2019):
    - Penalties should be proportional to max distance
    - A = B = 1.5-2.0 * D_max works well in practice

    Args:
        distance_matrix: NxN matrix of distances between nodes
        multiplier: Scaling factor for penalties (default 2.0)

    Returns:
        Dictionary with penalty coefficients:
        - 'A': Visit-once constraint (each customer visited exactly once)
        - 'B': Position-once constraint (each position has exactly one customer)
        - 'C': Capacity constraint (for future multi-vehicle extension)

    Example:
        >>> D = np.array([[0, 10, 15], [10, 0, 12], [15, 12, 0]])
        >>> penalties = calculate_penalties(D)
        >>> penalties['A']  # Should be 2.0 * 15 = 30.0
        30.0
    """
    # Get maximum distance (excluding diagonal zeros)
    D_max = np.max(distance_matrix)

    # Get mean of non-zero distances (useful for diagnostics)
    non_zero = distance_matrix[distance_matrix > 0]
    D_mean = np.mean(non_zero) if len(non_zero) > 0 else D_max

    # Calculate penalties
    # Higher multiplier = stricter constraint enforcement
    A = multiplier * D_max  # Visit-once constraint
    B = multiplier * D_max  # Position-once constraint
    C = (multiplier + 1.0) * D_max  # Capacity constraint (stricter)

    return {
        'A': A,
        'B': B,
        'C': C,
        'D_max': D_max,
        'D_mean': D_mean,
    }


def validate_penalties(
    penalties: Dict[str, float],
    distance_matrix: np.ndarray
) -> Dict[str, bool]:
    """
    Validate that penalty coefficients are appropriately scaled.

    Checks:
    1. Penalties are positive
    2. Penalties exceed max distance (theoretical lower bound)
    3. Penalties are not excessively large (may cause numerical issues)

    Args:
        penalties: Dictionary with A, B, C coefficients
        distance_matrix: The distance matrix for the problem

    Returns:
        Dictionary with validation results
    """
    D_max = np.max(distance_matrix)
    D_sum = np.sum(distance_matrix)

    results = {
        'A_positive': penalties['A'] > 0,
        'B_positive': penalties['B'] > 0,
        'A_exceeds_max': penalties['A'] > D_max,
        'B_exceeds_max': penalties['B'] > D_max,
        'A_not_excessive': penalties['A'] < D_sum,
        'B_not_excessive': penalties['B'] < D_sum,
    }

    results['all_valid'] = all(results.values())
    return results
