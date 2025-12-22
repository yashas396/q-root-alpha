"""QUBO construction for CVRP."""

import dimod
import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from q_route.core.distance_matrix import compute_distance_matrix
from q_route.core.penalty_calculator import calculate_penalties

if TYPE_CHECKING:
    from q_route.models.problem import CVRPProblem


class QUBOBuilder:
    """
    Builds QUBO (Quadratic Unconstrained Binary Optimization) formulation
    for the Capacitated Vehicle Routing Problem.

    The QUBO encodes the CVRP using binary variables x_{i,p} where:
    - i is the customer index (1 to N)
    - p is the position in the route (1 to N)
    - x_{i,p} = 1 means customer i is visited at position p

    The total Hamiltonian is:
        H_total = H_objective + A * H_visit + B * H_position

    Where:
    - H_objective: Minimizes total route distance
    - H_visit: Each customer visited exactly once
    - H_position: Each position has exactly one customer
    """

    def __init__(
        self,
        problem: "CVRPProblem",
        penalty_multiplier: float = 2.0
    ):
        """
        Initialize QUBO builder.

        Args:
            problem: CVRPProblem instance to solve
            penalty_multiplier: Scaling factor for constraint penalties
        """
        self.problem = problem
        self.n_customers = problem.n_customers
        self.penalty_multiplier = penalty_multiplier

        # Compute distance matrix (includes depot as node 0)
        self.distance_matrix = compute_distance_matrix(problem)

        # Calculate penalty coefficients
        self.penalties = calculate_penalties(
            self.distance_matrix,
            multiplier=penalty_multiplier
        )
        self.A = self.penalties['A']
        self.B = self.penalties['B']

        # Store variable mapping for decoding
        self._var_to_index: Dict[str, Tuple[int, int]] = {}
        self._build_variable_mapping()

    def _build_variable_mapping(self) -> None:
        """Build mapping from variable names to (customer, position) tuples."""
        for i in range(1, self.n_customers + 1):
            for p in range(1, self.n_customers + 1):
                var_name = self._var_name(i, p)
                self._var_to_index[var_name] = (i, p)

    def _var_name(self, customer: int, position: int) -> str:
        """
        Generate variable name for customer at position.

        Args:
            customer: Customer ID (1-indexed)
            position: Position in route (1-indexed)

        Returns:
            Variable name string, e.g., "x_1_2"
        """
        return f"x_{customer}_{position}"

    def build_bqm(self) -> dimod.BinaryQuadraticModel:
        """
        Build the Binary Quadratic Model (QUBO) for the CVRP.

        Returns:
            dimod.BinaryQuadraticModel ready for sampling

        The QUBO includes:
        1. Distance objective (minimize route length)
        2. Visit constraint penalty (each customer once)
        3. Position constraint penalty (each position once)
        """
        bqm = dimod.BinaryQuadraticModel(vartype='BINARY')
        n = self.n_customers
        D = self.distance_matrix
        A = self.A
        B = self.B

        # ========================================
        # Add Linear Terms (Diagonal of Q matrix)
        # ========================================
        for i in range(1, n + 1):
            for p in range(1, n + 1):
                var = self._var_name(i, p)

                # Constraint penalties favor selection (negative bias)
                # From expansion of (sum x - 1)^2:
                #   = sum x^2 + 2*sum_{p<p'} x_p*x_p' - 2*sum x + 1
                #   = sum x + 2*sum_{p<p'} x_p*x_p' - 2*sum x + 1 (since x^2=x)
                #   = -sum x + 2*sum_{p<p'} x_p*x_p' + 1
                # So linear coefficient is -A for visit, -B for position
                linear_bias = -A - B

                # Depot connections (distance objective)
                # First position: add distance from depot to customer i
                if p == 1:
                    linear_bias += D[0][i]
                # Last position: add distance from customer i to depot
                if p == n:
                    linear_bias += D[i][0]

                bqm.add_variable(var, linear_bias)

        # ========================================
        # Add Quadratic Terms (Off-diagonal of Q)
        # ========================================
        for i in range(1, n + 1):
            for p in range(1, n + 1):
                var_ip = self._var_name(i, p)

                # Visit constraint: same customer, different positions
                # Penalty for x_{i,p} and x_{i,p'} both being 1
                for p2 in range(p + 1, n + 1):
                    var_ip2 = self._var_name(i, p2)
                    bqm.add_interaction(var_ip, var_ip2, 2 * A)

                # Position constraint: same position, different customers
                # Penalty for x_{i,p} and x_{j,p} both being 1
                for j in range(i + 1, n + 1):
                    var_jp = self._var_name(j, p)
                    bqm.add_interaction(var_ip, var_jp, 2 * B)

                # Distance objective: consecutive positions
                # Cost for going from customer at position p to customer at p+1
                if p < n:
                    for j in range(1, n + 1):
                        var_j_next = self._var_name(j, p + 1)
                        # Add distance from customer i to customer j
                        bqm.add_interaction(var_ip, var_j_next, D[i][j])

        return bqm

    def decode_solution(self, sample: dict) -> List[int]:
        """
        Convert binary sample to route.

        Args:
            sample: Dictionary mapping variable names to 0/1 values

        Returns:
            Route as list of node IDs, starting and ending at depot (0)

        Example:
            >>> sample = {'x_1_2': 1, 'x_2_1': 1, 'x_3_3': 1, ...}
            >>> route = builder.decode_solution(sample)
            >>> route
            [0, 2, 1, 3, 0]  # Depot -> Cust2 -> Cust1 -> Cust3 -> Depot
        """
        route = [0]  # Start at depot

        n = self.n_customers
        for position in range(1, n + 1):
            customer_at_position = None

            for customer in range(1, n + 1):
                var_name = self._var_name(customer, position)
                if sample.get(var_name, 0) == 1:
                    if customer_at_position is not None:
                        # Multiple customers at same position (constraint violation)
                        # Take the first one found
                        pass
                    else:
                        customer_at_position = customer

            if customer_at_position is not None:
                route.append(customer_at_position)

        route.append(0)  # Return to depot
        return route

    def validate_solution(self, route: List[int]) -> Dict[str, any]:
        """
        Validate that a decoded route satisfies all constraints.

        Args:
            route: Route as list of node IDs

        Returns:
            Dictionary with:
            - is_feasible: bool
            - violations: list of violation descriptions
            - customers_visited: number of unique customers
        """
        violations = []
        n = self.n_customers

        # Check starts at depot
        if not route or route[0] != 0:
            violations.append("Route does not start at depot")

        # Check ends at depot
        if not route or route[-1] != 0:
            violations.append("Route does not end at depot")

        # Get customers visited (excluding depot)
        customers_in_route = [node for node in route if node != 0]
        customers_visited = set(customers_in_route)
        expected_customers = set(range(1, n + 1))

        # Check all customers visited
        missing = expected_customers - customers_visited
        if missing:
            violations.append(f"Missing customers: {sorted(missing)}")

        # Check no duplicate visits
        if len(customers_in_route) != len(customers_visited):
            duplicates = [
                c for c in customers_in_route
                if customers_in_route.count(c) > 1
            ]
            violations.append(f"Duplicate visits: {sorted(set(duplicates))}")

        # Check extra customers (shouldn't happen)
        extra = customers_visited - expected_customers
        if extra:
            violations.append(f"Unknown customers: {sorted(extra)}")

        # Check capacity (for single vehicle, should be pre-validated)
        total_demand = sum(
            self.problem.get_customer_by_id(c).demand
            for c in customers_visited
            if self.problem.get_customer_by_id(c) is not None
        )
        if total_demand > self.problem.vehicle_capacity:
            violations.append(
                f"Capacity exceeded: {total_demand} > {self.problem.vehicle_capacity}"
            )

        return {
            'is_feasible': len(violations) == 0,
            'violations': violations,
            'customers_visited': len(customers_visited),
            'expected_customers': n,
            'total_demand': total_demand,
        }

    def get_variable_count(self) -> int:
        """Return total number of binary variables in the QUBO."""
        return self.n_customers ** 2

    def get_qubo_info(self) -> Dict[str, any]:
        """Return metadata about the QUBO formulation."""
        return {
            'n_customers': self.n_customers,
            'n_variables': self.get_variable_count(),
            'penalty_A': self.A,
            'penalty_B': self.B,
            'D_max': self.penalties['D_max'],
            'D_mean': self.penalties['D_mean'],
            'penalty_multiplier': self.penalty_multiplier,
        }
