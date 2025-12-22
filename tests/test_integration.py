"""Integration tests for Q-Route Alpha."""

import pytest
import numpy as np
from pathlib import Path

from q_route.models.problem import Customer, CVRPProblem
from q_route.models.solution import CVRPSolution
from q_route.core.distance_matrix import compute_distance_matrix, get_route_distance
from q_route.core.penalty_calculator import calculate_penalties
from q_route.core.qubo_builder import QUBOBuilder
from q_route.solvers.sa_solver import SimulatedAnnealingSolver


class TestDistanceMatrix:
    """Tests for distance matrix computation."""

    def test_symmetric_matrix(self):
        """Distance matrix should be symmetric."""
        problem = CVRPProblem(
            depot=(0, 0),
            customers=[
                Customer(1, 3, 4, 5),
                Customer(2, 6, 0, 3),
            ],
            vehicle_capacity=10
        )
        D = compute_distance_matrix(problem)
        assert np.allclose(D, D.T), "Distance matrix should be symmetric"

    def test_zero_diagonal(self):
        """Diagonal should be zero (distance to self)."""
        problem = CVRPProblem(
            depot=(0, 0),
            customers=[Customer(1, 3, 4, 5)],
            vehicle_capacity=10
        )
        D = compute_distance_matrix(problem)
        assert np.allclose(np.diag(D), 0), "Diagonal should be zero"

    def test_known_distance(self):
        """Test with known Pythagorean triple (3-4-5)."""
        problem = CVRPProblem(
            depot=(0, 0),
            customers=[Customer(1, 3, 4, 5)],
            vehicle_capacity=10
        )
        D = compute_distance_matrix(problem)
        assert np.isclose(D[0][1], 5.0), "Distance from (0,0) to (3,4) should be 5"


class TestQUBOBuilder:
    """Tests for QUBO construction."""

    def test_variable_count(self):
        """QUBO should have N^2 variables for N customers."""
        problem = CVRPProblem(
            depot=(0, 0),
            customers=[
                Customer(1, 1, 0, 1),
                Customer(2, 0, 1, 1),
                Customer(3, -1, 0, 1),
            ],
            vehicle_capacity=5
        )
        builder = QUBOBuilder(problem)
        assert builder.get_variable_count() == 9  # 3^2 = 9

    def test_bqm_construction(self):
        """BQM should build without errors."""
        problem = CVRPProblem(
            depot=(0, 0),
            customers=[
                Customer(1, 1, 0, 1),
                Customer(2, 0, 1, 1),
            ],
            vehicle_capacity=5
        )
        builder = QUBOBuilder(problem)
        bqm = builder.build_bqm()
        assert bqm is not None
        assert len(bqm.variables) == 4  # 2^2 = 4


class TestSolver:
    """Tests for the simulated annealing solver."""

    @pytest.fixture
    def simple_problem(self):
        """Create a simple 3-customer problem."""
        return CVRPProblem(
            depot=(0, 0),
            customers=[
                Customer(1, 3, 4, 2),
                Customer(2, 6, 0, 2),
                Customer(3, 3, -4, 2),
            ],
            vehicle_capacity=10,
            name="test-3-node"
        )

    def test_solve_returns_solution(self, simple_problem):
        """Solver should return a CVRPSolution."""
        solver = SimulatedAnnealingSolver(num_reads=100, num_sweeps=100)
        solution = solver.solve(simple_problem)
        assert isinstance(solution, CVRPSolution)

    def test_solution_route_structure(self, simple_problem):
        """Route should start and end at depot."""
        solver = SimulatedAnnealingSolver(num_reads=100, num_sweeps=100, seed=42)
        solution = solver.solve(simple_problem)
        assert solution.route[0] == 0, "Route should start at depot"
        assert solution.route[-1] == 0, "Route should end at depot"

    def test_all_customers_visited(self, simple_problem):
        """All customers should be visited exactly once."""
        solver = SimulatedAnnealingSolver(num_reads=500, num_sweeps=500, seed=42)
        solution = solver.solve(simple_problem)

        # Get customer visits (excluding depot)
        visits = [n for n in solution.route if n != 0]

        # Should have exactly n_customers visits
        assert len(visits) == simple_problem.n_customers

        # Each customer visited once
        assert len(set(visits)) == simple_problem.n_customers

    def test_feasible_solution(self, simple_problem):
        """Solution should be feasible with enough reads."""
        solver = SimulatedAnnealingSolver(num_reads=1000, num_sweeps=1000, seed=42)
        solution = solver.solve(simple_problem)
        assert solution.is_feasible, f"Solution should be feasible. Violations: {solution.constraint_violations}"

    def test_positive_distance(self, simple_problem):
        """Total distance should be positive."""
        solver = SimulatedAnnealingSolver(num_reads=100, num_sweeps=100, seed=42)
        solution = solver.solve(simple_problem)
        assert solution.total_distance > 0

    def test_execution_time_recorded(self, simple_problem):
        """Execution time should be recorded."""
        solver = SimulatedAnnealingSolver(num_reads=100, num_sweeps=100)
        solution = solver.solve(simple_problem)
        assert solution.execution_time_seconds > 0


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_5_node_problem(self):
        """Solve the 5-node demo problem."""
        problem = CVRPProblem(
            depot=(0, 0),
            customers=[
                Customer(1, 10, 15, 4),
                Customer(2, -8, 12, 3),
                Customer(3, 5, -10, 5),
                Customer(4, -12, -5, 2),
                Customer(5, 8, 8, 6),
            ],
            vehicle_capacity=20,
            name="test-5-node"
        )

        solver = SimulatedAnnealingSolver(num_reads=1000, seed=42)
        solution = solver.solve(problem)

        # Should find feasible solution
        assert solution.is_feasible, f"Violations: {solution.constraint_violations}"

        # All customers visited
        customers_visited = set(solution.route) - {0}
        assert customers_visited == {1, 2, 3, 4, 5}

        # Reasonable distance (less than worst case)
        D = compute_distance_matrix(problem)
        max_possible = np.sum(D)
        assert solution.total_distance < max_possible

    def test_load_from_json(self, tmp_path):
        """Test loading problem from JSON."""
        # Create a temp JSON file
        json_content = '''{
            "name": "test",
            "depot": {"x": 0, "y": 0},
            "customers": [
                {"id": 1, "x": 5, "y": 0, "demand": 3},
                {"id": 2, "x": 0, "y": 5, "demand": 2}
            ],
            "vehicle_capacity": 10
        }'''

        json_path = tmp_path / "test_problem.json"
        json_path.write_text(json_content)

        # Load and solve
        problem = CVRPProblem.from_json(str(json_path))
        assert problem.n_customers == 2

        solver = SimulatedAnnealingSolver(num_reads=100, seed=42)
        solution = solver.solve(problem)
        assert solution is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
