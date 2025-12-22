"""Abstract base class for CVRP solvers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from q_route.models.problem import CVRPProblem
    from q_route.models.solution import CVRPSolution


class BaseSolver(ABC):
    """
    Abstract base class for CVRP solvers.

    All solver implementations should inherit from this class
    and implement the solve() and get_solver_info() methods.

    This abstraction allows swapping between different backends:
    - SimulatedAnnealingSolver (classical, local)
    - HybridSolver (D-Wave Leap cloud)
    - QPUSolver (D-Wave quantum hardware)
    - QAOASolver (IBM Qiskit, future)
    """

    @abstractmethod
    def solve(
        self,
        problem: "CVRPProblem",
        **kwargs
    ) -> "CVRPSolution":
        """
        Solve the CVRP instance.

        Args:
            problem: The CVRP problem to solve
            **kwargs: Solver-specific parameters

        Returns:
            CVRPSolution with route and metrics

        Raises:
            ValueError: If problem is infeasible
            RuntimeError: If solver fails
        """
        pass

    @abstractmethod
    def get_solver_info(self) -> Dict[str, Any]:
        """
        Return solver metadata.

        Returns:
            Dictionary containing:
            - name: Solver name
            - version: Solver version
            - backend: Underlying library/hardware
            - parameters: Current solver parameters
        """
        pass

    def validate_problem(self, problem: "CVRPProblem") -> None:
        """
        Validate that the problem is feasible before solving.

        Args:
            problem: CVRPProblem to validate

        Raises:
            ValueError: If problem is infeasible
        """
        if not problem.validate():
            raise ValueError(
                f"Problem is infeasible: total demand ({problem.total_demand}) "
                f"exceeds vehicle capacity ({problem.vehicle_capacity})"
            )

        if problem.n_customers < 1:
            raise ValueError("Problem must have at least one customer")

        if problem.n_customers > 20:
            # Warn about scalability
            import warnings
            warnings.warn(
                f"Problem has {problem.n_customers} customers. "
                "QUBO size grows as N², which may cause performance issues. "
                "Consider using hybrid decomposition for large instances.",
                UserWarning
            )
