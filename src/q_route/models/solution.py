"""CVRP Solution representation."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class CVRPSolution:
    """
    Solution to a CVRP instance.

    Attributes:
        route: Ordered list of node IDs (starts and ends with 0 for depot)
        total_distance: Total route distance
        energy: QUBO energy value of the solution
        is_feasible: Whether all constraints are satisfied
        constraint_violations: List of violation descriptions
        execution_time_seconds: Time taken to find solution
        num_reads: Number of annealing reads performed
        best_sample_occurrence: How many times best solution was found
        solver_info: Metadata about the solver used
    """

    route: List[int]
    total_distance: float
    energy: float
    is_feasible: bool
    constraint_violations: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    num_reads: int = 0
    best_sample_occurrence: int = 0
    solver_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_customers_visited(self) -> int:
        """Number of unique customers visited (excluding depot)."""
        return len(set(self.route)) - 1  # Subtract 1 for depot

    @property
    def route_length(self) -> int:
        """Total number of stops including depot visits."""
        return len(self.route)

    def get_customer_sequence(self) -> List[int]:
        """Return customer IDs in visit order (excluding depot)."""
        return [node for node in self.route if node != 0]

    def to_dict(self) -> dict:
        """Serialize solution to dictionary."""
        return {
            "route": self.route,
            "total_distance": self.total_distance,
            "energy": self.energy,
            "is_feasible": self.is_feasible,
            "constraint_violations": self.constraint_violations,
            "execution_time_seconds": self.execution_time_seconds,
            "num_reads": self.num_reads,
            "best_sample_occurrence": self.best_sample_occurrence,
            "solver_info": self.solver_info,
        }

    def to_json(self, path: str) -> None:
        """Save solution to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "CVRPSolution":
        """Create CVRPSolution from dictionary."""
        return cls(
            route=data["route"],
            total_distance=data["total_distance"],
            energy=data["energy"],
            is_feasible=data["is_feasible"],
            constraint_violations=data.get("constraint_violations", []),
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
            num_reads=data.get("num_reads", 0),
            best_sample_occurrence=data.get("best_sample_occurrence", 0),
            solver_info=data.get("solver_info", {}),
        )

    def __repr__(self) -> str:
        feasible_str = "feasible" if self.is_feasible else "INFEASIBLE"
        return (
            f"CVRPSolution(route={self.route}, "
            f"distance={self.total_distance:.2f}, "
            f"{feasible_str})"
        )

    def format_report(self) -> str:
        """Generate a human-readable report of the solution."""
        lines = [
            "=" * 50,
            "Q-Route Alpha Solution Report",
            "=" * 50,
            "",
            f"Route: {' -> '.join(map(str, self.route))}",
            f"Total Distance: {self.total_distance:.2f}",
            f"QUBO Energy: {self.energy:.4f}",
            "",
            f"Feasible: {'Yes' if self.is_feasible else 'No'}",
        ]

        if self.constraint_violations:
            lines.append("Violations:")
            for v in self.constraint_violations:
                lines.append(f"  - {v}")

        lines.extend([
            "",
            "Performance:",
            f"  Execution Time: {self.execution_time_seconds:.3f}s",
            f"  Num Reads: {self.num_reads}",
            f"  Best Sample Occurrence: {self.best_sample_occurrence}",
        ])

        if self.solver_info:
            lines.append("")
            lines.append("Solver Info:")
            for key, value in self.solver_info.items():
                lines.append(f"  {key}: {value}")

        lines.append("=" * 50)
        return "\n".join(lines)
