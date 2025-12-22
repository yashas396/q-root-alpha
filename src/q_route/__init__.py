"""
Q-Route Alpha: Quantum-Ready Logistics Optimizer

A QUBO-based solver for the Capacitated Vehicle Routing Problem (CVRP)
using quantum-inspired simulated annealing.
"""

__version__ = "0.1.0"
__author__ = "Quantum Gandiva AI"

from q_route.models.problem import Customer, CVRPProblem
from q_route.models.solution import CVRPSolution
from q_route.solvers.sa_solver import SimulatedAnnealingSolver

__all__ = [
    "Customer",
    "CVRPProblem",
    "CVRPSolution",
    "SimulatedAnnealingSolver",
    "__version__",
]
