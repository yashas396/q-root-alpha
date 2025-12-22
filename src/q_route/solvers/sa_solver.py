"""Simulated Annealing solver for CVRP."""

import time
from typing import Dict, Any, Optional, Tuple

from dwave.samplers import SimulatedAnnealingSampler
import dimod

from q_route.solvers.base_solver import BaseSolver
from q_route.models.problem import CVRPProblem
from q_route.models.solution import CVRPSolution
from q_route.core.qubo_builder import QUBOBuilder
from q_route.core.distance_matrix import get_route_distance, compute_distance_matrix


class SimulatedAnnealingSolver(BaseSolver):
    """
    Simulated Annealing solver for CVRP using D-Wave samplers.

    This solver uses quantum-inspired simulated annealing to solve
    the QUBO formulation of CVRP. It runs locally without requiring
    quantum hardware or D-Wave API access.

    The solver is quantum-ready: the same QUBO can be submitted
    to D-Wave quantum annealers by swapping the sampler.

    Attributes:
        num_reads: Number of independent annealing runs
        num_sweeps: Number of sweeps per annealing run
        beta_range: Inverse temperature range (auto if None)
        seed: Random seed for reproducibility
        penalty_multiplier: Scaling factor for constraint penalties
    """

    def __init__(
        self,
        num_reads: int = 1000,
        num_sweeps: int = 1000,
        beta_range: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
        penalty_multiplier: float = 2.0
    ):
        """
        Initialize the Simulated Annealing solver.

        Args:
            num_reads: Number of annealing runs (more = better quality, slower)
            num_sweeps: Sweeps per run (more = better convergence)
            beta_range: Inverse temperature range (start, end). Auto if None.
            seed: Random seed for reproducibility
            penalty_multiplier: Scaling for constraint penalties (higher = stricter)
        """
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed
        self.penalty_multiplier = penalty_multiplier

        # Initialize the sampler
        self._sampler = SimulatedAnnealingSampler()

    def solve(
        self,
        problem: CVRPProblem,
        **kwargs
    ) -> CVRPSolution:
        """
        Solve the CVRP using simulated annealing.

        Args:
            problem: CVRPProblem instance to solve
            **kwargs: Override default parameters:
                - num_reads: Override number of reads
                - num_sweeps: Override number of sweeps
                - seed: Override random seed

        Returns:
            CVRPSolution with optimal route and metrics
        """
        # Validate problem
        self.validate_problem(problem)

        # Get parameters (kwargs override defaults)
        num_reads = kwargs.get('num_reads', self.num_reads)
        num_sweeps = kwargs.get('num_sweeps', self.num_sweeps)
        seed = kwargs.get('seed', self.seed)
        beta_range = kwargs.get('beta_range', self.beta_range)

        # Start timing
        start_time = time.time()

        # Build QUBO
        builder = QUBOBuilder(problem, penalty_multiplier=self.penalty_multiplier)
        bqm = builder.build_bqm()

        # Prepare sampler parameters
        sampler_params = {
            'num_reads': num_reads,
            'num_sweeps': num_sweeps,
        }
        if seed is not None:
            sampler_params['seed'] = seed
        if beta_range is not None:
            sampler_params['beta_range'] = beta_range

        # Run simulated annealing
        sampleset = self._sampler.sample(bqm, **sampler_params)

        # Get best sample (lowest energy)
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy

        # Count occurrences of best solution
        best_occurrence = 0
        for sample, energy, num_occ in sampleset.data(
            fields=['sample', 'energy', 'num_occurrences']
        ):
            if abs(energy - best_energy) < 1e-10:
                best_occurrence += num_occ

        # Decode solution
        route = builder.decode_solution(best_sample)

        # Validate solution
        validation = builder.validate_solution(route)

        # Calculate route distance
        distance_matrix = compute_distance_matrix(problem)
        total_distance = get_route_distance(route, distance_matrix)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build solution object
        solution = CVRPSolution(
            route=route,
            total_distance=total_distance,
            energy=best_energy,
            is_feasible=validation['is_feasible'],
            constraint_violations=validation['violations'],
            execution_time_seconds=execution_time,
            num_reads=num_reads,
            best_sample_occurrence=best_occurrence,
            solver_info=self.get_solver_info()
        )

        return solution

    def get_solver_info(self) -> Dict[str, Any]:
        """Return solver metadata."""
        return {
            'name': 'SimulatedAnnealingSolver',
            'backend': 'dwave-samplers',
            'version': '1.0.0',
            'parameters': {
                'num_reads': self.num_reads,
                'num_sweeps': self.num_sweeps,
                'beta_range': self.beta_range,
                'seed': self.seed,
                'penalty_multiplier': self.penalty_multiplier,
            },
            'quantum_ready': True,
            'requires_api_key': False,
        }

    def sample_raw(
        self,
        bqm: dimod.BinaryQuadraticModel,
        **kwargs
    ) -> dimod.SampleSet:
        """
        Sample a BQM directly (for advanced usage).

        Args:
            bqm: Binary Quadratic Model to sample
            **kwargs: Sampler parameters

        Returns:
            dimod.SampleSet with all samples
        """
        params = {
            'num_reads': kwargs.get('num_reads', self.num_reads),
            'num_sweeps': kwargs.get('num_sweeps', self.num_sweeps),
        }
        if self.seed is not None:
            params['seed'] = self.seed
        if self.beta_range is not None:
            params['beta_range'] = self.beta_range

        params.update(kwargs)
        return self._sampler.sample(bqm, **params)
