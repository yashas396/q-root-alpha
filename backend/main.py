"""
Q-Route Alpha FastAPI Backend

REST API for quantum-inspired logistics optimization.
POST /solve - Submit CVRP problem, returns optimized route.
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# q_route is now included locally in backend/q_route/

# =============================================================================
# Configuration from Environment
# =============================================================================

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
_cors_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5174,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:3000"
)
# Allow all origins if "*" is set, otherwise use the comma-separated list
CORS_ORIGINS = ["*"] if _cors_env == "*" else _cors_env.split(",")
DEFAULT_NUM_READS = int(os.getenv("DEFAULT_NUM_READS", "1000"))
DEFAULT_NUM_SWEEPS = int(os.getenv("DEFAULT_NUM_SWEEPS", "1000"))
MAX_CUSTOMERS = int(os.getenv("MAX_CUSTOMERS", "50"))

from q_route.models.problem import Customer, CVRPProblem
from q_route.solvers.sa_solver import SimulatedAnnealingSolver

# =============================================================================
# Pydantic Models for API
# =============================================================================


class DepotInput(BaseModel):
    """Depot location input."""
    x: float
    y: float


class CustomerInput(BaseModel):
    """Customer input with location and demand."""
    id: int
    x: float
    y: float
    demand: int
    name: str = ""


class SolveRequest(BaseModel):
    """Request body for /solve endpoint."""
    depot: DepotInput
    customers: List[CustomerInput] = Field(..., min_length=1)
    vehicle_capacity: int = Field(..., gt=0)
    num_reads: int = Field(default=DEFAULT_NUM_READS, ge=100, le=10000)
    num_sweeps: int = Field(default=DEFAULT_NUM_SWEEPS, ge=100, le=5000)
    seed: Optional[int] = None

    @field_validator('customers')
    @classmethod
    def validate_unique_ids(cls, customers: List[CustomerInput]) -> List[CustomerInput]:
        ids = [c.id for c in customers]
        if len(ids) != len(set(ids)):
            raise ValueError("Customer IDs must be unique")
        if len(customers) > MAX_CUSTOMERS:
            raise ValueError(f"Maximum {MAX_CUSTOMERS} customers allowed")
        return customers


class SolveResponse(BaseModel):
    """Response from /solve endpoint."""
    route: List[int]
    total_distance: float
    energy: float
    is_feasible: bool
    constraint_violations: List[str]
    execution_time_seconds: float
    num_reads: int
    improvement_vs_random: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Q-Route Alpha API",
    description="Quantum-inspired logistics optimization API",
    version=APP_VERSION,
)

# CORS Configuration - Allow frontend origins (from environment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=APP_VERSION
    )


@app.post("/solve", response_model=SolveResponse)
async def solve_cvrp(request: SolveRequest):
    """
    Solve a Capacitated Vehicle Routing Problem.

    Takes a CVRP problem definition and returns an optimized route
    using quantum-inspired simulated annealing.
    """
    # Convert API models to domain models
    customers = [
        Customer(
            id=c.id,
            x=c.x,
            y=c.y,
            demand=c.demand,
            name=c.name
        )
        for c in request.customers
    ]

    problem = CVRPProblem(
        depot=(request.depot.x, request.depot.y),
        customers=customers,
        vehicle_capacity=request.vehicle_capacity,
        name="api-request"
    )

    # Validate feasibility
    if not problem.validate():
        raise HTTPException(
            status_code=400,
            detail=f"Problem is infeasible: total demand ({problem.total_demand}) "
                   f"exceeds vehicle capacity ({problem.vehicle_capacity})"
        )

    # Create solver and solve
    solver = SimulatedAnnealingSolver(
        num_reads=request.num_reads,
        num_sweeps=request.num_sweeps,
        seed=request.seed
    )

    solution = solver.solve(problem)

    # Calculate improvement vs random (optional metric)
    improvement = None
    try:
        from q_route.utils.metrics import random_baseline
        baseline = random_baseline(problem, n_samples=100)
        if baseline['mean_distance'] > 0:
            improvement = (
                (baseline['mean_distance'] - solution.total_distance)
                / baseline['mean_distance'] * 100
            )
    except Exception:
        pass  # Non-critical, continue without improvement metric

    return SolveResponse(
        route=solution.route,
        total_distance=solution.total_distance,
        energy=solution.energy,
        is_feasible=solution.is_feasible,
        constraint_violations=solution.constraint_violations,
        execution_time_seconds=solution.execution_time_seconds,
        num_reads=request.num_reads,
        improvement_vs_random=improvement
    )


# =============================================================================
# Run with: uvicorn main:app --reload --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
