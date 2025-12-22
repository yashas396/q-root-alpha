"""
TDD Tests for Q-Route Alpha FastAPI Backend

These tests are written FIRST before implementation.
Run with: pytest backend/test_main.py -v
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Health endpoint should return status: healthy."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestSolveEndpoint:
    """Test the POST /solve endpoint."""

    def test_solve_returns_200_with_valid_input(self, client, valid_problem):
        """Solve endpoint should return 200 with valid problem."""
        response = client.post("/solve", json=valid_problem)
        assert response.status_code == 200

    def test_solve_returns_route(self, client, valid_problem):
        """Solve endpoint should return a route array."""
        response = client.post("/solve", json=valid_problem)
        data = response.json()
        assert "route" in data
        assert isinstance(data["route"], list)
        # Route should start and end at depot (0)
        assert data["route"][0] == 0
        assert data["route"][-1] == 0

    def test_solve_returns_distance(self, client, valid_problem):
        """Solve endpoint should return total distance."""
        response = client.post("/solve", json=valid_problem)
        data = response.json()
        assert "total_distance" in data
        assert isinstance(data["total_distance"], (int, float))
        assert data["total_distance"] > 0

    def test_solve_returns_feasibility(self, client, valid_problem):
        """Solve endpoint should return feasibility status."""
        response = client.post("/solve", json=valid_problem)
        data = response.json()
        assert "is_feasible" in data
        assert isinstance(data["is_feasible"], bool)

    def test_solve_returns_energy(self, client, valid_problem):
        """Solve endpoint should return QUBO energy."""
        response = client.post("/solve", json=valid_problem)
        data = response.json()
        assert "energy" in data
        assert isinstance(data["energy"], (int, float))

    def test_solve_returns_execution_time(self, client, valid_problem):
        """Solve endpoint should return execution time."""
        response = client.post("/solve", json=valid_problem)
        data = response.json()
        assert "execution_time_seconds" in data
        assert data["execution_time_seconds"] > 0

    def test_solve_visits_all_customers(self, client, valid_problem):
        """Route should visit all customers exactly once."""
        response = client.post("/solve", json=valid_problem)
        data = response.json()
        route = data["route"]
        # Remove depot visits
        customers_visited = [n for n in route if n != 0]
        # Should have 5 customers
        assert len(customers_visited) == 5
        # Each customer visited once
        assert len(set(customers_visited)) == 5

    def test_solve_rejects_missing_depot(self, client):
        """Solve should reject request without depot."""
        invalid = {
            "customers": [{"id": 1, "x": 10, "y": 15, "demand": 5}],
            "vehicle_capacity": 20
        }
        response = client.post("/solve", json=invalid)
        assert response.status_code == 422

    def test_solve_rejects_missing_customers(self, client):
        """Solve should reject request without customers."""
        invalid = {
            "depot": {"x": 0, "y": 0},
            "vehicle_capacity": 20
        }
        response = client.post("/solve", json=invalid)
        assert response.status_code == 422

    def test_solve_rejects_infeasible_problem(self, client):
        """Solve should reject problem where demand exceeds capacity."""
        infeasible = {
            "depot": {"x": 0, "y": 0},
            "customers": [
                {"id": 1, "x": 10, "y": 15, "demand": 100}  # Demand > capacity
            ],
            "vehicle_capacity": 20
        }
        response = client.post("/solve", json=infeasible)
        # Should return 400 Bad Request for infeasible problem
        assert response.status_code == 400

    def test_solve_accepts_custom_num_reads(self, client, valid_problem):
        """Solve should accept optional num_reads parameter."""
        valid_problem["num_reads"] = 500
        response = client.post("/solve", json=valid_problem)
        assert response.status_code == 200


class TestCORS:
    """Test CORS configuration."""

    def test_cors_allows_localhost_origin(self, client):
        """CORS should allow requests from localhost:5173 (Vite dev server)."""
        response = client.options(
            "/solve",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            }
        )
        # Should not be forbidden
        assert response.status_code != 403


# Fixtures
@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    from main import app
    return TestClient(app)


@pytest.fixture
def valid_problem():
    """Valid 5-customer CVRP problem."""
    return {
        "depot": {"x": 0, "y": 0},
        "customers": [
            {"id": 1, "x": 10, "y": 15, "demand": 4},
            {"id": 2, "x": -8, "y": 12, "demand": 3},
            {"id": 3, "x": 5, "y": -10, "demand": 5},
            {"id": 4, "x": -12, "y": -5, "demand": 2},
            {"id": 5, "x": 8, "y": 8, "demand": 6},
        ],
        "vehicle_capacity": 20
    }
