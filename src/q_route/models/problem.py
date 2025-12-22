"""CVRP Problem definition models."""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import json
from pathlib import Path


@dataclass
class Customer:
    """
    Represents a customer node in the CVRP.

    Attributes:
        id: Unique customer identifier (1-indexed, 0 is reserved for depot)
        x: X-coordinate of customer location
        y: Y-coordinate of customer location
        demand: Demand quantity at this customer
        name: Optional human-readable name
    """

    id: int
    x: float
    y: float
    demand: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValueError(f"Customer ID must be positive, got {self.id}")
        if self.demand < 0:
            raise ValueError(f"Demand cannot be negative, got {self.demand}")

    @property
    def location(self) -> Tuple[float, float]:
        """Return (x, y) coordinate tuple."""
        return (self.x, self.y)

    def to_dict(self) -> dict:
        """Serialize customer to dictionary."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "demand": self.demand,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Customer":
        """Create Customer from dictionary."""
        return cls(
            id=data["id"],
            x=float(data["x"]),
            y=float(data["y"]),
            demand=int(data["demand"]),
            name=data.get("name", ""),
        )


@dataclass
class CVRPProblem:
    """
    Complete Capacitated Vehicle Routing Problem instance.

    Attributes:
        depot: (x, y) coordinates of the depot (node 0)
        customers: List of Customer objects
        vehicle_capacity: Maximum capacity of the vehicle
        name: Optional problem instance name
    """

    depot: Tuple[float, float]
    customers: List[Customer]
    vehicle_capacity: int
    name: str = "unnamed"

    def __post_init__(self) -> None:
        if self.vehicle_capacity <= 0:
            raise ValueError(
                f"Vehicle capacity must be positive, got {self.vehicle_capacity}"
            )
        if not self.customers:
            raise ValueError("At least one customer is required")

        # Validate unique customer IDs
        ids = [c.id for c in self.customers]
        if len(ids) != len(set(ids)):
            raise ValueError("Customer IDs must be unique")

    @property
    def n_customers(self) -> int:
        """Number of customers (excluding depot)."""
        return len(self.customers)

    @property
    def total_demand(self) -> int:
        """Total demand across all customers."""
        return sum(c.demand for c in self.customers)

    @property
    def n_nodes(self) -> int:
        """Total number of nodes including depot."""
        return self.n_customers + 1

    def validate(self) -> bool:
        """
        Check if problem instance is feasible.

        Returns:
            True if total demand <= vehicle capacity
        """
        return self.total_demand <= self.vehicle_capacity

    def get_customer_by_id(self, customer_id: int) -> Optional[Customer]:
        """Get customer by ID."""
        for c in self.customers:
            if c.id == customer_id:
                return c
        return None

    def get_node_location(self, node_id: int) -> Tuple[float, float]:
        """
        Get location of a node by ID.

        Args:
            node_id: 0 for depot, 1+ for customers

        Returns:
            (x, y) coordinate tuple
        """
        if node_id == 0:
            return self.depot
        customer = self.get_customer_by_id(node_id)
        if customer is None:
            raise ValueError(f"No customer with ID {node_id}")
        return customer.location

    def to_dict(self) -> dict:
        """Serialize problem to dictionary."""
        return {
            "name": self.name,
            "depot": {"x": self.depot[0], "y": self.depot[1]},
            "customers": [c.to_dict() for c in self.customers],
            "vehicle_capacity": self.vehicle_capacity,
        }

    def to_json(self, path: str) -> None:
        """Save problem to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "CVRPProblem":
        """Create CVRPProblem from dictionary."""
        depot_data = data["depot"]
        depot = (float(depot_data["x"]), float(depot_data["y"]))

        customers = [Customer.from_dict(c) for c in data["customers"]]

        return cls(
            depot=depot,
            customers=customers,
            vehicle_capacity=int(data["vehicle_capacity"]),
            name=data.get("name", "unnamed"),
        )

    @classmethod
    def from_json(cls, path: str) -> "CVRPProblem":
        """Load problem from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"CVRPProblem(name='{self.name}', "
            f"customers={self.n_customers}, "
            f"total_demand={self.total_demand}, "
            f"capacity={self.vehicle_capacity})"
        )
