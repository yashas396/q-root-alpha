"""Command-line interface for Q-Route Alpha."""

import argparse
import sys
import json
from pathlib import Path

from q_route import __version__
from q_route.models.problem import CVRPProblem
from q_route.solvers.sa_solver import SimulatedAnnealingSolver
from q_route.utils.metrics import solution_quality_report, format_quality_report
from q_route.utils.visualization import plot_route


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='q-route',
        description='Q-Route Alpha: Quantum-Ready Logistics Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  q-route solve examples/simple_5_node.json
  q-route solve problem.json --num-reads 2000 --visualize
  q-route solve problem.json --output solution.json --save-plot route.png
  q-route info examples/simple_5_node.json
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'Q-Route Alpha {__version__}'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Solve command
    solve_parser = subparsers.add_parser(
        'solve',
        help='Solve a CVRP problem instance'
    )
    solve_parser.add_argument(
        'problem_file',
        type=str,
        help='Path to JSON problem file'
    )
    solve_parser.add_argument(
        '--num-reads',
        type=int,
        default=1000,
        help='Number of annealing reads (default: 1000)'
    )
    solve_parser.add_argument(
        '--num-sweeps',
        type=int,
        default=1000,
        help='Number of sweeps per read (default: 1000)'
    )
    solve_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    solve_parser.add_argument(
        '--penalty-multiplier',
        type=float,
        default=2.0,
        help='Constraint penalty multiplier (default: 2.0)'
    )
    solve_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Save solution to JSON file'
    )
    solve_parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Show route visualization'
    )
    solve_parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Save visualization to file'
    )
    solve_parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark comparison against baselines'
    )
    solve_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )

    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show information about a problem file'
    )
    info_parser.add_argument(
        'problem_file',
        type=str,
        help='Path to JSON problem file'
    )

    # Demo command
    demo_parser = subparsers.add_parser(
        'demo',
        help='Run a demonstration with built-in problem'
    )
    demo_parser.add_argument(
        '--size',
        type=int,
        default=5,
        choices=[3, 5, 10],
        help='Problem size (number of customers)'
    )

    return parser


def cmd_solve(args) -> int:
    """Execute the solve command."""
    # Load problem
    try:
        problem = CVRPProblem.from_json(args.problem_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.problem_file}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return 1
    except Exception as e:
        print(f"Error loading problem: {e}")
        return 1

    if not args.quiet:
        print(f"Loaded problem: {problem}")
        print(f"Customers: {problem.n_customers}")
        print(f"Total demand: {problem.total_demand} / {problem.vehicle_capacity}")
        print()

    # Validate
    if not problem.validate():
        print("Error: Problem is infeasible (demand exceeds capacity)")
        return 1

    # Create solver
    solver = SimulatedAnnealingSolver(
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        seed=args.seed,
        penalty_multiplier=args.penalty_multiplier
    )

    if not args.quiet:
        print("Solving with Simulated Annealing...")
        print(f"  num_reads: {args.num_reads}")
        print(f"  num_sweeps: {args.num_sweeps}")
        print()

    # Solve
    solution = solver.solve(problem)

    # Output results
    if args.quiet:
        print(f"{solution.total_distance:.2f}")
    else:
        print(solution.format_report())

        if not solution.is_feasible:
            print("\nWARNING: Solution is not feasible!")
            for v in solution.constraint_violations:
                print(f"  - {v}")

    # Benchmark
    if args.benchmark and not args.quiet:
        print("\nRunning benchmark comparison...")
        report = solution_quality_report(
            problem,
            solution.total_distance,
            solution.execution_time_seconds
        )
        print(format_quality_report(report))

    # Save solution
    if args.output:
        solution.to_json(args.output)
        if not args.quiet:
            print(f"\nSolution saved to: {args.output}")

    # Visualization
    if args.visualize or args.save_plot:
        try:
            import matplotlib.pyplot as plt
            fig = plot_route(
                problem,
                solution,
                save_path=args.save_plot
            )
            if args.save_plot and not args.quiet:
                print(f"Plot saved to: {args.save_plot}")
            if args.visualize:
                plt.show()
        except ImportError:
            print("Warning: matplotlib not available for visualization")

    return 0 if solution.is_feasible else 1


def cmd_info(args) -> int:
    """Execute the info command."""
    try:
        problem = CVRPProblem.from_json(args.problem_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.problem_file}")
        return 1
    except Exception as e:
        print(f"Error loading problem: {e}")
        return 1

    print("=" * 50)
    print("CVRP Problem Information")
    print("=" * 50)
    print(f"Name: {problem.name}")
    print(f"File: {args.problem_file}")
    print()
    print(f"Depot: ({problem.depot[0]}, {problem.depot[1]})")
    print(f"Customers: {problem.n_customers}")
    print(f"Vehicle Capacity: {problem.vehicle_capacity}")
    print(f"Total Demand: {problem.total_demand}")
    print(f"Feasible: {'Yes' if problem.validate() else 'No'}")
    print()
    print("Customers:")
    for c in problem.customers:
        print(f"  #{c.id}: ({c.x:.1f}, {c.y:.1f}) demand={c.demand}")
    print()
    print(f"QUBO Variables: {problem.n_customers ** 2}")
    print("=" * 50)

    return 0


def cmd_demo(args) -> int:
    """Execute the demo command."""
    from q_route.models.problem import Customer

    # Create demo problem
    if args.size == 3:
        customers = [
            Customer(1, 3, 4, 5, "Customer A"),
            Customer(2, 6, 0, 3, "Customer B"),
            Customer(3, 3, -4, 4, "Customer C"),
        ]
        capacity = 15
    elif args.size == 5:
        customers = [
            Customer(1, 10, 15, 4, "Customer A"),
            Customer(2, -8, 12, 3, "Customer B"),
            Customer(3, 5, -10, 5, "Customer C"),
            Customer(4, -12, -5, 2, "Customer D"),
            Customer(5, 8, 8, 6, "Customer E"),
        ]
        capacity = 20
    else:  # size == 10
        customers = [
            Customer(1, 10, 15, 3),
            Customer(2, -8, 12, 2),
            Customer(3, 5, -10, 4),
            Customer(4, -12, -5, 2),
            Customer(5, 8, 8, 3),
            Customer(6, -5, 18, 2),
            Customer(7, 15, -8, 3),
            Customer(8, -15, 3, 2),
            Customer(9, 12, 5, 2),
            Customer(10, -3, -12, 2),
        ]
        capacity = 30

    problem = CVRPProblem(
        depot=(0, 0),
        customers=customers,
        vehicle_capacity=capacity,
        name=f"Demo-{args.size}-node"
    )

    print(f"Q-Route Alpha Demo ({args.size} customers)")
    print("=" * 50)
    print(f"Problem: {problem}")
    print()

    solver = SimulatedAnnealingSolver(num_reads=1000)
    print("Solving...")
    solution = solver.solve(problem)

    print(solution.format_report())

    # Show visualization
    try:
        import matplotlib.pyplot as plt
        plot_route(problem, solution)
        plt.show()
    except ImportError:
        print("(matplotlib not available for visualization)")

    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'solve':
        return cmd_solve(args)
    elif args.command == 'info':
        return cmd_info(args)
    elif args.command == 'demo':
        return cmd_demo(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
