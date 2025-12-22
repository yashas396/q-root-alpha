/**
 * ResultsPanel Component Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ResultsPanel from './ResultsPanel';

describe('ResultsPanel', () => {
  const mockProblem = {
    depot: { x: 0, y: 0 },
    customers: [
      { id: 1, x: 10, y: 15, demand: 4, name: 'A' },
      { id: 2, x: -8, y: 12, demand: 3, name: 'B' },
      { id: 3, x: 5, y: -10, demand: 5, name: 'C' },
    ],
    vehicle_capacity: 20,
  };

  const mockSolution = {
    route: [0, 1, 2, 3, 0],
    total_distance: 45.5,
    energy: -150.25,
    is_feasible: true,
    constraint_violations: [],
    execution_time_seconds: 0.523,
    improvement_vs_random: 25.5,
  };

  it('shows no results message when solution is null', () => {
    render(<ResultsPanel solution={null} problem={mockProblem} />);
    expect(screen.getByText('No Results Yet')).toBeInTheDocument();
  });

  it('shows instructions when no solution', () => {
    render(<ResultsPanel solution={null} problem={mockProblem} />);
    expect(screen.getByText(/Configure your problem/)).toBeInTheDocument();
  });

  it('renders optimization results title when solution exists', () => {
    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    expect(screen.getByText('Optimization Results')).toBeInTheDocument();
  });

  it('shows feasible badge for feasible solution', () => {
    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    expect(screen.getByText(/Feasible/)).toBeInTheDocument();
  });

  it('shows infeasible badge for infeasible solution', () => {
    const infeasibleSolution = { ...mockSolution, is_feasible: false };
    render(<ResultsPanel solution={infeasibleSolution} problem={mockProblem} />);
    expect(screen.getByText(/Infeasible/)).toBeInTheDocument();
  });

  it('displays total distance metric', () => {
    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    expect(screen.getByText('Total Distance')).toBeInTheDocument();
    expect(screen.getByText('45.50')).toBeInTheDocument();
  });

  it('displays QUBO energy metric', () => {
    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    expect(screen.getByText('QUBO Energy')).toBeInTheDocument();
  });

  it('displays execution time metric', () => {
    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    expect(screen.getByText('Execution Time')).toBeInTheDocument();
  });

  it('displays improvement vs random when available', () => {
    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    expect(screen.getByText('vs Random')).toBeInTheDocument();
    expect(screen.getByText('+25.5%')).toBeInTheDocument();
  });

  it('displays route string', () => {
    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    expect(screen.getByText('Optimized Route')).toBeInTheDocument();
    expect(screen.getByText('0 → 1 → 2 → 3 → 0')).toBeInTheDocument();
  });

  it('displays route details table', () => {
    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    expect(screen.getByText('Route Details')).toBeInTheDocument();
    expect(screen.getByText('Step')).toBeInTheDocument();
    expect(screen.getByText('From → To')).toBeInTheDocument();
  });

  it('shows constraint violations when present', () => {
    const violationSolution = {
      ...mockSolution,
      is_feasible: false,
      constraint_violations: ['Capacity exceeded', 'Missing customer'],
    };
    render(<ResultsPanel solution={violationSolution} problem={mockProblem} />);
    expect(screen.getByText('Constraint Violations')).toBeInTheDocument();
    expect(screen.getByText(/Capacity exceeded/)).toBeInTheDocument();
  });

  it('renders export JSON button', () => {
    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    expect(screen.getByText('Export JSON')).toBeInTheDocument();
  });

  it('triggers download when export button is clicked', () => {
    const mockClick = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag) => {
      const element = originalCreateElement(tag);
      if (tag === 'a') {
        element.click = mockClick;
      }
      return element;
    });

    render(<ResultsPanel solution={mockSolution} problem={mockProblem} />);
    fireEvent.click(screen.getByText('Export JSON'));

    expect(mockClick).toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});
