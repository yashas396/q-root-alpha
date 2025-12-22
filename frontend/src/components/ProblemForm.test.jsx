/**
 * ProblemForm Component Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ProblemForm from './ProblemForm';

describe('ProblemForm', () => {
  const mockOnSubmit = vi.fn();
  const mockOnProblemChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders problem setup title', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);
    expect(screen.getByText('Problem Setup')).toBeInTheDocument();
  });

  it('renders vehicle capacity input', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);
    expect(screen.getByText('Vehicle Capacity')).toBeInTheDocument();
  });

  it('renders optimization quality select', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);
    expect(screen.getByText('Optimization Quality (num_reads)')).toBeInTheDocument();
  });

  it('renders customer list', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);
    expect(screen.getByText(/Customers \(\d+\)/)).toBeInTheDocument();
  });

  it('renders Run Optimization button', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);
    expect(screen.getByText('Run Optimization')).toBeInTheDocument();
  });

  it('shows loading state when isLoading is true', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={true} />);
    expect(screen.getByText('Optimizing...')).toBeInTheDocument();
  });

  it('disables submit button when loading', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={true} />);
    const button = screen.getByRole('button', { name: /Optimizing/i });
    expect(button).toBeDisabled();
  });

  it('shows form and json mode toggles', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);
    expect(screen.getByText('Form')).toBeInTheDocument();
    expect(screen.getByText('JSON')).toBeInTheDocument();
  });

  it('switches to JSON mode when JSON button is clicked', async () => {
    const user = userEvent.setup();
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);

    await user.click(screen.getByText('JSON'));
    expect(screen.getByPlaceholderText(/Paste JSON/i)).toBeInTheDocument();
  });

  it('shows Add Customer button', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);
    expect(screen.getByText('+ Add Customer')).toBeInTheDocument();
  });

  it('displays feasibility status', () => {
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);
    // Default problem should be feasible
    expect(screen.getByText(/Feasible/)).toBeInTheDocument();
  });

  it('calls onProblemChange when capacity changes', async () => {
    const user = userEvent.setup();
    render(
      <ProblemForm
        onSubmit={mockOnSubmit}
        isLoading={false}
        onProblemChange={mockOnProblemChange}
      />
    );

    const capacityInput = screen.getByDisplayValue('20');
    await user.clear(capacityInput);
    await user.type(capacityInput, '30');

    expect(mockOnProblemChange).toHaveBeenCalled();
  });

  it('submits form with problem data', async () => {
    const user = userEvent.setup();
    render(<ProblemForm onSubmit={mockOnSubmit} isLoading={false} />);

    await user.click(screen.getByText('Run Optimization'));

    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        depot: expect.any(Object),
        customers: expect.any(Array),
        vehicle_capacity: expect.any(Number),
        num_reads: expect.any(Number),
      })
    );
  });
});
