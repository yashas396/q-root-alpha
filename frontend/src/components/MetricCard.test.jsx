/**
 * MetricCard Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import MetricCard from './MetricCard';

describe('MetricCard', () => {
  it('renders title and value', () => {
    render(<MetricCard title="Total Distance" value="45.5" />);
    expect(screen.getByText('Total Distance')).toBeInTheDocument();
    expect(screen.getByText('45.5')).toBeInTheDocument();
  });

  it('renders unit when provided', () => {
    render(<MetricCard title="Distance" value="45.5" unit="km" />);
    expect(screen.getByText('km')).toBeInTheDocument();
  });

  it('does not render unit when not provided', () => {
    render(<MetricCard title="Energy" value="-150.25" />);
    expect(screen.queryByText('km')).not.toBeInTheDocument();
  });

  it('renders comparison text when provided', () => {
    render(
      <MetricCard
        title="Distance"
        value="45.5"
        comparison="+25% vs baseline"
        status="success"
      />
    );
    expect(screen.getByText('+25% vs baseline')).toBeInTheDocument();
  });

  it('applies success status color', () => {
    render(
      <MetricCard
        title="Distance"
        value="45.5"
        comparison="Good"
        status="success"
      />
    );
    const comparison = screen.getByText('Good');
    expect(comparison).toHaveClass('text-[#24A148]');
  });

  it('applies error status color', () => {
    render(
      <MetricCard
        title="Distance"
        value="45.5"
        comparison="Bad"
        status="error"
      />
    );
    const comparison = screen.getByText('Bad');
    expect(comparison).toHaveClass('text-[#DA1E28]');
  });

  it('applies warning status color', () => {
    render(
      <MetricCard
        title="Distance"
        value="45.5"
        comparison="Warning"
        status="warning"
      />
    );
    const comparison = screen.getByText('Warning');
    expect(comparison).toHaveClass('text-[#FF832B]');
  });

  it('applies neutral status color by default', () => {
    render(
      <MetricCard
        title="Distance"
        value="45.5"
        comparison="Neutral"
        status="neutral"
      />
    );
    const comparison = screen.getByText('Neutral');
    expect(comparison).toHaveClass('text-[#4A4A68]');
  });
});
