/**
 * Header Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Header from './Header';

describe('Header', () => {
  it('renders the Q-Route Alpha title', () => {
    render(<Header isConnected={false} />);
    expect(screen.getByText('Q-ROUTE ALPHA')).toBeInTheDocument();
  });

  it('shows disconnected status when isConnected is false', () => {
    render(<Header isConnected={false} />);
    expect(screen.getByText('Backend Disconnected')).toBeInTheDocument();
  });

  it('shows connected status when isConnected is true', () => {
    render(<Header isConnected={true} />);
    expect(screen.getByText('Backend Connected')).toBeInTheDocument();
  });

  it('displays version badge', () => {
    render(<Header isConnected={false} />);
    expect(screen.getByText('v0.1.0')).toBeInTheDocument();
  });

  it('has green indicator for connected state', () => {
    render(<Header isConnected={true} />);
    const statusContainer = screen.getByText('Backend Connected').parentElement;
    const indicator = statusContainer.querySelector('div');
    expect(indicator).toHaveClass('bg-[#24A148]');
  });

  it('has red indicator for disconnected state', () => {
    render(<Header isConnected={false} />);
    const statusContainer = screen.getByText('Backend Disconnected').parentElement;
    const indicator = statusContainer.querySelector('div');
    expect(indicator).toHaveClass('bg-[#DA1E28]');
  });
});
