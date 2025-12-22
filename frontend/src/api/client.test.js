/**
 * API Client Tests
 */

import { describe, it, expect, vi, beforeAll, afterAll, afterEach } from 'vitest';
import { server } from '../test/mocks/server';
import { checkHealth, solveProblem, formatError } from './client';

// Setup MSW server
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('API Client', () => {
  describe('checkHealth', () => {
    it('returns health status from server', async () => {
      const result = await checkHealth();
      expect(result).toEqual({
        status: 'healthy',
        version: '0.1.0',
      });
    });
  });

  describe('solveProblem', () => {
    const validProblem = {
      depot: { x: 0, y: 0 },
      customers: [
        { id: 1, x: 10, y: 15, demand: 4, name: 'A' },
        { id: 2, x: -8, y: 12, demand: 3, name: 'B' },
        { id: 3, x: 5, y: -10, demand: 5, name: 'C' },
      ],
      vehicle_capacity: 20,
      num_reads: 1000,
    };

    it('returns solution with route', async () => {
      const result = await solveProblem(validProblem);
      expect(result.route).toBeDefined();
      expect(Array.isArray(result.route)).toBe(true);
    });

    it('returns solution with total_distance', async () => {
      const result = await solveProblem(validProblem);
      expect(typeof result.total_distance).toBe('number');
    });

    it('returns solution with is_feasible flag', async () => {
      const result = await solveProblem(validProblem);
      expect(typeof result.is_feasible).toBe('boolean');
    });

    it('includes execution time in response', async () => {
      const result = await solveProblem(validProblem);
      expect(result.execution_time_seconds).toBeDefined();
    });

    it('respects num_reads parameter', async () => {
      const result = await solveProblem({ ...validProblem, num_reads: 500 });
      expect(result.num_reads).toBe(500);
    });
  });

  describe('formatError', () => {
    it('formats server error with detail', () => {
      const error = {
        response: {
          status: 400,
          data: { detail: 'Problem is infeasible' },
        },
      };
      expect(formatError(error)).toBe('Problem is infeasible');
    });

    it('formats server error without detail', () => {
      const error = {
        response: {
          status: 500,
          data: {},
        },
      };
      expect(formatError(error)).toBe('Server error: 500');
    });

    it('formats network error', () => {
      const error = {
        request: {},
        message: 'Network Error',
      };
      expect(formatError(error)).toBe('Unable to connect to server. Is the backend running?');
    });

    it('formats generic error', () => {
      const error = {
        message: 'Something went wrong',
      };
      expect(formatError(error)).toBe('Something went wrong');
    });
  });
});
