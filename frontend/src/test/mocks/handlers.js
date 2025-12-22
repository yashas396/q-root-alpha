/**
 * MSW Request Handlers
 * Mock API responses for testing
 */

import { http, HttpResponse } from 'msw';

// Mock solution response
const mockSolution = {
  route: [0, 1, 2, 3, 0],
  total_distance: 45.5,
  energy: -150.25,
  is_feasible: true,
  constraint_violations: [],
  execution_time_seconds: 0.523,
  num_reads: 1000,
  improvement_vs_random: 25.5,
};

export const handlers = [
  // Health check endpoint
  http.get('http://localhost:8000/health', () => {
    return HttpResponse.json({
      status: 'healthy',
      version: '0.1.0',
    });
  }),

  // Solve endpoint
  http.post('http://localhost:8000/solve', async ({ request }) => {
    const body = await request.json();

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 100));

    // Return mock solution with route based on customer count
    const customerCount = body.customers?.length || 3;
    const route = [0, ...Array.from({ length: customerCount }, (_, i) => i + 1), 0];

    return HttpResponse.json({
      ...mockSolution,
      route,
      num_reads: body.num_reads || 1000,
    });
  }),

  // Error case handler
  http.post('http://localhost:8000/solve-error', () => {
    return HttpResponse.json(
      { detail: 'Problem is infeasible: demand exceeds capacity' },
      { status: 400 }
    );
  }),
];
