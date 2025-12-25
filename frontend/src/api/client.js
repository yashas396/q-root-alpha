/**
 * Q-Route Alpha API Client
 * Handles communication with the FastAPI backend
 */

import axios from 'axios';

// API base URL - configurable via environment variable
// In production (Docker): uses /api which nginx proxies to backend
// In development: uses VITE_API_URL or defaults to localhost:8080
const API_BASE = import.meta.env.PROD
  ? '/api'
  : (import.meta.env.VITE_API_URL || 'http://localhost:8080');

const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 second timeout for optimization
});

/**
 * Health check endpoint
 * @returns {Promise<{status: string, version: string}>}
 */
export async function checkHealth() {
  const response = await apiClient.get('/health');
  return response.data;
}

/**
 * Solve a CVRP problem
 * @param {Object} problem - The CVRP problem definition
 * @param {Object} problem.depot - Depot location {x, y}
 * @param {Array} problem.customers - Array of customers [{id, x, y, demand}]
 * @param {number} problem.vehicle_capacity - Vehicle capacity
 * @param {number} [problem.num_reads=1000] - Number of annealing reads
 * @returns {Promise<Object>} Solution with route, distance, etc.
 */
export async function solveProblem(problem) {
  const response = await apiClient.post('/solve', problem);
  return response.data;
}

/**
 * Format API error for display
 * @param {Error} error - Axios error object
 * @returns {string} User-friendly error message
 */
export function formatError(error) {
  if (error.response) {
    // Server responded with error
    const data = error.response.data;
    if (data.detail) {
      return data.detail;
    }
    return `Server error: ${error.response.status}`;
  } else if (error.request) {
    // No response received
    return 'Unable to connect to server. Is the backend running?';
  } else {
    // Request setup error
    return error.message;
  }
}

export default apiClient;
