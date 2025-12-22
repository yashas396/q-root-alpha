/**
 * Q-Route Alpha - Main Application
 * Quantum-Optimized Logistics Dashboard
 */

import { useState, useEffect } from 'react';
import Header from './components/Header';
import RouteMap from './components/RouteMap';
import ProblemForm from './components/ProblemForm';
import ResultsPanel from './components/ResultsPanel';
import { checkHealth, solveProblem, formatError } from './api/client';

// Default problem for initial display
const DEFAULT_PROBLEM = {
  depot: { x: 0, y: 0 },
  customers: [
    { id: 1, x: 10, y: 15, demand: 4, name: 'Customer A' },
    { id: 2, x: -8, y: 12, demand: 3, name: 'Customer B' },
    { id: 3, x: 5, y: -10, demand: 5, name: 'Customer C' },
    { id: 4, x: -12, y: -5, demand: 2, name: 'Customer D' },
    { id: 5, x: 8, y: 8, demand: 6, name: 'Customer E' },
  ],
  vehicle_capacity: 20,
  num_reads: 1000,
};

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [problem, setProblem] = useState(DEFAULT_PROBLEM);
  const [solution, setSolution] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        await checkHealth();
        setIsConnected(true);
      } catch (err) {
        setIsConnected(false);
        console.warn('Backend not connected:', err.message);
      }
    };
    checkBackend();
    // Re-check every 30 seconds
    const interval = setInterval(checkBackend, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle optimization submit
  const handleSubmit = async (problemData) => {
    setIsLoading(true);
    setError(null);
    setSolution(null);

    try {
      const result = await solveProblem(problemData);
      setSolution(result);
      setProblem(problemData);
    } catch (err) {
      setError(formatError(err));
    } finally {
      setIsLoading(false);
    }
  };

  // Handle problem changes (for live map update)
  const handleProblemChange = (newProblem) => {
    setProblem(newProblem);
  };

  return (
    <div className="min-h-screen bg-[#F8F8FC] flex flex-col">
      <Header isConnected={isConnected} />

      <main className="flex-1 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Title */}
          <div className="mb-6">
            <h1 className="text-2xl font-semibold text-[#1A1A2E]">
              Route Optimization Dashboard
            </h1>
            <p className="text-[#4A4A68] mt-1">
              Quantum-inspired optimization for logistics routing
            </p>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-[#DA1E28]/10 border border-[#DA1E28] rounded-lg">
              <div className="flex items-center gap-2 text-[#DA1E28]">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
                <span className="font-medium">Error</span>
              </div>
              <p className="mt-1 text-[#DA1E28]">{error}</p>
            </div>
          )}

          {/* Main Grid Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Form */}
            <div className="lg:col-span-1 space-y-6">
              <ProblemForm
                onSubmit={handleSubmit}
                isLoading={isLoading}
                onProblemChange={handleProblemChange}
              />
              <ResultsPanel solution={solution} problem={problem} />
            </div>

            {/* Right Column - Map */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg border border-[#D1D1E0] p-4">
                <h2 className="text-lg font-semibold text-[#1A1A2E] mb-4">
                  Route Visualization
                </h2>
                <div className="h-[600px]">
                  <RouteMap
                    depot={problem.depot}
                    customers={problem.customers}
                    route={solution?.route || []}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Footer Info */}
          <div className="mt-8 text-center text-sm text-[#4A4A68]">
            <p>
              Powered by{' '}
              <span className="text-[#0043CE] font-medium">
                D-Wave Simulated Annealing
              </span>{' '}
              | QUBO-based CVRP Optimization
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
