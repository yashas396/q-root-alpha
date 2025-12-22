/**
 * Problem Form Component
 * Input form for CVRP problem definition
 */

import { useState } from 'react';

// Default 5-node demo problem
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

export default function ProblemForm({ onSubmit, isLoading, onProblemChange }) {
  const [problem, setProblem] = useState(DEFAULT_PROBLEM);
  const [jsonInput, setJsonInput] = useState('');
  const [inputMode, setInputMode] = useState('form'); // 'form' or 'json'

  const totalDemand = problem.customers.reduce((sum, c) => sum + c.demand, 0);
  const isFeasible = totalDemand <= problem.vehicle_capacity;

  const handleCapacityChange = (e) => {
    const newProblem = {
      ...problem,
      vehicle_capacity: parseInt(e.target.value) || 0,
    };
    setProblem(newProblem);
    onProblemChange?.(newProblem);
  };

  const handleNumReadsChange = (e) => {
    const newProblem = {
      ...problem,
      num_reads: parseInt(e.target.value) || 1000,
    };
    setProblem(newProblem);
  };

  const handleCustomerChange = (index, field, value) => {
    const newCustomers = [...problem.customers];
    newCustomers[index] = {
      ...newCustomers[index],
      [field]: field === 'name' ? value : parseFloat(value) || 0,
    };
    const newProblem = { ...problem, customers: newCustomers };
    setProblem(newProblem);
    onProblemChange?.(newProblem);
  };

  const addCustomer = () => {
    const newId = Math.max(...problem.customers.map(c => c.id), 0) + 1;
    const newCustomers = [
      ...problem.customers,
      { id: newId, x: 0, y: 0, demand: 1, name: '' },
    ];
    const newProblem = { ...problem, customers: newCustomers };
    setProblem(newProblem);
    onProblemChange?.(newProblem);
  };

  const removeCustomer = (index) => {
    if (problem.customers.length <= 1) return;
    const newCustomers = problem.customers.filter((_, i) => i !== index);
    const newProblem = { ...problem, customers: newCustomers };
    setProblem(newProblem);
    onProblemChange?.(newProblem);
  };

  const loadFromJson = () => {
    try {
      const parsed = JSON.parse(jsonInput);
      setProblem(parsed);
      onProblemChange?.(parsed);
      setInputMode('form');
    } catch (e) {
      alert('Invalid JSON: ' + e.message);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isFeasible) {
      alert('Problem is infeasible: demand exceeds capacity');
      return;
    }
    onSubmit(problem);
  };

  return (
    <div className="bg-white rounded-lg border border-[#D1D1E0] p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-[#1A1A2E]">Problem Setup</h2>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setInputMode('form')}
            className={`px-3 py-1 text-sm rounded ${
              inputMode === 'form'
                ? 'bg-[#0043CE] text-white'
                : 'bg-[#E8E8F0] text-[#4A4A68]'
            }`}
          >
            Form
          </button>
          <button
            type="button"
            onClick={() => setInputMode('json')}
            className={`px-3 py-1 text-sm rounded ${
              inputMode === 'json'
                ? 'bg-[#0043CE] text-white'
                : 'bg-[#E8E8F0] text-[#4A4A68]'
            }`}
          >
            JSON
          </button>
        </div>
      </div>

      {inputMode === 'json' ? (
        <div className="space-y-3">
          <textarea
            value={jsonInput}
            onChange={(e) => setJsonInput(e.target.value)}
            placeholder="Paste JSON problem definition..."
            className="w-full h-48 p-3 border border-[#D1D1E0] rounded font-mono text-sm"
          />
          <button
            type="button"
            onClick={loadFromJson}
            className="px-4 py-2 bg-[#0043CE] text-white rounded hover:bg-[#0035a3]"
          >
            Load JSON
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Vehicle Capacity */}
          <div>
            <label className="block text-sm text-[#4A4A68] mb-1">
              Vehicle Capacity
            </label>
            <input
              type="number"
              value={problem.vehicle_capacity}
              onChange={handleCapacityChange}
              className="w-full px-3 py-2 border border-[#D1D1E0] rounded focus:outline-none focus:border-[#0043CE]"
            />
            <div className={`text-sm mt-1 ${isFeasible ? 'text-[#24A148]' : 'text-[#DA1E28]'}`}>
              Total demand: {totalDemand} / {problem.vehicle_capacity}
              {isFeasible ? ' ✓ Feasible' : ' ✗ Exceeds capacity'}
            </div>
          </div>

          {/* Num Reads */}
          <div>
            <label className="block text-sm text-[#4A4A68] mb-1">
              Optimization Quality (num_reads)
            </label>
            <select
              value={problem.num_reads}
              onChange={handleNumReadsChange}
              className="w-full px-3 py-2 border border-[#D1D1E0] rounded focus:outline-none focus:border-[#0043CE]"
            >
              <option value={500}>Quick (500 reads)</option>
              <option value={1000}>Balanced (1000 reads)</option>
              <option value={2000}>Thorough (2000 reads)</option>
            </select>
          </div>

          {/* Customers */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-[#4A4A68]">
                Customers ({problem.customers.length})
              </label>
              <button
                type="button"
                onClick={addCustomer}
                className="text-sm text-[#0043CE] hover:underline"
              >
                + Add Customer
              </button>
            </div>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {problem.customers.map((customer, index) => (
                <div
                  key={customer.id}
                  className="flex items-center gap-2 p-2 bg-[#F8F8FC] rounded text-sm"
                >
                  <span className="w-8 text-[#4A4A68]">#{customer.id}</span>
                  <input
                    type="number"
                    value={customer.x}
                    onChange={(e) => handleCustomerChange(index, 'x', e.target.value)}
                    placeholder="X"
                    className="w-16 px-2 py-1 border border-[#D1D1E0] rounded text-center"
                  />
                  <input
                    type="number"
                    value={customer.y}
                    onChange={(e) => handleCustomerChange(index, 'y', e.target.value)}
                    placeholder="Y"
                    className="w-16 px-2 py-1 border border-[#D1D1E0] rounded text-center"
                  />
                  <input
                    type="number"
                    value={customer.demand}
                    onChange={(e) => handleCustomerChange(index, 'demand', e.target.value)}
                    placeholder="Demand"
                    className="w-16 px-2 py-1 border border-[#D1D1E0] rounded text-center"
                  />
                  <button
                    type="button"
                    onClick={() => removeCustomer(index)}
                    className="text-[#DA1E28] hover:text-red-700 px-2"
                    disabled={problem.customers.length <= 1}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={isLoading || !isFeasible}
            className={`w-full py-3 rounded font-medium transition-colors ${
              isLoading || !isFeasible
                ? 'bg-[#9E9EB8] text-white cursor-not-allowed'
                : 'bg-[#0043CE] text-white hover:bg-[#0035a3]'
            }`}
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Optimizing...
              </span>
            ) : (
              'Run Optimization'
            )}
          </button>
        </form>
      )}
    </div>
  );
}
