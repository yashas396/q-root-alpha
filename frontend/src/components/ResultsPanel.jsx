/**
 * Results Panel Component
 * Displays optimization results and route details
 */

import MetricCard from './MetricCard';

export default function ResultsPanel({ solution, problem }) {
  if (!solution) {
    return (
      <div className="bg-white rounded-lg border border-[#D1D1E0] p-6 text-center">
        <div className="text-[#4A4A68]">
          <svg
            className="w-16 h-16 mx-auto mb-4 text-[#D1D1E0]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
            />
          </svg>
          <p className="text-lg font-medium mb-2">No Results Yet</p>
          <p className="text-sm">Configure your problem and click "Run Optimization"</p>
        </div>
      </div>
    );
  }

  const routeString = solution.route.join(' → ');

  return (
    <div className="bg-white rounded-lg border border-[#D1D1E0] p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-[#1A1A2E]">
          Optimization Results
        </h2>
        <span
          className={`px-3 py-1 rounded-full text-sm font-medium ${
            solution.is_feasible
              ? 'bg-[#24A148]/10 text-[#24A148]'
              : 'bg-[#DA1E28]/10 text-[#DA1E28]'
          }`}
        >
          {solution.is_feasible ? '✓ Feasible' : '✗ Infeasible'}
        </span>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          title="Total Distance"
          value={solution.total_distance.toFixed(2)}
          unit="km"
          status="success"
        />
        <MetricCard
          title="QUBO Energy"
          value={solution.energy.toFixed(2)}
          status="neutral"
        />
        <MetricCard
          title="Execution Time"
          value={solution.execution_time_seconds.toFixed(3)}
          unit="sec"
          status="neutral"
        />
        {solution.improvement_vs_random && (
          <MetricCard
            title="vs Random"
            value={`${solution.improvement_vs_random > 0 ? '+' : ''}${solution.improvement_vs_random.toFixed(1)}%`}
            status={solution.improvement_vs_random > 0 ? 'success' : 'error'}
          />
        )}
      </div>

      {/* Route */}
      <div>
        <h3 className="text-sm font-medium text-[#4A4A68] mb-2">Optimized Route</h3>
        <div className="bg-[#F8F8FC] rounded p-3">
          <code className="text-sm text-[#1A1A2E] break-all">
            {routeString}
          </code>
        </div>
      </div>

      {/* Route Details Table */}
      <div>
        <h3 className="text-sm font-medium text-[#4A4A68] mb-2">Route Details</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#D1D1E0]">
                <th className="text-left py-2 text-[#4A4A68] font-medium">Step</th>
                <th className="text-left py-2 text-[#4A4A68] font-medium">From → To</th>
                <th className="text-right py-2 text-[#4A4A68] font-medium">Demand</th>
              </tr>
            </thead>
            <tbody>
              {solution.route.slice(0, -1).map((fromNode, index) => {
                const toNode = solution.route[index + 1];
                const fromLabel = fromNode === 0 ? 'Depot' : `Customer ${fromNode}`;
                const toLabel = toNode === 0 ? 'Depot' : `Customer ${toNode}`;
                const customerDemand = toNode !== 0 && problem?.customers
                  ? problem.customers.find(c => c.id === toNode)?.demand || 0
                  : 0;

                return (
                  <tr key={index} className="border-b border-[#E8E8F0]">
                    <td className="py-2 text-[#4A4A68]">{index + 1}</td>
                    <td className="py-2 text-[#1A1A2E]">{fromLabel} → {toLabel}</td>
                    <td className="py-2 text-right text-[#4A4A68]">
                      {customerDemand > 0 ? `+${customerDemand}` : '-'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Constraint Violations */}
      {solution.constraint_violations && solution.constraint_violations.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-[#DA1E28] mb-2">Constraint Violations</h3>
          <ul className="text-sm text-[#DA1E28] space-y-1">
            {solution.constraint_violations.map((violation, index) => (
              <li key={index}>• {violation}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Export Button */}
      <button
        onClick={() => {
          const data = JSON.stringify(solution, null, 2);
          const blob = new Blob([data], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'q-route-solution.json';
          a.click();
        }}
        className="w-full py-2 border border-[#0043CE] text-[#0043CE] rounded hover:bg-[#0043CE]/5 transition-colors"
      >
        Export JSON
      </button>
    </div>
  );
}
