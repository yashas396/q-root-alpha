/**
 * Metric Card Component
 * Displays a single KPI metric with optional comparison
 */

export default function MetricCard({
  title,
  value,
  unit = '',
  comparison = null,
  status = 'neutral' // 'success', 'warning', 'error', 'neutral'
}) {
  const statusColors = {
    success: 'text-[#24A148]',
    warning: 'text-[#FF832B]',
    error: 'text-[#DA1E28]',
    neutral: 'text-[#4A4A68]',
  };

  return (
    <div className="bg-white rounded-lg border border-[#D1D1E0] p-4 shadow-sm">
      <div className="text-sm text-[#4A4A68] mb-1">{title}</div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-semibold text-[#1A1A2E]">
          {value}
        </span>
        {unit && (
          <span className="text-sm text-[#4A4A68]">{unit}</span>
        )}
      </div>
      {comparison && (
        <div className={`text-sm mt-1 ${statusColors[status]}`}>
          {comparison}
        </div>
      )}
    </div>
  );
}
