/**
 * Q-Route Alpha Header Component
 * Navigation bar with logo and status
 */

export default function Header({ isConnected }) {
  return (
    <header className="bg-white border-b border-gray-200 px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="text-2xl font-bold text-[#0043CE]">
            <span className="inline-block transform rotate-45 mr-1">◇</span>
            Q-ROUTE ALPHA
          </div>
          <span className="text-xs bg-[#E8E8F0] text-[#4A4A68] px-2 py-1 rounded">
            v0.1.0
          </span>
        </div>

        {/* Status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-[#24A148]' : 'bg-[#DA1E28]'
              }`}
            />
            <span className="text-sm text-[#4A4A68]">
              {isConnected ? 'Backend Connected' : 'Backend Disconnected'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
