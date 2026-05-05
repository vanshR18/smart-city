/**
 * Header.jsx
 * Top bar with title, WS connection badge, and simulation button.
 */

import { useState } from 'react'
import { Activity, Wifi, WifiOff, Zap } from 'lucide-react'
import { runSimulation } from '../api.js'

export default function Header({ connected, stats, onSimulate }) {
  const [simulating, setSimulating] = useState(false)

  const handleSim = async () => {
    setSimulating(true)
    try {
      const result = await runSimulation(20)
      onSimulate?.(result)
    } catch (e) {
      console.error('Simulation failed:', e)
    } finally {
      setSimulating(false)
    }
  }

  return (
    <header className="flex items-center justify-between px-5 py-3
                       bg-gray-900 border-b border-gray-800 shrink-0">

      {/* Left: title */}
      <div className="flex items-center gap-3">
        <Activity className="text-red-500" size={22} />
        <div>
          <h1 className="text-base font-bold text-white leading-tight">
            SmartCity AI
          </h1>
          <p className="text-xs text-gray-400">Lucknow Emergency Control</p>
        </div>
      </div>

      {/* Right: WS status + sim button */}
      <div className="flex items-center gap-3">
        <span className={`flex items-center gap-1.5 text-xs font-medium
          ${connected ? 'text-green-400' : 'text-gray-500'}`}>
          {connected
            ? <><Wifi size={14}/> Live</>
            : <><WifiOff size={14}/> Offline</>}
        </span>

        <button
          onClick={handleSim}
          disabled={simulating}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs
            font-medium bg-blue-600 hover:bg-blue-500 disabled:opacity-50
            disabled:cursor-not-allowed transition-colors"
        >
          <Zap size={13} />
          {simulating ? 'Running…' : 'Simulate 20 Events'}
        </button>
      </div>
    </header>
  )
}

function Stat({ label, value, color = 'text-white' }) {
  return (
    <div className="text-center">
      <div className={`text-sm font-bold ${color}`}>{value}</div>
      <div className="text-gray-500">{label}</div>
    </div>
  )
}