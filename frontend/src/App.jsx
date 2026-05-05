/**
 * App.jsx
 * ───────
 * Root layout:
 *
 *  ┌─────────────────────────────────────────────────────┐
 *  │  Header (title · stats · WS status · Simulate btn) │
 *  ├──────────────────────────────┬──────────────────────┤
 *  │                              │  Alerts Feed         │
 *  │   LiveMap                    ├──────────────────────┤
 *  │   (Leaflet + heatmap)        │  Analytics Panel     │
 *  │                              │  (recharts)          │
 *  └──────────────────────────────┴──────────────────────┘
 *
 * Data flow:
 *   useEvents()       → initial REST load + live WebSocket updates
 *   useWebSocket()    → WS connection status + new event handler
 */

import { useCallback } from 'react'
import Header          from './components/Header.jsx'
import StatsBar        from './components/StatsBar.jsx'
import LiveMap         from './components/LiveMap.jsx'
import AlertsFeed      from './components/AlertsFeed.jsx'
import AnalyticsPanel  from './components/AnalyticsPanel.jsx'
import { useEvents }   from './hooks/useEvents.js'
import { useWebSocket } from './hooks/useWebSocket.js'

export default function App() {
  const {
    events, stats, alerts,
    loading, error, handleLiveEvent,
  } = useEvents()

  const { connected } = useWebSocket(handleLiveEvent)

  if (loading) {
    return (
      <div className="h-screen flex flex-col items-center justify-center
                      bg-gray-950 text-gray-400 gap-3">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent
                        rounded-full animate-spin" />
        <p className="text-sm">Connecting to SmartCityAI…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-screen flex flex-col items-center justify-center
                      bg-gray-950 text-red-400 gap-2">
        <p className="text-lg font-semibold">⚠️ Backend Unreachable</p>
        <p className="text-sm text-gray-500">{error}</p>
        <p className="text-xs text-gray-600 mt-2">
          Run: <code className="bg-gray-800 px-1 rounded">uvicorn main:app --reload</code>
        </p>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-gray-950 overflow-hidden">

      <Header
        connected={connected}
        stats={stats}
        onSimulate={() => {}}
      />

      <StatsBar stats={stats} />

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">

        {/* Map — takes most of the screen */}
        <div className="flex-1 relative">
          <LiveMap events={events} />
        </div>

        {/* Right sidebar */}
        <aside className="w-72 flex flex-col border-l border-gray-800
                          bg-gray-900 overflow-hidden shrink-0">

          {/* Alerts feed */}
          <div className="flex flex-col border-b border-gray-800" style={{ height: '55%' }}>
            <div className="px-3 pt-3 pb-2 flex items-center justify-between shrink-0">
              <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Live Alerts
              </h2>
              <span className="text-xs text-gray-600">{alerts.length}</span>
            </div>
            <AlertsFeed alerts={alerts} />
          </div>

          {/* Analytics panel */}
          <div className="flex-1 overflow-y-auto p-3">
            <h2 className="text-xs font-semibold text-gray-400 uppercase
                           tracking-wider mb-3">
              Analytics
            </h2>
            <AnalyticsPanel events={events} />
          </div>

        </aside>
      </div>
    </div>
  )
}