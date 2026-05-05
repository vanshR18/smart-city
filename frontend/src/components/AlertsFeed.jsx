/**
 * AlertsFeed.jsx
 * Live scrolling list of HIGH and CRITICAL alerts.
 * New events prepend to the top with a flash animation.
 */

import { useState } from 'react'
import { formatDistanceToNow } from 'date-fns'
import { ChevronDown, ChevronUp } from 'lucide-react'
import RiskBadge from './RiskBadge.jsx'
import { EVENT_EMOJIS } from '../api.js'

export default function AlertsFeed({ alerts }) {
  const [expanded, setExpanded] = useState(null)

  if (!alerts.length) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
        No alerts yet — run a simulation
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto alerts-scroll divide-y divide-gray-800">
      {alerts.map((alert, i) => {
        const isOpen  = expanded === i
        const emoji   = EVENT_EMOJIS[alert.event_type] || '⚠️'
        const time    = alert.occurred_at
          ? formatDistanceToNow(new Date(alert.occurred_at), { addSuffix: true })
          : ''

        return (
          <div
            key={alert.id || i}
            className={`px-3 py-2 cursor-pointer transition-colors
              hover:bg-gray-800/60
              ${i === 0 ? 'animate-pulse-once' : ''}`}
            onClick={() => setExpanded(isOpen ? null : i)}
          >
            {/* Row: emoji + area + badge + chevron */}
            <div className="flex items-center gap-2">
              <span className="text-base">{emoji}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-white truncate">
                    {alert.area_name || 'Unknown'}
                  </span>
                  <RiskBadge level={alert.risk_level} score={alert.risk_score} />
                </div>
                <div className="text-xs text-gray-500 flex gap-2">
                  <span>{alert.event_type}</span>
                  <span>·</span>
                  <span>{time}</span>
                </div>
              </div>
              {isOpen
                ? <ChevronUp  size={14} className="text-gray-500 shrink-0" />
                : <ChevronDown size={14} className="text-gray-500 shrink-0" />}
            </div>

            {/* Expanded drawer */}
            {isOpen && (
              <div className="mt-2 ml-6 text-xs text-gray-400 space-y-1">
                {alert.raw_input && (
                  <p className="italic text-gray-500">"{alert.raw_input}"</p>
                )}
                {alert.explanation?.reasons?.map((r, ri) => (
                  <p key={ri}>• {r}</p>
                ))}
                <div className="flex gap-4 mt-1 text-gray-500">
                  <span>Score: <span className="text-white">{alert.risk_score?.toFixed(1)}</span></span>
                  <span>Signal: <span className="text-white">{alert.explanation?.dominant_signal || '—'}</span></span>
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}