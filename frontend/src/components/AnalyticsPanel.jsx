/**
 * AnalyticsPanel.jsx
 * Two charts:
 *   1. Incidents by event type (bar)
 *   2. Risk score distribution (bar — bucketed into 0-25, 25-50, 50-75, 75-100)
 */

import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'
import { RISK_COLORS, EVENT_EMOJIS } from '../api.js'

// ── By event type ─────────────────────────────────────────────────────────────
function TypeChart({ events }) {
  const data = useMemo(() => {
    const counts = {}
    events.forEach(e => {
      counts[e.event_type] = (counts[e.event_type] || 0) + 1
    })
    return Object.entries(counts)
      .map(([type, count]) => ({ type, count, label: `${EVENT_EMOJIS[type] || ''} ${type}` }))
      .sort((a, b) => b.count - a.count)
  }, [events])

  if (!data.length) return <Empty />

  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
        <XAxis
          dataKey="label"
          tick={{ fill: '#9ca3af', fontSize: 9 }}
          angle={-35}
          textAnchor="end"
          interval={0}
        />
        <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} width={28} />
        <Tooltip
          contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 6 }}
          labelStyle={{ color: '#f9fafb', fontSize: 12 }}
          itemStyle={{ color: '#d1d5db', fontSize: 11 }}
        />
        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill="#3b82f6" opacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── By risk score bucket ──────────────────────────────────────────────────────
function RiskDistChart({ events }) {
  const data = useMemo(() => {
    const buckets = { 'LOW (0–35)': 0, 'MEDIUM (35–55)': 0,
                      'HIGH (55–75)': 0, 'CRITICAL (75+)': 0 }
    events.forEach(e => {
      const s = e.risk_score || 0
      if      (s < 35) buckets['LOW (0–35)']++
      else if (s < 55) buckets['MEDIUM (35–55)']++
      else if (s < 75) buckets['HIGH (55–75)']++
      else             buckets['CRITICAL (75+)']++
    })
    return Object.entries(buckets).map(([name, count]) => ({ name, count }))
  }, [events])

  const COLORS = [
    RISK_COLORS.LOW,
    RISK_COLORS.MEDIUM,
    RISK_COLORS.HIGH,
    RISK_COLORS.CRITICAL,
  ]

  return (
    <ResponsiveContainer width="100%" height={140}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
        <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 9 }} angle={-20} textAnchor="end" interval={0} />
        <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} width={28} />
        <Tooltip
          contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 6 }}
          labelStyle={{ color: '#f9fafb', fontSize: 12 }}
          itemStyle={{ color: '#d1d5db', fontSize: 11 }}
        />
        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
          {data.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function Empty() {
  return (
    <div className="h-32 flex items-center justify-center text-gray-600 text-xs">
      No data yet
    </div>
  )
}

export default function AnalyticsPanel({ events }) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Incidents by Type
        </h3>
        <TypeChart events={events} />
      </div>
      <div>
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Risk Distribution
        </h3>
        <RiskDistChart events={events} />
      </div>
    </div>
  )
}