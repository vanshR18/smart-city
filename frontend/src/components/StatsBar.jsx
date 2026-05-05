/**
 * StatsBar.jsx
 * ────────────
 * Full-width metrics bar shown below the Header.
 * Pulls data from two sources:
 *   - props.stats       → /stats endpoint  (event counts by risk level)
 *   - props.alertStats  → /alerts/stats    (alert engine runtime metrics)
 *
 * Designed to sit between Header and the map/sidebar split.
 * Collapses gracefully when data is null (loading state).
 *
 * Usage in App.jsx:
 *   <StatsBar stats={stats} alertStats={alertStats} />
 */

import { useEffect, useState } from 'react'
import { fetchAlertStats } from '../api.js'

// ── Mini sub-components ───────────────────────────────────────────────────────

function StatCard({ label, value, sub, color, pulse, bar }) {
  const colorMap = {
    critical: 'text-red-400',
    high:     'text-orange-400',
    medium:   'text-yellow-400',
    low:      'text-green-400',
    blue:     'text-blue-400',
    default:  'text-white',
  }
  const barColorMap = {
    critical: 'bg-red-500',
    high:     'bg-orange-500',
    medium:   'bg-yellow-500',
    low:      'bg-green-500',
    blue:     'bg-blue-500',
    default:  'bg-gray-500',
  }

  const valueColor  = colorMap[color]  || colorMap.default
  const barColor    = barColorMap[color] || barColorMap.default

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 min-w-0">
      <div className="flex items-center gap-1.5 mb-1">
        {pulse && (
          <span className={`inline-block w-1.5 h-1.5 rounded-full shrink-0
            ${color === 'critical' ? 'bg-red-400 animate-pulse' : 'bg-green-400 animate-pulse'}`}
          />
        )}
        <span className="text-xs text-gray-400 truncate">{label}</span>
      </div>

      <div className={`text-lg font-semibold leading-none ${valueColor}`}>
        {value ?? '—'}
      </div>

      {sub && (
        <div className="text-xs text-gray-500 mt-0.5 truncate">{sub}</div>
      )}

      {bar != null && (
        <div className="mt-1.5 h-0.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${barColor}`}
            style={{ width: `${Math.min(Math.max(bar, 0), 100)}%` }}
          />
        </div>
      )}
    </div>
  )
}

function SectionLabel({ children }) {
  return (
    <div className="text-xs font-medium text-gray-600 uppercase tracking-widest
                    self-end pb-1 pl-0.5 whitespace-nowrap">
      {children}
    </div>
  )
}

// ── Uptime calculator ─────────────────────────────────────────────────────────
function useUptime(startedAt) {
  const [display, setDisplay] = useState('—')

  useEffect(() => {
    if (!startedAt) return
    const update = () => {
      const secs = Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000)
      if (secs < 60)   return setDisplay(`${secs}s`)
      if (secs < 3600) return setDisplay(`${Math.floor(secs / 60)}m ${secs % 60}s`)
      const h = Math.floor(secs / 3600)
      const m = Math.floor((secs % 3600) / 60)
      setDisplay(`${h}h ${m}m`)
    }
    update()
    const t = setInterval(update, 10_000)
    return () => clearInterval(t)
  }, [startedAt])

  return display
}

// ── Main component ────────────────────────────────────────────────────────────
export default function StatsBar({ stats }) {
  const [alertStats, setAlertStats] = useState(null)
  const uptime = useUptime(alertStats?.started_at)

  // Poll alert stats every 15 seconds independently
  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchAlertStats()
        setAlertStats(data)
      } catch (_) {}
    }
    load()
    const t = setInterval(load, 15_000)
    return () => clearInterval(t)
  }, [])

  // Derived values
  const total    = stats?.total_events  ?? 0
  const active   = stats?.active_events ?? 0
  const critical = stats?.critical      ?? 0
  const high     = stats?.high          ?? 0

  // Medium and Low: total − critical − high − normal
  // We don't have exact counts from /stats, so derive from alert engine
  const eventsProcessed = alertStats?.events_processed ?? 0
  const alertsSent      = alertStats?.alerts_sent      ?? 0
  const telegramSent    = alertStats?.telegram_sent    ?? 0
  const wsConnections   = alertStats?.active_ws_connections ?? 0
  const nlpMethod       = alertStats?.nlp_method ?? null

  // Alert rate: what % of processed events triggered an alert
  const alertRate = eventsProcessed > 0
    ? Math.round((alertsSent / eventsProcessed) * 100)
    : 0

  // Average risk score proxy: CRITICAL events count as 87, HIGH as 65
  // (rough estimate when we don't have the full distribution)
  const avgScore = total > 0
    ? Math.round(((critical * 87) + (high * 65)) / Math.max(critical + high, 1))
    : 0

  // Active as % of total (for bar)
  const activeBar = total > 0 ? Math.round((active / total) * 100) : 0

  return (
    <div className="shrink-0 px-4 py-2 bg-gray-900 border-b border-gray-800">
      <div className="grid gap-2"
           style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))' }}>

        {/* ── Overview ─────────────────────────────────────────── */}
        <SectionLabel>Overview</SectionLabel>

        <StatCard
          label="Total events"
          value={total.toLocaleString()}
          sub="all time"
          pulse
        />
        <StatCard
          label="Active"
          value={active.toLocaleString()}
          sub="unresolved"
          bar={activeBar}
          color="blue"
        />
        <StatCard
          label="Alert rate"
          value={`${alertRate}%`}
          sub="above threshold"
          bar={alertRate}
          color={alertRate > 50 ? 'high' : alertRate > 25 ? 'medium' : 'low'}
        />
        <StatCard
          label="Avg score"
          value={avgScore || '—'}
          sub="/ 100 estimate"
          bar={avgScore}
          color={avgScore >= 75 ? 'critical' : avgScore >= 55 ? 'high' : 'medium'}
        />

        {/* ── By risk level ─────────────────────────────────────── */}
        <SectionLabel>Risk levels</SectionLabel>

        <StatCard
          label="Critical"
          value={critical}
          sub="score ≥ 75"
          color="critical"
          pulse={critical > 0}
        />
        <StatCard
          label="High"
          value={high}
          sub="score 55–74"
          color="high"
        />
        <StatCard
          label="Events processed"
          value={eventsProcessed.toLocaleString()}
          sub="this session"
        />
        <StatCard
          label="Alerts fired"
          value={alertsSent}
          sub={`${telegramSent} Telegram`}
          color={alertsSent > 0 ? 'high' : 'default'}
        />

        {/* ── System ───────────────────────────────────────────── */}
        <SectionLabel>System</SectionLabel>

        <StatCard
          label="WS clients"
          value={wsConnections}
          sub="dashboards open"
          color={wsConnections > 0 ? 'low' : 'default'}
          pulse={wsConnections > 0}
        />
        <StatCard
          label="NLP model"
          value={nlpMethod ?? (alertStats ? 'rule-based' : '—')}
          sub={nlpMethod === 'distilbert' ? 'fine-tuned' : 'train to upgrade'}
          color={nlpMethod === 'distilbert' ? 'low' : 'default'}
        />
        <StatCard
          label="Uptime"
          value={uptime}
          sub="since start"
        />
        <StatCard
          label="Threshold"
          value={alertStats?.alert_threshold ?? 55}
          sub="min alert score"
          color="medium"
        />

      </div>
    </div>
  )
}