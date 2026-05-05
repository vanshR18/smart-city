/**
 * useEvents.js
 * ────────────
 * Manages the full event state:
 *   - Initial load via REST (last 50 events from DB)
 *   - Live updates via WebSocket (new events pushed from server)
 *   - Stats polling every 30 seconds
 *
 * Keeps a capped list of MAX_EVENTS so the map doesn't explode with 10k markers.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchEvents, fetchStats, fetchAlerts } from '../api.js'

const MAX_EVENTS       = 200    // max markers on map
const STATS_POLL_MS    = 30000  // refresh stats every 30s

export function useEvents() {
  const [events,    setEvents]    = useState([])
  const [stats,     setStats]     = useState(null)
  const [alerts,    setAlerts]    = useState([])
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState(null)

  const statsTimer = useRef(null)

  // ── Initial data load ─────────────────────────────────────────────────────
  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        const [eventsData, statsData, alertsData] = await Promise.all([
          fetchEvents({ limit: 50 }),
          fetchStats(),
          fetchAlerts({ limit: 30, hours_back: 24 }),
        ])
        setEvents(eventsData.events || [])
        setStats(statsData)
        setAlerts(alertsData.alerts || [])
        setError(null)
      } catch (err) {
        setError('Could not connect to SmartCityAI backend. Is it running?')
        console.error('[useEvents] Load error:', err)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  // ── Stats polling ─────────────────────────────────────────────────────────
  useEffect(() => {
    statsTimer.current = setInterval(async () => {
      try {
        const s = await fetchStats()
        setStats(s)
      } catch (_) {}
    }, STATS_POLL_MS)
    return () => clearInterval(statsTimer.current)
  }, [])

  // ── WebSocket new-event handler ───────────────────────────────────────────
  const handleLiveEvent = useCallback((event) => {
    if (event.type !== 'new_event') return

    setEvents(prev => {
      // Prepend new event, cap at MAX_EVENTS
      const updated = [event, ...prev]
      return updated.slice(0, MAX_EVENTS)
    })

    // Add to alerts feed if HIGH or CRITICAL
    if (['HIGH', 'CRITICAL'].includes(event.risk_level)) {
      setAlerts(prev => [event, ...prev].slice(0, 50))
    }

    // Bump total_events count in stats
    setStats(prev => prev
      ? { ...prev, total_events: (prev.total_events || 0) + 1 }
      : prev
    )
  }, [])

  return { events, stats, alerts, loading, error, handleLiveEvent }
}