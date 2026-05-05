/**
 * useWebSocket.js
 * ───────────────
 * Connects to the FastAPI WebSocket endpoint and keeps the connection alive.
 * Auto-reconnects after unexpected drops (network blip, server restart).
 *
 * Usage:
 *   const { connected, lastEvent } = useWebSocket(onEvent)
 */

import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL            = 'ws://localhost:8000/ws/live'
const RECONNECT_DELAY   = 3000   // ms before reconnect attempt
const MAX_RECONNECTS    = 10

export function useWebSocket(onEvent) {
  const [connected,    setConnected]    = useState(false)
  const [lastEvent,    setLastEvent]    = useState(null)
  const [reconnects,   setReconnects]   = useState(0)

  const wsRef          = useRef(null)
  const reconnectTimer = useRef(null)
  const isMounted      = useRef(true)

  const connect = useCallback(() => {
    if (!isMounted.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        if (!isMounted.current) return
        setConnected(true)
        setReconnects(0)
        console.log('[WS] Connected to SmartCityAI live feed')
      }

      ws.onmessage = (e) => {
        if (!isMounted.current) return
        try {
          const data = JSON.parse(e.data)
          setLastEvent(data)
          if (data.type === 'new_event' && onEvent) {
            onEvent(data)
          }
        } catch (err) {
          console.warn('[WS] Failed to parse message:', err)
        }
      }

      ws.onclose = (e) => {
        if (!isMounted.current) return
        setConnected(false)
        console.warn(`[WS] Disconnected (code ${e.code}). Reconnecting...`)

        setReconnects(prev => {
          if (prev < MAX_RECONNECTS) {
            reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY)
            return prev + 1
          }
          console.error('[WS] Max reconnects reached.')
          return prev
        })
      }

      ws.onerror = (err) => {
        console.error('[WS] Error:', err)
        ws.close()
      }

    } catch (err) {
      console.error('[WS] Failed to create WebSocket:', err)
    }
  }, [onEvent])

  useEffect(() => {
    isMounted.current = true
    connect()
    return () => {
      isMounted.current = false
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { connected, lastEvent, reconnects }
}