/**
 * LiveMap.jsx
 * ───────────
 * Leaflet map centred on Lucknow.
 * Shows:
 *   - Coloured CircleMarkers for each incident (colour = risk level)
 *   - Popup with event details on click
 *   - Heatmap layer (loaded separately from /risk/heatmap)
 *
 * Why CircleMarkers over default pins?
 *   - Render faster with many points (no DOM icon elements)
 *   - Colour encodes risk level instantly
 *   - Radius encodes risk score (bigger = higher score)
 */

import { useEffect, useState, useRef } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet'
import { fetchHeatmap } from '../api.js'
import { RISK_COLORS, EVENT_EMOJIS } from '../api.js'
import RiskBadge from './RiskBadge.jsx'

// Lucknow centre coordinates
const LUCKNOW = [26.8467, 80.9462]

// ── Heatmap layer using Leaflet.heat ──────────────────────────────────────────
// Leaflet.heat is loaded via CDN in index.html (not available as npm package).
// This component adds/updates the heat layer whenever heatData changes.
function HeatmapLayer({ heatData }) {
  const map      = useMap()
  const layerRef = useRef(null)

  useEffect(() => {
    if (!window.L?.heatLayer || !heatData?.length) return

    // Remove old layer
    if (layerRef.current) {
      map.removeLayer(layerRef.current)
    }

    // [[lat, lon, intensity], ...]  intensity is already 0–1 from API
    layerRef.current = window.L.heatLayer(heatData, {
      radius:  35,
      blur:    25,
      maxZoom: 14,
      max:     1.0,
      gradient: {
        0.0: '#22c55e',   // LOW
        0.4: '#f59e0b',   // MEDIUM
        0.7: '#ef4444',   // HIGH
        1.0: '#7f1d1d',   // CRITICAL
      },
    }).addTo(map)

    return () => {
      if (layerRef.current) map.removeLayer(layerRef.current)
    }
  }, [heatData, map])

  return null
}

// ── Marker radius by risk score ───────────────────────────────────────────────
function markerRadius(score) {
  if (score >= 75) return 10
  if (score >= 55) return 8
  if (score >= 35) return 6
  return 5
}

// ── Main component ────────────────────────────────────────────────────────────
export default function LiveMap({ events }) {
  const [heatData,     setHeatData]     = useState([])
  const [showHeatmap,  setShowHeatmap]  = useState(true)
  const [showMarkers,  setShowMarkers]  = useState(true)
  const [selectedType, setSelectedType] = useState('ALL')

  // Load heatmap data from API
  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchHeatmap(24)
        setHeatData(data.leaflet_heat || [])
      } catch (e) {
        console.warn('Heatmap load failed:', e)
      }
    }
    load()
    const t = setInterval(load, 60_000)   // refresh every minute
    return () => clearInterval(t)
  }, [])

  // Filter events by type
  const EVENT_TYPES = ['ALL', 'ACCIDENT', 'FIRE', 'FLOOD', 'CRIME', 'CROWD', 'MEDICAL']
  const filtered = selectedType === 'ALL'
    ? events
    : events.filter(e => e.event_type === selectedType)

  return (
    <div className="relative h-full flex flex-col">

      {/* Map controls toolbar */}
      <div className="absolute top-2 left-2 z-[1000] flex flex-wrap gap-1.5">
        {/* Layer toggles */}
        <Toggle active={showHeatmap}  onClick={() => setShowHeatmap(v => !v)}  label="Heatmap" />
        <Toggle active={showMarkers}  onClick={() => setShowMarkers(v => !v)}  label="Markers" />

        {/* Event type filter */}
        <div className="flex bg-gray-900/90 rounded-md overflow-hidden
                        border border-gray-700 text-xs">
          {EVENT_TYPES.map(t => (
            <button
              key={t}
              onClick={() => setSelectedType(t)}
              className={`px-2 py-1 transition-colors
                ${selectedType === t
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white'}`}
            >
              {t === 'ALL' ? 'ALL' : `${EVENT_EMOJIS[t]} ${t}`}
            </button>
          ))}
        </div>
      </div>

      {/* Event count badge */}
      <div className="absolute top-2 right-2 z-[1000] bg-gray-900/90
                      border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-300">
        {filtered.length} events
      </div>

      {/* Leaflet map */}
      <MapContainer
        center={LUCKNOW}
        zoom={12}
        className="h-full w-full"
        zoomControl={false}
      >
        {/* Dark tile layer — CartoDB Dark Matter */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='© <a href="https://carto.com/">CARTO</a>'
          maxZoom={19}
        />

        {/* Heatmap */}
        {showHeatmap && <HeatmapLayer heatData={heatData} />}

        {/* Incident markers */}
        {showMarkers && filtered.map((event, i) => {
          if (!event.latitude || !event.longitude) return null
          const color  = RISK_COLORS[event.risk_level] || '#6b7280'
          const radius = markerRadius(event.risk_score || 50)
          const emoji  = EVENT_EMOJIS[event.event_type] || '⚠️'

          return (
            <CircleMarker
              key={event.id || i}
              center={[event.latitude, event.longitude]}
              radius={radius}
              pathOptions={{
                color:       color,
                fillColor:   color,
                fillOpacity: 0.75,
                weight:      1.5,
              }}
            >
              <Popup>
                <div className="text-sm min-w-[180px]">
                  <div className="font-bold text-base mb-1">
                    {emoji} {event.event_type}
                  </div>
                  <div className="text-gray-600 mb-1">📍 {event.area_name}</div>
                  <RiskBadge level={event.risk_level} score={event.risk_score} size="lg" />
                  {event.raw_input && (
                    <p className="mt-2 text-xs text-gray-500 italic line-clamp-3">
                      "{event.raw_input}"
                    </p>
                  )}
                  {event.explanation?.reasons?.length > 0 && (
                    <p className="mt-1 text-xs text-gray-600">
                      {event.explanation.reasons.join(' · ')}
                    </p>
                  )}
                  <div className="mt-1 text-xs text-gray-400">
                    {event.occurred_at
                      ? new Date(event.occurred_at).toLocaleTimeString()
                      : ''}
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          )
        })}
      </MapContainer>
    </div>
  )
}

function Toggle({ active, onClick, label }) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-1 rounded-md text-xs font-medium border transition-colors
        ${active
          ? 'bg-blue-600 border-blue-500 text-white'
          : 'bg-gray-900/90 border-gray-700 text-gray-400 hover:text-white'}`}
    >
      {label}
    </button>
  )
}