/**
 * api.js
 * ──────
 * All backend calls in one place.
 * Using /api prefix because Vite proxies it to http://localhost:8000.
 * In production, replace BASE_URL with your deployed Railway/Render URL.
 */

import axios from 'axios'

const BASE_URL = '/api'

const http = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
})

// ── Events ────────────────────────────────────────────────────────────────────
export const fetchEvents = (params = {}) =>
  http.get('/events', { params }).then(r => r.data)

export const fetchStats = () =>
  http.get('/stats').then(r => r.data)

// ── Alerts ────────────────────────────────────────────────────────────────────
export const fetchAlerts = (params = {}) =>
  http.get('/alerts', { params }).then(r => r.data)

export const fetchAlertStats = () =>
  http.get('/alerts/stats').then(r => r.data)

export const sendTestAlert = () =>
  http.post('/alerts/test').then(r => r.data)

// ── Heatmap & Risk ────────────────────────────────────────────────────────────
export const fetchHeatmap = (hoursBack = 24) =>
  http.get('/risk/heatmap', { params: { hours_back: hoursBack } }).then(r => r.data)

export const fetchTimeProfile = (eventType = 'overall') =>
  http.get('/risk/time-profile', { params: { event_type: eventType } }).then(r => r.data)

// ── NLP prediction ────────────────────────────────────────────────────────────
export const predictText = (text) =>
  http.post('/predict/text', { text }).then(r => r.data)

// ── Simulation ────────────────────────────────────────────────────────────────
export const runSimulation = (n = 20) =>
  http.post('/simulate/batch', null, { params: { n } }).then(r => r.data)

export const seedHistorical = () =>
  http.post('/simulate/seed-historical').then(r => r.data)

// ── Health ────────────────────────────────────────────────────────────────────
export const fetchHealth = () =>
  http.get('/health').then(r => r.data)

// ── Risk level colours & labels ───────────────────────────────────────────────
export const RISK_COLORS = {
  CRITICAL: '#ef4444',
  HIGH:     '#f97316',
  MEDIUM:   '#f59e0b',
  LOW:      '#22c55e',
  NORMAL:   '#6b7280',
}

export const EVENT_EMOJIS = {
  ACCIDENT: '🚗',
  FIRE:     '🔥',
  FLOOD:    '🌊',
  CRIME:    '🚨',
  CROWD:    '👥',
  MEDICAL:  '🏥',
  NORMAL:   '✅',
}