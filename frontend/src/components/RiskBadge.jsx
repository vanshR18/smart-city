/**
 * RiskBadge.jsx
 * Reusable coloured pill for risk levels.
 * Used on map popups, alert cards, and the stats bar.
 */

const STYLES = {
  CRITICAL: 'bg-red-900 text-red-200 ring-1 ring-red-500',
  HIGH:     'bg-orange-900 text-orange-200 ring-1 ring-orange-500',
  MEDIUM:   'bg-yellow-900 text-yellow-200 ring-1 ring-yellow-500',
  LOW:      'bg-green-900 text-green-200 ring-1 ring-green-500',
}

export default function RiskBadge({ level, score, size = 'sm' }) {
  const style = STYLES[level] || 'bg-gray-700 text-gray-300'
  const text  = size === 'lg'
    ? `${level}${score != null ? `  ${score.toFixed(0)}` : ''}`
    : level

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full
      font-mono font-semibold tracking-wide ${style}
      ${size === 'lg' ? 'text-sm' : 'text-xs'}`}>
      {text}
    </span>
  )
}