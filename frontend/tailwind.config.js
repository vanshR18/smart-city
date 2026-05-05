/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        critical: '#7f1d1d',
        high:     '#ef4444',
        medium:   '#f59e0b',
        low:      '#22c55e',
      },
    },
  },
  plugins: [],
}