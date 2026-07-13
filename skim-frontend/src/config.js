// Backend base URL. Override with VITE_API_BASE in .env.
export const API_BASE =
  (import.meta.env.VITE_API_BASE || 'http://localhost:8090').replace(/\/$/, '')
