// Per-session id. Lives in sessionStorage so a refresh keeps the current chat's
// document, but a new tab is a brand-new, isolated session (can't see other
// sessions' PDFs). Sent as X-Session-Id on every backend request.
const KEY = 'skim_session_id'

function makeId() {
  if (crypto && crypto.randomUUID) return crypto.randomUUID()
  return 'sess-' + Math.random().toString(36).slice(2) + Date.now().toString(36)
}

export function getSessionId() {
  let id = sessionStorage.getItem(KEY)
  if (!id) {
    id = makeId()
    sessionStorage.setItem(KEY, id)
  }
  return id
}
