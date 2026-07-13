// Backend API client. All requests carry the session id so the server can
// isolate this session's document(s). SSE endpoints (upload/stream, chat) are
// read via fetch + a ReadableStream reader and parsed frame-by-frame.
import { API_BASE } from './config.js'
import { getSessionId } from './session.js'

function headers(extra = {}) {
  return { 'X-Session-Id': getSessionId(), ...extra }
}

// Parse an SSE response body, invoking onEvent({ event, data }) per frame.
async function readSSE(response, onEvent) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = 'message'
      let dataStr = ''
      for (const rawLine of frame.split('\n')) {
        const line = rawLine.replace(/\r$/, '')
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataStr += line.slice(5).trim()
      }
      if (!dataStr) continue
      let data
      try { data = JSON.parse(dataStr) } catch { data = dataStr }
      onEvent({ event, data })
    }
  }
}

// --- documents ---
export async function uploadDocumentStreaming(file, { onStage, onDone, onError } = {}) {
  const form = new FormData()
  form.append('file', file)
  let res
  try {
    res = await fetch(`${API_BASE}/documents/upload/stream`, {
      method: 'POST', headers: headers(), body: form,
    })
  } catch (e) {
    onError && onError('Could not reach the server. Is the backend running?')
    return
  }
  if (!res.ok || !res.body) {
    let detail = `Upload failed (HTTP ${res.status}).`
    try { detail = (await res.json()).detail || detail } catch { /* ignore */ }
    onError && onError(detail)
    return
  }
  await readSSE(res, ({ event, data }) => {
    if (event === 'stage') onStage && onStage(data)
    else if (event === 'done') onDone && onDone(data)
    else if (event === 'error') onError && onError(data.detail || 'Failed to process the PDF.')
  })
}

export async function listDocuments() {
  const res = await fetch(`${API_BASE}/documents`, { headers: headers() })
  if (!res.ok) return []
  const json = await res.json()
  return json.documents || []
}

export async function deleteDocument(docId) {
  const res = await fetch(`${API_BASE}/documents/${docId}`, {
    method: 'DELETE', headers: headers(),
  })
  return res.ok
}

// --- chat ---
export async function streamChat({ query, docId }, { onToken, onMetadata, onDone, onError } = {}) {
  let res
  try {
    res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ query, doc_id: docId }),
    })
  } catch (e) {
    onError && onError('Could not reach the server.')
    return
  }
  if (!res.ok || !res.body) {
    onError && onError(`Chat failed (HTTP ${res.status}).`)
    return
  }
  await readSSE(res, ({ event, data }) => {
    if (event === 'token') onToken && onToken(data.text || '')
    else if (event === 'metadata') onMetadata && onMetadata(data)
    else if (event === 'done') onDone && onDone(data)
  })
}
