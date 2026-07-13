import { useCallback, useEffect, useRef, useState } from 'react'
import UploadScreen from './screens/UploadScreen.jsx'
import ProcessingScreen from './screens/ProcessingScreen.jsx'
import ChatScreen from './screens/ChatScreen.jsx'
import {
  deleteDocument, listDocuments, streamChat, uploadDocumentStreaming,
} from './api.js'
import { titleFromName } from './util.js'

let _idSeq = 0
const nextId = (p) => `${p}${Date.now()}-${_idSeq++}`

const introText = (title) =>
  `All done — I've read "${title}" cover to cover. Ask me anything and I'll point you to the exact page it came from.`

const introMessage = (title) => ({
  id: 'intro', role: 'assistant', text: introText(title), done: true, sources: [],
})

// Per-document conversations persist in sessionStorage, so navigating between
// chats (or refreshing the tab) keeps each document's history for this session.
const CHATS_KEY = 'skim_conversations'
function loadConversations() {
  try {
    const data = JSON.parse(sessionStorage.getItem(CHATS_KEY) || '{}')
    // no stream can be in-flight right after a reload -> mark every message done
    for (const id of Object.keys(data)) {
      data[id] = (data[id] || []).map((m) => ({ ...m, done: true }))
    }
    return data
  } catch { return {} }
}

export default function App() {
  const [screen, setScreen] = useState('upload')
  const [fileName, setFileName] = useState('')
  const [docTitle, setDocTitle] = useState('Document')
  const [docId, setDocId] = useState(null)
  const [progress, setProgress] = useState(0)
  const [stepIndex, setStepIndex] = useState(0)
  const [processingError, setProcessingError] = useState(null)
  // docId -> that document's message list. Each chat stays isolated + persistent.
  const [conversations, setConversations] = useState(loadConversations)
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [documents, setDocuments] = useState([])
  const [isMobile, setIsMobile] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [copiedId, setCopiedId] = useState(null)

  const inputRef = useRef(null)

  // --- responsive ---
  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 860)
    onResize()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const loadDocuments = useCallback(async () => {
    setDocuments(await listDocuments())
  }, [])

  useEffect(() => { loadDocuments() }, [loadDocuments])

  // messages shown = the active document's conversation
  const messages = docId ? (conversations[docId] || []) : []

  // persist every conversation for this browser session
  useEffect(() => {
    try { sessionStorage.setItem(CHATS_KEY, JSON.stringify(conversations)) } catch { /* ignore */ }
  }, [conversations])

  // update one document's message list (targeted, so a background stream lands
  // in the right chat even if the user has switched away)
  const updateDocMessages = useCallback((targetDocId, updater) => {
    setConversations((all) => {
      const current = all[targetDocId] || []
      const next = typeof updater === 'function' ? updater(current) : updater
      return { ...all, [targetDocId]: next }
    })
  }, [])

  // --- upload -> processing -> chat ---
  const handleFile = useCallback((file) => {
    setScreen('processing')
    setFileName(file.name)
    setDocTitle(titleFromName(file.name))
    setProgress(0)
    setStepIndex(0)
    setProcessingError(null)

    uploadDocumentStreaming(file, {
      onStage: (data) => {
        // data.step is the just-completed step (0 extract, 1 chunk, 2 index)
        const next = (data.step ?? 0) + 1
        setStepIndex(next)
        setProgress(next * 25)
      },
      onDone: (data) => {
        setProgress(100)
        setStepIndex(4)
        const title = titleFromName(data.filename || file.name)
        setDocId(data.doc_id)
        setDocTitle(title)
        setTimeout(() => {
          setConversations((all) => ({ ...all, [data.doc_id]: [introMessage(title)] }))
          setScreen('chat')
          loadDocuments()
        }, 480)
      },
      onError: (msg) => setProcessingError(msg),
    })
  }, [loadDocuments])

  // --- chat ---
  const send = useCallback((preset) => {
    const text = (preset != null ? preset : input).trim()
    if (!text || streaming || !docId) return
    const activeDocId = docId
    const userId = nextId('u')
    const botId = nextId('a')
    updateDocMessages(activeDocId, (m) => [
      ...m,
      { id: userId, role: 'user', text, done: true, sources: [] },
      { id: botId, role: 'assistant', text: '', done: false, sources: [] },
    ])
    setInput('')
    setStreaming(true)
    if (inputRef.current) inputRef.current.style.height = 'auto'

    streamChat({ query: text, docId: activeDocId }, {
      onToken: (t) => updateDocMessages(activeDocId, (m) => m.map((x) => x.id === botId ? { ...x, text: x.text + t } : x)),
      onMetadata: (meta) => updateDocMessages(activeDocId, (m) => m.map((x) => x.id === botId ? { ...x, sources: meta.sources || [] } : x)),
      onDone: () => { updateDocMessages(activeDocId, (m) => m.map((x) => x.id === botId ? { ...x, done: true } : x)); setStreaming(false) },
      onError: (msg) => {
        updateDocMessages(activeDocId, (m) => m.map((x) => x.id === botId ? { ...x, text: x.text || msg, done: true } : x))
        setStreaming(false)
      },
    })
  }, [input, streaming, docId, updateDocMessages])

  const onInput = (e) => {
    setInput(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = Math.min(140, el.scrollHeight) + 'px'
  }
  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  const copyMsg = (id, text) => {
    try { navigator.clipboard.writeText(text) } catch { /* ignore */ }
    setCopiedId(id)
    setTimeout(() => setCopiedId((cur) => (cur === id ? null : cur)), 1600)
  }

  const exportChat = () => {
    const lines = messages.map((m) => (m.role === 'user' ? 'You: ' : 'Skim: ') + m.text).join('\n\n')
    const blob = new Blob([`Skim — ${docTitle}\n\n${lines}\n`], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'skim-conversation.txt'
    document.body.appendChild(a); a.click(); a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const backToUpload = () => {
    setScreen('upload')
    // keep saved conversations; just leave the current chat view
    setFileName(''); setInput(''); setSidebarOpen(false)
    setProgress(0); setStepIndex(0); setProcessingError(null); setDocId(null)
  }

  const selectDoc = (doc) => {
    const title = titleFromName(doc.filename)
    setDocId(doc.doc_id)
    setDocTitle(title)
    // restore this document's existing conversation; seed an intro only on first open
    setConversations((all) => (all[doc.doc_id] ? all : { ...all, [doc.doc_id]: [introMessage(title)] }))
    setInput(''); setStreaming(false); setSidebarOpen(false)
  }

  const toggleSidebar = () => {
    if (isMobile) setSidebarOpen((o) => !o)
    else setCollapsed(false)
  }

  // --- render ---
  const shell = { height: '100vh', width: '100%', overflow: 'hidden', position: 'relative', background: '#FBF7F0' }
  const showSuggestions = screen === 'chat' && messages.filter((m) => m.role === 'user').length === 0
  const sendDisabled = input.trim().length === 0 || streaming

  return (
    <div style={shell}>
      {screen === 'upload' && <UploadScreen onFile={handleFile} />}
      {screen === 'processing' && (
        <ProcessingScreen
          fileName={fileName} progress={progress} stepIndex={stepIndex}
          error={processingError} onRetry={backToUpload}
        />
      )}
      {screen === 'chat' && (
        <ChatScreen
          docTitle={docTitle} messages={messages} documents={documents} activeDocId={docId}
          input={input} streaming={streaming} sendDisabled={sendDisabled}
          copiedId={copiedId} showSuggestions={showSuggestions}
          isMobile={isMobile} sidebarOpen={sidebarOpen} collapsed={collapsed}
          onSend={() => send()} onInput={onInput} onKeyDown={onKeyDown}
          onSuggestion={(t) => send(t)} onCopy={copyMsg}
          onSelectDoc={selectDoc} onNewDocument={backToUpload} onExport={exportChat}
          onToggleSidebar={toggleSidebar} onCollapse={() => setCollapsed(true)}
          onCloseSidebar={() => setSidebarOpen(false)} inputRef={inputRef}
        />
      )}
    </div>
  )
}
