import { useEffect, useRef } from 'react'
import LogoMark from '../components/LogoMark.jsx'
import {
  ArrowUpIcon, ChevronsLeftIcon, DownloadIcon, HamburgerIcon, PlusIcon,
} from '../components/icons.jsx'
import { titleFromName, uniquePages } from '../util.js'
import Markdown from '../components/Markdown.jsx'

const SUGGESTIONS = [
  'Give me a 3-bullet summary',
  'What are the key dates or numbers?',
  'Explain the most important section',
]

// ---------------- Sidebar ----------------
function Sidebar({ isMobile, sidebarOpen, collapsed, documents, activeDocId,
  onSelectDoc, onNewDocument, onCollapse }) {
  const base = { flex: 'none', width: '270px', background: '#F6F0E6', borderRight: '1px solid #EDE6DB', padding: '18px 16px', display: 'flex', flexDirection: 'column', overflow: 'hidden', whiteSpace: 'nowrap' }
  let style
  if (isMobile) {
    style = { ...base, position: 'absolute', top: 0, bottom: 0, left: 0, zIndex: 40, boxShadow: '0 0 40px rgba(43,38,34,.25)', transform: sidebarOpen ? 'translateX(0)' : 'translateX(-100%)', transition: 'transform .28s cubic-bezier(.4,0,.2,1)' }
  } else if (collapsed) {
    style = { ...base, width: '0px', padding: '18px 0', borderRight: 'none', transition: 'width .28s cubic-bezier(.4,0,.2,1), padding .28s ease' }
  } else {
    style = { ...base, transition: 'width .28s cubic-bezier(.4,0,.2,1), padding .28s ease' }
  }

  return (
    <aside style={style}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '9px', padding: '4px 4px 18px' }}>
        <LogoMark size={28} radius={9} fontSize={16} shadow="none" />
        <span style={{ fontFamily: "'Bricolage Grotesque'", fontWeight: 700, fontSize: '17px', flex: 1 }}>Skim</span>
        {!isMobile && (
          <button onClick={onCollapse} title="Collapse sidebar" style={{ flex: 'none', width: '30px', height: '30px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1.5px solid #EDE6DB', background: '#fff', borderRadius: '9px', cursor: 'pointer', color: '#8A8178' }}>
            <ChevronsLeftIcon size={16} />
          </button>
        )}
      </div>

      <button onClick={onNewDocument} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '7px', width: '100%', background: '#FF7A59', color: '#fff', border: 'none', borderRadius: '13px', padding: '11px', cursor: 'pointer', fontFamily: "'Nunito'", fontWeight: 700, fontSize: '14.5px', boxShadow: '0 3px 0 #e8613f', marginBottom: '20px', whiteSpace: 'nowrap' }}>
        <PlusIcon size={17} strokeWidth={2.4} />New document
      </button>

      <div style={{ fontSize: '11px', fontWeight: 800, color: '#B0A79A', textTransform: 'uppercase', letterSpacing: '.09em', padding: '0 4px 8px' }}>Recent</div>
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '3px' }}>
        {documents.map((doc) => {
          const active = doc.doc_id === activeDocId
          return (
            <button key={doc.doc_id} onClick={() => onSelectDoc(doc)} style={{ display: 'flex', alignItems: 'center', gap: '9px', width: '100%', textAlign: 'left', background: active ? '#FFF1EA' : 'transparent', border: 'none', cursor: 'pointer', borderRadius: '11px', padding: '9px 11px', fontFamily: "'Nunito'", fontWeight: active ? 700 : 600, fontSize: '14px', color: active ? '#e8613f' : '#5C554D' }}>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{titleFromName(doc.filename)}</span>
            </button>
          )
        })}
      </div>
      <div style={{ borderTop: '1px solid #EDE6DB', paddingTop: '12px', marginTop: '8px', fontSize: '12px', color: '#B0A79A', lineHeight: 1.5 }}>Open access · anyone with the link can use Skim. No account, no history saved.</div>
    </aside>
  )
}

// ---------------- Header ----------------
function Header({ docTitle, showMenuButton, onToggleSidebar, onExport }) {
  return (
    <header style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: '12px', padding: '14px 20px', borderBottom: '1px solid #EDE6DB', background: 'rgba(251,247,240,.85)', backdropFilter: 'blur(8px)' }}>
      {showMenuButton && (
        <button onClick={onToggleSidebar} title="Show sidebar" style={{ flex: 'none', width: '38px', height: '38px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1.5px solid #EDE6DB', background: '#fff', borderRadius: '11px', cursor: 'pointer', color: '#2B2622' }}>
          <HamburgerIcon size={18} />
        </button>
      )}
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ fontFamily: "'Bricolage Grotesque'", fontWeight: 700, fontSize: '15.5px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{docTitle}</div>
        <div style={{ fontSize: '12.5px', color: '#2FB8A6', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: '#2FB8A6', display: 'inline-block' }} />Indexed · ready to answer
        </div>
      </div>
      <button onClick={onExport} style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: '7px', border: '1.5px solid #EDE6DB', background: '#fff', borderRadius: '11px', padding: '9px 14px', cursor: 'pointer', fontFamily: "'Nunito'", fontWeight: 700, fontSize: '13.5px', color: '#2B2622' }}>
        <DownloadIcon size={16} />Export
      </button>
    </header>
  )
}

// ---------------- Message row ----------------
function MessageRow({ msg, copied, onCopy }) {
  const isUser = msg.role === 'user'
  const isAssistant = !isUser
  const rowStyle = { display: 'flex', gap: '12px', alignItems: 'flex-start', justifyContent: isUser ? 'flex-end' : 'flex-start', animation: 'skimUp .35s ease both' }
  const colStyle = { display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start', maxWidth: '82%', minWidth: 0 }
  const bubbleStyle = isUser
    ? { background: '#FF7A59', color: '#fff', padding: '12px 17px', borderRadius: '20px 20px 6px 20px', fontSize: '15.5px', fontWeight: 600, lineHeight: 1.5, boxShadow: '0 6px 16px -8px rgba(255,122,89,.6)', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }
    : { background: '#fff', color: '#2B2622', padding: '13px 18px', borderRadius: '6px 20px 20px 20px', fontSize: '15.5px', fontWeight: 500, lineHeight: 1.6, border: '1.5px solid #EDE6DB', boxShadow: '0 6px 18px -14px rgba(43,38,34,.35)', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }
  const showCaret = isAssistant && !msg.done
  const showActions = isAssistant && msg.done && msg.text.length > 0
  const pages = isAssistant && msg.done ? uniquePages(msg.sources) : []

  return (
    <div style={rowStyle}>
      {isAssistant && (
        <div style={{ width: '32px', height: '32px', flex: 'none', borderRadius: '10px', background: '#FF7A59', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: "'Bricolage Grotesque'", fontWeight: 800, fontSize: '17px', boxShadow: '0 3px 8px -3px rgba(255,122,89,.6)' }}>S</div>
      )}
      <div style={colStyle}>
        <div style={bubbleStyle}>
          {isAssistant ? <Markdown text={msg.text} /> : msg.text}
          {showCaret && <span style={{ display: 'inline-block', width: '8px', height: '16px', background: '#2FB8A6', borderRadius: '2px', marginLeft: '3px', verticalAlign: '-2px', animation: 'skimBlink 1s step-end infinite' }} />}
        </div>
        {pages.length > 0 && (
          <div style={{ marginTop: '6px', fontSize: '12.5px', fontWeight: 700, color: '#B0A79A' }}>
            Sources: {pages.map((p) => `p. ${p}`).join(' · ')}
          </div>
        )}
        {showActions && (
          <button onClick={() => onCopy(msg.id, msg.text)} style={{ marginTop: '7px', display: 'inline-flex', alignItems: 'center', gap: '5px', background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: "'Nunito'", fontWeight: 700, fontSize: '12.5px', color: copied ? '#2FB8A6' : '#B0A79A', padding: '2px 2px' }}>
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
      </div>
    </div>
  )
}

// ---------------- Composer ----------------
function Composer({ input, onInput, onKeyDown, onSend, sendDisabled, inputRef }) {
  const sendBtnStyle = {
    flex: 'none', width: '40px', height: '40px', borderRadius: '13px', border: 'none',
    cursor: sendDisabled ? 'default' : 'pointer', fontSize: '16px',
    background: sendDisabled ? '#F0EADF' : '#FF7A59', color: sendDisabled ? '#C9BEAD' : '#fff',
    boxShadow: sendDisabled ? 'none' : '0 3px 0 #e8613f', transition: 'all .15s',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  }
  return (
    <div style={{ flex: 'none', padding: '12px 20px 18px', background: 'linear-gradient(to top,#FBF7F0 70%,rgba(251,247,240,0))' }}>
      <div style={{ maxWidth: '760px', margin: '0 auto', display: 'flex', alignItems: 'flex-end', gap: '10px', background: '#fff', border: '1.5px solid #EDE6DB', borderRadius: '18px', padding: '8px 8px 8px 18px', boxShadow: '0 10px 28px -18px rgba(43,38,34,.3)' }}>
        <textarea ref={inputRef} value={input} onInput={onInput} onKeyDown={onKeyDown} rows={1} placeholder="Ask anything about this document…" style={{ flex: 1, border: 'none', outline: 'none', resize: 'none', background: 'transparent', fontFamily: "'Nunito'", fontWeight: 500, fontSize: '15.5px', color: '#2B2622', lineHeight: 1.45, maxHeight: '140px', padding: '8px 0' }} />
        <button onClick={onSend} disabled={sendDisabled} style={sendBtnStyle}>
          <ArrowUpIcon size={18} strokeWidth={2.2} />
        </button>
      </div>
      <div style={{ maxWidth: '760px', margin: '8px auto 0', textAlign: 'center', fontSize: '12px', color: '#B0A79A' }}>Skim can make mistakes — answers cite the pages they came from.</div>
    </div>
  )
}

// ---------------- ChatScreen ----------------
export default function ChatScreen(props) {
  const {
    docTitle, messages, documents, activeDocId, input, streaming, sendDisabled,
    copiedId, showSuggestions, isMobile, sidebarOpen, collapsed,
    onSend, onInput, onKeyDown, onSuggestion, onCopy, onSelectDoc, onNewDocument,
    onExport, onToggleSidebar, onCollapse, onCloseSidebar, inputRef,
  } = props

  const scrollRef = useRef(null)
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  return (
    <div style={{ height: '100%', display: 'flex', position: 'relative' }}>
      {isMobile && sidebarOpen && (
        <div onClick={onCloseSidebar} style={{ position: 'absolute', inset: 0, background: 'rgba(43,38,34,.4)', zIndex: 30 }} />
      )}

      <Sidebar
        isMobile={isMobile} sidebarOpen={sidebarOpen} collapsed={collapsed}
        documents={documents} activeDocId={activeDocId}
        onSelectDoc={onSelectDoc} onNewDocument={onNewDocument} onCollapse={onCollapse}
      />

      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Header
          docTitle={docTitle}
          showMenuButton={isMobile || collapsed}
          onToggleSidebar={onToggleSidebar}
          onExport={onExport}
        />

        <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: '26px 20px 8px' }}>
          <div style={{ maxWidth: '760px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {messages.map((msg) => (
              <MessageRow key={msg.id} msg={msg} copied={copiedId === msg.id} onCopy={onCopy} />
            ))}

            {showSuggestions && (
              <div style={{ paddingLeft: '44px', display: 'flex', flexDirection: 'column', gap: '9px', animation: 'skimUp .4s ease both' }}>
                <div style={{ fontSize: '12.5px', fontWeight: 800, color: '#B0A79A', textTransform: 'uppercase', letterSpacing: '.07em' }}>Try asking</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '9px' }}>
                  {SUGGESTIONS.map((text) => (
                    <button key={text} onClick={() => onSuggestion(text)} style={{ textAlign: 'left', background: '#fff', border: '1.5px solid #EDE6DB', borderRadius: '13px', padding: '10px 15px', cursor: 'pointer', fontFamily: "'Nunito'", fontWeight: 600, fontSize: '14px', color: '#2B2622', boxShadow: '0 2px 5px -3px rgba(43,38,34,.2)' }}>{text}</button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <Composer
          input={input} onInput={onInput} onKeyDown={onKeyDown} onSend={onSend}
          sendDisabled={sendDisabled} inputRef={inputRef}
        />
      </div>
    </div>
  )
}
