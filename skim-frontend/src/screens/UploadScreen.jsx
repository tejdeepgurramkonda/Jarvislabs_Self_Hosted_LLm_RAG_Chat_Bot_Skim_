import { useRef, useState } from 'react'
import LogoMark from '../components/LogoMark.jsx'
import { DocumentIcon, UploadArrowIcon } from '../components/icons.jsx'

// Screen 1 — Upload. Styles ported verbatim from the prototype; the "or try a
// sample" section is intentionally removed (real backend, no bundled samples).
export default function UploadScreen({ onFile }) {
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)

  const openPicker = () => inputRef.current && inputRef.current.click()
  const onFileChange = (e) => {
    const f = e.target.files && e.target.files[0]
    if (f) onFile(f)
    e.target.value = '' // allow re-selecting the same file
  }
  const onDrop = (e) => {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files && e.dataTransfer.files[0]
    if (f) onFile(f)
  }
  const onDragOver = (e) => { e.preventDefault(); if (!drag) setDrag(true) }
  const onDragLeave = (e) => { e.preventDefault(); setDrag(false) }

  const dropZoneStyle = {
    width: '100%', cursor: 'pointer', textAlign: 'center', padding: '38px 24px',
    background: drag ? '#FFF1EA' : '#fff',
    border: '2.5px dashed ' + (drag ? '#FF7A59' : '#E3D9C9'),
    borderRadius: '24px',
    boxShadow: drag ? '0 18px 40px -20px rgba(255,122,89,.5)' : '0 12px 34px -22px rgba(43,38,34,.3)',
    transition: 'all .18s ease', transform: drag ? 'scale(1.01)' : 'none',
  }

  return (
    <div style={{ height: '100%', overflowY: 'auto', display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative' }}>
      <div style={{ position: 'absolute', top: '-120px', left: '-80px', width: '340px', height: '340px', background: 'radial-gradient(circle,#FFE0D4 0%,rgba(255,224,212,0) 70%)', pointerEvents: 'none' }} />
      <div style={{ position: 'absolute', top: '60px', right: '-100px', width: '380px', height: '380px', background: 'radial-gradient(circle,#D8F1EC 0%,rgba(216,241,236,0) 70%)', pointerEvents: 'none' }} />

      <div style={{ width: '100%', maxWidth: '720px', padding: '26px 22px 60px', display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative', zIndex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '11px', alignSelf: 'flex-start', marginBottom: 'min(9vh,86px)' }}>
          <LogoMark size={34} radius={11} fontSize={19} shadow="0 4px 0 #e8613f" />
          <span style={{ fontFamily: "'Bricolage Grotesque'", fontWeight: 700, fontSize: '20px', letterSpacing: '-.02em' }}>Skim</span>
          <span style={{ fontSize: '12px', fontWeight: 700, color: '#2FB8A6', background: '#DDF3EF', padding: '3px 9px', borderRadius: '20px', marginLeft: '2px' }}>no login needed</span>
        </div>

        <div style={{ width: '96px', height: '96px', background: 'linear-gradient(135deg,#FF9576,#FF7A59)', animation: 'skimBlob 7s ease-in-out infinite,skimFloat 4s ease-in-out infinite', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px', boxShadow: '0 14px 30px -10px rgba(255,122,89,.6)' }}>
          <DocumentIcon size={44} stroke="#fff" strokeWidth={1.8} />
        </div>

        <h1 style={{ fontFamily: "'Bricolage Grotesque'", fontWeight: 800, fontSize: 'clamp(30px,6vw,46px)', lineHeight: 1.03, letterSpacing: '-.03em', textAlign: 'center', margin: '0 0 12px', maxWidth: '14ch' }}>Drop a PDF. Get answers.</h1>
        <p style={{ fontSize: 'clamp(15px,2.4vw,18px)', lineHeight: 1.5, color: '#8A8178', textAlign: 'center', margin: '0 0 34px', maxWidth: '40ch' }}>Medical, legal, food, tech — any document. Skim reads every page so you can just ask.</p>

        <div onClick={openPicker} onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave} style={dropZoneStyle}>
          <div style={{ width: '58px', height: '58px', borderRadius: '18px', background: '#FFE9E1', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
            <UploadArrowIcon size={26} stroke="#FF7A59" strokeWidth={2} />
          </div>
          <div style={{ fontFamily: "'Bricolage Grotesque'", fontWeight: 700, fontSize: '19px', marginBottom: '5px' }}>Drop your PDF here</div>
          <div style={{ fontSize: '14px', color: '#8A8178', marginBottom: '18px' }}>or click to browse — single PDF, up to 40MB</div>
          <span style={{ display: 'inline-block', background: '#FF7A59', color: '#fff', fontWeight: 700, fontSize: '15px', padding: '11px 24px', borderRadius: '14px', boxShadow: '0 4px 0 #e8613f' }}>Choose a file</span>
          <input ref={inputRef} type="file" accept="application/pdf" onChange={onFileChange} style={{ display: 'none' }} />
        </div>
      </div>
    </div>
  )
}
