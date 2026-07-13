import { MagnifierIcon } from '../components/icons.jsx'

// Screen 2 — Processing. Progress bar + 4-step checklist driven by REAL backend
// SSE stages (App maps extract/chunk/index/done -> progress 25/50/75/100 and
// stepIndex 1/2/3). No simulated timers.
const STEP_LABELS = [
  'Extracting text from pages',
  'Splitting into smart chunks',
  'Building the search index',
  'Warming up the model',
]

function Step({ label, state }) {
  const base = { width: '24px', height: '24px', flex: 'none', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '13px', fontWeight: 800 }
  let dotStyle, mark
  if (state === 'done') { dotStyle = { ...base, background: '#2FB8A6', color: '#fff' }; mark = '✓' }
  else if (state === 'active') { dotStyle = { ...base, background: '#FFE9E1', color: '#FF7A59', border: '2px solid #FF7A59' }; mark = '•' }
  else { dotStyle = { ...base, background: '#F0EADF', color: '#C9BEAD' }; mark = '' }
  const labelStyle = { fontSize: '14.5px', fontWeight: state === 'todo' ? 500 : 700, color: state === 'todo' ? '#B0A79A' : '#2B2622' }
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '7px 0' }}>
      <div style={dotStyle}>{mark}</div>
      <span style={labelStyle}>{label}</span>
    </div>
  )
}

export default function ProcessingScreen({ fileName, progress, stepIndex, error, onRetry }) {
  const bg = (
    <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 50% 40%,#FFF3EC 0%,#FBF7F0 60%)', pointerEvents: 'none' }} />
  )
  const wrap = { height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '24px', position: 'relative' }

  if (error) {
    return (
      <div style={wrap}>
        {bg}
        <div style={{ width: '100%', maxWidth: '440px', position: 'relative', zIndex: 1, textAlign: 'center' }}>
          <h2 style={{ fontFamily: "'Bricolage Grotesque'", fontWeight: 800, fontSize: '27px', letterSpacing: '-.02em', margin: '0 0 6px' }}>Couldn't read that file</h2>
          <p style={{ fontSize: '15px', color: '#8A8178', margin: '0 0 22px' }}>{error}</p>
          <button onClick={onRetry} style={{ background: '#FF7A59', color: '#fff', border: 'none', borderRadius: '14px', padding: '11px 24px', cursor: 'pointer', fontFamily: "'Nunito'", fontWeight: 700, fontSize: '15px', boxShadow: '0 4px 0 #e8613f' }}>Choose another file</button>
        </div>
      </div>
    )
  }

  const progressPct = Math.round(progress) + '%'
  return (
    <div style={wrap}>
      {bg}
      <div style={{ width: '100%', maxWidth: '440px', position: 'relative', zIndex: 1, textAlign: 'center' }}>
        <div style={{ width: '110px', height: '110px', margin: '0 auto 26px', position: 'relative' }}>
          <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(135deg,#5FD3C2,#2FB8A6)', animation: 'skimBlob 5s ease-in-out infinite', boxShadow: '0 16px 34px -12px rgba(47,184,166,.65)' }} />
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', animation: 'skimFloat 3s ease-in-out infinite' }}>
            <MagnifierIcon size={44} stroke="#fff" strokeWidth={2} />
          </div>
        </div>
        <h2 style={{ fontFamily: "'Bricolage Grotesque'", fontWeight: 800, fontSize: '27px', letterSpacing: '-.02em', margin: '0 0 6px' }}>Reading your document…</h2>
        <p style={{ fontSize: '15px', color: '#8A8178', margin: '0 0 26px' }}>{fileName}</p>

        <div style={{ background: '#fff', border: '1.5px solid #EDE6DB', borderRadius: '20px', padding: '22px', boxShadow: '0 12px 34px -18px rgba(43,38,34,.25)', textAlign: 'left' }}>
          <div style={{ height: '9px', background: '#F0EADF', borderRadius: '20px', overflow: 'hidden', marginBottom: '20px' }}>
            <div style={{ height: '100%', width: progressPct, background: 'linear-gradient(90deg,#FF9576,#FF7A59)', borderRadius: '20px', transition: 'width .4s ease' }} />
          </div>
          {STEP_LABELS.map((label, i) => {
            const state = (i < stepIndex || progress >= 100) ? 'done' : (i === stepIndex ? 'active' : 'todo')
            return <Step key={i} label={label} state={state} />
          })}
        </div>
        <p style={{ fontSize: '13px', color: '#B0A79A', margin: '18px 0 0' }}>Nothing is stored after you close the tab.</p>
      </div>
    </div>
  )
}
