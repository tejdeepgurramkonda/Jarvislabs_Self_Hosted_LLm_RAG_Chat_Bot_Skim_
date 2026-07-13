// The coral "S" logo mark. Sizes/radii vary by placement (upload=34/11,
// sidebar=28/9, chat avatar=32/10) — ported from the prototype.
export default function LogoMark({ size = 34, radius = 11, fontSize = 19, shadow = '0 4px 0 #e8613f' }) {
  return (
    <div style={{
      width: size, height: size, background: '#FF7A59', color: '#fff',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: "'Bricolage Grotesque'", fontWeight: 800, fontSize,
      borderRadius: radius, boxShadow: shadow, flex: 'none',
    }}>S</div>
  )
}
