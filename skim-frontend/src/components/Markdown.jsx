// Minimal, dependency-free markdown renderer for assistant messages.
// Handles what the LLM actually emits: paragraphs, numbered/bulleted lists,
// **bold**, `inline code`, and line breaks. Renders to React elements (no
// dangerouslySetInnerHTML), so it's XSS-safe, and it degrades gracefully on the
// partial text produced during streaming.

const codeStyle = {
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  fontSize: '0.9em', background: '#F1EADF', borderRadius: '5px', padding: '1px 5px',
}
const listStyle = { margin: '0', paddingLeft: '1.4em' }
const liStyle = { margin: '2px 0' }

const ORDERED = /^\s*\d+\.\s+/
const UNORDERED = /^\s*[-*]\s+/

// Render inline spans: **bold** and `code`. Everything else is plain text.
function renderInline(text, keyPrefix) {
  const nodes = []
  const regex = /\*\*([^*]+)\*\*|`([^`]+)`/g
  let last = 0
  let match
  let k = 0
  while ((match = regex.exec(text)) !== null) {
    if (match.index > last) nodes.push(text.slice(last, match.index))
    if (match[1] !== undefined) {
      nodes.push(<strong key={`${keyPrefix}-b${k++}`}>{match[1]}</strong>)
    } else {
      nodes.push(<code key={`${keyPrefix}-c${k++}`} style={codeStyle}>{match[2]}</code>)
    }
    last = match.index + match[0].length
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

// Collect consecutive list items of `kind`, tolerating blank lines between items.
function collectList(lines, start, pattern) {
  const items = []
  let i = start
  while (i < lines.length) {
    if (pattern.test(lines[i])) {
      items.push(lines[i].replace(pattern, ''))
      i++
    } else if (lines[i].trim() === '') {
      let j = i + 1
      while (j < lines.length && lines[j].trim() === '') j++
      if (j < lines.length && pattern.test(lines[j])) { i = j } else break
    } else {
      break
    }
  }
  return { items, next: i }
}

export default function Markdown({ text }) {
  const lines = (text || '').split('\n')
  const blocks = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    if (lines[i].trim() === '') { i++; continue }
    const style = { marginTop: blocks.length ? '0.7em' : 0 }

    if (ORDERED.test(lines[i])) {
      const { items, next } = collectList(lines, i, ORDERED)
      blocks.push(
        <ol key={key++} style={{ ...listStyle, ...style }}>
          {items.map((it, j) => <li key={j} style={liStyle}>{renderInline(it, `o${key}-${j}`)}</li>)}
        </ol>,
      )
      i = next
    } else if (UNORDERED.test(lines[i])) {
      const { items, next } = collectList(lines, i, UNORDERED)
      blocks.push(
        <ul key={key++} style={{ ...listStyle, ...style }}>
          {items.map((it, j) => <li key={j} style={liStyle}>{renderInline(it, `u${key}-${j}`)}</li>)}
        </ul>,
      )
      i = next
    } else {
      const para = []
      while (i < lines.length && lines[i].trim() !== ''
        && !ORDERED.test(lines[i]) && !UNORDERED.test(lines[i])) {
        para.push(lines[i]); i++
      }
      blocks.push(
        <p key={key++} style={{ margin: 0, ...style }}>
          {para.map((ln, j) => (
            <span key={j}>{renderInline(ln, `p${key}-${j}`)}{j < para.length - 1 && <br />}</span>
          ))}
        </p>,
      )
    }
  }

  return <>{blocks}</>
}
