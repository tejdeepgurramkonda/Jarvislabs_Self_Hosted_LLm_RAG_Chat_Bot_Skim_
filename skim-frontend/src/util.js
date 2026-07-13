// Derive a human title from a filename: drop .pdf, _/- -> spaces, collapse space.
// Matches the prototype's titleFromName exactly.
export function titleFromName(name) {
  return (name || '')
    .replace(/\.pdf$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim() || 'Document'
}

// Unique page numbers from a metadata `sources` array, ascending.
export function uniquePages(sources) {
  const pages = (sources || [])
    .map((s) => s.page)
    .filter((p) => p != null)
  return [...new Set(pages)].sort((a, b) => a - b)
}
