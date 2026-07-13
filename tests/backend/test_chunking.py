"""Chunking unit tests (CK1, CK2) against the real recursive splitter."""

from __future__ import annotations

from app.services.ingestion import split_text


def _max_overlap(a: str, b: str) -> int:
    """Longest suffix of `a` that appears at/near the start of `b`."""
    best = 0
    for L in range(1, min(len(a), len(b)) + 1):
        if a[-L:] in b[: L + 40]:
            best = L
    return best


def test_CK1_size_and_overlap(record):
    size, overlap = 800, 120
    text = (("Sentence about apples number %d. Apples are a crunchy red fruit. " % i for i in range(60)))
    text = " ".join(text) + "\n\n" + " ".join("Paragraph two about oranges %d." % i for i in range(60))
    chunks = split_text(text, size, overlap)
    max_len = max(len(c) for c in chunks)
    overlaps = [_max_overlap(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]
    best_overlap = max(overlaps) if overlaps else 0
    record("CK1", "Chunking", "chunk size + overlap behave as configured",
           inp=f"split_text(len={len(text)}, size={size}, overlap={overlap})",
           ideal=">=2 chunks; each <= chunk_size; consecutive chunks share overlap; no empty chunks",
           actual=f"n_chunks={len(chunks)}; max_len={max_len}; best_consecutive_overlap={best_overlap}")
    assert len(chunks) >= 2
    assert max_len <= size
    assert all(c.strip() for c in chunks)
    assert best_overlap >= 20  # real overlap present between consecutive chunks


def test_CK2_short_text_single_chunk(record):
    chunks = split_text("one short line of text", 800, 120)
    record("CK2", "Chunking", "short text -> single chunk",
           inp="split_text('one short line of text', 800, 120)",
           ideal="exactly 1 chunk, content preserved",
           actual=f"n_chunks={len(chunks)}; chunk={chunks[0]!r}")
    assert len(chunks) == 1
    assert chunks[0] == "one short line of text"
