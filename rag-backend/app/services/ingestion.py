"""Document ingestion: PDF -> text -> overlapping chunks -> embeddings -> FAISS.

Pipeline (POST /documents/upload):
  1. save the uploaded PDF under data/uploads/<doc_id>.pdf
  2. extract text per page with pypdf
  3. split each page into overlapping chunks (recursive character splitter)
  4. embed + L2-normalize the chunks (embeddings.py)
  5. add to the FAISS store with metadata {doc_id, filename, page, chunk_idx}
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from pypdf import PdfReader

from ..configs.config import settings
from ..utils.logger import get_logger
from . import embeddings
from .vectorstore import store

log = get_logger(__name__)

# Separators tried in order — paragraph, line, sentence, word, character.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# --------------------------------------------------------------------------- #
# Recursive character text splitter (size + overlap, configurable)
# --------------------------------------------------------------------------- #
def _merge_splits(splits: list[str], sep: str, chunk_size: int, overlap: int) -> list[str]:
    sep_len = len(sep)
    docs: list[str] = []
    current: list[str] = []
    total = 0
    for piece in splits:
        plen = len(piece)
        # would adding this piece overflow the target size?
        if current and total + plen + (sep_len if current else 0) > chunk_size:
            doc = sep.join(current).strip()
            if doc:
                docs.append(doc)
            # drop from the front until we're back under overlap budget
            while current and (
                total > overlap
                or (total + plen + (sep_len if current else 0) > chunk_size and total > 0)
            ):
                total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                current = current[1:]
        current.append(piece)
        total += plen + (sep_len if len(current) > 1 else 0)
    doc = sep.join(current).strip()
    if doc:
        docs.append(doc)
    return docs


def split_text(text: str, chunk_size: int, overlap: int, separators: list[str] | None = None) -> list[str]:
    """Recursively split text into ~chunk_size pieces with `overlap` char overlap."""
    separators = separators if separators is not None else _SEPARATORS
    final: list[str] = []

    # pick the finest separator that actually appears in this text
    separator = separators[-1]
    remaining_seps: list[str] = []
    for i, s in enumerate(separators):
        if s == "":
            separator = s
            break
        if s in text:
            separator = s
            remaining_seps = separators[i + 1:]
            break

    splits = list(text) if separator == "" else text.split(separator)

    good: list[str] = []
    for s in splits:
        if len(s) < chunk_size:
            good.append(s)
        else:
            if good:
                final.extend(_merge_splits(good, separator, chunk_size, overlap))
                good = []
            if not remaining_seps:
                final.append(s)  # can't split further; keep oversized piece
            else:
                final.extend(split_text(s, chunk_size, overlap, remaining_seps))
    if good:
        final.extend(_merge_splits(good, separator, chunk_size, overlap))
    return [c for c in final if c.strip()]


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #
def extract_pages(pdf_path) -> list[tuple[int, str]]:
    """Return [(page_number_1_indexed, text), ...] for pages with text."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((idx, text))
    return pages


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class IngestionError(Exception):
    """Raised when a document yields no usable text."""


def ingest_pdf_streaming(file_bytes: bytes, filename: str, session_id: str) -> "Iterator[dict]":
    """Full ingest pipeline as a generator, yielding one event per completed stage.

    Stages map 1:1 to the frontend's 4-step checklist:
        {"stage": "extract", "step": 0, "pages": N}   text pulled from the PDF
        {"stage": "chunk",   "step": 1, "chunks": N}  split into overlapping chunks
        {"stage": "index",   "step": 2, "chunks": N}  embedded + added to FAISS
        {"stage": "done",    "step": 3, doc_id, filename, chunk_count, page_count}

    Every chunk is tagged with `session_id` so retrieval / listing can be isolated
    per session. Raises IngestionError for empty / unsplittable PDFs.
    """
    doc_id = uuid.uuid4().hex
    dest = settings.uploads_dir / f"{doc_id}.pdf"
    dest.write_bytes(file_bytes)
    log.info("Saved upload '%s' -> %s (%d bytes)", filename, dest.name, len(file_bytes))

    # --- stage: extract ---
    pages = extract_pages(dest)
    if not pages:
        dest.unlink(missing_ok=True)
        raise IngestionError("No extractable text found in the PDF (is it a scanned image?).")
    yield {"stage": "extract", "step": 0, "pages": len(pages)}

    # --- stage: chunk (each page, keeping page + a document-global chunk index) ---
    chunk_texts: list[str] = []
    metadatas: list[dict] = []
    chunk_idx = 0
    for page_no, page_text in pages:
        for piece in split_text(page_text, settings.chunk_size, settings.chunk_overlap):
            chunk_texts.append(piece)
            metadatas.append({
                "doc_id": doc_id,
                "session_id": session_id,
                "filename": filename,
                "page": page_no,
                "chunk_idx": chunk_idx,
                "text": piece,
            })
            chunk_idx += 1

    if not chunk_texts:
        dest.unlink(missing_ok=True)
        raise IngestionError("PDF text could not be split into chunks.")
    log.info("Extracted %d pages -> %d chunks for doc %s", len(pages), len(chunk_texts), doc_id)
    yield {"stage": "chunk", "step": 1, "chunks": len(chunk_texts)}

    # --- stage: index (embed + add to FAISS; the slow stage) ---
    vectors = embeddings.embed_texts(chunk_texts)
    added = store.add(vectors, metadatas)
    log.info("Indexed %d chunks for doc %s (session=%s)", added, doc_id, session_id)
    yield {"stage": "index", "step": 2, "chunks": added}

    # --- stage: done ---
    yield {
        "stage": "done",
        "step": 3,
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": added,
        "page_count": len(pages),
    }


def ingest_pdf(file_bytes: bytes, filename: str, session_id: str) -> dict:
    """Blocking ingest. Drives ingest_pdf_streaming to completion and returns the
    final {doc_id, filename, chunk_count, page_count} payload."""
    result: dict = {}
    for ev in ingest_pdf_streaming(file_bytes, filename, session_id):
        if ev["stage"] == "done":
            result = {k: ev[k] for k in ("doc_id", "filename", "chunk_count", "page_count")}
    return result
