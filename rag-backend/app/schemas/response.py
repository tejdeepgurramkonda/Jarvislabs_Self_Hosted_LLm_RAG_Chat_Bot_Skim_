"""Pydantic v2 response schemas — the outbound contract for the API.

Document-ingestion schemas live here now; chat/RAG response schemas are added in
later phases.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Returned by POST /documents/upload."""
    doc_id: str = Field(..., description="Server-assigned id for the document")
    filename: str
    chunk_count: int = Field(..., ge=0, description="Number of chunks indexed")
    page_count: int = Field(..., ge=0, description="Pages with extractable text")


class DocumentSummary(BaseModel):
    """One entry in GET /documents."""
    doc_id: str
    filename: str | None = None
    chunk_count: int = Field(..., ge=0)
    page_count: int = Field(..., ge=0)


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total_documents: int = Field(..., ge=0)
    total_vectors: int = Field(..., ge=0)


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int = Field(..., ge=0)
    status: str


# --------------------------------------------------------------------------- #
# Retrieval (Phase C)
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    """A single context chunk with its cosine similarity score."""
    doc_id: str
    filename: str | None = None
    page: int | None = None
    chunk_idx: int
    score: float = Field(..., description="Cosine similarity (higher = more relevant)")
    text: str


class RetrieveResponse(BaseModel):
    """Returned by the temporary retrieval-inspection route."""
    query: str
    found: bool = Field(..., description="True if any chunk cleared the threshold")
    top_score: float | None = Field(None, description="Best raw similarity before thresholding")
    threshold: float = Field(..., description="Threshold applied")
    chunks: list[RetrievedChunk]


# --------------------------------------------------------------------------- #
# Chat response envelope (Phase E)
# --------------------------------------------------------------------------- #
class SourceRef(BaseModel):
    """A citation for one chunk that fed the answer."""
    doc_id: str
    filename: str | None = None
    page: int | None = None
    chunk_idx: int
    score: float


class ChatMetadata(BaseModel):
    """The structured payload sent as the final SSE event, AFTER the answer text.

    `status` is the validation/outcome signal the UI can branch on:
      - "answered"    normal grounded answer
      - "no_context"  retrieval found nothing relevant -> fallback message
      - "llm_error"   the LLM call failed/timed out -> fallback message
    """
    status: str = Field(..., pattern="^(answered|no_context|llm_error)$")
    sources: list[SourceRef] = Field(default_factory=list)
    top_score: float | None = None
    fallback: bool = Field(False, description="True when the answer text was a fallback message")

