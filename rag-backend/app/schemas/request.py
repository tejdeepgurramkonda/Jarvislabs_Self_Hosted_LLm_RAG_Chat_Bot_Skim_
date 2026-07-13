"""Pydantic v2 request schemas — the inbound contract for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body for POST /chat."""
    query: str = Field(..., min_length=1, max_length=4000, description="User's question")
    doc_id: str | None = Field(default=None, description="Scope retrieval to this document")
    top_k: int | None = Field(default=None, ge=1, le=50)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class RetrieveRequest(BaseModel):
    """Body for the temporary retrieval-inspection route (Phase C)."""
    query: str = Field(..., min_length=1, description="Natural-language query")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Override config top_k")
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override config similarity threshold"
    )
