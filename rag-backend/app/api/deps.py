"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Header, HTTPException


def get_session_id(x_session_id: str | None = Header(default=None)) -> str:
    """Require an X-Session-Id header.

    Every browser session mints its own id (frontend: sessionStorage). It scopes
    uploads, listing, deletion, and retrieval so a session can only ever see its
    own documents. Missing/blank -> 400 (the frontend always sends it).
    """
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header.")
    return x_session_id.strip()
