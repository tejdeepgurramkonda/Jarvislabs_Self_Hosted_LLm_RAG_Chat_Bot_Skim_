"""POST /chat — stream a grounded answer as Server-Sent Events, then a validated
metadata event.

SSE contract:
  event: token      data: {"text": "<delta>"}     # one per token, in order
  event: metadata   data: {ChatMetadata}           # sources + validation status
  event: done       data: {"finished": true}

The request body is validated by Pydantic (ChatRequest) up front -> 422 on bad
input. The FINAL metadata payload is validated against ChatMetadata before it is
sent; if that validation fails (or anything unexpected happens mid-stream), we
emit a graceful fallback message + a safe metadata event instead of crashing the
connection.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ..schemas.request import ChatRequest
from ..schemas.response import ChatMetadata
from ..services.rag import LLM_ERROR_FALLBACK, run_chat_stream
from ..utils.logger import get_logger
from .deps import get_session_id

router = APIRouter(tags=["chat"])
log = get_logger(__name__)


def _sse(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _safe_metadata() -> ChatMetadata:
    """Metadata used when the real payload can't be produced/validated."""
    return ChatMetadata(status="llm_error", sources=[], top_score=None, fallback=True)


def _event_stream(req: ChatRequest, session_id: str) -> Iterator[str]:
    log.info("chat: streaming answer for query=%r (session=%s doc=%s)",
             req.query[:80], session_id, req.doc_id)
    try:
        for ev in run_chat_stream(req.query, top_k=req.top_k, threshold=req.threshold,
                                  session_id=session_id, doc_id=req.doc_id):
            if ev["type"] == "token":
                yield _sse("token", {"text": ev["text"]})
                continue

            # ev["type"] == "final": validate before emitting
            try:
                meta = ChatMetadata(**ev["payload"])
            except ValidationError as exc:
                log.warning("Final metadata failed validation: %s", exc)
                # graceful fallback: make sure the user sees a message, then safe meta
                yield _sse("token", {"text": LLM_ERROR_FALLBACK})
                meta = _safe_metadata()
            yield _sse("metadata", meta.model_dump())
    except Exception as exc:  # noqa: BLE001 - last-resort guard, never crash the stream
        log.exception("Unexpected error during chat stream: %s", exc)
        yield _sse("token", {"text": LLM_ERROR_FALLBACK})
        yield _sse("metadata", _safe_metadata().model_dump())

    yield _sse("done", {"finished": True})


@router.post("/chat")
def chat(req: ChatRequest, session_id: str = Depends(get_session_id)) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(req, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering so tokens flush live
        },
    )
