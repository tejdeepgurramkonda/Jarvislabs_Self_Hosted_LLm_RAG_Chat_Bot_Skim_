"""RAG orchestration: retrieve -> build prompt -> stream a grounded answer, then
emit a structured metadata payload.

`run_chat_stream` is a generator of typed events consumed by the /chat route:

    {"type": "token", "text": "<delta>"}          # zero or more, in order
    {"type": "final", "payload": {...}}            # exactly one, last

The "final" payload is a plain dict shaped like ChatMetadata; the route validates
it against the Pydantic schema before sending the terminal event. Splitting
"produce" (here) from "validate + serialize" (route) keeps this layer transport-
agnostic and makes the validation step explicit.

Outcomes (payload["status"]):
  * "answered"    grounded answer streamed from the LLM
  * "no_context"  retrieval found nothing relevant -> fallback text
  * "llm_error"   the LLM call failed/timed out    -> fallback text
"""

from __future__ import annotations

from collections.abc import Iterator

from ..utils.logger import get_logger
from ..utils.prompt_templates import build_messages
from . import llm_client
from .retrieval import retrieve

log = get_logger(__name__)

NO_CONTEXT_FALLBACK = "I couldn't find that in your documents."
LLM_ERROR_FALLBACK = "Sorry, I'm having trouble answering right now. Please try again."


def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": c["doc_id"],
            "filename": c.get("filename"),
            "page": c.get("page"),
            "chunk_idx": c["chunk_idx"],
            "score": c["score"],
        }
        for c in chunks
    ]


def run_chat_stream(query: str, top_k: int | None = None,
                    threshold: float | None = None,
                    session_id: str | None = None, doc_id: str | None = None) -> Iterator[dict]:
    result = retrieve(query, top_k=top_k, threshold=threshold,
                      session_id=session_id, doc_id=doc_id)
    sources = _sources_from_chunks(result["chunks"])

    # --- no relevant context -> graceful fallback, no LLM call ---
    if not result["found"]:
        log.info("No relevant context (top_score=%s) -> no_context fallback", result["top_score"])
        yield {"type": "token", "text": NO_CONTEXT_FALLBACK}
        yield {"type": "final", "payload": {
            "status": "no_context",
            "sources": [],
            "top_score": result["top_score"],
            "fallback": True,
        }}
        return

    # --- stream the grounded answer ---
    messages = build_messages(result["query"], result["chunks"])
    emitted_any = False
    try:
        for token in llm_client.stream_chat(messages):
            emitted_any = True
            yield {"type": "token", "text": token}
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        log.warning("LLM stream failed: %s", exc)
        if not emitted_any:
            # nothing shown yet -> show the fallback message
            yield {"type": "token", "text": LLM_ERROR_FALLBACK}
        yield {"type": "final", "payload": {
            "status": "llm_error",
            "sources": sources,
            "top_score": result["top_score"],
            "fallback": True,
        }}
        return

    yield {"type": "final", "payload": {
        "status": "answered",
        "sources": sources,
        "top_score": result["top_score"],
        "fallback": False,
    }}
