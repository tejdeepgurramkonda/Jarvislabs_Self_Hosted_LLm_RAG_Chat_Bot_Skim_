"""Query-time retrieval over the FAISS store, with cross-encoder reranking.

Two-stage pipeline:
  1. embed the query with the SAME model used at ingest time, L2-normalized
  2. inner-product (== cosine) search for `faiss_top_k` CANDIDATE chunks (recall)
  3. GATE: keep candidates whose cosine >= `similarity_threshold`. This — the
     first-stage recall signal — is what decides "is there relevant context?"
     (cosine separates real questions from noise cleanly; the reranker score does
     not, so it must NOT be the gate).
  4. rerank the surviving candidates with the cross-encoder (reranker.py) — this
     only RE-ORDERS them for precision
  5. keep the best `final_top_k` (optionally filtered by `rerank_threshold`)

Only the FAISS candidates are reranked — never the whole database. If nothing
clears the cosine gate, `found` is False and the caller (rag.py) emits the
graceful "I couldn't find that" fallback.
"""

from __future__ import annotations

from ..configs.config import settings
from ..utils.logger import get_logger
from . import embeddings, reranker
from .vectorstore import store

log = get_logger(__name__)


def retrieve(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    session_id: str | None = None,
    doc_id: str | None = None,
) -> dict:
    """Retrieve the most relevant chunks for a query via FAISS + reranking.

    Args:
        query: the natural-language question.
        top_k: number of chunks to KEEP after reranking (defaults to
            ``settings.final_top_k``).
        threshold: first-stage COSINE floor used as the recall/no-context gate
            (defaults to ``settings.similarity_threshold``).
        session_id / doc_id: scope the search to a session / single document.

    Returns:
        {
          "query": str,
          "chunks": [ {..., "score": <cosine>, "rerank_score": <[0,1]>}, ... ],
          "top_score": float | None,   # best first-stage cosine (recall signal)
          "found": bool,               # True if any chunk cleared the cosine gate
        }
    """
    final_top_k = top_k if top_k is not None else settings.final_top_k
    min_similarity = threshold if threshold is not None else settings.similarity_threshold

    query = (query or "").strip()
    if not query:
        return {"query": query, "chunks": [], "top_score": None, "found": False}

    # --- stage 1: recall — pull candidates from FAISS (cosine, best first) ---
    qvec = embeddings.embed_query(query)
    candidates = store.search(qvec, settings.faiss_top_k, session_id=session_id, doc_id=doc_id)
    log.info("FAISS returned %d candidates", len(candidates))
    top_score = candidates[0]["score"] if candidates else None

    # --- gate: is there any relevant context at all? (first-stage cosine) ---
    relevant = [c for c in candidates if c["score"] >= min_similarity]
    if not relevant:
        log.info(
            "No candidate cleared the similarity gate (best cosine=%s, floor=%.2f) -> no context",
            f"{top_score:.3f}" if top_score is not None else "n/a", min_similarity,
        )
        return {"query": query, "chunks": [], "top_score": top_score, "found": False}

    # --- stage 2: precision — rerank the recalled candidates, keep the best ---
    ranked = reranker.rerank(query, relevant)
    kept = [c for c in ranked if c["rerank_score"] >= settings.rerank_threshold][:final_top_k]

    log.info("Returning %d chunks", len(kept))
    return {
        "query": query,
        "chunks": kept,
        "top_score": top_score,
        "found": bool(kept),
    }
