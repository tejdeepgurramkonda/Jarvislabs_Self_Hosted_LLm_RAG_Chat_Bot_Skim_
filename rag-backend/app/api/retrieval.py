"""TEMPORARY retrieval-inspection routes (Phase C).

These let you eyeball which chunks come back for a query and at what similarity
score, so you can tune top_k / similarity_threshold. They are NOT part of the
final chat contract and can be removed once retrieval quality looks right.

  GET  /retrieve/test?query=...&top_k=&threshold=
  POST /retrieve/test    { "query": "...", "top_k": ..., "threshold": ... }
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..configs.config import settings
from ..schemas.request import RetrieveRequest
from ..schemas.response import RetrievedChunk, RetrieveResponse
from ..services.retrieval import retrieve

router = APIRouter(prefix="/retrieve", tags=["retrieval (debug)"])


def _to_response(result: dict, threshold: float | None) -> RetrieveResponse:
    thr = threshold if threshold is not None else settings.similarity_threshold
    return RetrieveResponse(
        query=result["query"],
        found=result["found"],
        top_score=result["top_score"],
        threshold=thr,
        chunks=[RetrievedChunk(**c) for c in result["chunks"]],
    )


@router.get("/test", response_model=RetrieveResponse)
def retrieve_test_get(
    query: str = Query(..., min_length=1),
    top_k: int | None = Query(default=None, ge=1, le=50),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> RetrieveResponse:
    result = retrieve(query, top_k=top_k, threshold=threshold)
    return _to_response(result, threshold)


@router.post("/test", response_model=RetrieveResponse)
def retrieve_test_post(req: RetrieveRequest) -> RetrieveResponse:
    result = retrieve(req.query, top_k=req.top_k, threshold=req.threshold)
    return _to_response(result, req.threshold)
