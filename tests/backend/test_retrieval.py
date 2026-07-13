"""Retrieval tests (R1-R4): real embeddings + FAISS cosine search + thresholding."""

from __future__ import annotations

from app.configs.config import settings
from app.services.retrieval import retrieve


def test_R1_relevant_query_returns_right_chunks(ingest, session_id, record):
    ingest("paris", session_id)
    ingest("biology", session_id)
    res = retrieve("Where is the Eiffel Tower located?", session_id=session_id)
    top = res["chunks"][0] if res["chunks"] else {}
    top_text = top.get("text", "")
    record("R1", "Retrieval", "obvious answer retrieved in top_k",
           inp="ingest paris+biology; retrieve('Where is the Eiffel Tower located?')",
           ideal="found=true; top chunk mentions Paris/Eiffel; top_score high; <= top_k results",
           actual=f"found={res['found']}; n={len(res['chunks'])}; top_score={res['top_score']:.3f}; top_text={top_text[:60]!r}")
    assert res["found"] is True
    assert "Paris" in top_text or "Eiffel" in top_text
    assert res["top_score"] >= settings.similarity_threshold
    assert len(res["chunks"]) <= settings.top_k


def test_R2_irrelevant_query_below_threshold(ingest, session_id, record):
    ingest("paris", session_id)
    res = retrieve("What is the boiling point of mercury in Kelvin?", session_id=session_id)
    ts = res["top_score"]
    record("R2", "Retrieval", "no-match query -> threshold path (no_context)",
           inp="ingest paris; retrieve('boiling point of mercury ...')",
           ideal="found=false; best score below threshold; empty kept list",
           actual=f"found={res['found']}; kept={len(res['chunks'])}; top_score={ts}")
    assert res["found"] is False
    assert res["chunks"] == []
    assert ts is None or ts < settings.similarity_threshold


def test_R3_cosine_correctness(ingest, session_id, record):
    ingest("biology", session_id)
    # query is (nearly) an exact chunk -> cosine should be ~1.0
    exact = "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen."
    res = retrieve(exact, session_id=session_id, threshold=0.0)
    scores = [c["score"] for c in res["chunks"]]
    in_range = all(-1.0001 <= s <= 1.0001 for s in scores)
    descending = scores == sorted(scores, reverse=True)
    record("R3", "Retrieval", "cosine similarity correct (normalized IP)",
           inp="retrieve(exact chunk text, threshold=0)",
           ideal="top score ~1.0; all scores in [-1,1]; sorted descending",
           actual=f"top={scores[0]:.4f}; in_range={in_range}; descending={descending}; scores={[round(s,3) for s in scores]}")
    assert scores and scores[0] > 0.95
    assert in_range and descending


def test_R4_session_scoping(ingest, make_headers, record):
    ingest("paris", "sessionA")
    ingest("biology", "sessionB")
    a = retrieve("Eiffel Tower", session_id="sessionA")
    b = retrieve("Eiffel Tower", session_id="sessionB")
    a_docs = {c["doc_id"] for c in a["chunks"]}
    b_texts = " ".join(c["text"] for c in b["chunks"])
    record("R4", "Retrieval", "retrieval is session-scoped",
           inp="ingest paris->A, biology->B; retrieve('Eiffel Tower') scoped to each",
           ideal="A returns paris chunks; B never returns paris content",
           actual=f"A_found={a['found']} A_docs={len(a_docs)}; B_found={b['found']} B_has_paris={'Eiffel' in b_texts}")
    assert a["found"] is True
    assert "Eiffel" not in b_texts  # B cannot see A's document
