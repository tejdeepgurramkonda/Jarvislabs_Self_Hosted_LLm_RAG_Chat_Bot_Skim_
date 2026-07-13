"""Validation & fallback tests (V1-V6). LLM mocked."""

from __future__ import annotations

import mocks
from app.services.rag import NO_CONTEXT_FALLBACK, LLM_ERROR_FALLBACK
from fixtures.make_pdfs import pdf_bytes
from sse_utils import parse_sse, events_of, tokens_text


def _upload(client, headers):
    return client.post("/documents/upload", headers=headers,
                       files={"file": ("paris.pdf", pdf_bytes("paris"), "application/pdf")})


def test_V1_malformed_body_422(client, headers, record):
    cases = {
        "empty query": {"query": ""},
        "missing query": {"doc_id": "x"},
        "top_k=0": {"query": "hi", "top_k": 0},
        "threshold=1.5": {"query": "hi", "threshold": 1.5},
    }
    results = {name: client.post("/chat", headers=headers, json=body).status_code
               for name, body in cases.items()}
    record("V1", "Validation", "malformed request body -> 422",
           inp="POST /chat with invalid bodies: " + ", ".join(cases),
           ideal="every case -> 422 (schema rejects)",
           actual=str(results))
    assert all(code == 422 for code in results.values())


def test_V2_missing_session_header_400(client, record):
    resp = client.post("/chat", json={"query": "hello"})
    record("V2", "Validation", "missing X-Session-Id -> 400",
           inp="POST /chat without X-Session-Id",
           ideal="400 (Missing X-Session-Id header)",
           actual=f"{resp.status_code}; {resp.json()}")
    assert resp.status_code == 400


def test_V3_no_context_fallback(client, headers, monkeypatch, record):
    rec = mocks.StreamRecorder(["should not be used"])
    monkeypatch.setattr("app.services.llm_client.stream_chat", rec)
    # empty store -> nothing relevant
    resp = client.post("/chat", headers=headers, json={"query": "anything at all"})
    events = parse_sse(resp.text)
    meta = events_of(events, "metadata")[0].data
    record("V3", "Validation", "no relevant context -> graceful fallback, no LLM call",
           inp="POST /chat with empty index",
           ideal="token=NO_CONTEXT_FALLBACK; status=no_context; sources=[]; fallback=true; LLM not called",
           actual=f"text={tokens_text(events)!r}; status={meta.get('status')}; sources={len(meta.get('sources',[]))}; "
                  f"fallback={meta.get('fallback')}; llm_called={rec.called}")
    assert NO_CONTEXT_FALLBACK in tokens_text(events)
    assert meta["status"] == "no_context" and meta["sources"] == [] and meta["fallback"] is True
    assert rec.called is False


def test_V4_llm_failure_fallback(client, headers, monkeypatch, record):
    monkeypatch.setattr("app.services.llm_client.stream_chat", mocks.raise_immediately)
    _upload(client, headers)
    resp = client.post("/chat", headers=headers, json={"query": "Where is the Eiffel Tower?"})
    events = parse_sse(resp.text)
    meta = events_of(events, "metadata")[0].data
    record("V4", "Validation", "LLM failure (immediate) -> graceful fallback",
           inp="POST /chat; mocked stream_chat raises before any token",
           ideal="token=LLM_ERROR_FALLBACK; status=llm_error; fallback=true; no crash; done present",
           actual=f"text={tokens_text(events)!r}; status={meta.get('status')}; fallback={meta.get('fallback')}; "
                  f"done={len(events_of(events,'done'))}")
    assert resp.status_code == 200
    assert LLM_ERROR_FALLBACK in tokens_text(events)
    assert meta["status"] == "llm_error" and meta["fallback"] is True
    assert len(events_of(events, "done")) == 1


def test_V5_midstream_failure(client, headers, monkeypatch, record):
    monkeypatch.setattr("app.services.llm_client.stream_chat",
                        mocks.yield_then_raise(["The Eiffel", " Tower"]))
    _upload(client, headers)
    resp = client.post("/chat", headers=headers, json={"query": "Where is the Eiffel Tower?"})
    events = parse_sse(resp.text)
    text = tokens_text(events)
    meta = events_of(events, "metadata")[0].data
    record("V5", "Validation", "mid-stream LLM failure -> partial answer, no duplicate fallback",
           inp="POST /chat; mock yields 2 tokens then raises",
           ideal="partial tokens shown; status=llm_error; stream ends with done; NO fallback text prepended (already emitted)",
           actual=f"text={text!r}; status={meta.get('status')}; fallback={meta.get('fallback')}; done={len(events_of(events,'done'))}")
    assert "Eiffel" in text
    assert LLM_ERROR_FALLBACK not in text          # not prepended, since tokens were already emitted
    assert meta["status"] == "llm_error" and meta["fallback"] is True
    assert len(events_of(events, "done")) == 1


def _bad_final_stream(query, top_k=None, threshold=None, session_id=None, doc_id=None):
    """Stand-in run_chat_stream that emits a token then an INVALID final payload."""
    yield {"type": "token", "text": "partial answer "}
    yield {"type": "final", "payload": {"status": "bogus_status", "sources": [], "top_score": 0.5}}


def test_V6_bad_final_payload_safe_fallback(client, headers, monkeypatch, record):
    monkeypatch.setattr("app.api.chat.run_chat_stream", _bad_final_stream)
    resp = client.post("/chat", headers=headers, json={"query": "anything"})
    events = parse_sse(resp.text)
    metas = events_of(events, "metadata")
    meta = metas[0].data if metas else {}
    valid_meta = meta.get("status") in ("answered", "no_context", "llm_error")
    record("V6", "Validation", "invalid final payload -> safe fallback, not a broken response",
           inp="patch run_chat_stream to yield final status='bogus_status'",
           ideal="route catches ValidationError -> fallback token + safe metadata (llm_error) + done; response well-formed",
           actual=f"text={tokens_text(events)!r}; metadata_status={meta.get('status')}; valid_schema={valid_meta}; "
                  f"done={len(events_of(events,'done'))}")
    assert resp.status_code == 200
    assert LLM_ERROR_FALLBACK in tokens_text(events)
    assert valid_meta and meta["status"] == "llm_error"
    assert len(events_of(events, "done")) == 1
