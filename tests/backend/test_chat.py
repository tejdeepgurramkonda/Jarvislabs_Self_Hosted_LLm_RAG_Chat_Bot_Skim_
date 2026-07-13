"""Chat generation tests (C1-C3) with the LLM MOCKED. Real retrieval feeds the prompt."""

from __future__ import annotations

import mocks
from app.schemas.response import ChatMetadata
from fixtures.make_pdfs import pdf_bytes
from sse_utils import parse_sse, events_of, tokens_text


def _upload(client, headers):
    return client.post("/documents/upload", headers=headers,
                       files={"file": ("paris.pdf", pdf_bytes("paris"), "application/pdf")})


def test_C1_happy_path_streams_and_validates(client, headers, monkeypatch, record):
    rec = mocks.StreamRecorder(["The Eiffel", " Tower is", " in Paris."])
    monkeypatch.setattr("app.services.llm_client.stream_chat", rec)
    _upload(client, headers)
    resp = client.post("/chat", headers=headers, json={"query": "Where is the Eiffel Tower?"})
    events = parse_sse(resp.text)
    toks = events_of(events, "token")
    metas = events_of(events, "metadata")
    meta = metas[0].data if metas else {}
    valid = True
    try:
        ChatMetadata(**meta)
    except Exception as e:  # noqa: BLE001
        valid = False
    record("C1", "Chat", "happy path streams tokens + validated metadata",
           inp="POST /chat 'Where is the Eiffel Tower?' (LLM mocked, 3 tokens)",
           ideal="multiple token events (not one blob); metadata status=answered, sources non-empty, fallback=false, valid ChatMetadata; done event",
           actual=f"token_events={len(toks)}; text={tokens_text(events)!r}; status={meta.get('status')}; "
                  f"sources={len(meta.get('sources', []))}; fallback={meta.get('fallback')}; schema_valid={valid}; "
                  f"done={len(events_of(events,'done'))}")
    assert resp.status_code == 200
    assert len(toks) >= 2                       # incremental, not one blob
    assert meta.get("status") == "answered"
    assert len(meta.get("sources", [])) >= 1
    assert meta.get("fallback") is False
    assert valid
    assert len(events_of(events, "done")) == 1


def test_C2_sse_event_order(client, headers, monkeypatch, record):
    monkeypatch.setattr("app.services.llm_client.stream_chat", mocks.StreamRecorder(["Paris."]))
    _upload(client, headers)
    resp = client.post("/chat", headers=headers, json={"query": "Where is the Eiffel Tower?"})
    seq = [e.event for e in parse_sse(resp.text)]
    last_token = max(i for i, e in enumerate(seq) if e == "token")
    meta_i = seq.index("metadata")
    done_i = seq.index("done")
    record("C2", "Chat", "SSE event order: tokens -> metadata -> done",
           inp="POST /chat (mocked)",
           ideal="all token events precede metadata, which precedes done",
           actual=f"sequence={seq}")
    assert last_token < meta_i < done_i


def test_C3_grounded_prompt_passed_to_llm(client, headers, monkeypatch, record):
    rec = mocks.StreamRecorder(["ok"])
    monkeypatch.setattr("app.services.llm_client.stream_chat", rec)
    _upload(client, headers)
    client.post("/chat", headers=headers, json={"query": "Where is the Eiffel Tower?"})
    user_msg = rec.messages[1]["content"] if rec.messages else ""
    record("C3", "Chat", "retrieved context is injected into the LLM prompt",
           inp="capture messages passed to mocked stream_chat",
           ideal="messages built from retrieved chunks: user content has CONTEXT with fixture text",
           actual=f"called={rec.called}; roles={[m['role'] for m in (rec.messages or [])]}; "
                  f"context_has_eiffel={'Eiffel Tower' in user_msg}")
    assert rec.called
    assert "CONTEXT:" in user_msg and "Eiffel Tower" in user_msg
