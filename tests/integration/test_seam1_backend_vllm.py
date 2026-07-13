"""SEAM 1 — Backend <-> AI service (vLLM). Real network across the boundary.

Positives hit the live backend; negatives (bad key, unreachable) use an isolated
in-process backend so a bad target never disturbs the running server.
"""

from __future__ import annotations

import pytest

from fixtures.make_pdfs import KNOWN_ANSWERS

pytestmark = pytest.mark.integration

ZEPHYR_Q = KNOWN_ANSWERS["zephyr"][0]
LLM_ERROR = "Sorry, I'm having trouble answering right now. Please try again."


def test_S1_1_backend_reaches_vllm(live, require_llm, record):
    body = live.get("/health").json()
    record("S1.1", "SEAM1", "backend LLM client reaches live vLLM (auth OK)",
           inp="GET {BACKEND_URL}/health",
           ideal="200; llm.reachable=true; model 'qwen'",
           actual=f"reachable={body['llm']['reachable']}; model={body['llm']['model']}; base={body['llm']['base_url']}")
    assert body["llm"]["reachable"] is True
    assert "qwen" in str(body["llm"]["model"]).lower()


def test_S1_2_streaming_tokens_flow_through_backend(sess, require_llm, record):
    sess.upload_stream("zephyr")
    res = sess.chat(ZEPHYR_Q)
    toks = res.by("token")
    meta = res.by("metadata")[0].data if res.by("metadata") else {}
    record("S1.2", "SEAM1", "tokens stream incrementally BE<-vLLM",
           inp=f"upload zephyr; POST /chat {ZEPHYR_Q!r}",
           ideal="multiple token frames (incremental); metadata.status=answered",
           actual=f"status={res.status}; token_frames={len(toks)}; ttft={res.ttft}; meta_status={meta.get('status')}")
    assert res.status == 200
    assert len(toks) >= 2
    assert meta.get("status") == "answered"


def test_S1_3_bad_key_clean_error(inprocess, cfg, require_llm, record):
    inprocess.set_llm(cfg.vllm_base_url, "totally-wrong-key-123")
    inprocess.ingest("zephyr", "neg-key")
    # threshold=0 guarantees retrieval returns context so the LLM is actually called
    res = inprocess.chat(ZEPHYR_Q, "neg-key", threshold=0.0)
    meta = res.by("metadata")[0].data if res.by("metadata") else {}
    text = res.tokens_text()
    record("S1.3", "SEAM1", "bad API key -> clean fallback (no crash)",
           inp="in-process BE -> real vLLM base, WRONG key; /chat with a doc",
           ideal="vLLM 401 caught -> llm_error fallback; no crash/stacktrace to client",
           actual=f"status={res.status}; meta_status={meta.get('status')}; fallback={meta.get('fallback')}; text={text[:60]!r}")
    assert res.status == 200
    assert meta.get("status") == "llm_error"
    assert meta.get("fallback") is True
    assert LLM_ERROR in text


def test_S1_4_unreachable_vllm_fallback(inprocess, record):
    # point at a closed local port -> connection refused, no external dependency
    inprocess.set_llm("http://127.0.0.1:9", "EMPTY")
    inprocess.ingest("zephyr", "neg-url")
    res = inprocess.chat(ZEPHYR_Q, "neg-url", threshold=0.0)
    meta = res.by("metadata")[0].data if res.by("metadata") else {}
    text = res.tokens_text()
    record("S1.4", "SEAM1", "vLLM unreachable/timeout -> fallback (no crash)",
           inp="in-process BE -> bad LLM_BASE_URL http://127.0.0.1:9; /chat with a doc",
           ideal="connection error caught -> llm_error fallback; stream still completes with done",
           actual=f"status={res.status}; frames={[e.event for e in res.events]}; meta_status={meta.get('status')}; text={text[:60]!r}")
    assert res.status == 200
    assert meta.get("status") == "llm_error"
    assert LLM_ERROR in text
    assert len(res.by("done")) == 1


def test_S1_5_retrieved_context_reaches_model(sess, require_llm, record):
    sess.upload_stream("zephyr")
    res = sess.chat(ZEPHYR_Q)
    text = res.tokens_text()
    expected = KNOWN_ANSWERS["zephyr"][1]  # "512"
    record("S1.5", "SEAM1", "retrieved context actually reaches the model",
           inp=f"upload zephyr (invented fact); ask {ZEPHYR_Q!r}",
           ideal=f"answer reflects the doc (contains {expected!r}) — grounding, not model priors",
           actual=f"answer={text[:120]!r}")
    assert expected in text
