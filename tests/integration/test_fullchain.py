"""FULL CHAIN — Frontend -> Backend -> AI -> back (end-to-end, API level)."""

from __future__ import annotations

import pytest

from fixtures.make_pdfs import ABSENT_QUESTION, KNOWN_ANSWERS

pytestmark = pytest.mark.integration

NO_CONTEXT = "I couldn't find that in your documents."


def test_F1_grounded_answer_end_to_end(sess, require_llm, record, perf):
    up = sess.upload_stream("zephyr")
    doc_id = up.by("done")[0].data["doc_id"]
    q, expected = KNOWN_ANSWERS["zephyr"]
    res = sess.chat(q)
    text = res.tokens_text()
    meta = res.by("metadata")[0].data if res.by("metadata") else {}
    src_docs = {s.get("doc_id") for s in meta.get("sources", [])}
    n_tok = len(res.by("token"))
    tps = round(n_tok / (res.total - (res.ttft or 0)), 1) if res.total and n_tok else None
    perf("F1 (grounded)", ttft_s=round(res.ttft, 3) if res.ttft else None,
         total_s=round(res.total, 3) if res.total else None, token_frames=n_tok, approx_tok_per_s=tps)
    record("F1", "FULL", "grounded answer streams end-to-end with sources",
           inp=f"upload zephyr; ask {q!r}",
           ideal=f"streams token-by-token; answer contains {expected!r}; status=answered; sources cite the uploaded doc",
           actual=f"tokens={n_tok}; status={meta.get('status')}; contains_expected={expected in text}; "
                  f"src_has_doc={doc_id in src_docs}; ttft={res.ttft}; answer={text[:100]!r}")
    assert n_tok >= 2
    assert meta.get("status") == "answered"
    assert expected in text
    assert doc_id in src_docs


def test_F2_no_context_fallback_travels_back(sess, require_live, record):
    sess.upload_stream("zephyr")
    res = sess.chat(ABSENT_QUESTION)
    text = res.tokens_text()
    meta = res.by("metadata")[0].data if res.by("metadata") else {}
    record("F2", "FULL", "no-context fallback travels back to the client",
           inp=f"upload zephyr; ask absent question {ABSENT_QUESTION!r}",
           ideal="client gets NO_CONTEXT_FALLBACK text; status=no_context; sources=[]",
           actual=f"status={meta.get('status')}; sources={len(meta.get('sources', []))}; text={text[:80]!r}")
    assert meta.get("status") == "no_context"
    assert NO_CONTEXT in text
    assert meta.get("sources", []) == []


def test_F3_multiturn_keeps_document_context(sess, require_llm, record):
    sess.upload_stream("zephyr")
    q1, exp1 = KNOWN_ANSWERS["zephyr"]           # temperature -> 512
    r1 = sess.chat(q1)
    r2 = sess.chat("Who was the lead designer of that reactor?")  # -> Vantathe (from doc)
    t1, t2 = r1.tokens_text(), r2.tokens_text()
    m1 = r1.by("metadata")[0].data if r1.by("metadata") else {}
    m2 = r2.by("metadata")[0].data if r2.by("metadata") else {}
    record("F3", "FULL", "multi-turn: same session stays grounded on the doc",
           inp="upload zephyr; Q1 temperature; Q2 follow-up about the designer (same session)",
           ideal="both answered from the same doc (Q1~512, Q2~Vantathe). NB backend is stateless (no chat memory by design) — this checks doc-context persistence",
           actual=f"Q1 status={m1.get('status')} has512={exp1 in t1}; Q2 status={m2.get('status')} hasVantathe={'Vantathe' in t2}; Q2={t2[:80]!r}",
           note="Backend does not store conversation history; each turn is retrieved independently from the session's doc.")
    assert m1.get("status") == "answered" and exp1 in t1
    assert m2.get("status") == "answered" and "Vantathe" in t2
