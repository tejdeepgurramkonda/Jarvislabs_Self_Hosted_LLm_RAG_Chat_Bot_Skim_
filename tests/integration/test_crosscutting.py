"""CROSS-CUTTING — end-to-end streaming, schema of the final event, realistic UX sequence."""

from __future__ import annotations

import pytest

from fixtures.make_pdfs import KNOWN_ANSWERS

pytestmark = pytest.mark.integration


def test_X1_streaming_is_incremental(sess, require_llm, record, perf):
    sess.upload_stream("zephyr")
    res = sess.chat(KNOWN_ANSWERS["zephyr"][0])
    toks = res.by("token")
    # incremental if frames arrive spread over time, not all at the end
    spread = (res.total - res.ttft) if (res.total and res.ttft is not None) else 0
    n = len(toks)
    tps = round(n / (res.total - res.ttft), 1) if (res.total and res.ttft and n) else None
    perf("X1 (stream)", ttft_s=round(res.ttft, 3) if res.ttft else None,
         total_s=round(res.total, 3) if res.total else None, token_frames=n, approx_tok_per_s=tps)
    record("X1", "CROSS", "streaming is incremental end-to-end (not one blob)",
           inp="stream /chat, time the frames",
           ideal=">=2 token frames arriving over time; record TTFT + latency + tok/s",
           actual=f"token_frames={n}; ttft={round(res.ttft,3) if res.ttft else None}s; total={round(res.total,3) if res.total else None}s; spread={round(spread,3)}s; tok/s~{tps}")
    assert n >= 2
    assert res.ttft is not None and res.total is not None
    assert spread > 0  # last token strictly after the first -> genuinely streamed


def test_X2_final_event_after_text_and_schema(sess, require_llm, record):
    from app.schemas.response import ChatMetadata
    sess.upload_stream("zephyr")
    res = sess.chat(KNOWN_ANSWERS["zephyr"][0])
    seq = [e.event for e in res.events]
    last_token = max((i for i, e in enumerate(seq) if e == "token"), default=-1)
    meta_i = seq.index("metadata")
    meta = res.by("metadata")[0].data
    valid = True
    err = ""
    try:
        ChatMetadata(**meta)
    except Exception as e:  # noqa: BLE001
        valid, err = False, str(e)
    record("X2", "CROSS", "final structured event after text + matches schema",
           inp="inspect stream order + validate metadata",
           ideal="all token frames precede the single metadata; metadata validates against ChatMetadata",
           actual=f"order_ok={last_token < meta_i}; sequence_tail={seq[-4:]}; schema_valid={valid}{(' err='+err) if err else ''}")
    assert last_token < meta_i
    assert valid


def test_X3_realistic_upload_ask_recent_ask(sess, require_llm, record):
    up = sess.upload_stream("zephyr")
    doc_id = up.by("done")[0].data["doc_id"]
    r1 = sess.chat(KNOWN_ANSWERS["zephyr"][0])                       # ask
    recent = sess.client.get("/documents", headers=sess.headers()).json()  # switch "recent" tab
    doc_ids = [d["doc_id"] for d in recent["documents"]]
    r2 = sess.chat("What coolant does it use?", doc_id=doc_id)       # ask again scoped to the doc
    t2 = r2.tokens_text()
    m1 = r1.by("metadata")[0].data if r1.by("metadata") else {}
    m2 = r2.by("metadata")[0].data if r2.by("metadata") else {}
    record("X3", "CROSS", "realistic sequence: upload -> ask -> recent -> ask again",
           inp="upload docA; ask; GET /documents; select docA; ask again with doc_id=docA",
           ideal="every step ok; recent lists docA; second (doc-scoped) answer is grounded",
           actual=f"ask1 status={m1.get('status')}; recent_has_doc={doc_id in doc_ids}; "
                  f"ask2 status={m2.get('status')}; ask2_grounded={'BlueSalt' in t2}; ask2={t2[:80]!r}")
    assert m1.get("status") == "answered"
    assert doc_id in doc_ids
    assert m2.get("status") == "answered" and "BlueSalt" in t2
