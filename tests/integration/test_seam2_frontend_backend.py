"""SEAM 2 — Frontend <-> Backend (API level, emulating the browser exactly:
Origin header + X-Session-Id + SSE reading, as skim-frontend/src/api.js does).
"""

from __future__ import annotations

import pytest

from fixtures.make_pdfs import pdf_bytes

pytestmark = pytest.mark.integration


def test_S2_1_upload_stream_path_the_ui_uses(sess, require_live, record):
    res = sess.upload_stream("zephyr")
    stages = [e.data.get("stage") for e in res.by("stage") if isinstance(e.data, dict)]
    done = res.by("done")[0].data if res.by("done") else {}
    listing = sess.client.get("/documents", headers=sess.headers()).json()
    record("S2.1", "SEAM2", "upload via /documents/upload/stream (frontend's path)",
           inp="POST /documents/upload/stream (multipart) w/ X-Session-Id + Origin",
           ideal="SSE stage frames (extract/chunk/index) then done{doc_id,chunk_count}; GET /documents shows it",
           actual=f"content_type={res.content_type!r}; stages={stages}; done={done}; listed={listing['total_documents']} docs / {listing['total_vectors']} vectors")
    assert "text/event-stream" in res.content_type
    assert {"extract", "chunk", "index"}.issubset(set(stages))
    assert done.get("doc_id") and done.get("chunk_count", 0) > 0
    assert listing["total_documents"] == 1


def test_S2_2_chat_returns_consumable_sse(sess, require_live, record):
    sess.upload_stream("zephyr")
    res = sess.chat("What temperature does the Zephyr-7 reactor run at?")
    kinds = {e.event for e in res.events}
    record("S2.2", "SEAM2", "chat returns an SSE stream the UI can consume",
           inp="POST /chat (SSE)",
           ideal="content-type text/event-stream; well-formed token/metadata/done frames",
           actual=f"content_type={res.content_type!r}; event_kinds={sorted(kinds)}; token_frames={len(res.by('token'))}")
    assert "text/event-stream" in res.content_type
    assert {"token", "metadata", "done"}.issubset(kinds)


def test_S2_3_cors_allows_frontend_origin(live, cfg, require_live, record):
    origin = cfg.frontend_origin
    pre = live.request("OPTIONS", "/chat", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-session-id",
    })
    acao = pre.headers.get("access-control-allow-origin")
    acac = pre.headers.get("access-control-allow-credentials")
    # a disallowed origin should NOT be echoed
    bad = live.get("/health", headers={"Origin": "http://evil.example"})
    bad_acao = bad.headers.get("access-control-allow-origin")
    record("S2.3", "SEAM2", "CORS allows the frontend origin",
           inp=f"OPTIONS /chat preflight from Origin {origin}; and a disallowed origin",
           ideal="preflight echoes allow-origin=frontend origin, allow-credentials=true; disallowed origin not echoed",
           actual=f"preflight status={pre.status_code}; allow_origin={acao!r}; allow_credentials={acac!r}; disallowed_echoed={bad_acao!r}")
    assert acao == origin
    assert acac == "true"
    assert bad_acao != "http://evil.example"


def test_S2_4_error_shape_has_detail(live, sess, require_live, record):
    # missing X-Session-Id -> 400
    r_missing = live.post("/chat", json={"query": "hi"})
    # bad body (empty query) with a valid header -> 422
    r_bad = live.post("/chat", headers=sess.headers({"Content-Type": "application/json"}), json={"query": ""})
    # non-PDF upload -> 415
    r_type = live.post("/documents/upload", headers=sess.headers(),
                       files={"file": ("notes.txt", b"hi", "text/plain")})
    def has_detail(r):
        try:
            return "detail" in r.json()
        except Exception:  # noqa: BLE001
            return False
    record("S2.4", "SEAM2", "error responses carry a `detail` field (UI shape)",
           inp="missing header (400), empty query (422), non-PDF (415)",
           ideal="each is a JSON body with a `detail` field, as api.js reads",
           actual=f"missing={r_missing.status_code}/detail={has_detail(r_missing)}; "
                  f"bad_body={r_bad.status_code}/detail={has_detail(r_bad)}; "
                  f"bad_type={r_type.status_code}/detail={has_detail(r_type)}")
    assert r_missing.status_code == 400 and has_detail(r_missing)
    assert r_bad.status_code == 422 and has_detail(r_bad)
    assert r_type.status_code == 415 and has_detail(r_type)
