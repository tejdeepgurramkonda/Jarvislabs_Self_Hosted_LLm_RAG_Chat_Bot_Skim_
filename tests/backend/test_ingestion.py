"""Document ingestion endpoint tests (I1-I7). Real chunking/embedding/FAISS."""

from __future__ import annotations

from fpdf import FPDF

from fixtures.make_pdfs import pdf_bytes


def _upload(client, headers, name, data, content_type="application/pdf"):
    return client.post("/documents/upload", headers=headers,
                       files={"file": (name, data, content_type)})


def test_I1_valid_pdf_ingested_indexed_persisted(client, headers, session_id, app_store, app_settings, record):
    resp = _upload(client, headers, "paris.pdf", pdf_bytes("paris"))
    body = resp.json()
    # store gained vectors, metadata present, files persisted to the temp dir
    recs = [r for r in app_store.records.values() if r.get("session_id") == session_id]
    sample = recs[0] if recs else {}
    files_written = app_settings.faiss_index_path.exists() and app_settings.chunk_store_path.exists()
    record("I1", "Ingestion", "valid PDF -> chunks embedded + indexed + persisted",
           inp="POST /documents/upload (paris.pdf) + X-Session-Id",
           ideal="200; chunk_count>0, page_count>=1, doc_id; vectors added; metadata has doc_id/session_id/filename/page/chunk_idx/text; index files written",
           actual=f"{resp.status_code}; body={body}; vectors={app_store.index.ntotal}; meta_keys={sorted(sample)}; files_written={files_written}")
    assert resp.status_code == 200
    assert body["chunk_count"] > 0 and body["page_count"] >= 1 and body["doc_id"]
    assert app_store.index.ntotal == body["chunk_count"]
    assert recs and all(k in sample for k in ("doc_id", "session_id", "filename", "page", "chunk_idx", "text"))
    assert files_written


def test_I2_non_pdf_rejected(client, headers, record):
    resp = _upload(client, headers, "notes.txt", b"just some text", "text/plain")
    record("I2", "Ingestion", "non-PDF rejected cleanly",
           inp="upload notes.txt (text/plain)",
           ideal="415, no crash",
           actual=f"{resp.status_code}; {resp.json()}")
    assert resp.status_code == 415


def test_I3_corrupt_pdf_handled(client, headers, record):
    resp = _upload(client, headers, "bad.pdf", b"%PDF-1.4 not really a valid pdf body")
    body = resp.json()
    handled = resp.status_code != 200 and "detail" in body  # clean JSON error, no hang/crash
    is_4xx = 400 <= resp.status_code < 500
    record("I3", "Ingestion", "corrupt PDF handled gracefully",
           inp="upload bad.pdf (garbage bytes, application/pdf)",
           ideal="clean 4xx (415/422) ideally; must not hang/crash",
           actual=f"{resp.status_code}; {body}",
           status="PASS" if is_4xx else "PARTIAL",
           note="" if is_4xx else "Handled cleanly (JSON error, no crash) but returns 500, not a 4xx. "
                                  "Suggest mapping PdfReadError/PdfStreamError to a 422 in upload_document.")
    # graceful handling is the hard requirement; 4xx-vs-500 is flagged as PARTIAL above
    assert handled


def test_I4a_empty_file_rejected(client, headers, record):
    resp = _upload(client, headers, "empty.pdf", b"")
    record("I4a", "Ingestion", "empty (0-byte) file rejected",
           inp="upload empty.pdf (0 bytes)",
           ideal="400 (empty file), no crash",
           actual=f"{resp.status_code}; {resp.json()}")
    assert resp.status_code == 400


def test_I4b_no_text_pdf_rejected(client, headers, record):
    blank = FPDF()
    blank.add_page()  # a page with no text
    resp = _upload(client, headers, "blank.pdf", bytes(blank.output()))
    record("I4b", "Ingestion", "PDF with no extractable text handled",
           inp="upload blank.pdf (page, no text)",
           ideal="422 IngestionError ('no extractable text'), no crash",
           actual=f"{resp.status_code}; {resp.json()}")
    assert resp.status_code == 422


def test_I5_listing_session_scoped(client, headers, session_id, make_headers, record):
    _upload(client, headers, "paris.pdf", pdf_bytes("paris"))
    mine = client.get("/documents", headers=headers).json()
    other = client.get("/documents", headers=make_headers("someone-else")).json()
    record("I5", "Ingestion", "GET /documents lists doc, session-scoped",
           inp="upload as session A, then GET /documents as A and as B",
           ideal="A sees 1 doc with correct totals; B sees 0",
           actual=f"A: total_documents={mine['total_documents']} total_vectors={mine['total_vectors']}; B: total_documents={other['total_documents']}")
    assert mine["total_documents"] == 1
    assert mine["total_vectors"] == mine["documents"][0]["chunk_count"] > 0
    assert other["total_documents"] == 0


def test_I6_delete_removes_from_index_and_store(client, headers, make_headers, app_store, app_settings, record):
    up = _upload(client, headers, "paris.pdf", pdf_bytes("paris")).json()
    doc_id = up["doc_id"]
    before = app_store.index.ntotal
    # another session cannot delete it
    cross = client.delete(f"/documents/{doc_id}", headers=make_headers("intruder"))
    # owner deletes it
    resp = client.delete(f"/documents/{doc_id}", headers=headers)
    after = app_store.index.ntotal
    pdf_gone = not (app_settings.uploads_dir / f"{doc_id}.pdf").exists()
    unknown = client.delete("/documents/does-not-exist", headers=headers)
    record("I6", "Ingestion", "DELETE removes chunks from index + store + file",
           inp="upload, cross-session delete, owner delete, unknown-id delete",
           ideal="cross-session -> 404; owner -> 200 deleted_chunks>0, vectors drop, file unlinked; unknown -> 404",
           actual=f"cross={cross.status_code}; owner={resp.status_code} {resp.json()}; vectors {before}->{after}; pdf_gone={pdf_gone}; unknown={unknown.status_code}")
    assert cross.status_code == 404
    assert resp.status_code == 200 and resp.json()["deleted_chunks"] == before > 0
    assert after == 0 and pdf_gone
    assert unknown.status_code == 404


def test_I7_missing_session_header(client, record):
    resp = client.post("/documents/upload", files={"file": ("paris.pdf", pdf_bytes("paris"), "application/pdf")})
    record("I7", "Ingestion", "missing X-Session-Id -> 400",
           inp="POST /documents/upload without X-Session-Id",
           ideal="400 (Missing X-Session-Id header)",
           actual=f"{resp.status_code}; {resp.json()}")
    assert resp.status_code == 400
