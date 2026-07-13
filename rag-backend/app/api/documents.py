"""Document management endpoints (all scoped to the caller's X-Session-Id).

  POST   /documents/upload         multipart PDF -> ingest -> {doc_id, chunk_count}
  POST   /documents/upload/stream  multipart PDF -> SSE per-stage ingest progress
  GET    /documents                list this session's documents
  DELETE /documents/{doc_id}       remove one of this session's documents
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..configs.config import settings
from ..schemas.response import (
    DeleteResponse,
    DocumentListResponse,
    UploadResponse,
)
from ..services.ingestion import IngestionError, ingest_pdf, ingest_pdf_streaming
from ..services.vectorstore import store
from ..utils.logger import get_logger
from .deps import get_session_id

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger(__name__)

MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # 40MB (matches the UI copy)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _read_pdf(file: UploadFile) -> tuple[bytes, str]:
    """Validate content-type/size and return (bytes, filename). Raises HTTPException."""
    filename = file.filename or "upload.pdf"
    is_pdf = filename.lower().endswith(".pdf") or (file.content_type == "application/pdf")
    if not is_pdf:
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF is too large (max 40MB).")
    return data, filename


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
) -> UploadResponse:
    data, filename = await _read_pdf(file)
    try:
        result = ingest_pdf(data, filename, session_id)
    except IngestionError as exc:
        log.warning("Ingestion rejected '%s': %s", filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error ingesting '%s'", filename)
        raise HTTPException(status_code=500, detail="Failed to process the PDF.") from exc
    return UploadResponse(**result)


@router.post("/upload/stream")
async def upload_document_stream(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
) -> StreamingResponse:
    """Stream ingest progress as Server-Sent Events.

    Emits `event: stage` frames (extract / chunk / index) then a terminal
    `event: done` with {doc_id, filename, chunk_count, page_count}. A friendly
    failure is sent as `event: error` instead of crashing the stream.
    """
    data, filename = await _read_pdf(file)

    def _stream() -> Iterator[str]:
        try:
            for ev in ingest_pdf_streaming(data, filename, session_id):
                event = "done" if ev.get("stage") == "done" else "stage"
                yield _sse(event, ev)
        except IngestionError as exc:
            log.warning("Ingestion rejected '%s': %s", filename, exc)
            yield _sse("error", {"detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected error ingesting '%s'", filename)
            yield _sse("error", {"detail": "Failed to process the PDF."})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(session_id: str = Depends(get_session_id)) -> DocumentListResponse:
    docs = store.list_documents(session_id=session_id)
    total_vectors = sum(d["chunk_count"] for d in docs)
    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
        total_vectors=total_vectors,
    )


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, session_id: str = Depends(get_session_id)) -> DeleteResponse:
    removed = store.delete_document(doc_id, session_id=session_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document found with id '{doc_id}'.")

    # also drop the saved PDF file (best-effort)
    pdf_path = settings.uploads_dir / f"{doc_id}.pdf"
    try:
        pdf_path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("Could not delete file %s: %s", pdf_path, exc)

    log.info("Deleted document %s (%d chunks, session=%s)", doc_id, removed, session_id)
    return DeleteResponse(doc_id=doc_id, deleted_chunks=removed, status="deleted")
