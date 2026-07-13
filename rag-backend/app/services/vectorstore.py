"""FAISS-backed vector store with a parallel chunk/metadata store.

Design:
  * Vectors are unit-normalized (see embeddings.py), and we use an inner-product
    index (IndexFlatIP) so a search score IS the cosine similarity.
  * The flat index is wrapped in an IndexIDMap2 so each chunk gets a stable int64
    id. That lets DELETE /documents/{id} remove a document's chunks by id without
    rebuilding the whole index.
  * A parallel dict `records[id] -> {doc_id, filename, page, chunk_idx, text}`
    holds everything FAISS doesn't. Both the index and this dict are persisted to
    data/index/ so they survive restarts.

The module exposes a single process-wide `store` instance.
"""

from __future__ import annotations

import json
import threading

import faiss
import numpy as np

from ..configs.config import settings
from ..utils.logger import get_logger

log = get_logger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.dim = settings.embedding_dim
        self.index: faiss.IndexIDMap2 | None = None
        self.records: dict[int, dict] = {}   # faiss id -> metadata + text
        self.next_id: int = 0
        self._load()

    # ---------- persistence ----------
    def _new_index(self) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def _load(self) -> None:
        idx_path = settings.faiss_index_path
        store_path = settings.chunk_store_path
        if idx_path.exists() and store_path.exists():
            try:
                self.index = faiss.read_index(str(idx_path))
                with open(store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.records = {int(k): v for k, v in data.get("records", {}).items()}
                self.next_id = int(data.get("next_id", 0))
                log.info("Loaded vector store: %d vectors, %d chunk records",
                         self.index.ntotal, len(self.records))
                return
            except Exception as exc:  # noqa: BLE001 - corrupt store shouldn't wedge startup
                log.warning("Failed to load existing index (%s); starting fresh", exc)
        self.index = self._new_index()
        self.records = {}
        self.next_id = 0

    def _persist(self) -> None:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(settings.faiss_index_path))
        payload = {"next_id": self.next_id,
                   "records": {str(k): v for k, v in self.records.items()}}
        tmp = settings.chunk_store_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp.replace(settings.chunk_store_path)  # atomic-ish swap

    # ---------- writes ----------
    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> int:
        """Add vectors + aligned metadata. Returns number added."""
        if len(vectors) == 0:
            return 0
        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas length mismatch")
        with self._lock:
            ids = np.arange(self.next_id, self.next_id + len(vectors), dtype="int64")
            self.index.add_with_ids(vectors.astype("float32"), ids)
            for i, meta in zip(ids.tolist(), metadatas):
                self.records[i] = meta
            self.next_id += len(vectors)
            self._persist()
        return len(vectors)

    def delete_document(self, doc_id: str, session_id: str | None = None) -> int:
        """Remove all chunks for a doc_id. Returns number removed.

        When session_id is given, only chunks owned by that session are removed
        (a session can't delete another session's document).
        """
        with self._lock:
            ids = [
                i for i, r in self.records.items()
                if r.get("doc_id") == doc_id
                and (session_id is None or r.get("session_id") == session_id)
            ]
            if not ids:
                return 0
            self.index.remove_ids(np.array(ids, dtype="int64"))
            for i in ids:
                del self.records[i]
            self._persist()
        return len(ids)

    # ---------- reads ----------
    def search(self, query_vec: np.ndarray, k: int,
               session_id: str | None = None, doc_id: str | None = None) -> list[dict]:
        """Cosine (inner-product) search, best first.

        Optional session_id / doc_id filters keep results scoped. Because the flat
        index can't filter server-side, we over-fetch a candidate pool, drop
        records that don't match the filters, then take the top k.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        filtered = session_id is not None or doc_id is not None
        pool = min(self.index.ntotal, max(k * 10, 50)) if filtered else min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vec.astype("float32"), pool)
        results: list[dict] = []
        for score, i in zip(scores[0].tolist(), ids[0].tolist()):
            if i == -1:
                continue
            rec = self.records.get(i)
            if rec is None:
                continue
            if session_id is not None and rec.get("session_id") != session_id:
                continue
            if doc_id is not None and rec.get("doc_id") != doc_id:
                continue
            results.append({**rec, "score": float(score), "faiss_id": i})
            if len(results) >= k:
                break
        return results

    def list_documents(self, session_id: str | None = None) -> list[dict]:
        """Aggregate stored chunks into one entry per document.

        When session_id is given, only that session's documents are returned.
        """
        docs: dict[str, dict] = {}
        for r in self.records.values():
            if session_id is not None and r.get("session_id") != session_id:
                continue
            d = docs.setdefault(r["doc_id"], {
                "doc_id": r["doc_id"],
                "filename": r.get("filename"),
                "chunk_count": 0,
                "pages": set(),
            })
            d["chunk_count"] += 1
            if r.get("page") is not None:
                d["pages"].add(r["page"])
        out = []
        for d in docs.values():
            out.append({
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "chunk_count": d["chunk_count"],
                "page_count": len(d["pages"]),
            })
        return out

    def stats(self) -> dict:
        return {"vectors": int(self.index.ntotal) if self.index else 0,
                "documents": len(self.list_documents())}


# process-wide singleton
store = VectorStore()
