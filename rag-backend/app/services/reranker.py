"""Cross-encoder reranking stage (BAAI/bge-reranker-base).

Second-stage reranker for the retrieval pipeline. A cross-encoder reads each
(query, chunk) pair TOGETHER and outputs a single relevance score, which is far
more accurate than the first-stage bi-encoder cosine similarity (that embeds the
query and the chunk independently). We use it to re-order the FAISS candidates so
only the truly relevant chunks reach the prompt.

Mirrors embeddings.py: the model is loaded ONCE (at app startup) and reused for
every request — never reloaded per query. This module is deliberately independent
of retrieval: it only scores and sorts pairs. The selection policy (threshold,
final top-k) lives in the retrieval layer.
"""

from __future__ import annotations

import numpy as np

from ..configs.config import settings
from ..utils.logger import get_logger

log = get_logger(__name__)

# Process-wide singleton, loaded lazily and reused (see load_model / get_model).
_model = None


def get_model():
    """Return the cached CrossEncoder, loading it once on first use.

    Loading is deferred so importing this module (e.g. for --help) doesn't pay the
    heavy model-load cost; `load_model()` warms it eagerly at startup.
    """
    global _model
    if _model is None:
        # Imported here so the import cost is paid only when the model is needed.
        import torch
        from sentence_transformers import CrossEncoder

        log.info("Loading reranker '%s'...", settings.rerank_model)
        _model = CrossEncoder(
            settings.rerank_model,
            activation_fn=torch.nn.Sigmoid(),  # map raw logits -> relevance in [0, 1]
            max_length=512,
            device="cpu",
        )
        log.info("Reranker loaded successfully.")
    return _model


def load_model() -> None:
    """Eagerly load the reranker at application startup (called from app.main)."""
    get_model()


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Score every (query, chunk) pair and return the chunks sorted best-first.

    Each returned item is a shallow copy of the input chunk with a `rerank_score`
    (relevance in [0, 1]) added. This is pure scoring + sorting — it applies no
    threshold and no top-k cut (that is the retrieval layer's responsibility).
    """
    if not chunks:
        return []

    model = get_model()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    log.info("Reranking %d chunks", len(chunks))

    scores = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    scored = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, np.asarray(scores, dtype="float32"))
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    if scored:
        log.info("Top reranker score: %.2f", scored[0]["rerank_score"])
    return scored
