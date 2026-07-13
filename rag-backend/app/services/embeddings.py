"""Embedding model wrapper (sentence-transformers all-MiniLM-L6-v2, CPU).

The model is loaded lazily ONCE and reused. All vectors are L2-normalized so a
FAISS inner-product search is equivalent to cosine similarity.
"""

from __future__ import annotations

import numpy as np

from ..configs.config import settings
from ..utils.logger import get_logger

log = get_logger(__name__)

_model = None


def get_model():
    """Lazily load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        # Imported here so app startup / --help doesn't pay the heavy import cost.
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model '%s' (CPU)...", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model, device="cpu")
        dim = _model.get_sentence_embedding_dimension()
        if dim != settings.embedding_dim:
            log.warning("embedding_dim in config (%s) != model dim (%s); using model dim",
                        settings.embedding_dim, dim)
        log.info("Embedding model loaded (dim=%s)", dim)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts -> float32 array (n, dim), L2-normalized rows.

    Normalization is done by the encoder (normalize_embeddings=True), which makes
    every row a unit vector -> inner product == cosine similarity.
    """
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype="float32")
    model = get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,   # unit vectors => IP == cosine
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """Embed a single query -> float32 array (1, dim), L2-normalized."""
    return embed_texts([text])
