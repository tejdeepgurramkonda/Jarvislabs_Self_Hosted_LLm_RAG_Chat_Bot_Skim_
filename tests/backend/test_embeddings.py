"""Embedding unit tests (E1-E4) using the real MiniLM model (cached, offline)."""

from __future__ import annotations

import numpy as np

from app.configs.config import settings
from app.services import embeddings


def test_E1_deterministic(record):
    v1 = embeddings.embed_texts(["hello world"])
    v2 = embeddings.embed_texts(["hello world"])
    same = np.array_equal(v1, v2)
    record("E1", "Embeddings", "same input -> same vector",
           inp="embed_texts(['hello world']) x2",
           ideal="identical vectors",
           actual=f"array_equal={same}; max_abs_diff={float(np.abs(v1 - v2).max())}")
    assert same


def test_E2_l2_normalized(record):
    v = embeddings.embed_texts(["a normalized vector should be unit length"])
    norm = float(np.linalg.norm(v[0]))
    record("E2", "Embeddings", "vectors are L2-normalized",
           inp="norm(embed_texts([...])[0])",
           ideal="L2 norm ~= 1.0 (+/- 1e-5)",
           actual=f"norm={norm:.6f}")
    assert abs(norm - 1.0) < 1e-5


def test_E3_dimension_and_dtype(record):
    v = embeddings.embed_texts(["one", "two", "three"])
    record("E3", "Embeddings", "correct dimension + dtype",
           inp="embed_texts(['one','two','three'])",
           ideal=f"shape (3, {settings.embedding_dim}), float32",
           actual=f"shape={v.shape}; dtype={v.dtype}")
    assert v.shape == (3, settings.embedding_dim)
    assert v.dtype == np.float32


def test_E4_empty_input(record):
    v = embeddings.embed_texts([])
    record("E4", "Embeddings", "empty input handled",
           inp="embed_texts([])",
           ideal=f"shape (0, {settings.embedding_dim}), no error",
           actual=f"shape={v.shape}")
    assert v.shape == (0, settings.embedding_dim)
