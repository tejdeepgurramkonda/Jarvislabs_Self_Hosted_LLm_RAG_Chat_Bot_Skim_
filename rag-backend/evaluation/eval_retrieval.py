"""Offline retrieval evaluation: precision@k and recall@k on a labeled set.

This is an OFFLINE quality check, NOT a per-request live check. Given a small
labeled set (queries + which chunks are truly relevant), it builds an isolated
FAISS index from the labeled corpus and measures how well the retriever ranks the
relevant chunks into the top k.

It reuses the SAME embedding model and normalized inner-product (cosine) search
as production (app/services), but on a throwaway in-memory index so it never
touches data/index/.

Definitions (per query, given the top-k retrieved chunk ids):
    precision@k = (# retrieved in top-k that are relevant) / k
    recall@k    = (# retrieved in top-k that are relevant) / (# relevant)
Reported numbers are the mean over all queries (macro-average).

Usage (from the rag-backend/ directory):
    python evaluation/eval_retrieval.py                 # k = config top_k
    python evaluation/eval_retrieval.py --k 1 --k 3     # multiple cut-offs
    python evaluation/eval_retrieval.py --file my_set.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

# Allow `import app...` when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.configs.config import settings          # noqa: E402
from app.services import embeddings              # noqa: E402

DEFAULT_SET = Path(__file__).resolve().parent / "labeled_set.json"


# --------------------------------------------------------------------------- #
# Isolated index built from the labeled corpus (does not touch the real store)
# --------------------------------------------------------------------------- #
class _EvalIndex:
    def __init__(self, corpus: list[dict]) -> None:
        self.ids = [c["id"] for c in corpus]
        texts = [c["text"] for c in corpus]
        vectors = embeddings.embed_texts(texts)          # L2-normalized, float32
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine
        self.index.add(vectors)

    def search(self, query: str, k: int) -> list[str]:
        """Return the ids of the top-k chunks for a query, best first."""
        qvec = embeddings.embed_query(query)
        k = min(k, self.index.ntotal)
        _scores, idxs = self.index.search(qvec, k)
        return [self.ids[i] for i in idxs[0].tolist() if i != -1]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def evaluate(labeled: dict, ks: list[int]) -> dict:
    index = _EvalIndex(labeled["corpus"])
    queries = labeled["queries"]
    max_k = max(ks)

    per_query = []
    sums = {k: {"p": 0.0, "r": 0.0} for k in ks}
    for q in queries:
        relevant = set(q["relevant"])
        retrieved = index.search(q["query"], max_k)
        row = {"query": q["query"], "retrieved": retrieved, "relevant": sorted(relevant)}
        for k in ks:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            row[f"p@{k}"] = p
            row[f"r@{k}"] = r
            sums[k]["p"] += p
            sums[k]["r"] += r
        per_query.append(row)

    n = len(queries) or 1
    means = {k: {"precision": sums[k]["p"] / n, "recall": sums[k]["r"] / n} for k in ks}
    return {"per_query": per_query, "means": means, "num_queries": len(queries)}


# --------------------------------------------------------------------------- #
# CLI / reporting
# --------------------------------------------------------------------------- #
def _print_report(labeled: dict, ks: list[int], report: dict) -> None:
    print(f"\nEmbedding model : {settings.embedding_model}")
    print(f"Labeled set     : {report['num_queries']} queries, {len(labeled['corpus'])} chunks")
    print(f"Cut-offs (k)    : {ks}\n")

    for row in report["per_query"]:
        print(f"Q: {row['query']}")
        print(f"   relevant : {row['relevant']}")
        print(f"   retrieved: {row['retrieved']}")
        metrics = "   " + "  ".join(
            f"P@{k}={row[f'p@{k}']:.2f} R@{k}={row[f'r@{k}']:.2f}" for k in ks
        )
        print(metrics + "\n")

    print("=" * 48)
    print("MACRO-AVERAGE")
    for k in ks:
        m = report["means"][k]
        print(f"  k={k}:  precision@{k}={m['precision']:.3f}   recall@{k}={m['recall']:.3f}")
    print("=" * 48)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation (precision@k / recall@k).")
    parser.add_argument("--file", type=Path, default=DEFAULT_SET, help="Path to labeled set JSON.")
    parser.add_argument("--k", type=int, action="append", dest="ks",
                        help="Cut-off k (repeatable). Defaults to config top_k.")
    args = parser.parse_args()

    ks = args.ks or [settings.top_k]
    ks = sorted(set(ks))

    with open(args.file, "r", encoding="utf-8") as f:
        labeled = json.load(f)

    report = evaluate(labeled, ks)
    _print_report(labeled, ks, report)


if __name__ == "__main__":
    main()
