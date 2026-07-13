"""
Single-user streaming TTFT / ITL measurement against a running vLLM OpenAI server.

The offline benchmark (experiments/*) can't produce per-token timing — the vLLM batch
API returns the whole completion at once. This script streams RAG-length prompts (so TTFT
reflects realistic prefill over a long retrieved context) at concurrency 1 and records:

  - TTFT (time to first *content* token)
  - ITL  (mean inter-token latency = (e2e - ttft) / (content_tokens - 1))

It then merges ttft_ms_mean / ttft_ms_p90 / itl_ms_mean into benchmarks/results/<label>.json
(the file already holds memory + throughput + E2E from the offline run).

Usage:
    python benchmarks/stream_measure.py --url http://localhost:8000/v1 \
        --api-key dummy --model qwen --label bf16 --repeats 5
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

# Long-ish RAG-style prompts: a multi-paragraph context + a question. The long context
# is what makes TTFT meaningful (prefill-bound), which is the realistic RAG situation.
_CTX1 = (
    "Context:\n"
    "Vector databases store high-dimensional embeddings and support approximate nearest "
    "neighbor (ANN) search. Popular indexes include HNSW, which builds a navigable "
    "small-world graph for fast graph-traversal search, and IVF, which partitions vectors "
    "into clusters and searches only the nearest clusters. HNSW gives high recall at low "
    "latency but uses more memory; IVF is more memory-efficient but recall depends on how "
    "many clusters (nprobe) are searched. In a retrieval-augmented generation pipeline, a "
    "user query is embedded, the top-k nearest passages are retrieved from the index, and "
    "those passages are concatenated into the prompt as context for the language model. "
    "Chunk size matters: chunks that are too large dilute relevance and waste context "
    "window, while chunks that are too small lose surrounding meaning. A reranker can be "
    "applied after retrieval to reorder candidates by relevance before they reach the model.\n\n"
    "Question: In a RAG pipeline using this stack, explain the trade-off between HNSW and "
    "IVF indexes, and why chunk size affects answer quality."
)
_CTX2 = (
    "Context:\n"
    "Quantization reduces the numerical precision of model weights to save memory and "
    "sometimes speed up inference. FP16 and BF16 keep 16 bits per weight and are considered "
    "full precision for serving. FP8 halves that to 8 bits and, on Ada/Hopper GPUs with "
    "hardware FP8 support, can improve throughput while keeping quality close to BF16. "
    "INT4 methods such as GPTQ and AWQ compress weights to roughly 4 bits: GPTQ minimizes "
    "layer-wise reconstruction error, while AWQ protects the most salient weight channels "
    "identified from activation statistics. INT4 cuts weight memory by about 4x versus BF16, "
    "leaving far more VRAM for the KV cache, but can slightly degrade quality on hard "
    "reasoning tasks. The right choice depends on whether the deployment is memory-bound or "
    "compute-bound and how sensitive the workload is to small quality regressions.\n\n"
    "Question: For a memory-constrained RAG server on a 24 GB GPU, which quantization would "
    "you choose and what is the main risk?"
)
PROMPTS = [_CTX1, _CTX2]


async def one_request(client, url, api_key, model, prompt, max_tokens):
    """Stream one chat request; return (ttft_s, e2e_s, content_tokens)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.perf_counter()
    first = None
    tokens = 0
    async with client.stream("POST", f"{url}/chat/completions", json=payload, headers=headers) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"]
            except Exception:
                continue
            if delta.get("content"):  # count only real content tokens
                if first is None:
                    first = time.perf_counter()
                tokens += 1
    end = time.perf_counter()
    ttft = (first - start) if first else float("nan")
    return ttft, end - start, tokens


async def measure(url, api_key, model, repeats, max_tokens):
    ttfts, itls = [], []
    async with httpx.AsyncClient(timeout=180.0) as client:
        for _ in range(repeats):
            for prompt in PROMPTS:
                ttft, e2e, toks = await one_request(client, url, api_key, model, prompt, max_tokens)
                if ttft == ttft and toks > 1:  # not NaN, has tokens
                    ttfts.append(ttft)
                    itls.append((e2e - ttft) / (toks - 1))
    return ttfts, itls


def percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    ttfts, itls = asyncio.run(measure(args.url, args.api_key, args.model, args.repeats, args.max_tokens))
    ttft_mean = round(1000 * sum(ttfts) / len(ttfts), 1) if ttfts else None
    ttft_p90 = round(1000 * percentile(ttfts, 90), 1) if ttfts else None
    itl_mean = round(1000 * sum(itls) / len(itls), 2) if itls else None

    result_path = Path("benchmarks/results") / f"{args.label}.json"
    data = json.loads(result_path.read_text())
    data["ttft_ms_mean"] = ttft_mean
    data["ttft_ms_p90"] = ttft_p90
    data["itl_ms_mean"] = itl_mean
    result_path.write_text(json.dumps(data, indent=2))
    print(f"[{args.label}] ttft_ms_mean={ttft_mean} ttft_ms_p90={ttft_p90} "
          f"itl_ms_mean={itl_mean} (n={len(ttfts)}) -> {result_path}")


if __name__ == "__main__":
    main()
