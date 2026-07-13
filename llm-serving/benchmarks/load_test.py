"""
Concurrency / stress test against a running vLLM OpenAI-compatible server.

This is the tool for the metrics that only appear UNDER LOAD: how TTFT and
throughput change as concurrent users pile up. It streams responses so it can
time the first token precisely.

Usage:
    pip install httpx
    python benchmarks/load_test.py \
        --url https://<your-jarvislabs-url>/v1 \
        --api-key <key or "dummy"> \
        --model qwen

It ramps through concurrency levels [1, 5, 10, 20, 50] and prints a table.
"""

import argparse
import asyncio
import time

import httpx


async def one_request(client, url, api_key, model, prompt, max_tokens):
    """Fire one streaming chat request; return (ttft, e2e, token_count)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.perf_counter()
    first = None
    tokens = 0
    async with client.stream("POST", f"{url}/chat/completions", json=payload, headers=headers) as r:
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            if line.strip() == "data: [DONE]":
                break
            if first is None:
                first = time.perf_counter()
            tokens += 1
    end = time.perf_counter()
    ttft = (first - start) if first else float("nan")
    return ttft, end - start, tokens


async def run_level(url, api_key, model, concurrency, prompt, max_tokens):
    """Fire `concurrency` requests at once and aggregate their metrics."""
    limits = httpx.Limits(max_connections=concurrency + 5)
    async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
        wall_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                one_request(client, url, api_key, model, prompt, max_tokens)
                for _ in range(concurrency)
            ]
        )
        wall = time.perf_counter() - wall_start

    ttfts = [r[0] for r in results if r[0] == r[0]]  # drop NaNs
    total_tokens = sum(r[2] for r in results)
    return {
        "concurrency": concurrency,
        "avg_ttft_ms": round(1000 * sum(ttfts) / len(ttfts), 1) if ttfts else None,
        "avg_e2e_s": round(sum(r[1] for r in results) / len(results), 2),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / wall, 1) if wall > 0 else 0.0,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Base URL, e.g. https://xxx/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--prompt", default="Explain how a rainbow forms, in detail.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--levels", default="1,5,10,20,50")
    args = ap.parse_args()

    rows = []
    for c in [int(x) for x in args.levels.split(",")]:
        print(f"Running concurrency={c} ...")
        rows.append(
            await run_level(args.url, args.api_key, args.model, c, args.prompt, args.max_tokens)
        )

    try:
        from tabulate import tabulate

        cols = list(rows[0].keys())
        print("\n" + tabulate([[r[c] for c in cols] for r in rows], headers=cols, tablefmt="github"))
    except Exception:
        print("\n" + "\n".join(str(r) for r in rows))


if __name__ == "__main__":
    asyncio.run(main())
