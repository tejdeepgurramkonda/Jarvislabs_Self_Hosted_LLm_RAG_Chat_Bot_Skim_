"""
Compare every saved experiment result into one table (and charts).

Run after you've run some experiments:
    python benchmarks/compare.py

Reads benchmarks/results/*.json and prints a side-by-side comparison of
model-weights VRAM, load time, E2E latency, and throughput, then saves a
throughput chart and a memory chart.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# The columns we care about, in display order.
COLUMNS = [
    "label",
    "quantization",
    "weights_gb",
    "load_time_s",
    "ttft_ms_mean",
    "itl_ms_mean",
    "tokens_per_sec_mean",
    "e2e_s_mean",
    "completion_tokens_mean",
]

# Preferred row order (falls back to alphabetical for anything unlisted).
ORDER = ["bf16", "fp16", "fp8", "gptq", "awq"]


def load_results() -> list[dict]:
    rows = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
    rows.sort(key=lambda r: (ORDER.index(r["label"]) if r.get("label") in ORDER else 99,
                             r.get("label", "")))
    return rows


def main() -> None:
    rows = load_results()
    if not rows:
        print(f"No results in {RESULTS_DIR}. Run an experiment first, e.g. "
              "`python -m experiments.bf16.run`.")
        return

    table = [{c: r.get(c) for c in COLUMNS} for r in rows]

    try:
        from tabulate import tabulate

        print(tabulate([[t[c] for c in COLUMNS] for t in table], headers=COLUMNS, tablefmt="github"))
    except Exception:
        print(" | ".join(COLUMNS))
        for t in table:
            print(" | ".join(str(t[c]) for c in COLUMNS))

    # Charts: throughput and model-weights memory.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r["label"] for r in rows]

        tps = [r.get("tokens_per_sec_mean") or 0 for r in rows]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, tps, color="#4C78A8")
        plt.ylabel("tokens / sec (mean)")
        plt.title("Throughput by quantization method")
        plt.tight_layout()
        out_tps = RESULTS_DIR / "throughput.png"
        plt.savefig(out_tps)
        plt.close()

        mem = [r.get("weights_gb") or 0 for r in rows]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, mem, color="#F58518")
        plt.ylabel("model weights (GB)")
        plt.title("Model-weights VRAM by quantization method")
        plt.tight_layout()
        out_mem = RESULTS_DIR / "weights.png"
        plt.savefig(out_mem)
        plt.close()

        print(f"\nCharts saved -> {out_tps}, {out_mem}")
    except Exception as exc:  # matplotlib optional
        print(f"\n(Charts skipped: {exc})")


if __name__ == "__main__":
    main()
