"""
Benchmark primitives.

The core idea: wrap a *streaming* generation and record a timestamp for every
token as it arrives. From that single timeline we can derive every metric you
care about:

  - TTFT  (Time To First Token): first_token_time - start   -> "responsiveness"
  - ITL   (Inter-Token Latency): gaps between consecutive tokens -> "smoothness"
  - E2E   (End-to-End latency):  last_token_time - start
  - Throughput: completion_tokens / E2E   (tokens per second)

These are the standard metrics used to compare inference engines and
quantization methods.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    import torch
except Exception:  # torch may be absent in a pure-analysis env
    torch = None


# ----------------------------------------------------------------------------
# One generation's timeline
# ----------------------------------------------------------------------------
@dataclass
class StreamTrace:
    start: float
    token_times: list[float] = field(default_factory=list)  # absolute perf_counter stamps
    end: float = 0.0

    @property
    def completion_tokens(self) -> int:
        return len(self.token_times)

    @property
    def ttft(self) -> float:
        return (self.token_times[0] - self.start) if self.token_times else float("nan")

    @property
    def e2e(self) -> float:
        return self.end - self.start

    @property
    def itls(self) -> list[float]:
        t = self.token_times
        return [t[i] - t[i - 1] for i in range(1, len(t))]

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.e2e if self.e2e > 0 else 0.0


def time_stream(stream: Iterable[str]) -> StreamTrace:
    """Consume a token stream (an iterable of text pieces) and time each piece."""
    trace = StreamTrace(start=time.perf_counter())
    for piece in stream:
        if piece:
            trace.token_times.append(time.perf_counter())
    trace.end = time.perf_counter()
    return trace


# ----------------------------------------------------------------------------
# Aggregation across many runs
# ----------------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(label: str, traces: list[StreamTrace], extra: Optional[dict] = None) -> dict:
    ttfts = [t.ttft for t in traces if t.completion_tokens]
    itls = [x for t in traces for x in t.itls]
    tps = [t.tokens_per_second for t in traces if t.e2e > 0]
    e2es = [t.e2e for t in traces if t.e2e > 0]
    out = {
        "label": label,
        "runs": len(traces),
        "ttft_ms_mean": round(1000 * (sum(ttfts) / len(ttfts)), 1) if ttfts else None,
        "ttft_ms_p90": round(1000 * percentile(ttfts, 90), 1) if ttfts else None,
        "itl_ms_mean": round(1000 * (sum(itls) / len(itls)), 2) if itls else None,
        "tokens_per_sec_mean": round(sum(tps) / len(tps), 1) if tps else None,
        "e2e_s_mean": round(sum(e2es) / len(e2es), 2) if e2es else None,
    }
    if extra:
        out.update(extra)
    return out


def summarize_generations(label: str, gens: list[dict], extra: Optional[dict] = None) -> dict:
    """Aggregate a list of engine.generate() result dicts.

    Used for the vLLM offline path, where a single generate() call returns the full
    response with a correct completion_tokens count (unlike the streaming timer, which
    the offline vLLM API would collapse to a single "token"). E2E latency and throughput
    are meaningful here; per-token TTFT/ITL are not available offline (that needs the vLLM
    server, measured in the serving phase), so they're reported as None.
    """
    e2es = [g["total_time_s"] for g in gens if g.get("total_time_s")]
    tps = [g["tokens_per_second"] for g in gens if g.get("tokens_per_second")]
    ctoks = [g["completion_tokens"] for g in gens if g.get("completion_tokens") is not None]
    out = {
        "label": label,
        "runs": len(gens),
        "ttft_ms_mean": None,   # offline vLLM has no per-token stream
        "ttft_ms_p90": None,
        "itl_ms_mean": None,
        "tokens_per_sec_mean": round(sum(tps) / len(tps), 1) if tps else None,
        "e2e_s_mean": round(sum(e2es) / len(e2es), 2) if e2es else None,
        "completion_tokens_mean": round(sum(ctoks) / len(ctoks), 1) if ctoks else None,
    }
    if extra:
        out.update(extra)
    return out


# ----------------------------------------------------------------------------
# GPU + saving + display
# ----------------------------------------------------------------------------
def gpu_memory_report() -> dict:
    if torch is None or not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device": props.name,
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "total_gb": round(props.total_memory / 1024**3, 2),
    }


def save_result(result: dict, results_dir: str = "benchmarks/results") -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir) / f"{result['label']}.json"
    path.write_text(json.dumps(result, indent=2))
    return str(path)


def format_table(rows: list[dict]) -> str:
    """Pretty comparison table. Uses tabulate if available, else a plain fallback."""
    if not rows:
        return "(no results)"
    cols = list(rows[0].keys())
    try:
        from tabulate import tabulate

        return tabulate([[r.get(c) for c in cols] for r in rows], headers=cols, tablefmt="github")
    except Exception:
        line = " | ".join(cols)
        sep = "-" * len(line)
        body = "\n".join(" | ".join(str(r.get(c)) for c in cols) for r in rows)
        return f"{line}\n{sep}\n{body}"
