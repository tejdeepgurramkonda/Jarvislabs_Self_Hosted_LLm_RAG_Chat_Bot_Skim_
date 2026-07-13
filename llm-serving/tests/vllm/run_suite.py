"""
Primary entrypoint: run every check in TEST_PLAN.md order against the live vLLM
server, then write TEST_RESULTS.md and print a pass/fail summary.

Errors are caught per-check (see checks.py), so one failing test never stops the
run. Usage:

    python run_suite.py            # from tests/vllm/
    python tests/vllm/run_suite.py # from repo root

Reads BASE_URL / API_KEY / MODEL from tests/vllm/.env.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on the status emoji; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from checks import ALL_CHECKS
from helpers import PASS, FAIL, PARTIAL, NA, ERROR, make_context, trim

HERE = Path(__file__).resolve().parent
RESULTS_MD = HERE / "TEST_RESULTS.md"

STATUS_EMOJI = {PASS: "✅", FAIL: "❌", PARTIAL: "🟡", NA: "⚪", ERROR: "💥"}


def run() -> int:
    ctx = make_context()
    print(f"Target: {ctx.cfg.base_url}  model={ctx.cfg.model}\n")

    for check in ALL_CHECKS:
        r = check(ctx)
        ctx.record(r)
        print(f"  {STATUS_EMOJI.get(r.status, '?')} {r.id:4} {r.status:7} {r.description}")

    write_results(ctx)
    print_summary(ctx)
    # non-zero exit if anything outright failed or errored
    bad = sum(1 for r in ctx.results if r.status in (FAIL, ERROR))
    return 1 if bad else 0


def _counts(results):
    c = {PASS: 0, FAIL: 0, PARTIAL: 0, NA: 0, ERROR: 0}
    for r in results:
        c[r.status] = c.get(r.status, 0) + 1
    return c


def print_summary(ctx):
    c = _counts(ctx.results)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {c[PASS]} pass, {c[FAIL]} fail, {c[PARTIAL]} partial, "
          f"{c[NA]} n/a, {c[ERROR]} error  (of {len(ctx.results)})")
    flagged = [r for r in ctx.results if r.status in (FAIL, ERROR, PARTIAL)]
    if flagged:
        print("Flagged:")
        for r in flagged:
            print(f"  {STATUS_EMOJI[r.status]} {r.id} {r.status}: {r.note or r.description}")
    print(f"\nFull results: {RESULTS_MD}")


def write_results(ctx):
    c = _counts(ctx.results)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    a = lines.append

    a("# vLLM Functional Test — Results\n")
    a(f"- **Run at:** {now}")
    a(f"- **Target:** `{ctx.cfg.base_url}`")
    a(f"- **Model:** `{ctx.cfg.model}`")
    a(f"- **max_model_len (reported):** {ctx.max_model_len}\n")

    a("## Summary\n")
    a("| Status | Count |")
    a("|--------|-------|")
    a(f"| ✅ PASS | {c[PASS]} |")
    a(f"| ❌ FAIL | {c[FAIL]} |")
    a(f"| 🟡 PARTIAL | {c[PARTIAL]} |")
    a(f"| ⚪ N/A | {c[NA]} |")
    a(f"| 💥 ERROR | {c[ERROR]} |")
    a(f"| **Total** | **{len(ctx.results)}** |\n")

    # quick status line per test
    a("| ID | Area | Test | Status |")
    a("|----|------|------|--------|")
    for r in ctx.results:
        a(f"| {r.id} | {r.area} | {r.description} | {STATUS_EMOJI.get(r.status,'?')} {r.status} |")
    a("")

    # performance table
    if "S2" in ctx.perf:
        p = ctx.perf["S2"]
        a("## Streaming performance (S2)\n")
        a("| Metric | Value |")
        a("|--------|-------|")
        a(f"| Time to first token | {p['ttft_s']} s |")
        a(f"| Total time | {p['total_s']} s |")
        a(f"| Completion tokens | {p['completion_tokens']} |")
        a(f"| Tokens/sec | {p['tokens_per_sec']} |")
        a(f"| Content chunks | {p['n_chunks']} |\n")

    conc = [ctx.perf[k] for k in ("X1", "X2", "X3") if k in ctx.perf]
    if conc:
        a("## Concurrency (X1/X2/X3)\n")
        a("| Concurrency | OK | Fail | Avg TTFT (s) | Wall (s) | Aggregate tok/s |")
        a("|-------------|----|------|--------------|----------|-----------------|")
        for p in conc:
            a(f"| {p['n']} | {p['ok']} | {p['fail']} | {p['avg_ttft_s']} | "
              f"{p['wall_s']} | {p['agg_tokens_per_sec']} |")
        a("")

    # per-test detail
    a("## Detailed results\n")
    for r in ctx.results:
        a(f"### {r.id} — {r.description}  {STATUS_EMOJI.get(r.status,'?')} **{r.status}**\n")
        a(f"- **Area:** {r.area}")
        a(f"- **Request:** {r.request}")
        a(f"- **Ideal:** {r.ideal}")
        a(f"- **Actual:** {trim(r.actual, 900)}")
        if r.note:
            a(f"- **Note:** {r.note}")
        a("")

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run())
