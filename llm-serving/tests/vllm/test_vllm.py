"""
Pytest wrapper around the same checks used by run_suite.py.

Each test calls its check function and asserts the recorded status is acceptable
(PASS, or PARTIAL/N-A where the plan allows graceful degradation). This makes the
suite runnable via `pytest tests/vllm/ -v` for CI/re-runs, while `run_suite.py`
remains the way to produce the TEST_RESULTS.md document.

Note: these tests hit the LIVE server, so they need a reachable BASE_URL in .env.
"""

from __future__ import annotations

import pytest

import checks
from helpers import PASS, PARTIAL, NA, make_context

# A single shared context (and thus one health/model discovery) for the module.
CTX = None


@pytest.fixture(scope="session")
def ctx():
    global CTX
    if CTX is None:
        CTX = make_context()
    return CTX


OK = {PASS, PARTIAL, NA}


@pytest.mark.parametrize("check", checks.ALL_CHECKS, ids=lambda c: c.__name__.replace("check_", ""))
def test_check(check, ctx):
    r = check(ctx)
    assert r.status in OK, f"{r.id} {r.status}: {r.note}\n  actual: {r.actual}"
