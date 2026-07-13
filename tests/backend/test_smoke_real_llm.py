"""SM1 — real-LLM smoke test (MARKED + SKIPPED by default).

This is the ONE test that hits the live vLLM server, to confirm end-to-end wiring.
It is skipped unless RUN_REAL_LLM=1, and also auto-skips if the endpoint isn't
reachable. To run it:

    # resume JarvisLabs, start the server, then set the live endpoint:
    export LLM_BASE_URL=https://<host>.notebooksn.jarvislabs.net/v1
    export LLM_API_KEY=<key>
    RUN_REAL_LLM=1 pytest tests/backend/test_smoke_real_llm.py -v
"""

from __future__ import annotations

import os

import pytest

from fixtures.make_pdfs import pdf_bytes
from sse_utils import parse_sse, events_of, tokens_text

pytestmark = pytest.mark.real_llm

RUN = os.environ.get("RUN_REAL_LLM") == "1"


@pytest.mark.skipif(not RUN, reason="real-LLM smoke disabled (set RUN_REAL_LLM=1 to enable)")
def test_SM1_real_llm_end_to_end(client, headers, record):
    # confirm the endpoint is actually reachable, else skip (not fail)
    from app.services.llm_client import check_llm
    probe = check_llm()
    if not probe["reachable"]:
        pytest.skip(f"vLLM endpoint not reachable: {probe['detail']}")

    client.post("/documents/upload", headers=headers,
                files={"file": ("paris.pdf", pdf_bytes("paris"), "application/pdf")})
    resp = client.post("/chat", headers=headers, json={"query": "Where is the Eiffel Tower located?"})
    events = parse_sse(resp.text)
    text = tokens_text(events)
    meta = events_of(events, "metadata")[0].data
    record("SM1", "Smoke", "real end-to-end /chat against live vLLM",
           inp="RUN_REAL_LLM=1; POST /chat 'Where is the Eiffel Tower located?'",
           ideal="streams a real grounded answer; metadata.status=answered",
           actual=f"answer={text[:120]!r}; status={meta.get('status')}; sources={len(meta.get('sources',[]))}")
    assert len(events_of(events, "token")) >= 1
    assert meta["status"] == "answered"
