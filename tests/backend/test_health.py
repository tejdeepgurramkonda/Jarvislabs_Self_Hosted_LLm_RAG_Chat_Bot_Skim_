"""Health endpoint tests (H1, H2). The LLM probe is mocked."""

from __future__ import annotations

import mocks


def test_H1_health_ok_reports_llm(client, monkeypatch, record):
    monkeypatch.setattr("app.api.health.check_llm", mocks.check_llm_ok)
    resp = client.get("/health")
    body = resp.json()
    record("H1", "Health", "/health ok + reports LLM config",
           inp="GET /health (check_llm mocked reachable)",
           ideal="200; status=ok; llm.reachable=true; base_url & model present",
           actual=f"{resp.status_code}; {body}")
    assert resp.status_code == 200
    assert body["status"] == "ok"
    assert body["llm"]["reachable"] is True
    assert body["llm"]["base_url"] and body["llm"]["model"]


def test_H2_health_degrades_when_llm_down(client, monkeypatch, record):
    monkeypatch.setattr("app.api.health.check_llm", mocks.check_llm_down)
    resp = client.get("/health")
    body = resp.json()
    record("H2", "Health", "health degrades gracefully when LLM down",
           inp="GET /health (check_llm mocked failing)",
           ideal="200 (never crashes); llm.reachable=false; detail set",
           actual=f"{resp.status_code}; reachable={body['llm']['reachable']}; detail={body['llm']['detail']!r}")
    assert resp.status_code == 200
    assert body["llm"]["reachable"] is False
    assert body["llm"]["detail"]
