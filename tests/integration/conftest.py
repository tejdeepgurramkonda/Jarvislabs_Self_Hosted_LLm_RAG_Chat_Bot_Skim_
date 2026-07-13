"""Integration-test harness for the 3-service RAG system (real services, no mocks).

- `live` is an httpx client to the RUNNING backend (`BACKEND_URL`); most tests use it.
- `require_live` / `require_llm` skip cleanly (never error) when a service is down.
- SEAM-1 negatives use `inprocess` — an isolated in-process backend (throwaway temp
  index, monkeypatched LLM target) so pointing at a bad URL/key never disturbs the
  running server, while the network call to vLLM stays real.
- Data is throwaway: each test uses a unique X-Session-Id and deletes its docs.
- The run generates TEST_RESULTS.md (per-test seam/input/actual/ideal/status + perf).
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

import httpx
import pytest

from sse_utils import stream_sse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RAG_BACKEND = REPO_ROOT / "rag-backend"

sys.path.insert(0, str(HERE))                 # local imports (sse_utils, fixtures)
sys.path.insert(0, str(RAG_BACKEND))          # allow `import app...` for in-process backend

# Redirect the in-process backend's storage to a throwaway dir BEFORE app import.
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="integ_"))
os.environ["INDEX_DIR"] = str(_TMP_ROOT / "index")
os.environ["UPLOADS_DIR"] = str(_TMP_ROOT / "uploads")


# --------------------------------------------------------------------------- #
# Config (test .env, with fallbacks to the app's real config files)
# --------------------------------------------------------------------------- #
def _parse_env(path: Path) -> dict:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


class Cfg:
    def __init__(self):
        integ = _parse_env(HERE / ".env")
        backend = _parse_env(RAG_BACKEND / ".env")
        vllm = _parse_env(REPO_ROOT / "tests" / "vllm" / ".env")
        env = os.environ

        self.backend_url = (env.get("BACKEND_URL") or integ.get("BACKEND_URL")
                            or "http://localhost:8090").rstrip("/")
        self.frontend_origin = (env.get("FRONTEND_ORIGIN") or integ.get("FRONTEND_ORIGIN")
                                or "http://localhost:5173").rstrip("/")
        # vLLM creds (for the bad-key seam test): test .env -> backend .env -> vllm .env
        self.vllm_base_url = (integ.get("VLLM_BASE_URL") or backend.get("LLM_BASE_URL")
                              or vllm.get("BASE_URL") or "http://localhost:8000")
        self.vllm_api_key = (integ.get("VLLM_API_KEY") or backend.get("LLM_API_KEY")
                             or vllm.get("API_KEY") or "EMPTY")


@pytest.fixture(scope="session")
def cfg() -> Cfg:
    return Cfg()


# --------------------------------------------------------------------------- #
# Live backend client + readiness gates
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def live(cfg) -> httpx.Client:
    with httpx.Client(base_url=cfg.backend_url, timeout=90.0) as c:
        yield c


@pytest.fixture(scope="session")
def _backend_up(cfg) -> bool:
    try:
        r = httpx.get(cfg.backend_url + "/health", timeout=10)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def _llm_up(cfg, _backend_up) -> bool:
    if not _backend_up:
        return False
    try:
        return bool(httpx.get(cfg.backend_url + "/health", timeout=20).json()["llm"]["reachable"])
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture
def require_live(_backend_up, cfg):
    if not _backend_up:
        pytest.skip(f"backend not reachable at {cfg.backend_url} — start it (uvicorn ... --port 8090)")


@pytest.fixture
def require_llm(require_live, _llm_up, cfg):
    if not _llm_up:
        pytest.skip("vLLM not reachable via backend /health — resume JarvisLabs + fill rag-backend/.env")


# --------------------------------------------------------------------------- #
# A throwaway session against the LIVE backend (auto-cleans its documents)
# --------------------------------------------------------------------------- #
class LiveSession:
    def __init__(self, client: httpx.Client, origin: str):
        self.client = client
        self.id = "itest-" + uuid.uuid4().hex[:12]
        self.origin = origin
        self.created: list[str] = []

    def headers(self, extra: dict | None = None) -> dict:
        h = {"X-Session-Id": self.id, "Origin": self.origin}
        if extra:
            h.update(extra)
        return h

    def upload_stream(self, doc_key: str):
        """Upload via the endpoint the frontend uses; returns the StreamResult."""
        from fixtures.make_pdfs import pdf_bytes
        files = {"file": (f"{doc_key}.pdf", pdf_bytes(doc_key), "application/pdf")}
        res = stream_sse(self.client, "POST", "/documents/upload/stream",
                         headers=self.headers(), files=files)
        for ev in res.by("done"):
            if isinstance(ev.data, dict) and ev.data.get("doc_id"):
                self.created.append(ev.data["doc_id"])
        return res

    def chat(self, query: str, doc_id: str | None = None):
        body = {"query": query, "doc_id": doc_id}
        return stream_sse(self.client, "POST", "/chat",
                          headers=self.headers({"Content-Type": "application/json"}), json=body)

    def cleanup(self):
        for doc_id in self.created:
            try:
                self.client.delete(f"/documents/{doc_id}", headers=self.headers())
            except Exception:  # noqa: BLE001
                pass


@pytest.fixture
def sess(live, cfg):
    s = LiveSession(live, cfg.frontend_origin)
    yield s
    s.cleanup()


# --------------------------------------------------------------------------- #
# In-process backend for SEAM-1 negatives (isolated; real network to vLLM)
# --------------------------------------------------------------------------- #
class InProcessBackend:
    def __init__(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.services.vectorstore import store
        self._app = app
        self.store = store
        self.client = TestClient(app)
        self._reset_store()

    def _reset_store(self):
        s = self.store
        with s._lock:
            s.index = s._new_index()
            s.records = {}
            s.next_id = 0

    def set_llm(self, base_url: str, api_key: str):
        from app.configs.config import settings
        from app.services import llm_client
        settings.llm_base_url = base_url
        settings.llm_api_key = api_key
        llm_client._client = None  # force rebuild with new target

    def ingest(self, doc_key: str, session_id: str) -> dict:
        from app.services import ingestion
        from fixtures.make_pdfs import pdf_bytes
        return ingestion.ingest_pdf(pdf_bytes(doc_key), f"{doc_key}.pdf", session_id)

    def chat(self, query: str, session_id: str, doc_id: str | None = None,
             threshold: float | None = None):
        body: dict = {"query": query, "doc_id": doc_id}
        if threshold is not None:
            body["threshold"] = threshold
        return stream_sse(self.client, "POST", "/chat",
                          headers={"X-Session-Id": session_id, "Content-Type": "application/json"},
                          json=body)


@pytest.fixture
def inprocess():
    be = InProcessBackend()
    yield be
    be._reset_store()


# --------------------------------------------------------------------------- #
# Results + perf recorder -> TEST_RESULTS.md
# --------------------------------------------------------------------------- #
_RESULTS: dict[str, dict] = {}
_OUTCOME: dict[str, str] = {}
_PERF: list[dict] = []


@pytest.fixture
def record(request):
    def _record(tid, seam, desc, inp, ideal, actual, status="PASS", note=""):
        _RESULTS[request.node.nodeid] = {
            "id": tid, "seam": seam, "desc": desc, "input": inp,
            "ideal": ideal, "actual": actual, "status": status, "note": note,
        }
    return _record


@pytest.fixture
def perf():
    def _perf(name, **kv):
        _PERF.append({"name": name, **kv})
    return _perf


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        _OUTCOME[item.nodeid] = rep.outcome
    elif rep.when == "setup" and rep.skipped:
        _OUTCOME[item.nodeid] = "skipped"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: real-service integration test")


def _write_results():
    for nodeid, r in _RESULTS.items():
        oc = _OUTCOME.get(nodeid, "unknown")
        if oc == "failed":
            r["status"] = "FAIL"
        elif oc == "skipped":
            r["status"] = "SKIP"
    seam_order = ["SEAM1", "SEAM2", "FULL", "CROSS"]
    rows = sorted(_RESULTS.values(),
                  key=lambda r: (seam_order.index(r["seam"]) if r["seam"] in seam_order else 9, r["id"]))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    emoji = {"PASS": "✅", "FAIL": "❌", "PARTIAL": "🟡", "SKIP": "⚪"}
    import datetime as dt
    L: list[str] = []
    a = L.append
    a("# RAG System — Integration Test Results\n")
    a(f"- **Run at:** {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    a("- **Type:** connections BETWEEN services, real requests/network/streaming (no mocks).")
    a("- **Labels:** SEAM1/SEAM2 = pairwise integration; FULL/CROSS = end-to-end (API-level; real-browser layer deferred).\n")
    a("## Summary\n")
    a("| Status | Count |\n|--------|-------|")
    for k in ("PASS", "FAIL", "PARTIAL", "SKIP"):
        a(f"| {emoji[k]} {k} | {counts.get(k, 0)} |")
    a(f"| **Total** | **{len(rows)}** |\n")
    a("| ID | Seam | Test | Status |\n|----|------|------|--------|")
    for r in rows:
        a(f"| {r['id']} | {r['seam']} | {r['desc']} | {emoji.get(r['status'],'?')} {r['status']} |")
    if _PERF:
        a("\n## Performance (streamed full-chain calls)\n")
        keys = sorted({k for p in _PERF for k in p if k != "name"})
        a("| call | " + " | ".join(keys) + " |")
        a("|" + "------|" * (len(keys) + 1))
        for p in _PERF:
            a(f"| {p['name']} | " + " | ".join(str(p.get(k, "")) for k in keys) + " |")
    a("\n## Details\n")
    for r in rows:
        a(f"### {r['id']} — {r['desc']}  {emoji.get(r['status'],'?')} **{r['status']}**\n")
        a(f"- **Seam:** {r['seam']}")
        a(f"- **Input:** {r['input']}")
        a(f"- **Ideal:** {r['ideal']}")
        a(f"- **Actual:** {r['actual']}")
        if r["note"]:
            a(f"- **Note:** {r['note']}")
        a("")
    (HERE / "TEST_RESULTS.md").write_text("\n".join(L), encoding="utf-8")


def pytest_sessionfinish(session, exitstatus):
    import shutil
    try:
        if _RESULTS:
            _write_results()
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
