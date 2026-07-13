"""
Shared helpers for the vLLM functional test suite.

Loads config from `.env`, builds OpenAI + raw httpx clients, provides a streaming
timer (TTFT / tokens-per-sec), and a small `Result` recorder used by both the
pytest suite (`test_vllm.py`) and the standalone runner (`run_suite.py`).

Nothing here touches the FastAPI backend or the frontend — it only speaks HTTP to
the OpenAI-compatible vLLM server named by `MODEL` in `.env`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    base_url: str          # e.g. https://host/v1
    api_key: str           # real key, or "EMPTY" when the server has no auth
    model: str             # served model name, e.g. "qwen"

    @property
    def root_url(self) -> str:
        """Server root (strip a trailing /v1) — used for /health."""
        b = self.base_url.rstrip("/")
        return b[:-3].rstrip("/") if b.endswith("/v1") else b


def load_config() -> Config:
    base_url = os.environ.get("BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError(
            "BASE_URL is not set. Fill tests/vllm/.env (BASE_URL=https://.../v1)."
        )
    return Config(
        base_url=base_url,
        api_key=os.environ.get("API_KEY", "").strip() or "EMPTY",
        model=os.environ.get("MODEL", "qwen").strip() or "qwen",
    )


def make_client(cfg: Config, api_key: str | None = None, timeout: float = 60.0) -> OpenAI:
    return OpenAI(
        base_url=cfg.base_url,
        api_key=api_key if api_key is not None else cfg.api_key,
        timeout=timeout,
        max_retries=0,  # we want to observe the first response, not retried behaviour
    )


def raw_client(cfg: Config, timeout: float = 60.0, auth: bool = True) -> httpx.Client:
    """Raw httpx client for status-code level tests (models list, malformed body).

    Includes the Authorization header by default because the vLLM server is launched
    with --api-key, so every /v1 route requires it. Pass auth=False to omit it.
    """
    headers = {}
    if auth and cfg.api_key and cfg.api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return httpx.Client(base_url=cfg.base_url, timeout=timeout, headers=headers)


# --------------------------------------------------------------------------- #
# Result recording
# --------------------------------------------------------------------------- #
PASS, FAIL, PARTIAL, NA, ERROR = "PASS", "FAIL", "PARTIAL", "N/A", "ERROR"


@dataclass
class Result:
    id: str
    area: str
    description: str
    request: str
    ideal: str
    actual: str = ""
    status: str = ""
    note: str = ""


@dataclass
class Context:
    """Shared state threaded through every check."""
    cfg: Config
    client: OpenAI
    max_model_len: int | None = None          # learned from /v1/models (H1)
    perf: dict = field(default_factory=dict)  # perf/concurrency numbers for the tables
    results: list[Result] = field(default_factory=list)

    def record(self, r: Result) -> Result:
        self.results.append(r)
        return r


def make_context() -> Context:
    cfg = load_config()
    return Context(cfg=cfg, client=make_client(cfg))


# --------------------------------------------------------------------------- #
# Streaming timer
# --------------------------------------------------------------------------- #
@dataclass
class StreamStats:
    text: str
    ttft: float                 # seconds to first content token
    total: float                # seconds to final chunk
    n_chunks: int               # number of content-bearing chunks
    completion_tokens: int | None
    tokens_per_sec: float | None
    finish_reason: str | None


def timed_stream(client: OpenAI, *, model: str, messages: list[dict], **kw) -> StreamStats:
    """Run a streaming chat completion and measure TTFT + tokens/sec.

    Uses stream_options.include_usage so vLLM reports exact completion_tokens in
    the final chunk; falls back to a whitespace token estimate if absent.
    """
    start = time.perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    n_chunks = 0
    completion_tokens: int | None = None
    finish_reason: str | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **kw,
    )
    for chunk in stream:
        if chunk.usage is not None:
            completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta.content if choice.delta else None
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - start
            parts.append(delta)
            n_chunks += 1
    total = time.perf_counter() - start

    text = "".join(parts)
    if completion_tokens is None:
        completion_tokens = len(text.split()) or None

    gen_time = max(total - (ttft or 0.0), 1e-6)
    tps = (completion_tokens / gen_time) if completion_tokens else None

    return StreamStats(
        text=text,
        ttft=ttft if ttft is not None else total,
        total=total,
        n_chunks=n_chunks,
        completion_tokens=completion_tokens,
        tokens_per_sec=tps,
        finish_reason=finish_reason,
    )


def trim(s: str, n: int = 600) -> str:
    s = (s or "").replace("\r", " ")
    return s if len(s) <= n else s[:n] + f" …[+{len(s) - n} chars]"
