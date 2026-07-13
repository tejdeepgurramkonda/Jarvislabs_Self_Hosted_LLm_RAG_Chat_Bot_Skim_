"""Thin wrapper around the OpenAI SDK pointed at my vLLM server.

The vLLM server speaks the OpenAI Chat Completions API, so we reuse the official
`openai` client with a custom base_url. Streaming is added in Phase D; here we
expose a lazily-created client plus a cheap reachability probe for /health.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from ..configs.config import settings
from ..utils.logger import get_logger

log = get_logger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a process-wide OpenAI client aimed at <vllm_base_url>/v1."""
    global _client
    if _client is None:
        log.info("Creating OpenAI client -> %s (model=%s)", settings.llm_v1_url, settings.llm_model)
        _client = OpenAI(
            base_url=settings.llm_v1_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # health probe / streaming should fail fast, not silently retry
        )
    return _client


def stream_chat(messages: list[dict], max_tokens: int | None = None,
                temperature: float | None = None) -> Iterator[str]:
    """Stream a chat completion from vLLM, yielding text deltas token-by-token.

    Raises on connection/timeout errors so the caller (rag.py) can emit its
    graceful fallback. The final chunk carries no content and is skipped.
    """
    client = get_client()
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        max_tokens=max_tokens if max_tokens is not None else settings.max_tokens,
        temperature=temperature if temperature is not None else settings.temperature,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        token = getattr(delta, "content", None)
        if token:
            yield token


def check_llm() -> dict:
    """Quick, cheap probe of the vLLM endpoint for /health.

    Returns a dict with reachability + which model answered, never raises.
    """
    result = {"reachable": False, "model": settings.llm_model, "detail": None}
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        result["reachable"] = True
        # vLLM echoes the served model id; surface it for sanity.
        result["model"] = getattr(resp, "model", settings.llm_model)
    except Exception as exc:  # noqa: BLE001 - health must never crash the app
        log.warning("LLM health probe failed: %s", exc)
        result["detail"] = f"{type(exc).__name__}: {exc}"
    return result
