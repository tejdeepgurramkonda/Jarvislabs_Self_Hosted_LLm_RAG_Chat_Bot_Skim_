"""Fake LLM seams used to keep the suite deterministic and offline.

Only the remote LLM is mocked; chunking/embedding/retrieval use the real code.
The chat pipeline calls `app.services.llm_client.stream_chat(messages, ...)`; these
fakes stand in for it via monkeypatch. `check_llm` fakes stand in for /health.
"""

from __future__ import annotations

from collections.abc import Iterator


class StreamRecorder:
    """A callable stand-in for `stream_chat` that records the messages it received.

    Usage:
        rec = StreamRecorder(["Hello", " from", " mock"])
        monkeypatch.setattr("app.services.llm_client.stream_chat", rec)
        ...
        assert rec.called and "CONTEXT" in rec.messages[1]["content"]
    """

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.called = False
        self.messages: list[dict] | None = None
        self.kwargs: dict | None = None

    def __call__(self, messages: list[dict], **kwargs) -> Iterator[str]:
        self.called = True
        self.messages = messages
        self.kwargs = kwargs
        yield from self.tokens


def raise_immediately(messages: list[dict], **kwargs) -> Iterator[str]:
    """stream_chat that fails before yielding anything (e.g. connection/timeout)."""
    raise RuntimeError("simulated LLM connection failure")
    yield  # pragma: no cover - makes this a generator


def yield_then_raise(tokens: list[str]):
    """Factory: stream_chat that yields a few tokens, then fails mid-stream."""
    def _gen(messages: list[dict], **kwargs) -> Iterator[str]:
        for t in tokens:
            yield t
        raise RuntimeError("simulated mid-stream LLM failure")
    return _gen


# ---- /health fakes ----
def check_llm_ok() -> dict:
    return {"reachable": True, "model": "qwen", "detail": None}


def check_llm_down() -> dict:
    return {"reachable": False, "model": "qwen",
            "detail": "APIConnectionError: simulated unreachable"}
