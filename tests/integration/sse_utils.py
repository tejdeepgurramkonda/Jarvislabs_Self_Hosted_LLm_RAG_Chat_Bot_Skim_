"""SSE reader for integration tests — reads a real streamed HTTP response
(httpx) frame-by-frame, the same way the frontend's api.js does, and records
timing (TTFT) so we can prove tokens arrive incrementally.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


@dataclass
class SSEEvent:
    event: str
    data: object          # dict when JSON-decodable, else str
    t: float              # seconds since request start when this frame completed


@dataclass
class StreamResult:
    status: int
    content_type: str
    events: list[SSEEvent] = field(default_factory=list)
    ttft: float | None = None      # time to first `token` frame
    total: float | None = None     # time to last frame

    def by(self, name: str) -> list[SSEEvent]:
        return [e for e in self.events if e.event == name]

    def tokens_text(self) -> str:
        return "".join(
            e.data.get("text", "") for e in self.events
            if e.event == "token" and isinstance(e.data, dict)
        )


def _parse_frame(frame: str) -> tuple[str, str]:
    event = "message"
    data_parts: list[str] = []
    for raw in frame.split("\n"):
        line = raw.rstrip("\r")
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_parts.append(line[len("data:"):].strip())
    return event, "\n".join(data_parts)


def stream_sse(client, method: str, url: str, **kwargs) -> StreamResult:
    """Open a streaming request and collect SSE frames with timing.

    Works with httpx.Client. Any non-2xx returns a StreamResult with the status
    and no events (caller inspects .status). Never raises on HTTP status.
    """
    start = time.perf_counter()
    with client.stream(method, url, **kwargs) as resp:
        ct = resp.headers.get("content-type", "")
        res = StreamResult(status=resp.status_code, content_type=ct)
        if resp.status_code >= 400 or "text/event-stream" not in ct:
            # drain so the connection closes cleanly; body may be JSON error
            try:
                res.events.append(SSEEvent("_body", resp.read().decode("utf-8", "replace"),
                                           time.perf_counter() - start))
            except Exception:  # noqa: BLE001
                pass
            return res
        buffer = ""
        for chunk in resp.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                if not frame.strip():
                    continue
                event, data_str = _parse_frame(frame)
                if not data_str:
                    continue
                try:
                    data: object = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    data = data_str
                now = time.perf_counter() - start
                if event == "token" and res.ttft is None:
                    res.ttft = now
                res.events.append(SSEEvent(event, data, now))
        res.total = time.perf_counter() - start
    return res
