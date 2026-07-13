"""Tiny Server-Sent-Events parser for tests.

Turns a raw `text/event-stream` body into a list of (event, data) pairs, where
data is JSON-decoded to a dict when possible, else the raw string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class SSEEvent:
    event: str
    data: object  # dict when JSON, else str


def parse_sse(raw: str) -> list[SSEEvent]:
    events: list[SSEEvent] = []
    for block in raw.split("\n\n"):
        block = block.strip("\n")
        if not block.strip():
            continue
        event = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        raw_data = "\n".join(data_lines)
        try:
            data: object = json.loads(raw_data)
        except (json.JSONDecodeError, ValueError):
            data = raw_data
        events.append(SSEEvent(event=event, data=data))
    return events


def events_of(events: list[SSEEvent], name: str) -> list[SSEEvent]:
    return [e for e in events if e.event == name]


def tokens_text(events: list[SSEEvent]) -> str:
    """Concatenate all token-event texts in order."""
    out = []
    for e in events:
        if e.event == "token" and isinstance(e.data, dict):
            out.append(e.data.get("text", ""))
    return "".join(out)
