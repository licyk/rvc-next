"""Reading and writing worker events as JSON lines."""

from __future__ import annotations

import json
import sys
import threading
from typing import IO, Any

from rvc_next.protocol.messages import LogEvent, WorkerEvent, from_dict, to_dict

PREFIX = "@rvc "
"""Marks an event line, so stray prints from libraries are kept as log text, not parsed."""


class Emitter:
    """Write events to a stream (stdout by default), one per line, safe across threads."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()

    def emit(self, event: WorkerEvent) -> None:
        line = PREFIX + json.dumps(to_dict(event), ensure_ascii=False, default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()


def parse_line(line: str) -> WorkerEvent:
    """An event line becomes its event; any other line becomes an info log line."""
    text = line.rstrip("\r\n")
    if text.startswith(PREFIX):
        try:
            data: Any = json.loads(text[len(PREFIX) :])
        except ValueError:
            data = None
        if isinstance(data, dict):
            event = from_dict(data)
            if event is not None:
                return event
    return LogEvent(level="info", message=text)
