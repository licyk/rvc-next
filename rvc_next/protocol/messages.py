"""Worker events, written one JSON object per line on a worker's stdout.

A worker receives one JSON request file (its path is the only argument) and reports with these
events. The types follow the original ``tools/pymss_webui.py`` protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EventType = Literal["progress", "log", "metric", "result", "error", "step", "output"]


@dataclass
class ProgressEvent:
    """Overall progress 0..1, or None while indeterminate; ``step`` names the current step."""

    value: float | None
    step: str | None = None
    message: str | None = None
    type: str = "progress"


@dataclass
class StepEvent:
    """A step changed state (pending, running, done, skipped, failed, cancelled)."""

    id: str
    state: str
    progress: float | None = None
    title: str | None = None
    type: str = "step"


@dataclass
class LogEvent:
    level: str
    """debug, info, warning or error."""
    message: str
    type: str = "log"


@dataclass
class MetricEvent:
    """A training measurement: losses and the learning rate at an epoch and step."""

    epoch: int
    step: int
    values: dict[str, float]
    lr: float | None = None
    type: str = "metric"


@dataclass
class OutputEvent:
    """A file the worker finished writing (a separated stem, a checkpoint)."""

    path: str
    label: str
    kind: str = "stem"
    source: str | None = None
    type: str = "output"


@dataclass
class ResultEvent:
    data: dict[str, Any] = field(default_factory=dict)
    type: str = "result"


@dataclass
class ErrorEvent:
    """``code`` is a domain code the core maps to its errors: asset_missing, invalid_input, compute_unavailable, …"""

    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    type: str = "error"


WorkerEvent = ProgressEvent | StepEvent | LogEvent | MetricEvent | OutputEvent | ResultEvent | ErrorEvent

_TYPES: dict[str, type] = {
    "progress": ProgressEvent,
    "step": StepEvent,
    "log": LogEvent,
    "metric": MetricEvent,
    "output": OutputEvent,
    "result": ResultEvent,
    "error": ErrorEvent,
}


def to_dict(event: WorkerEvent) -> dict[str, Any]:
    return asdict(event)


def from_dict(data: dict[str, Any]) -> WorkerEvent | None:
    """Rebuild an event; None for an unknown type or malformed fields."""
    cls = _TYPES.get(str(data.get("type")))
    if cls is None:
        return None
    try:
        return cls(**data)
    except TypeError:
        return None


# Exit code a worker uses to ask for a retry in a fresh process (the DirectML fp32 fallback).
EXIT_RETRY = 75
EXIT_CANCELLED = 130

# ``detail.reason`` of a ``compute_unavailable`` error when the device ran out of memory; the core
# unloads every cached model when it sees it (as ComfyUI does on an OOM).
OUT_OF_MEMORY = "out_of_memory"
