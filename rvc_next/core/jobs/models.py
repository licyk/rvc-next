"""Job records."""

from typing import Any, Literal

from pydantic import Field

from rvc_next.core.record import Record

JobKind = Literal["convert", "separate", "train", "index", "download", "merge", "extract", "export"]
JobState = Literal["queued", "waiting", "running", "cancelling", "cancelled", "failed", "completed", "interrupted"]
StepState = Literal["pending", "running", "done", "skipped", "failed", "cancelled"]
Resource = Literal["gpu", "cpu", "io"]

ACTIVE_STATES: frozenset[str] = frozenset({"queued", "waiting", "running", "cancelling"})
FINISHED_STATES: frozenset[str] = frozenset({"cancelled", "failed", "completed", "interrupted"})


class ErrorInfo(Record):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class JobStep(Record):
    id: str
    title: str
    state: StepState = "pending"
    progress: float | None = None


class Transfer(Record):
    """Bytes moved by a download, with the rate over the last few seconds."""

    done_bytes: int
    total_bytes: int
    bytes_per_second: float | None = None
    """None until a rate can be measured."""
    eta_seconds: float | None = None


class Job(Record):
    id: str
    kind: JobKind
    title: str
    state: JobState = "queued"
    waiting_for: Literal["gpu", "cpu", "asset"] | None = None
    progress: float | None = None
    """0..1; None while indeterminate."""
    step: str | None = None
    steps: list[JobStep] = Field(default_factory=list)
    error: ErrorInfo | None = None
    result: dict[str, Any] | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    can_retry: bool = False
    can_run_anyway: bool = False
    transfer: Transfer | None = None
    """Set while a download moves bytes; None otherwise."""


class JobPage(Record):
    items: list[Job]
    next_cursor: str | None = None


class JobLog(Record):
    job_id: str
    offset: int
    next_offset: int
    lines: list[str]
