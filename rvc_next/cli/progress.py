"""Show a job's progress in the terminal, fed by the same events the web UI receives."""

from __future__ import annotations

import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TaskProgressColumn, TextColumn, TimeElapsedColumn

from rvc_next.cli.output import err_console

if TYPE_CHECKING:
    from rvc_next.core.context import Services
    from rvc_next.core.events import EventBase
    from rvc_next.core.jobs.models import Job


@contextmanager
def job_progress(services: Services, enabled: bool = True) -> Generator[None, None, None]:
    """A Rich progress bar on stderr for every job that updates while the block runs."""
    from rvc_next.core.events.models import JobLogEvent, JobUpdatedEvent

    if not enabled or not sys.stderr.isatty():
        unsubscribe: Callable[[], None] = services.events.subscribe(lambda e: None)
        try:
            yield
        finally:
            unsubscribe()
        return
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=err_console, transient=True)
    tasks: dict[str, TaskID] = {}

    def handler(event: EventBase) -> None:
        if isinstance(event, JobUpdatedEvent):
            job = event.job
            label = job.title + (f" · {_step_title(job)}" if job.step else "")
            if job.id not in tasks:
                tasks[job.id] = progress.add_task(label, total=1.0)
            progress.update(tasks[job.id], description=label, completed=job.progress or 0.0, total=1.0 if job.progress is not None else None)
        elif isinstance(event, JobLogEvent):
            for line in event.lines:
                if line.startswith(("error", "Traceback")):
                    progress.console.print(line, style="red", markup=False, highlight=False)

    unsubscribe = services.events.subscribe(handler)
    try:
        with progress:
            yield
    finally:
        unsubscribe()


def _step_title(job: Job) -> str:
    for s in job.steps:
        if s.id == job.step:
            return s.title
    return job.step or ""


def finish(job: Job) -> None:
    """Report a foreground job's end, and exit with the domain error's code if it failed."""
    import typer

    from rvc_next.cli.output import console

    if job.state == "completed":
        return
    if job.state == "cancelled":
        err_console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit(130)
    error = job.error
    message = error.message if error else job.state
    err_console.print(f"[red]{message}[/red]")
    if error and error.detail.get("assets"):
        console.print(f"Install with: rvc-next assets download {' '.join(error.detail['assets'])}", style="yellow")
    raise typer.Exit(_exit_code(error.code if error else ""))


def _exit_code(code: str) -> int:
    from rvc_next.core import errors

    for cls in (errors.NotFoundError, errors.ConflictError, errors.InvalidPathError, errors.ValidationError, errors.AssetMissingError, errors.DeviceError, errors.ComputeError):
        if cls.code == code:
            return cls.exit_code
    return 3 if code == "busy" else 4 if code == "invalid_model" else 1
