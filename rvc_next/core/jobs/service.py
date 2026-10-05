"""The job service: queue, resources, runners and history.

Every long operation is a job. A kind registers a factory that turns a request (plain JSON) into a
``JobSpec``; the request is stored, so a job can be retried after a restart. The server runs jobs on
worker threads chosen by a scheduler; the command line runs one job in the foreground.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rvc_next.core.clock import new_id, now_iso
from rvc_next.core.compute.service import ComputeService, MemoryHolder
from rvc_next.core.db import Database
from rvc_next.core.engine_errors import is_out_of_memory, map_engine_error, worker_error
from rvc_next.core.errors import ConflictError, NotFoundError, RvcNextError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import JobLogEvent, JobsClearedEvent, JobUpdatedEvent
from rvc_next.core.jobs.models import ACTIVE_STATES, FINISHED_STATES, ErrorInfo, Job, JobLog, JobPage, JobStep, Transfer
from rvc_next.core.process import kill_process_tree, start_process, worker_command
from rvc_next.core.settings import SettingsService

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 0.25
# Download speed is measured over this many seconds.
TRANSFER_WINDOW = 5.0
LOG_EVENT_INTERVAL = 0.25
DB_PROGRESS_INTERVAL = 2.0
MAX_LOG_EVENT_LINES = 50
JOBS_HOLDER = "jobs"


class JobCancelled(Exception):
    """Raised inside a job when it was cancelled."""


@dataclass
class JobSpec:
    title: str
    run: Callable[[JobContext], dict[str, Any] | None]
    resources: frozenset[str] = frozenset({"cpu"})
    steps: list[JobStep] = field(default_factory=list)
    priority: int = 0
    """Higher runs first; previews use 10."""


JobFactory = Callable[[dict[str, Any]], JobSpec]


@dataclass
class _Active:
    job: Job
    spec: JobSpec
    cancel: threading.Event = field(default_factory=threading.Event)
    run_anyway: bool = False
    holds_gpu: bool = False
    process: Any = None
    last_progress_emit: float = 0.0
    last_db_write: float = 0.0
    pending_lines: list[str] = field(default_factory=list)
    pending_offset: int | None = None
    last_log_emit: float = 0.0
    engine_token: Any = None
    transfer_samples: deque[tuple[float, int]] = field(default_factory=deque)


class JobContext:
    """What a running job uses to report and to check for cancellation."""

    def __init__(self, service: JobService, active: _Active) -> None:
        self._service = service
        self._active = active

    @property
    def job_id(self) -> str:
        return self._active.job.id

    @property
    def cancelled(self) -> bool:
        return self._active.cancel.is_set()

    def check_cancel(self) -> None:
        if self._active.cancel.is_set():
            raise JobCancelled()

    def cancel_token(self) -> Any:
        """An engine ``CancelToken`` that follows this job's cancellation."""
        if self._active.engine_token is None:
            from rvc_next.engine.cancel import CancelToken

            token = CancelToken()
            if self._active.cancel.is_set():
                token.cancel()
            self._active.engine_token = token
        return self._active.engine_token

    def progress(self, value: float | None, step: str | None = None) -> None:
        self._service._progress(self._active, value, step)

    def step(self, step_id: str, state: str, progress: float | None = None) -> None:
        self._service._step(self._active, step_id, state, progress)

    def transfer(self, done_bytes: int, total_bytes: int) -> None:
        """Bytes downloaded so far out of ``total_bytes``; the job gets a speed and an ETA from them."""
        self._service._transfer(self._active, done_bytes, total_bytes)

    def add_steps(self, steps: list[JobStep]) -> None:
        self._service._add_steps(self._active, steps)

    def log(self, line: str) -> None:
        self._service._log(self._active, line)

    def run_worker(self, module: str, request: dict[str, Any], on_event: Callable[[Any], None] | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
        """Run ``python -m <module> <request.json>``; return the result event's data.

        Progress, step and log events are applied to the job unless ``on_event`` handles them (it
        sees every event first and returns nothing). An error event becomes the matching domain error.
        """
        return self._service._run_worker(self._active, module, request, on_event, env)


class JobService:
    def __init__(self, settings: SettingsService, db: Database, events: EventBus, compute: ComputeService) -> None:
        self._settings = settings
        self._db = db
        self._events = events
        self._compute = compute
        self._logs = settings.data_dir / "jobs"
        self._factories: dict[str, JobFactory] = {}
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._active: dict[str, _Active] = {}
        self._running: dict[str, int] = {"gpu": 0, "cpu": 0, "io": 0}
        self._thread: threading.Thread | None = None
        # Freeing GPU memory waits while any job is queued or running (its models are in use).
        compute.add_holder(MemoryHolder("jobs", busy=lambda: "jobs" if self._active else None))
        self._stop = False
        self._mark_interrupted()

    # -- registration --------------------------------------------------------

    def register(self, kind: str, factory: JobFactory) -> None:
        self._factories[kind] = factory

    def capacity(self, resource: str) -> int:
        if resource == "gpu":
            return self._settings.settings.compute.gpu_jobs
        if resource == "cpu":
            return max(1, (os.cpu_count() or 2) // 2)
        return 4

    # -- submission ----------------------------------------------------------

    def submit(self, kind: str, request: dict[str, Any]) -> Job:
        """Queue a job; the scheduler runs it when its resources are free."""
        active = self._create(kind, request)
        with self._cond:
            self._active[active.job.id] = active
            self._cond.notify_all()
        self._publish(active.job)
        self._compute.emit_usage(force=True)
        return active.job.model_copy(deep=True)

    def run_now(self, kind: str, request: dict[str, Any]) -> Job:
        """Run a job on the calling thread (the command line). Ctrl+C cancels it."""
        active = self._create(kind, request)
        with self._lock:
            self._active[active.job.id] = active
        self._publish(active.job)
        self._compute.emit_usage(force=True)
        try:
            self._execute(active)
        except KeyboardInterrupt:
            active.cancel.set()
            if active.engine_token is not None:
                active.engine_token.cancel()
            kill_process_tree(active.process)
            self._finish(active, "cancelled")
        return active.job.model_copy(deep=True)

    def _create(self, kind: str, request: dict[str, Any]) -> _Active:
        factory = self._factories.get(kind)
        if factory is None:
            raise ValidationError(f"Unknown job kind: {kind}")
        spec = factory(request)
        job = Job(
            id=new_id("j"),
            kind=kind,  # ty: ignore[invalid-argument-type]
            title=spec.title,
            state="queued",
            steps=[s.model_copy() for s in spec.steps],
            request=request,
            created_at=now_iso(),
            can_run_anyway=False,
        )
        self._db.execute(
            "INSERT INTO jobs(id, kind, title, state, request, progress, step, steps, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job.id, kind, job.title, job.state, json.dumps(request), None, None, json.dumps([s.model_dump() for s in job.steps]), job.created_at),
        )
        return _Active(job=job, spec=spec)

    # -- scheduling ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._stop = False
            self._thread = threading.Thread(target=self._schedule_loop, name="jobs", daemon=True)
            self._thread.start()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            actives = list(self._active.values())
            self._cond.notify_all()
        for a in actives:
            a.cancel.set()
            if a.engine_token is not None:
                a.engine_token.cancel()
            kill_process_tree(a.process)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _schedule_loop(self) -> None:
        while True:
            with self._cond:
                if self._stop:
                    return
                self._schedule()
                self._cond.wait(timeout=1.0)

    def _schedule(self) -> None:
        pending = sorted((a for a in self._active.values() if a.job.state in ("queued", "waiting")), key=lambda a: (-a.spec.priority, a.job.created_at))
        for a in pending:
            waiting_for = None
            for res in a.spec.resources:
                if self._running.get(res, 0) >= self.capacity(res):
                    waiting_for = res if res in ("gpu", "cpu") else None
                    break
            if waiting_for is None and "gpu" in a.spec.resources and not a.run_anyway and not self._compute.try_acquire(JOBS_HOLDER):
                waiting_for = "gpu"
            if waiting_for is not None or (waiting_for is None and any(self._running.get(r, 0) >= self.capacity(r) for r in a.spec.resources)):
                state_changed = a.job.state != "waiting" or a.job.waiting_for != waiting_for
                if state_changed:
                    a.job.state = "waiting"
                    a.job.waiting_for = waiting_for
                    a.job.can_run_anyway = waiting_for == "gpu" and self._compute.lease_holder == "live"
                    self._persist(a)
                    self._publish(a.job)
                continue
            for res in a.spec.resources:
                self._running[res] = self._running.get(res, 0) + 1
            a.holds_gpu = "gpu" in a.spec.resources and not a.run_anyway
            a.job.state = "running"
            a.job.waiting_for = None
            a.job.can_run_anyway = False
            threading.Thread(target=self._thread_main, args=(a,), name=f"job-{a.job.id}", daemon=True).start()

    def _thread_main(self, a: _Active) -> None:
        try:
            self._execute(a)
        finally:
            with self._cond:
                for res in a.spec.resources:
                    self._running[res] = max(0, self._running.get(res, 0) - 1)
                if a.holds_gpu and self._running.get("gpu", 0) == 0:
                    self._compute.release_lease(JOBS_HOLDER)
                self._cond.notify_all()

    def notify(self) -> None:
        """Wake the scheduler (the GPU lease changed)."""
        with self._cond:
            self._cond.notify_all()

    # -- execution ------------------------------------------------------------

    def _execute(self, a: _Active) -> None:
        a.job.state = "running"
        a.job.started_at = now_iso()
        self._persist(a)
        self._publish(a.job)
        ctx = JobContext(self, a)
        try:
            if a.cancel.is_set():
                raise JobCancelled()
            result = a.spec.run(ctx)
            if a.cancel.is_set():
                raise JobCancelled()
            a.job.result = result or {}
            a.job.progress = 1.0
            self._finish(a, "completed")
        except JobCancelled:
            self._finish(a, "cancelled")
        except Exception as e:
            from rvc_next.engine.errors import Cancelled

            if isinstance(e, Cancelled) or a.cancel.is_set():
                self._finish(a, "cancelled")
                return
            mapped = map_engine_error(e)
            if mapped is None or type(mapped) is RvcNextError:
                self._log(a, traceback.format_exc())
                logger.error("Job %s failed: %s", a.job.id, e)
            err = mapped or RvcNextError(str(e) or type(e).__name__)
            if is_out_of_memory(err):
                self._log(a, "Out of GPU memory: every loaded model was unloaded. Retry the job, or lower the batch size.")
                self._compute.release_after_oom(f"job {a.job.id} ({a.job.kind})")
            a.job.error = ErrorInfo(code=err.code, message=err.message, detail=err.detail)
            self._finish(a, "failed")

    def _finish(self, a: _Active, state: str) -> None:
        a.job.state = state  # ty: ignore[invalid-assignment]
        a.job.transfer = None
        a.job.finished_at = now_iso()
        a.job.can_retry = state in ("failed", "cancelled", "interrupted") and a.job.kind in self._factories
        a.job.can_run_anyway = False
        for s in a.job.steps:
            if s.state == "running":
                s.state = "cancelled" if state == "cancelled" else ("failed" if state == "failed" else "done")
            elif s.state == "pending" and state != "completed":
                s.state = "cancelled" if state == "cancelled" else "skipped"
        self._flush_log(a, force=True)
        self._persist(a)
        self._publish(a.job)
        with self._lock:
            self._active.pop(a.job.id, None)
        self._compute.emit_usage(force=True)

    def _run_worker(self, a: _Active, module: str, request: dict[str, Any], on_event: Callable[[Any], None] | None, env: dict[str, str] | None) -> dict[str, Any]:
        from rvc_next.protocol.jsonl import parse_line
        from rvc_next.protocol.messages import ErrorEvent, LogEvent, ProgressEvent, ResultEvent, StepEvent

        self._logs.mkdir(parents=True, exist_ok=True)
        req_path = self._logs / f"{a.job.id}.{new_id()}.request.json"
        req_path.write_text(json.dumps(request), encoding="utf-8")
        result: dict[str, Any] = {}
        error: ErrorEvent | None = None
        try:
            if a.cancel.is_set():
                raise JobCancelled()
            proc = start_process(worker_command(module, str(req_path)), env=env)
            a.process = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                event = parse_line(line)
                if on_event is not None:
                    on_event(event)
                if isinstance(event, ProgressEvent):
                    if on_event is None:
                        self._progress(a, event.value, event.step)
                elif isinstance(event, StepEvent):
                    if on_event is None:
                        self._step(a, event.id, event.state, event.progress)
                elif isinstance(event, LogEvent):
                    self._log(a, event.message)
                elif isinstance(event, ResultEvent):
                    result = event.data
                elif isinstance(event, ErrorEvent):
                    error = event
                    self._log(a, f"error: {event.message}")
            code = proc.wait()
        finally:
            a.process = None
            req_path.unlink(missing_ok=True)
        if a.cancel.is_set():
            raise JobCancelled()
        if error is not None:
            raise worker_error(error.code, error.message, error.detail)
        if code != 0:
            raise WorkerExit(code)
        return result

    # -- reporting -------------------------------------------------------------

    def _progress(self, a: _Active, value: float | None, step: str | None) -> None:
        a.job.progress = None if value is None else max(0.0, min(1.0, float(value)))
        if step is not None:
            a.job.step = step
        now = time.monotonic()
        if now - a.last_progress_emit >= PROGRESS_INTERVAL:
            a.last_progress_emit = now
            self._publish(a.job)
        if now - a.last_db_write >= DB_PROGRESS_INTERVAL:
            self._persist(a)

    def _transfer(self, a: _Active, done: int, total: int) -> None:
        now = time.monotonic()
        samples = a.transfer_samples
        if samples and done < samples[-1][1]:
            samples.clear()  # a file started again from the beginning
        samples.append((now, done))
        while len(samples) > 2 and now - samples[0][0] > TRANSFER_WINDOW:
            samples.popleft()
        speed = eta = None
        (t0, d0), (t1, d1) = samples[0], samples[-1]
        if t1 - t0 >= 0.5:
            speed = max(0.0, (d1 - d0) / (t1 - t0))
            eta = (total - done) / speed if speed > 0 else None
        a.job.transfer = Transfer(done_bytes=done, total_bytes=total, bytes_per_second=speed, eta_seconds=eta)
        if now - a.last_progress_emit >= PROGRESS_INTERVAL:
            a.last_progress_emit = now
            self._publish(a.job)

    def _step(self, a: _Active, step_id: str, state: str, progress: float | None) -> None:
        for s in a.job.steps:
            if s.id == step_id:
                s.state = state  # ty: ignore[invalid-assignment]
                s.progress = progress if progress is not None else (1.0 if state == "done" else s.progress)
                break
        else:
            a.job.steps.append(JobStep(id=step_id, title=step_id, state=state, progress=progress))  # ty: ignore[invalid-argument-type]
        if state == "running":
            a.job.step = step_id
        self._persist(a)
        self._publish(a.job)

    def _add_steps(self, a: _Active, steps: list[JobStep]) -> None:
        known = {s.id for s in a.job.steps}
        a.job.steps.extend(s for s in steps if s.id not in known)
        self._persist(a)
        self._publish(a.job)

    def _log(self, a: _Active, text: str) -> None:
        self._logs.mkdir(parents=True, exist_ok=True)
        path = self._logs / f"{a.job.id}.log"
        lines = text.rstrip("\n").split("\n")
        with self._lock:
            offset = path.stat().st_size if path.exists() else 0
            with open(path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            if a.pending_offset is None:
                a.pending_offset = offset
            a.pending_lines.extend(lines)
        self._flush_log(a)

    def _flush_log(self, a: _Active, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not a.pending_lines or (not force and now - a.last_log_emit < LOG_EVENT_INTERVAL):
                return
            lines, offset = a.pending_lines[-MAX_LOG_EVENT_LINES:], a.pending_offset or 0
            a.pending_lines, a.pending_offset = [], None
            a.last_log_emit = now
        self._events.publish(JobLogEvent(job_id=a.job.id, offset=offset, lines=lines))

    def _persist(self, a: _Active) -> None:
        j = a.job
        a.last_db_write = time.monotonic()
        self._db.execute(
            "UPDATE jobs SET state = ?, title = ?, progress = ?, step = ?, steps = ?, error = ?, result = ?, started_at = ?, finished_at = ? WHERE id = ?",
            (
                j.state,
                j.title,
                j.progress,
                j.step,
                json.dumps([s.model_dump() for s in j.steps]),
                j.error.model_dump_json() if j.error else None,
                json.dumps(j.result) if j.result is not None else None,
                j.started_at,
                j.finished_at,
                j.id,
            ),
        )

    def _publish(self, job: Job) -> None:
        self._events.publish(JobUpdatedEvent(job=job.model_copy(deep=True)))

    # -- control ---------------------------------------------------------------

    def cancel(self, job_id: str) -> Job:
        with self._cond:
            a = self._active.get(job_id)
            if a is None:
                job = self.get(job_id)
                if job.state in FINISHED_STATES:
                    return job
                raise NotFoundError(f"Job {job_id} is not running")
            a.cancel.set()
            if a.engine_token is not None:
                a.engine_token.cancel()
            if a.job.state in ("queued", "waiting"):
                self._active.pop(job_id, None)
                a.job.state = "cancelled"
                a.job.finished_at = now_iso()
                a.job.can_retry = a.job.kind in self._factories
                self._persist(a)
                self._publish(a.job)
                return a.job.model_copy(deep=True)
            a.job.state = "cancelling"
            proc = a.process
        self._publish(a.job)
        if proc is not None:
            threading.Thread(target=kill_process_tree, args=(proc,), daemon=True).start()
        return a.job.model_copy(deep=True)

    def run_anyway(self, job_id: str) -> Job:
        """Start a GPU job that waits for Live, bypassing the lease."""
        with self._cond:
            a = self._active.get(job_id)
            if a is None or a.job.state not in ("queued", "waiting"):
                raise ConflictError("Only a waiting job can be started anyway")
            a.run_anyway = True
            self._cond.notify_all()
            return a.job.model_copy(deep=True)

    def retry(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state not in ("failed", "cancelled", "interrupted"):
            raise ConflictError("Only a failed, cancelled or interrupted job can be retried")
        return self.submit(job.kind, job.request)

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.state in ACTIVE_STATES:
            raise ConflictError("Cancel the job before removing it")
        self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._remove_log(job_id)
        self._events.publish(JobsClearedEvent(ids=[job_id]))

    def clear_finished(self) -> list[str]:
        rows = self._db.fetchall(f"SELECT id FROM jobs WHERE state IN ({','.join('?' * len(FINISHED_STATES))})", tuple(FINISHED_STATES))
        ids = [r["id"] for r in rows]
        if ids:
            with self._db.transaction() as conn:
                conn.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids])
            for i in ids:
                self._remove_log(i)
            self._events.publish(JobsClearedEvent(ids=ids))
        return ids

    def _remove_log(self, job_id: str) -> None:
        (self._logs / f"{job_id}.log").unlink(missing_ok=True)

    # -- reading ---------------------------------------------------------------

    def get(self, job_id: str) -> Job:
        with self._lock:
            a = self._active.get(job_id)
            if a is not None:
                return a.job.model_copy(deep=True)
        row = self._db.fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise NotFoundError(f"Unknown job: {job_id}")
        return self._row(row)

    def list_jobs(self, state: str | None = None, kind: str | None = None, cursor: str | None = None, limit: int = 50) -> JobPage:
        where, params = [], []
        if state == "active":
            where.append(f"state IN ({','.join('?' * len(ACTIVE_STATES))})")
            params += sorted(ACTIVE_STATES)
        elif state == "finished":
            where.append(f"state IN ({','.join('?' * len(FINISHED_STATES))})")
            params += sorted(FINISHED_STATES)
        elif state:
            where.append("state = ?")
            params.append(state)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if cursor:
            created, _, cid = cursor.partition("|")
            where.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params += [created, created, cid]
        sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC, id DESC LIMIT ?"
        rows = self._db.fetchall(sql, (*params, limit + 1))
        items = []
        with self._lock:
            for r in rows[:limit]:
                a = self._active.get(r["id"])
                items.append(a.job.model_copy(deep=True) if a else self._row(r))
        next_cursor = f"{rows[limit - 1]['created_at']}|{rows[limit - 1]['id']}" if len(rows) > limit else None
        return JobPage(items=items, next_cursor=next_cursor)

    def active(self) -> list[Job]:
        with self._lock:
            return [a.job.model_copy(deep=True) for a in self._active.values()]

    def read_log(self, job_id: str, offset: int = 0, limit_bytes: int = 256 * 1024) -> JobLog:
        self.get(job_id)
        path = self._logs / f"{job_id}.log"
        if not path.exists():
            return JobLog(job_id=job_id, offset=offset, next_offset=offset, lines=[])
        with open(path, "rb") as f:
            f.seek(max(0, offset))
            data = f.read(limit_bytes)
        if len(data) == limit_bytes and b"\n" in data:
            data = data[: data.rindex(b"\n") + 1]
        text = data.decode("utf-8", errors="replace")
        return JobLog(job_id=job_id, offset=offset, next_offset=offset + len(data), lines=text.splitlines())

    def _row(self, r: Any) -> Job:
        job = Job(
            id=r["id"],
            kind=r["kind"],
            title=r["title"],
            state=r["state"],
            progress=r["progress"],
            step=r["step"],
            steps=[JobStep.model_validate(s) for s in json.loads(r["steps"] or "[]")],
            error=ErrorInfo.model_validate_json(r["error"]) if r["error"] else None,
            result=json.loads(r["result"]) if r["result"] else None,
            request=json.loads(r["request"]),
            created_at=r["created_at"],
            started_at=r["started_at"],
            finished_at=r["finished_at"],
        )
        job.can_retry = job.state in ("failed", "cancelled", "interrupted") and job.kind in self._factories
        return job

    def _mark_interrupted(self) -> None:
        """A job still active when the server stopped is ``interrupted`` now (another live server excepted)."""
        from rvc_next.core.net.runtime_file import read_runtime_file

        if read_runtime_file(self._settings.data_dir) is not None:
            return
        states = sorted(ACTIVE_STATES)
        self._db.execute(f"UPDATE jobs SET state = 'interrupted', finished_at = ? WHERE state IN ({','.join('?' * len(states))})", (now_iso(), *states))


class WorkerExit(RvcNextError):
    """A worker ended with a non-zero exit code and no error event."""

    def __init__(self, code: int) -> None:
        super().__init__(f"The worker process exited with code {code}", {"exit_code": code})
        self.exit_code = code
