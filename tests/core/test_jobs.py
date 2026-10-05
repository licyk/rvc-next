import threading
import time

import pytest

from rvc_next.core.errors import ConflictError, ValidationError
from rvc_next.core.jobs.service import JobSpec


def wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def register(services, kind, run, resources=frozenset({"cpu"})):
    services.jobs.register(kind, lambda req: JobSpec(title=f"{kind} {req.get('n', '')}", run=lambda ctx: run(ctx, req), resources=resources))


def test_foreground_job_completes_with_result_and_log(services):
    def run(ctx, req):
        ctx.log("hello\nworld")
        ctx.progress(0.5)
        return {"n": req["n"] * 2}

    register(services, "convert", run)
    job = services.jobs.run_now("convert", {"n": 21})
    assert job.state == "completed" and job.result == {"n": 42} and job.progress == 1.0
    log = services.jobs.read_log(job.id)
    assert log.lines == ["hello", "world"]
    assert services.jobs.get(job.id).state == "completed"


def test_failure_maps_domain_errors(services):
    def run(ctx, req):
        raise ValidationError("bad pitch", {"field": "pitch"})

    register(services, "convert", run)
    job = services.jobs.run_now("convert", {})
    assert job.state == "failed" and job.error.code == "invalid_input" and job.error.detail == {"field": "pitch"}
    assert job.can_retry


def test_scheduler_respects_gpu_capacity_and_live_lease(make_services):
    services = make_services(start_background=True)
    gate = threading.Event()
    started: list[str] = []

    def run(ctx, req):
        started.append(req["n"])
        gate.wait(5)
        return {}

    register(services, "convert", run, frozenset({"gpu"}))
    a = services.jobs.submit("convert", {"n": "a"})
    b = services.jobs.submit("convert", {"n": "b"})
    assert wait_for(lambda: services.jobs.get(a.id).state == "running")
    assert wait_for(lambda: services.jobs.get(b.id).state == "waiting")
    assert services.jobs.get(b.id).waiting_for == "gpu"
    gate.set()
    assert wait_for(lambda: services.jobs.get(b.id).state == "completed")

    # Live holds the lease: a GPU job waits and can be started anyway.
    gate.clear()
    services.compute.acquire_live()
    c = services.jobs.submit("convert", {"n": "c"})
    assert wait_for(lambda: services.jobs.get(c.id).state == "waiting" and services.jobs.get(c.id).can_run_anyway)
    services.jobs.run_anyway(c.id)
    assert wait_for(lambda: services.jobs.get(c.id).state == "running")
    gate.set()
    assert wait_for(lambda: services.jobs.get(c.id).state == "completed")
    services.compute.release_lease("live")


def test_cancel_in_process_and_queued(make_services):
    services = make_services(start_background=True)

    def run(ctx, req):
        for _ in range(500):
            ctx.check_cancel()
            time.sleep(0.01)
        return {}

    register(services, "convert", run)
    job = services.jobs.submit("convert", {})
    assert wait_for(lambda: services.jobs.get(job.id).state == "running")
    services.jobs.cancel(job.id)
    assert wait_for(lambda: services.jobs.get(job.id).state == "cancelled")
    retried = services.jobs.retry(job.id)
    assert retried.id != job.id
    services.jobs.cancel(retried.id)
    assert wait_for(lambda: services.jobs.get(retried.id).state == "cancelled")


def test_subprocess_worker_events_and_kill(make_services, tmp_path):
    services = make_services(start_background=True)
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, sys, time\n"
        "from rvc_next.protocol.jsonl import Emitter\n"
        "from rvc_next.protocol.messages import ProgressEvent, LogEvent, ResultEvent\n"
        "req = json.load(open(sys.argv[1]))\n"
        "e = Emitter()\n"
        "e.emit(LogEvent('info', 'started'))\n"
        "print('stray print')\n"
        "e.emit(ProgressEvent(0.5, 'half'))\n"
        "if req.get('sleep'):\n"
        "    time.sleep(60)\n"
        "e.emit(ResultEvent({'ok': True}))\n"
    )
    import sys

    sys.path.insert(0, str(tmp_path))

    def run(ctx, req):
        return ctx.run_worker("fake_worker", req, env={"PYTHONPATH": str(tmp_path) + ":" + ":".join(sys.path)})

    register(services, "separate", run)
    job = services.jobs.submit("separate", {})
    assert wait_for(lambda: services.jobs.get(job.id).state == "completed", 30)
    assert services.jobs.get(job.id).result == {"ok": True}
    assert "stray print" in services.jobs.read_log(job.id).lines
    slow = services.jobs.submit("separate", {"sleep": True})
    assert wait_for(lambda: services.jobs.get(slow.id).progress == 0.5, 30)
    services.jobs.cancel(slow.id)
    assert wait_for(lambda: services.jobs.get(slow.id).state == "cancelled", 15)


def test_running_jobs_become_interrupted_on_restart(make_services):
    services = make_services()
    services.db.execute("INSERT INTO jobs(id, kind, title, state, request, steps, created_at) VALUES ('x', 'convert', 't', 'running', '{}', '[]', '2026-01-01')")
    again = make_services()
    assert again.jobs.get("x").state == "interrupted"
    with pytest.raises(ConflictError):
        again.jobs.run_anyway("x")
    again.jobs.delete("x")
    assert again.jobs.list_jobs().items == []
