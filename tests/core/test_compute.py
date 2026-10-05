"""Free GPU memory: allowed whenever nothing is working; an out-of-memory error unloads everything."""

import threading

import pytest

from rvc_next.core.compute.service import MemoryHolder
from rvc_next.core.engine_errors import is_out_of_memory, worker_error
from rvc_next.core.errors import ConflictError
from tests.core.test_jobs import register, wait_for


def probe(services, busy=None):
    released: list[int] = []
    services.compute.add_holder(MemoryHolder("probe", busy=lambda: busy, cached=lambda: ["model"], release=lambda: released.append(1)))
    return released


def test_free_memory_whenever_no_job_or_live_session_runs(make_services) -> None:
    services = make_services(start_background=True)
    released = probe(services)
    usage = services.compute.usage()
    # Nothing loaded in the server's runtime is no reason to refuse: other holders may hold models.
    assert usage.can_release and usage.busy == [] and "probe:model" in usage.cached

    gate = threading.Event()
    register(services, "convert", lambda ctx, req: gate.wait(5) and {})
    job = services.jobs.submit("convert", {})
    assert wait_for(lambda: services.jobs.get(job.id).state == "running")
    usage = services.compute.usage()
    assert usage.busy == ["jobs"] and not usage.can_release
    with pytest.raises(ConflictError) as e:
        services.compute.release()
    assert e.value.detail == {"reason": "busy", "busy": ["jobs"]}
    assert released == []
    gate.set()
    assert wait_for(lambda: services.compute.usage().can_release)
    assert services.compute.release().can_release and released == [1]

    live = services.live
    live._state = live._state.model_copy(update={"state": "running"})
    assert services.compute.usage().busy == ["live"]
    live._state = live._state.model_copy(update={"state": "stopped"})
    assert services.compute.usage().can_release


def test_out_of_memory_in_a_job_unloads_every_model(services) -> None:
    import torch

    released = probe(services)

    def run(ctx, req):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB")

    register(services, "convert", run)
    job = services.jobs.run_now("convert", {})
    assert job.state == "failed" and job.error.code == "compute_unavailable" and job.error.detail == {"reason": "out_of_memory"}
    assert released == [1]
    assert any("every loaded model was unloaded" in line for line in services.jobs.read_log(job.id).lines)


def test_out_of_memory_reported_by_a_worker_is_recognised() -> None:
    assert is_out_of_memory(worker_error("compute_unavailable", "Out of GPU memory", {"reason": "out_of_memory"}))
    assert not is_out_of_memory(worker_error("compute_unavailable", "No CUDA device", {}))
