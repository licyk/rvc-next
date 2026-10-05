"""Compute: the in-process runtime, the GPU lease and memory reporting."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rvc_next.core.compute.models import ComputeDevice, ComputeInfo, ComputeUsage
from rvc_next.core.errors import ConflictError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import ComputeChangedEvent
from rvc_next.core.settings import Settings, SettingsService

if TYPE_CHECKING:
    from rvc_next.engine.runtime import Runtime

logger = logging.getLogger(__name__)

LIVE_HOLDER = "live"


@dataclass(frozen=True)
class MemoryHolder:
    """Something besides the server's runtime that keeps models in GPU memory, or uses them.

    Jobs and Live register one each. ``busy`` returns why freeing memory must wait (None when
    idle); ``cached`` names the models it holds; ``release`` unloads them.
    """

    name: str
    busy: Callable[[], str | None]
    cached: Callable[[], list[str]] = lambda: []
    release: Callable[[], None] = lambda: None


class ComputeService:
    """Owns the server's (or the command's) engine runtime.

    The runtime keeps HuBERT, the F0 models, the last three voices and their indexes. It is built on
    first use, so commands that never convert never import torch. The GPU lease has one holder at
    a time: an in-process GPU job, a GPU subprocess job, or Live. Live takes priority.
    """

    def __init__(self, settings: SettingsService, events: EventBus) -> None:
        self._settings = settings
        self._events = events
        self._lock = threading.RLock()
        self._runtime: Runtime | None = None
        self._runtime_key: tuple | None = None
        self._lease_holder: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_emit = 0.0
        self._live_vram_mb: float | None = None
        self._holders: list[MemoryHolder] = []
        settings.on_change(self._on_settings)

    def add_holder(self, holder: MemoryHolder) -> None:
        self._holders.append(holder)

    # -- runtime ------------------------------------------------------------

    def _key(self, s: Settings) -> tuple:
        return (s.compute.device, s.compute.precision, s.compute.cuda_graph_offline, str(self._settings.assets_dir))

    def runtime(self) -> Runtime:
        from rvc_next.engine.runtime import Runtime

        with self._lock:
            key = self._key(self._settings.settings)
            if self._runtime is None or self._runtime_key != key:
                if self._runtime is not None:
                    self._runtime.release()
                c = self._settings.settings.compute
                self._runtime = Runtime(self._settings.assets_dir, device=c.device, precision=c.precision, cuda_graph=c.cuda_graph_offline)
                self._runtime_key = key
            return self._runtime

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def _on_settings(self, s: Settings) -> None:
        with self._lock:
            if self._runtime is not None and self._runtime_key != self._key(s):
                self._runtime.release()
                self._runtime = None
                self._runtime_key = None

    def busy(self) -> list[str]:
        """Why GPU memory cannot be freed now: ``jobs`` while a job is queued or running, ``live``
        while Live runs. Empty when nothing is working."""
        out = []
        for h in self._holders:
            try:
                if h.busy():
                    out.append(h.name)
            except Exception:
                logger.exception("Memory holder %s failed", h.name)
        return out

    def release(self) -> ComputeUsage:
        """Free GPU memory: unload every cached model, here and in idle workers (the original's
        卸载音色省显存, ComfyUI's ``unload_all_models`` + ``soft_empty_cache``). Refused while a job or
        Live runs: their models are in use and would only be loaded again."""
        busy = self.busy()
        if busy:
            raise ConflictError("GPU memory can be freed when no job or Live session is running", {"reason": "busy", "busy": busy})
        self._unload_all()
        usage = self.usage()
        self._events.publish(ComputeChangedEvent(usage=usage))
        return usage

    def release_after_oom(self, source: str) -> None:
        """A device ran out of memory: unload every cached model, as ComfyUI does, so a retry starts
        from an empty GPU. Models a running job still uses stay alive through its references and
        are freed when it ends; a running Live session keeps its own."""
        logger.error("Out of GPU memory in %s: unloading every loaded model", source)
        self._unload_all()
        self.emit_usage(force=True)

    def _unload_all(self) -> None:
        with self._lock:
            if self._runtime is not None:
                self._runtime.release()
        for h in self._holders:
            try:
                if not h.busy():
                    h.release()
            except Exception:
                logger.exception("Unloading the models of %s failed", h.name)

    # -- reporting -----------------------------------------------------------

    def info(self) -> ComputeInfo:
        """Every device, with the precision the GPU rule chose. Imports torch."""
        from rvc_next.engine.runtime import choose_device, list_devices, torch_backend, torch_versions

        c = self._settings.settings.compute
        choice = choose_device(c.device, c.precision)
        devices = [
            ComputeDevice(
                id=p.id,
                name=p.name,
                kind=p.kind,  # ty: ignore[invalid-argument-type]
                memory_mb=round(p.memory_gb * 1024, 1) if p.memory_gb else None,
                sm=p.sm or None,
                eligible=p.eligible,
                precision="fp16" if p.fp16 else "fp32",
                reason=p.reason,
            )
            for p in list_devices()
        ]
        return ComputeInfo(
            selected=choice.id,
            precision="fp16" if choice.fp16 else "fp32",
            devices=devices,
            torch_version=torch_versions()[0],
            cuda_version=torch_versions()[1],
            backend=torch_backend(),
            cuda_graph=bool(self._runtime.cuda_graph) if self._runtime else False,
        )

    def usage(self) -> ComputeUsage:
        with self._lock:
            rt = self._runtime
            device = rt.choice.id if rt else self._settings.settings.compute.device
            cached = rt.cached() if rt else []
            vram = rt.vram() if rt else None
            holder = self._lease_holder
        for h in self._holders:
            try:
                cached += [f"{h.name}:{m}" for m in h.cached()]
            except Exception:
                logger.exception("Memory holder %s failed", h.name)
        busy = self.busy()
        return ComputeUsage(
            device=device,
            vram_used_mb=round(vram[0], 1) if vram else None,
            vram_total_mb=round(vram[1], 1) if vram else None,
            cached=cached,
            lease_holder=holder,
            busy=busy,
            can_release=not busy,
        )

    def emit_usage(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit < 1.0:
            return
        self._last_emit = now
        self._events.publish(ComputeChangedEvent(usage=self.usage()))

    # -- GPU lease -----------------------------------------------------------

    @property
    def lease_holder(self) -> str | None:
        return self._lease_holder

    def try_acquire(self, holder: str) -> bool:
        with self._lock:
            if self._lease_holder == holder:
                return True
            if self._lease_holder is not None:
                return False
            self._lease_holder = holder
        self.emit_usage(force=True)
        return True

    def acquire_live(self) -> None:
        """Live takes the lease at once; running GPU jobs keep their memory but new ones wait."""
        with self._lock:
            changed = self._lease_holder != LIVE_HOLDER
            self._lease_holder = LIVE_HOLDER
        if changed:
            self.emit_usage(force=True)

    def release_lease(self, holder: str) -> None:
        with self._lock:
            if self._lease_holder != holder:
                return
            self._lease_holder = None
        self.emit_usage(force=True)

    # -- idle unloading ------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._idle_loop, name="compute-idle", daemon=True)
            self._thread.start()

    def _idle_loop(self) -> None:
        while not self._stop.wait(30):
            minutes = self._settings.settings.compute.unload_after_minutes
            with self._lock:
                rt = self._runtime
            if rt is not None and minutes > 0 and rt.release_idle(minutes * 60):
                logger.info("Unloaded models after %s idle minutes", minutes)
                self.emit_usage(force=True)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        with self._lock:
            if self._runtime is not None:
                self._runtime.release()
                self._runtime = None
