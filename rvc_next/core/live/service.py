"""Live: supervises the live worker and owns device enumeration.

One session at a time. Device lists come from a short enumeration subprocess, so a running stream
is never disturbed. Saved selections are resolved against a fresh list (§7.6) before every start
and every device change; an output that would silently become the system default is refused
unless the user confirms. After a device loss the service re-enumerates every 2 seconds and
reopens the same device when it returns.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rvc_next.core.assets.service import AssetService
from rvc_next.core.clock import now_iso
from rvc_next.core.compute.service import LIVE_HOLDER, ComputeService, MemoryHolder
from rvc_next.core.errors import BusyError, ConflictError, DeviceError, RvcNextError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import DevicesChangedEvent, LiveStateEvent, LiveStatsEvent
from rvc_next.core.files import slugify
from rvc_next.core.jobs.service import JobService
from rvc_next.core.live.models import (
    DeviceCheck,
    DeviceList,
    DeviceProblem,
    DeviceSelection,
    LatencyMeasurement,
    LatencyTestRequest,
    LiveConfig,
    LiveDevices,
    LiveRecording,
    LiveState,
    LiveStats,
    RecordingRequest,
    ResolvedDevice,
    TestToneRequest,
)
from rvc_next.core.live.supervisor import LiveSupervisor
from rvc_next.core.models.service import VoiceModelService
from rvc_next.core.params import StreamParamsModel, VoiceParamsModel, f0_assets, require_effects
from rvc_next.core.safety import unique_path
from rvc_next.core.settings import SettingsService
from rvc_next.protocol import live as P
from rvc_next.protocol.messages import OUT_OF_MEMORY

if TYPE_CHECKING:
    from rvc_next.core.audio.service import AudioFileService

logger = logging.getLogger(__name__)

DEVICES_WORKER = "rvc_next.workers.devices"
RECONNECT_INTERVAL = 2.0
START_LIST_MAX_AGE = 10.0
"""Seconds a device list stays good enough for Start. PortAudio's indexes hold only while no device
is added or removed; the Live page re-enumerates every 5 s, so Start from it rarely enumerates."""
METER_SETTLE = 0.3
"""Seconds for the live worker to close the input meter before a measurement opens the input."""
METER_LEASE = 15.0
"""Seconds the input meter stays wanted after the last request for it. The device panel asks again
every 5 s while it is shown, so a closed or hidden page lets the microphone go."""
METER_RETRY = (1.0, 2.0, 5.0, 10.0)
"""Seconds before opening the meter again after its device refused or was lost, by attempt."""
METER_RESEND = 0.5
"""The least time between two requests for the same meter, when the worker answers without one."""
ACTIVE = ("starting", "loading", "prewarming", "running", "reconnecting")


class BrowserLink:
    """A browser's audio connection (``/live/browser-audio``): its microphone samples go to the live
    worker, the converted ones come back through ``send``. One at a time; a new one replaces it."""

    def __init__(self, service: LiveService, send: Callable[[bytes], None]) -> None:
        self._service = service
        self.send = send
        self.received = 0
        """Frames from the browser."""

    def feed(self, pcm: bytes) -> None:
        """Microphone samples (mono 16-bit PCM at the session's rate); dropped while no browser session runs."""
        self.received += 1
        if self._service._browser is self and self._service._browser_session():
            self._service._supervisor.send_audio(pcm)

    def close(self) -> None:
        self._service._close_browser(self)


class LiveService:
    def __init__(
        self,
        settings: SettingsService,
        events: EventBus,
        models: VoiceModelService,
        assets: AssetService,
        compute: ComputeService,
        jobs: JobService,
        audio: AudioFileService | None = None,
    ) -> None:
        self._settings = settings
        self._audio = audio
        self._events = events
        self._models = models
        self._assets = assets
        self._compute = compute
        self._jobs = jobs
        self._lock = threading.RLock()
        # Start and the hot updates read the state, message the worker and write the state back as one
        # step: a voice switch and the speaker reset it causes, sent at once, must not interleave.
        self._control = threading.RLock()
        # model path → voice id for every voice sent to the worker, to read its answer to a voice switch.
        self._voice_ids: dict[str, str] = {}
        self._state = LiveState()
        self._stats = LiveStats()
        self._devices: DeviceList | None = None
        self._devices_at = 0.0
        """``time.monotonic()`` of the enumeration ``_devices`` came from."""
        self._enumerating = threading.Lock()
        """One enumeration at a time: callers that ask meanwhile share its list."""
        self._lost: set[str] = set()
        self._stop_error: dict[str, Any] | None = None
        """Why the core stopped the session itself (a device lost without reconnecting): the worker's
        ``stopped`` that follows becomes ``error`` with it, so the reason stays on screen."""
        self._reconnect: threading.Thread | None = None
        self._closing = False
        # The input meter while Live is idle. Asking for it only extends a lease (``meter``); a thread
        # of its own opens it, follows the selected input and retries a refused device
        # (``_sync_meter``), so no request waits for the worker to start or a device to open.
        self._meter_until = 0.0
        self._meter_sent: dict[str, Any] | None = None
        """What the worker was last asked to meter (input and gain), until it says the meter is off."""
        self._meter_failures = 0
        self._meter_retry_at = 0.0
        self._meter_wake = threading.Event()
        self._meter_thread: threading.Thread | None = None
        self._meter_lock = threading.Lock()
        """Held while deciding about the meter and telling the worker, so a measurement's pause and
        the meter thread cannot cross."""
        self._awaiting_start = False
        self._measuring = False
        self._recording_seen = False
        """The worker has confirmed the current recording (a state naming its file)."""
        self.backend = "sounddevice"
        """``fake`` in tests, with ``fake`` as the fake backend's configuration."""
        self.fake: dict[str, Any] | None = None
        self._supervisor = LiveSupervisor(self._on_event, self._on_exit, self._on_audio)
        self._browser: BrowserLink | None = None
        settings.on_keys_change(self._on_settings_keys)
        self._worker_cached: list[str] = []
        """The models the live worker keeps loaded, from its last state message."""
        compute.add_holder(MemoryHolder(LIVE_HOLDER, busy=self._busy, cached=self._cached, release=self._release_memory))

    # -- browser audio -------------------------------------------------------------------

    def open_browser_link(self, send: Callable[[bytes], None]) -> BrowserLink:
        """Connect a browser's audio. ``send`` (thread-safe) carries converted samples to it."""
        link = BrowserLink(self, send)
        with self._lock:
            old, self._browser = self._browser, link
        if old is not None and self._browser_session():
            logger.info("Live: another browser took over the browser audio")
        return link

    def _close_browser(self, link: BrowserLink) -> None:
        with self._lock:
            if self._browser is not link:
                return
            self._browser = None
        if self._browser_session():
            logger.info("Live: the browser's audio disconnected; stopping")
            self.stop()

    def _browser_session(self) -> bool:
        with self._lock:
            return self._state.state in ACTIVE and self._state.config is not None and self._state.config.browser is not None

    def _on_audio(self, pcm: bytes) -> None:
        link = self._browser
        if link is not None:
            link.send(pcm)

    @staticmethod
    def _browser_devices(sample_rate: int, devices: LiveDevices) -> dict[str, Any]:
        """The worker's devices for a browser session: the browser both ways, the saved gains."""
        from rvc_next.engine.audio_io.browser import browser_device

        ep = {"device": browser_device(sample_rate), "channels": None, "sample_rate": sample_rate, "exclusive": False}
        return {
            "input": ep,
            "output": dict(ep),
            "monitor": None,
            "monitor_source": "converted",
            "monitor_gain_db": 0.0,
            "output_gain_db": devices.output_gain_db,
            "input_gain_db": devices.input_gain_db,
        }

    # -- GPU memory ----------------------------------------------------------------------

    def _busy(self) -> str | None:
        with self._lock:
            return LIVE_HOLDER if self._state.state in (*ACTIVE, "stopping") else None

    def _cached(self) -> list[str]:
        with self._lock:
            return list(self._worker_cached) if self._supervisor.running else []

    def _release_memory(self) -> None:
        """Free GPU memory: the idle worker drops its engine and models (it stays up for the meter)."""
        if not self._supervisor.running:
            return
        try:
            self._supervisor.send(P.ReleaseMemory())
        except RvcNextError:
            return
        with self._lock:
            self._worker_cached = []

    # -- state ---------------------------------------------------------------------------

    def state(self) -> LiveState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def stats(self) -> LiveStats:
        with self._lock:
            return self._stats.model_copy()

    def _set_state(self, **changes: Any) -> LiveState:
        with self._lock:
            self._state = self._state.model_copy(update=changes)
            snapshot = self._state.model_copy(deep=True)
        self._events.publish(LiveStateEvent(state=snapshot))
        self._meter_wake.set()  # a session started or stopped, passthrough changed: the meter may follow
        return snapshot

    def _on_event(self, message: Any) -> None:
        if isinstance(message, P.State):
            self._on_state(message)
        elif isinstance(message, P.Stats):
            stats = LiveStats.model_validate({k: getattr(message, k) for k in P.STATS_FIELDS})
            with self._lock:
                self._stats = stats
            self._events.publish(LiveStatsEvent(stats=stats))
        elif isinstance(message, P.DeviceLost) and self._browser_session():
            # Nothing to re-enumerate: the browser stopped sending (a closed or frozen tab).
            logger.warning("Live: the browser stopped sending audio; stopping")
            with self._lock:
                self._stop_error = {
                    "code": "device_unavailable",
                    "message": "This browser stopped sending audio (the tab was closed, frozen or lost its connection)",
                    "detail": {"reason": "missing", "role": "input", "browser": True, "lost": True, "error": message.message},
                }
            self._stop_session()
        elif isinstance(message, P.DeviceLost) and self.state().state not in ACTIVE:
            # The meter (or passthrough while stopped) could not open its device, or lost it: no
            # session to reconnect; the meter tries again later.
            logger.info("Live: the idle %s device is unavailable (%s): %s", message.direction, message.reason, message.message)
            self._meter_failed()
        elif isinstance(message, P.DeviceLost):
            logger.warning("Live: %s device lost (%s): %s", message.direction, message.reason, message.message)
            with self._lock:
                self._lost.add(message.direction)
            if self._settings.settings.live.auto_reconnect:
                self._start_reconnect()
        elif isinstance(message, P.Recorded):
            self._on_recorded(message)
        elif isinstance(message, P.Log):
            getattr(logger, message.level if message.level in ("debug", "info", "warning", "error") else "info")("live worker: %s", message.message)

    def _on_state(self, message: P.State) -> None:
        state = message.state
        with self._lock:
            was_busy = self._state.state in (*ACTIVE, "stopping")
            cached_changed = message.cached != self._worker_cached
            self._worker_cached = list(message.cached)
        if cached_changed:
            self._compute.emit_usage(force=True)
        error = message.error or {}
        if state == "error" and (error.get("detail") or {}).get("reason") == OUT_OF_MEMORY:
            # The worker has unloaded its own models; unload the server's too, as ComfyUI does.
            self._compute.release_after_oom("Live")
        with self._lock:
            if self._awaiting_start:
                # The worker announces "stopped" when it connects; Start has not reached it yet.
                if state == "stopped":
                    return
                self._awaiting_start = False
        with self._lock:
            if message.meter:
                self._meter_failures = 0
            elif self._meter_sent is not None:
                # The worker's meter is off (closed for a session, lost, or not yet open when the
                # worker announced itself): ask again, though not at once.
                self._meter_sent = None
                self._meter_retry_at = max(self._meter_retry_at, time.monotonic() + METER_RESEND)
        changes: dict[str, Any] = {"state": state, "meter": message.meter, "passthrough": message.passthrough, "stage": message.stage}
        recording = self._state.recording
        if recording is not None:
            # A state sent before the worker saw Record still says "not recording": wait for its answer.
            if message.recording == recording.path:
                self._recording_seen = True
            elif message.recording is None and (self._recording_seen or message.error or state not in ACTIVE):
                changes["recording"] = None
        if message.topology is not None:
            changes["topology"] = message.topology
        if message.sample_rate is not None:
            changes["sample_rate"] = message.sample_rate
        changes["error"] = message.error
        if message.voice_path is not None:
            # The answer to a voice switch names the voice now converting: the new one, or the one it
            # kept when the new one would not load. The state follows the worker, not the request.
            in_use = self._voice_ids.get(message.voice_path)
            config = self._state.config
            if in_use is not None and in_use != self._state.voice_id:
                changes["voice_id"] = in_use
                if config is not None:
                    changes["config"] = config.model_copy(update={"voice_id": in_use})
                self._settings.update({"live": {"last_voice": in_use}})
        if state == "running" and self._state.state != "running":
            changes["started_at"] = now_iso()
            with self._lock:
                self._lost.clear()
        if state in ("stopped", "error"):
            changes["started_at"] = None
            with self._lock:
                stop_error, self._stop_error = self._stop_error, None
            if state == "stopped" and stop_error is not None and message.error is None:
                changes.update(state="error", error=stop_error)
        self._set_state(**changes)
        if was_busy != (state in (*ACTIVE, "stopping")):
            self._compute.emit_usage(force=True)
        if state in ("stopped", "error"):
            self._release_gpu()
        if state == "reconnecting" and not self._settings.settings.live.auto_reconnect:
            # The worker's error names the lost device; keep it once the session has stopped.
            with self._lock:
                self._stop_error = message.error
            self._stop_session()

    def _stop_session(self) -> None:
        """Ask the worker to stop the session, as the core's own decision (``_stop_error`` says why)."""
        with self._lock:
            self._awaiting_start = False
        try:
            self._supervisor.send(P.Stop())
        except RvcNextError:
            pass

    def _on_exit(self, expected: bool, error: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._worker_cached = []
            self._meter_sent = None
        if not expected:
            self._meter_failed()
        self._release_gpu()
        if not self._closing:
            self._compute.emit_usage(force=True)
        if self._closing:
            return
        current = self.state()
        if not expected:
            error = error or {"code": "internal_error", "message": "The live worker stopped unexpectedly", "detail": {}}
            logger.error("Live: %s", error["message"])
            if (error.get("detail") or {}).get("reason") == OUT_OF_MEMORY:
                self._compute.release_after_oom("Live")
        if not expected or current.state in ACTIVE:
            self._set_state(state="error" if not expected else "stopped", error=error if not expected else None, meter=False, passthrough=False, stage=None, started_at=None)

    def _release_gpu(self) -> None:
        if self._compute.lease_holder == LIVE_HOLDER:
            self._compute.release_lease(LIVE_HOLDER)
            self._jobs.notify()

    # -- devices ---------------------------------------------------------------------------

    def _worker_env(self) -> dict[str, str]:
        return {"SD_ENABLE_ASIO": "1"} if self._settings.settings.live.enable_asio else {}

    def _run_devices(self, request: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        from rvc_next.core.engine_errors import worker_error
        from rvc_next.protocol.jsonl import parse_line
        from rvc_next.protocol.messages import ErrorEvent, ResultEvent

        live = self._settings.settings.live
        full = {
            "enable_asio": live.enable_asio,
            "loopback": live.show_all_devices,
            "backend": self.backend,
            **({"fake": self.fake} if self.fake is not None else {}),
            **request,
        }
        fd, name = tempfile.mkstemp(prefix="rvc-next-devices-", suffix=".json")
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(full, f)
        try:
            import os

            out = subprocess.run(
                [sys.executable, "-m", DEVICES_WORKER, name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env={**os.environ, **self._worker_env()},
            )
        except subprocess.TimeoutExpired as e:
            raise DeviceError("Listing audio devices timed out", "busy") from e
        finally:
            Path(name).unlink(missing_ok=True)
        result: dict[str, Any] | None = None
        for line in out.stdout.splitlines():
            event = parse_line(line)
            if isinstance(event, ResultEvent):
                result = event.data
            elif isinstance(event, ErrorEvent):
                raise worker_error(event.code, event.message, event.detail)
        if result is None:
            detail = (out.stderr or out.stdout).strip().splitlines()[-1:] or ["no output"]
            raise DeviceError(f"Audio devices are not available: {detail[0]}", "missing")
        return result

    def devices(self, refresh: bool = False, max_age: float | None = None) -> DeviceList:
        """The audio devices, as the menus and device resolution see them (with ``live.show_all_devices``
        the inputs include output devices recorded as loopback inputs; without it they are left out).

        ``refresh`` enumerates again; ``max_age`` enumerates again only when the cached list is
        older than that many seconds (an enumeration is a subprocess, most of a second, several
        seconds on Windows with many drivers). Enumerations never overlap: a call made while one
        runs waits for it and takes its list when that one started after the call or, without
        ``refresh``, is recent enough, so the page's polls and checks cannot pile up subprocesses."""
        asked = time.monotonic()

        def recent(at: float) -> bool:
            return at >= asked or (not refresh and (max_age is None or time.monotonic() - at <= max_age))

        with self._lock:
            cached, at = self._devices, self._devices_at
        if cached is not None and not refresh and recent(at):
            return self._view(cached)
        with self._enumerating:
            with self._lock:
                cached, at = self._devices, self._devices_at
            if cached is not None and recent(at):
                return self._view(cached)
            started = time.monotonic()
            data = self._run_devices({"action": "enumerate"})
            fresh = DeviceList.model_validate(data)
            changed = cached is None or _signature(cached) != _signature(fresh)
            with self._lock:
                self._devices = fresh
                self._devices_at = started
        if changed and cached is not None:
            self._events.publish(DevicesChangedEvent(devices=self._view(fresh)))
            with self._lock:
                self._meter_failures, self._meter_retry_at = 0, 0.0  # a device came or went: the meter tries at once
            self._meter_wake.set()
        return self._view(fresh)

    def _view(self, listing: DeviceList) -> DeviceList:
        # Loopback sources from the provider are enumerated only with the setting on; a PortAudio
        # build's own loopback devices (``[Loopback]``, ``Monitor of``) are always there and hidden here.
        if self._settings.settings.live.show_all_devices:
            return listing
        return listing.model_copy(update={"inputs": [p for p in listing.inputs if not p.is_loopback]})

    def _on_settings_keys(self, keys: list[str]) -> None:
        if any(k.startswith(("live.devices", "live.stream")) for k in keys):
            self._sync_latency_test()
        if any(k.startswith("live.devices") for k in keys):
            with self._lock:
                self._meter_failures, self._meter_retry_at = 0, 0.0
            self._meter_wake.set()  # another input or gain: the meter follows
        if "live.show_all_devices" in keys:
            # The loopback sources are listed (or not) by the enumeration itself: list again, off the
            # caller's thread; a changed list goes out as devices_changed.
            threading.Thread(target=self._relist, name="rvc-live-relist", daemon=True).start()

    def _relist(self) -> None:
        try:
            self.devices(refresh=True)
        except RvcNextError as e:
            logger.warning("Live: listing the audio devices failed: %s", e.message)

    def backend_info(self) -> dict[str, Any]:
        """For ``doctor``: the sounddevice version and what PortAudio sees."""
        from importlib import metadata

        try:
            version = metadata.version("sounddevice")
        except metadata.PackageNotFoundError:
            return {"error": "sounddevice is not installed"}
        try:
            devices = self.devices(refresh=True)
        except RvcNextError as e:
            return {"version": version, "error": e.message}
        return {
            "version": version,
            "portaudio": f"sounddevice {version}; host APIs: {', '.join(h.name for h in devices.host_apis) or 'none'}; {len(devices.inputs)} inputs, {len(devices.outputs)} outputs",
            "host_apis": [h.name for h in devices.host_apis],
            "inputs": len(devices.inputs),
            "outputs": len(devices.outputs),
        }

    def resolve(self, devices: LiveDevices, device_list: DeviceList | None = None) -> list[ResolvedDevice]:
        from rvc_next.engine.audio_io.devices import resolve_selection

        listing = (device_list or self.devices()).model_dump()
        out: list[ResolvedDevice] = []
        roles: list[tuple[str, DeviceSelection | None, str]] = [("input", devices.input, "input"), ("output", devices.output, "output")]
        if devices.monitor is not None:
            roles.append(("monitor", devices.monitor, "output"))
        for role, selection, direction in roles:
            status, device, message = resolve_selection(selection.model_dump() if selection else None, listing, direction)
            out.append(ResolvedDevice.model_validate({"role": role, "status": status, "device": device, "message": message}))
        return out

    @staticmethod
    def _endpoint(selection: DeviceSelection | None, resolved: ResolvedDevice | None) -> dict[str, Any] | None:
        if selection is None or resolved is None:
            return None
        return {
            "device": resolved.device.model_dump() if resolved.device else None,
            "channels": selection.channels,
            "sample_rate": selection.sample_rate,
            "exclusive": selection.exclusive,
        }

    def _worker_devices(self, devices: LiveDevices, resolved: list[ResolvedDevice]) -> dict[str, Any]:
        by_role = {r.role: r for r in resolved}
        return {
            "input": self._endpoint(devices.input, by_role.get("input")),
            "output": self._endpoint(devices.output, by_role.get("output")),
            "monitor": self._endpoint(devices.monitor, by_role.get("monitor")) if devices.monitor else None,
            "monitor_source": devices.monitor_source,
            "monitor_gain_db": devices.monitor_gain_db,
            "output_gain_db": devices.output_gain_db,
            "input_gain_db": devices.input_gain_db,
        }

    def check(self, devices: LiveDevices) -> DeviceCheck:
        """Resolve the selection and ask PortAudio whether it opens, before Start."""
        from rvc_next.engine.stream.latency import estimate_latency_ms

        resolved = self.resolve(devices, self.devices(refresh=True))
        problems: list[DeviceProblem] = []
        for r in resolved:
            if r.status == "missing":
                problems.append(DeviceProblem(role=r.role, reason="missing", message=r.message or f"No {r.role} device", action="choose"))
            elif r.status == "default_fallback":
                problems.append(DeviceProblem(role=r.role, reason="fallback", message=r.message or "Using the system default", action="choose"))
        by_role = {r.role: r for r in resolved}
        feedback = _feedback(resolved)
        if feedback is not None:
            problems.append(feedback)

        def sel(role: str, selection: DeviceSelection | None) -> dict[str, Any] | None:
            return _probe_selection(selection, by_role.get(role))

        topology = sample_rate = None
        if not any(p.reason in ("missing", "feedback") for p in problems):
            result = self._run_devices(
                {"action": "check", "input": sel("input", devices.input), "output": sel("output", devices.output), "monitor": sel("monitor", devices.monitor)}
            )
            topology = result.get("topology") if result.get("topology") in ("duplex", "split") else None
            sample_rate = result.get("sample_rate")
            if not result.get("ok"):
                reason = result.get("reason") or "format"
                exclusive = any(s and s.exclusive for s in (devices.input, devices.output))
                action = {
                    "format": "use_48k",
                    "channels": "disable_exclusive" if exclusive else "choose",
                    "busy": "disable_exclusive" if exclusive else "choose",
                    "permission": "grant_permission",
                }.get(reason, "choose")
                problems.append(DeviceProblem(role=result.get("role") or "output", reason=reason, message=result.get("message") or "The device refused the format", action=action))
        latency = None
        if topology is not None:
            in_dev = by_role["input"].device if "input" in by_role else None
            out_dev = by_role["output"].device if "output" in by_role else None
            latency = estimate_latency_ms(
                self._settings.settings.live.stream.to_engine(),
                topology == "split",
                (in_dev.latency_ms[0] if in_dev else 0.0),
                (out_dev.latency_ms[0] if out_dev else 0.0),
            )
        ok = not any(p.reason != "fallback" or p.role == "output" for p in problems)
        return DeviceCheck(
            ok=ok, resolved=resolved, problems=problems, topology=topology, sample_rate=sample_rate, est_latency_ms=round(latency, 1) if latency is not None else None
        )

    def test_tone(self, request: TestToneRequest) -> LiveState:
        saved = self._settings.settings.live.devices
        selection = request.device or (saved.monitor if request.role == "monitor" else saved.output) or DeviceSelection()
        resolved = self.resolve(LiveDevices(input=DeviceSelection(), output=selection))
        out = next(r for r in resolved if r.role == "output")
        if out.device is None:
            raise DeviceError(out.message or "No output device", "missing", {"role": request.role})
        result = self._run_devices(
            {
                "action": "test_tone",
                "device": {**selection.model_dump(), "device_id": out.device.id, "host_api": out.device.host_api, "portaudio_index": out.device.portaudio_index},
            }
        )
        if not result.get("ok"):
            raise DeviceError(result.get("message") or "The test sound could not play", result.get("reason") or "format", {"role": request.role})
        return self.state()

    # -- measured latency ---------------------------------------------------------------------

    def measure_latency(self, request: LatencyTestRequest) -> LatencyMeasurement:
        """Play test bursts on the output and find them in the input (a cable, a virtual cable's
        loopback, or speakers near the microphone): the real round trip of the devices and the
        session's buffers, plus the engine's own delay. Live must be stopped; the input meter is
        paused meanwhile. The result also goes into the state (``latency_test``)."""
        from rvc_next.engine.stream.latency import engine_delay_ms, estimate_latency_ms

        live = self._settings.settings.live
        devices = request.devices or live.devices
        stream = request.stream or live.stream
        with self._control:
            current = self.state()
            if current.state in ACTIVE:
                raise BusyError("Stop Live before measuring the latency")
            if current.passthrough:
                raise BusyError("Turn off Hear yourself before measuring the latency")
            with self._lock:
                if self._measuring:
                    raise BusyError("A latency measurement is already running")
                self._measuring = True
        try:
            resolved = self.resolve(LiveDevices(input=devices.input, output=devices.output), self.devices(refresh=True))
            by_role = {r.role: r for r in resolved}
            for r in resolved:
                if r.device is None:
                    raise _missing(r, devices)
            with self._meter_lock:
                # ``_measuring`` keeps the meter thread from opening it again meanwhile.
                paused = self._supervisor.running and (self._meter_sent is not None or self.state().meter)
                if paused:
                    self._supervisor.send(P.Meter(on=False))
                    self._meter_sent = None
            if paused:
                time.sleep(METER_SETTLE)
            result = self._run_devices(
                {
                    "action": "measure_latency",
                    "input": _probe_selection(devices.input, by_role["input"]),
                    "output": _probe_selection(devices.output, by_role["output"]),
                    "block_ms": stream.block_ms,
                    "level_db": request.level_db,
                },
                timeout=60.0 + 10.0 * stream.block_ms / 1000,
            )
        finally:
            with self._lock:
                self._measuring = False
            self._meter_wake.set()
        if not result.get("ok"):
            raise DeviceError(result.get("message") or "The devices could not be opened", result.get("reason") or "busy", {"role": result.get("role") or "output"})
        engine = stream.to_engine()
        round_trip = result.get("round_trip_ms") if result.get("found") else None
        engine_ms = engine_delay_ms(engine)
        reported = [float(x) for x in result.get("reported_ms") or [0.0, 0.0]]
        measurement = LatencyMeasurement(
            ok=bool(result.get("found")),
            reason=result.get("failure"),
            latency_ms=round(round_trip + engine_ms, 1) if round_trip is not None else None,
            round_trip_ms=round(round_trip, 1) if round_trip is not None else None,
            engine_ms=engine_ms,
            estimated_ms=round(estimate_latency_ms(engine, result.get("topology") == "split", reported[0], reported[1]), 1),
            jitter_ms=round(result["jitter_ms"], 2) if result.get("jitter_ms") is not None else None,
            pings=int(result.get("pings") or 0),
            detected=int(result.get("detected") or 0),
            snr_db=result.get("snr_db"),
            input_peak_db=float(result.get("input_peak_db", -120.0)),
            clipped=bool(result.get("clipped")),
            underruns=int(result.get("underruns") or 0),
            topology=result.get("topology") or "duplex",
            sample_rate=int(result.get("sample_rate") or 0),
            reported_ms=reported,
            devices=devices,
            stream=stream,
            measured_at=now_iso(),
        )
        self._set_state(latency_test=measurement)
        self._sync_latency_test()
        return measurement

    def _sync_latency_test(self) -> None:
        """Keep ``latency_test`` true to the current devices and stream settings (the running
        session's, else the saved ones): drop it when the input, the output or the block changed;
        recompute the engine's part when the crossfade or input denoise did."""
        from rvc_next.engine.stream.latency import engine_delay_ms, estimate_latency_ms

        with self._lock:
            m = self._state.latency_test
            if m is None:
                return
            config = self._state.config if self._state.state in ACTIVE else None
        live = self._settings.settings.live
        devices, stream = (config.devices, config.stream) if config is not None else (live.devices, live.stream)
        if (m.devices.input, m.devices.output) != (devices.input, devices.output) or m.stream.block_ms != stream.block_ms:
            self._set_state(latency_test=None)
            return
        if (m.stream.crossfade_ms, m.stream.input_denoise) == (stream.crossfade_ms, stream.input_denoise):
            return
        engine = stream.to_engine()
        engine_ms = engine_delay_ms(engine)
        updated = m.model_copy(
            update={
                "stream": stream,
                "engine_ms": engine_ms,
                "latency_ms": round(m.round_trip_ms + engine_ms, 1) if m.round_trip_ms is not None else None,
                "estimated_ms": round(estimate_latency_ms(engine, m.topology == "split", m.reported_ms[0], m.reported_ms[1]), 1),
            }
        )
        self._set_state(latency_test=updated)

    # -- the session -------------------------------------------------------------------------

    def _worker_request(self) -> dict[str, Any]:
        c = self._settings.settings.compute
        req: dict[str, Any] = {
            "assets_dir": str(self._settings.assets_dir),
            "device": c.device,
            "precision": c.precision,
            "cuda_graph": c.cuda_graph_live,
            "enable_asio": self._settings.settings.live.enable_asio,
            "backend": self.backend,
        }
        if self.fake is not None:
            req["fake"] = self.fake
        return req

    def _voice_paths(self, voice_id: str, speaker_id: int) -> tuple[str, str | None]:
        voice = self._models.get(voice_id)
        index = self._models.index_for(voice, speaker_id)
        self._voice_ids[voice.model_path] = voice_id
        return voice.model_path, str(index) if index else None

    def _require_assets(self, voice_id: str, params: VoiceParamsModel) -> None:
        from rvc_next.engine.features.embedders import asset_id

        voice = self._models.get(voice_id)
        ids = [asset_id(voice.embedder)]
        if voice.pitch_guidance:
            ids += f0_assets(params.f0_method, self._settings.settings.compute.device == "dml")
        self._assets.require(ids, "Live conversion")

    def start(self, config: LiveConfig | None = None) -> LiveState:
        """Start a session; ``config`` None resumes the previous one from settings."""
        with self._control:
            live = self._settings.settings.live
            if config is None:
                if not live.last_voice:
                    raise RvcNextError("Choose a voice first")
                config = LiveConfig(voice_id=live.last_voice, params=live.last_params, stream=live.stream, devices=live.devices)
            with self._lock:
                if self._state.state in ACTIVE:
                    raise BusyError("Live is already running")
            voice = self._models.get(config.voice_id)
            require_effects(config.stream.effects)
            if config.params.speaker_id >= max(voice.speaker_slots, 1):
                raise ValidationError(f"Speaker {config.params.speaker_id} is outside this voice's 0–{voice.speaker_slots - 1}")
            self._require_assets(config.voice_id, config.params)
            # The checks above are instant; what follows takes seconds (a device enumeration in a
            # subprocess, then starting the live worker, which imports its libraries before it
            # connects). Say so at once, so the screen does not sit unchanged after the click.
            before = self.state()
            with self._lock:
                # Until Start reaches the worker, a "stopped" from it (a new worker announces itself so,
                # an idle one answers the meter so) must not undo "starting".
                self._awaiting_start = True
                self._stop_error = None
            self._set_state(state="starting", stage="worker" if config.browser else "devices", voice_id=config.voice_id, config=config, error=None)
            try:
                self._launch(config)
            except BaseException:
                # Nothing reached the worker: back to how things were, with the error raised to the caller.
                with self._lock:
                    self._awaiting_start = False
                self._set_state(state=before.state, stage=None, voice_id=before.voice_id, config=before.config, error=before.error)
                raise
            self._settings.update(
                {"live": {"devices": config.devices.model_dump(), "stream": config.stream.model_dump(), "last_voice": config.voice_id, "last_params": config.params.model_dump()}}
            )
            self._sync_latency_test()
            return self.state()

    def _launch(self, config: LiveConfig) -> None:
        """The slow part of ``start``: resolve the devices against a fresh list, start the worker
        unless it runs, take the GPU lease and send Start. The state already says ``starting``."""
        if config.browser is not None:
            self._launch_browser(config)
            return
        resolved = self.resolve(config.devices, self.devices(max_age=START_LIST_MAX_AGE))
        by_role = {r.role: r for r in resolved}
        for r in resolved:
            if r.status == "missing":
                raise _missing(r, config.devices)
        _refuse_feedback(resolved)
        out = by_role["output"]
        if out.status == "default_fallback" and not config.allow_output_fallback:
            raise DeviceError(
                f"{out.message}. Confirm to send converted audio to {out.device.name if out.device else 'the default output'}.", "missing", {"role": "output", "fallback": True}
            )
        voice_path, index_path = self._voice_paths(config.voice_id, config.params.speaker_id)
        if not self._supervisor.running:
            self._set_state(stage="worker", resolved=resolved)
        self._supervisor.ensure(self._worker_request(), env=self._worker_env())
        if self._settings.settings.compute.device != "cpu":
            self._compute.acquire_live()
            self._jobs.notify()
        self._set_state(state="starting", stage=None, resolved=resolved, error=None, meter=False, passthrough=False)
        self._supervisor.send(
            P.Start(
                config={
                    "voice_path": voice_path,
                    "index_path": index_path,
                    "params": config.params.model_dump(),
                    "stream": config.stream.model_dump(),
                    "devices": self._worker_devices(config.devices, resolved),
                }
            )
        )

    def _launch_browser(self, config: LiveConfig) -> None:
        """``_launch`` for the browser's microphone and speakers: no devices to resolve."""
        assert config.browser is not None
        if self._browser is None:
            raise ConflictError("The browser's audio is not connected: start from the Live page with “This browser” chosen", {"reason": "browser_not_connected"})
        voice_path, index_path = self._voice_paths(config.voice_id, config.params.speaker_id)
        if not self._supervisor.running:
            self._set_state(stage="worker", resolved=[])
        self._supervisor.ensure(self._worker_request(), env=self._worker_env())
        if self._settings.settings.compute.device != "cpu":
            self._compute.acquire_live()
            self._jobs.notify()
        self._set_state(state="starting", stage=None, resolved=[], error=None, meter=False, passthrough=False)
        self._supervisor.send(
            P.Start(
                config={
                    "voice_path": voice_path,
                    "index_path": index_path,
                    "params": config.params.model_dump(),
                    "stream": config.stream.model_dump(),
                    "devices": self._browser_devices(config.browser.sample_rate, config.devices),
                }
            )
        )

    def stop(self) -> LiveState:
        with self._lock:
            self._awaiting_start = False
        if self._supervisor.running:
            try:
                self._supervisor.send(P.Stop())
            except RvcNextError:
                pass
        else:
            self._set_state(state="stopped", meter=False, passthrough=False)
            self._release_gpu()
        return self.state()

    def update_voice(self, params: VoiceParamsModel) -> LiveState:
        with self._control:
            current = self.state()
            if current.state in ACTIVE and self._supervisor.running:
                if current.config and current.voice_id and params.f0_method != current.config.params.f0_method:
                    self._require_assets(current.voice_id, params)  # a pitch model that is not installed
                if current.config and params.speaker_id != current.config.params.speaker_id and current.voice_id:
                    voice_path, index_path = self._voice_paths(current.voice_id, params.speaker_id)
                    self._supervisor.send(P.SetVoice(voice_path=voice_path, index_path=index_path, speaker_id=params.speaker_id))
                self._supervisor.send(P.UpdateVoice(params=params.model_dump()))
                if current.config:
                    self._set_state(config=current.config.model_copy(update={"params": params}))
            self._settings.update({"live": {"last_params": params.model_dump()}})
            return self.state()

    def update_stream(self, stream: StreamParamsModel) -> LiveState:
        require_effects(stream.effects)
        with self._control:
            current = self.state()
            if current.state in ACTIVE and self._supervisor.running:
                self._supervisor.send(P.UpdateStream(stream=stream.model_dump()))
                if current.config:
                    self._set_state(config=current.config.model_copy(update={"stream": stream}))
            self._settings.update({"live": {"stream": stream.model_dump()}})
            self._sync_latency_test()
            return self.state()

    def set_voice(self, voice_id: str) -> LiveState:
        with self._control:
            current = self.state()
            voice = self._models.get(voice_id)
            params = current.config.params if current.config else self._settings.settings.live.last_params
            if params.speaker_id >= max(voice.speaker_slots, 1):
                params = params.model_copy(update={"speaker_id": 0})
            if current.state in ACTIVE and self._supervisor.running:
                self._require_assets(voice_id, params)
                voice_path, index_path = self._voice_paths(voice_id, params.speaker_id)
                self._supervisor.send(P.SetVoice(voice_path=voice_path, index_path=index_path, speaker_id=params.speaker_id))
                if current.config:
                    self._set_state(voice_id=voice_id, config=current.config.model_copy(update={"voice_id": voice_id, "params": params}))
            self._settings.update({"live": {"last_voice": voice_id}})
            return self.state()

    def set_devices(self, devices: LiveDevices) -> LiveState:
        with self._control:
            self._settings.update({"live": {"devices": devices.model_dump()}})
            current = self.state()
            if current.state in ACTIVE and self._supervisor.running and current.config and current.config.browser:
                # The browser stays the device; only the gains apply.
                self._supervisor.send(P.SetDevices(devices=self._browser_devices(current.config.browser.sample_rate, devices)))
                self._set_state(config=current.config.model_copy(update={"devices": devices}))
            elif current.state in ACTIVE and self._supervisor.running:
                resolved = self.resolve(devices, self.devices(refresh=True))
                for r in resolved:
                    if r.status == "missing":
                        raise _missing(r, devices)
                _refuse_feedback(resolved)
                self._supervisor.send(P.SetDevices(devices=self._worker_devices(devices, resolved)))
                config = current.config.model_copy(update={"devices": devices}) if current.config else None
                self._set_state(resolved=resolved, config=config)
            self._sync_latency_test()
            return self.state()

    def meter(self, on: bool) -> LiveState:
        """Want the input meter for the next ``METER_LEASE`` seconds (``on``), or no longer.

        Returns at once: the meter thread starts the worker if needed and opens the selected input
        whenever Live is idle (stopped or failed, no Hear yourself, no measurement), so the level
        shows before the first Start, again after a session, and once a refused or unplugged
        device can be opened."""
        with self._lock:
            self._meter_until = time.monotonic() + METER_LEASE if on else 0.0
            if on and (self._meter_thread is None or not self._meter_thread.is_alive()) and not self._closing:
                self._meter_thread = threading.Thread(target=self._meter_loop, name="rvc-live-meter", daemon=True)
                self._meter_thread.start()
        self._meter_wake.set()
        return self.state()

    def _meter_failed(self) -> None:
        """The meter's device refused or went away, or the worker died: try again after a pause that grows."""
        with self._lock:
            self._meter_sent = None
            delay = METER_RETRY[min(self._meter_failures, len(METER_RETRY) - 1)]
            self._meter_failures += 1
            self._meter_retry_at = time.monotonic() + delay
        self._meter_wake.set()

    def _meter_loop(self) -> None:
        while not self._closing:
            self._meter_wake.clear()
            try:
                wait = self._sync_meter()
            except RvcNextError as e:  # the worker would not start, or went away
                logger.warning("Live: the input meter: %s", e.message)
                self._meter_failed()
                wait = None
            except Exception:
                logger.exception("Live: the input meter failed")
                self._meter_failed()
                wait = None
            self._meter_wake.wait(wait)

    def _meter_idle(self) -> bool:
        """Live has nothing open that meters the input already, and no measurement needs it."""
        with self._lock:
            state = self._state
            return state.state not in (*ACTIVE, "stopping") and not state.passthrough and not self._measuring

    def _sync_meter(self) -> float | None:
        """Bring the worker's meter in line with the lease and the state; the seconds until the next look (None: on a change)."""
        now = time.monotonic()
        with self._lock:
            until, retry_at, sent = self._meter_until, self._meter_retry_at, self._meter_sent
        wanted = now < until
        if not wanted or not self._meter_idle():
            # A session, passthrough or a measurement has the input (the worker closes the meter for
            # the first two, the measurement itself for the third); a meter no longer wanted is closed here.
            with self._meter_lock:
                if (sent is not None or self.state().meter) and self._meter_idle() and self._supervisor.running:
                    self._supervisor.send(P.Meter(on=False))
                with self._lock:
                    self._meter_sent = None
            return until - now if wanted else None
        if now < retry_at:
            return min(retry_at, until) - now
        target = self._meter_target()
        if target is None:
            self._meter_failed()
            return None
        if target == sent and self._supervisor.running:
            return until - now
        self._supervisor.ensure(self._worker_request(), env=self._worker_env())  # seconds, on the first use
        with self._meter_lock:
            if not self._meter_idle():
                return None
            self._supervisor.send(P.Meter(on=True, input=target["input"], input_gain_db=target["input_gain_db"]))
            with self._lock:
                self._meter_sent = target
        return until - now

    def _meter_target(self) -> dict[str, Any] | None:
        """The selected input as the worker opens it, and its gain; None when no input device is there."""
        devices = self._settings.settings.live.devices
        try:
            resolved = self.resolve(LiveDevices(input=devices.input, output=devices.output), self.devices())
        except RvcNextError as e:
            logger.warning("Live: no device list for the input meter: %s", e.message)
            return None
        r = next((r for r in resolved if r.role == "input"), None)
        if r is None or r.device is None:
            return None
        return {"input": self._endpoint(devices.input, r), "input_gain_db": devices.input_gain_db}

    def passthrough(self, on: bool) -> LiveState:
        devices = self._settings.settings.live.devices
        resolved = self.resolve(devices, self.devices()) if on else []
        _refuse_feedback(resolved)
        if not self._supervisor.running:
            if not on:
                return self._set_state(passthrough=False)
            self._supervisor.ensure(self._worker_request(), env=self._worker_env())
            self._supervisor.send(P.SetDevices(devices=self._worker_devices(devices, resolved)))
        self._supervisor.send(P.Passthrough(on=on))
        return self._set_state(passthrough=on)

    # -- recording ---------------------------------------------------------------------------------

    def start_recording(self, request: RecordingRequest) -> LiveState:
        """Record the running session to a WAV file in the outputs folder; it becomes an output when it stops."""
        with self._control:
            current = self.state()
            if current.state != "running":
                raise ConflictError("Recording needs Live to be running")
            if current.recording is not None:
                return current
            voice = self._models.get(current.voice_id) if current.voice_id else None
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H-%M-%S")
            folder = self._settings.outputs_dir / stamp[:10] / "live"
            stem = f"Live {stamp[11:]}" + (f".{slugify(voice.name)}" if voice else "")
            path = unique_path(folder / f"{stem}.{request.source}.wav")
            self._recording_seen = False
            self._supervisor.send(P.Record(on=True, path=str(path), source=request.source))
            return self._set_state(recording=LiveRecording(path=str(path), source=request.source, started_at=now_iso()))

    def stop_recording(self) -> LiveState:
        """Stop recording; the file is added to the outputs when the worker has closed it."""
        with self._control:
            if self.state().recording is None:
                return self.state()
            try:
                self._supervisor.send(P.Record(on=False))
            except RvcNextError:
                pass
            return self._set_state(recording=None)

    def _on_recorded(self, message: P.Recorded) -> None:
        if message.dropped_blocks:
            logger.warning("Live recording %s: %d blocks dropped (the disk fell behind)", message.path, message.dropped_blocks)
        path = Path(message.path)
        if self._audio is None or not path.is_file():
            return
        current = self.state()
        params = current.config.params if current.config else None
        self._audio.add_output(None, path, "recording", message.source, model_id=current.voice_id, voice=params)
        with self._lock:
            still = self._state.recording is not None and self._state.recording.path == message.path
        if still:
            # The session ended (or changed rate) under the recording.
            self._set_state(recording=None)

    # -- recovery ---------------------------------------------------------------------------------

    def _start_reconnect(self) -> None:
        with self._lock:
            if self._reconnect is not None and self._reconnect.is_alive():
                return
            self._reconnect = threading.Thread(target=self._reconnect_loop, name="live-reconnect", daemon=True)
            self._reconnect.start()

    def _reconnect_loop(self) -> None:
        while not self._closing:
            time.sleep(RECONNECT_INTERVAL)
            current = self.state()
            if current.state not in ("reconnecting", "running") or current.config is None:
                return
            with self._lock:
                lost = set(self._lost)
            if not lost:
                return
            try:
                listing = self.devices(refresh=True)
            except RvcNextError:
                continue
            devices = current.config.devices
            resolved = self.resolve(devices, listing)
            if all(r.status in ("exact", "matched") for r in resolved if r.role in lost):
                logger.info("Live: reconnecting %s", ", ".join(sorted(lost)))
                try:
                    self._supervisor.send(P.SetDevices(devices=self._worker_devices(devices, resolved)))
                    with self._lock:
                        self._lost.clear()
                    self._set_state(resolved=resolved)
                except RvcNextError:
                    return

    # -- lifecycle --------------------------------------------------------------------------------

    def start_background(self) -> None:
        """Nothing runs until a session or a meter is requested."""

    def close(self) -> None:
        self._closing = True
        self._meter_wake.set()
        if self._supervisor.running:
            self._supervisor.shutdown()
        self._release_gpu()


def _probe_selection(selection: DeviceSelection | None, resolved: ResolvedDevice | None) -> dict[str, Any] | None:
    """A selection as the devices worker opens it: the saved choice plus the resolved device's index."""
    if selection is None or resolved is None or resolved.device is None:
        return None
    d = resolved.device
    return {**selection.model_dump(), "device_id": d.id, "host_api": d.host_api, "portaudio_index": d.portaudio_index, "loopback_source": d.loopback_source}


def _feedback(resolved: list[ResolvedDevice]) -> DeviceProblem | None:
    """A loopback input that records the output (or monitor) the converted voice plays on: it would
    hear itself, louder each round. Matched by name, which is exact on Windows (the loopback is the
    output endpoint itself); a PulseAudio sink's name rarely matches PortAudio's ALSA names."""
    from rvc_next.engine.audio_io.devices import normalize_name

    by_role = {r.role: r for r in resolved}
    inp = by_role.get("input")
    recorded = inp.device.loopback_of if inp is not None and inp.device is not None else None
    if not recorded:
        return None
    for role in ("output", "monitor"):
        r = by_role.get(role)
        if r is not None and r.device is not None and normalize_name(r.device.name) == normalize_name(recorded):
            return DeviceProblem(
                role="input", reason="feedback", message=f"The input records {r.device.name}, which plays the {role}: the converted voice would feed back into it", action="choose"
            )
    return None


def _missing(resolved: ResolvedDevice, devices: LiveDevices) -> DeviceError:
    """``DeviceError`` (reason ``missing``) for a role whose device is not there, naming the role and
    the device that was chosen for it."""
    selection: DeviceSelection | None = getattr(devices, resolved.role, None)
    return DeviceError(resolved.message or f"No {resolved.role} device", "missing", {"role": resolved.role, "device": selection.name if selection else None})


def _refuse_feedback(resolved: list[ResolvedDevice]) -> None:
    """Raise ``DeviceError`` (reason ``feedback``) before a session, a device change or passthrough would loop."""
    problem = _feedback(resolved)
    if problem is not None:
        raise DeviceError(problem.message, "feedback", {"role": "input"})


def _signature(listing: DeviceList) -> str:
    data = listing.model_dump()
    data.pop("enumerated_at", None)
    for direction in ("inputs", "outputs"):
        for p in data.get(direction, []):
            for v in p.get("variants", []):
                v.pop("portaudio_index", None)
    return json.dumps(data, sort_keys=True)
