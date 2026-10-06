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
from pathlib import Path
from typing import Any

from rvc_next.core.assets.service import AssetService
from rvc_next.core.clock import now_iso
from rvc_next.core.compute.service import LIVE_HOLDER, ComputeService, MemoryHolder
from rvc_next.core.errors import BusyError, DeviceError, RvcNextError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import DevicesChangedEvent, LiveStateEvent, LiveStatsEvent
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
    LiveState,
    LiveStats,
    PhysicalDevice,
    ResolvedDevice,
    TestToneRequest,
)
from rvc_next.core.live.supervisor import LiveSupervisor
from rvc_next.core.models.service import VoiceModelService
from rvc_next.core.params import StreamParamsModel, VoiceParamsModel
from rvc_next.core.settings import SettingsService
from rvc_next.protocol import live as P
from rvc_next.protocol.messages import OUT_OF_MEMORY

logger = logging.getLogger(__name__)

DEVICES_WORKER = "rvc_next.workers.devices"
RECONNECT_INTERVAL = 2.0
START_LIST_MAX_AGE = 10.0
"""Seconds a device list stays good enough for Start. PortAudio's indexes hold only while no device
is added or removed; the Live page re-enumerates every 5 s, so Start from it rarely enumerates."""
METER_SETTLE = 0.3
"""Seconds for the live worker to close the input meter before a measurement opens the input."""
ACTIVE = ("starting", "loading", "prewarming", "running", "reconnecting")


class LiveService:
    def __init__(self, settings: SettingsService, events: EventBus, models: VoiceModelService, assets: AssetService, compute: ComputeService, jobs: JobService) -> None:
        self._settings = settings
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
        self._lost: set[str] = set()
        self._reconnect: threading.Thread | None = None
        self._closing = False
        self._awaiting_start = False
        self._measuring = False
        self.backend = "sounddevice"
        """``fake`` in tests, with ``fake`` as the fake backend's configuration."""
        self.fake: dict[str, Any] | None = None
        self._supervisor = LiveSupervisor(self._on_event, self._on_exit)
        settings.on_keys_change(self._on_settings_keys)
        self._worker_cached: list[str] = []
        """The models the live worker keeps loaded, from its last state message."""
        compute.add_holder(MemoryHolder(LIVE_HOLDER, busy=self._busy, cached=self._cached, release=self._release_memory))

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
        return snapshot

    def _on_event(self, message: Any) -> None:
        if isinstance(message, P.State):
            self._on_state(message)
        elif isinstance(message, P.Stats):
            stats = LiveStats.model_validate({k: getattr(message, k) for k in P.STATS_FIELDS})
            with self._lock:
                self._stats = stats
            self._events.publish(LiveStatsEvent(stats=stats))
        elif isinstance(message, P.DeviceLost):
            logger.warning("Live: %s device lost (%s): %s", message.direction, message.reason, message.message)
            with self._lock:
                self._lost.add(message.direction)
            if self._settings.settings.live.auto_reconnect:
                self._start_reconnect()
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
        changes: dict[str, Any] = {"state": state, "meter": message.meter, "passthrough": message.passthrough, "stage": message.stage}
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
        self._set_state(**changes)
        if was_busy != (state in (*ACTIVE, "stopping")):
            self._compute.emit_usage(force=True)
        if state in ("stopped", "error"):
            self._release_gpu()
        if state == "reconnecting" and not self._settings.settings.live.auto_reconnect:
            try:
                self._supervisor.send(P.Stop())
            except RvcNextError:
                pass

    def _on_exit(self, expected: bool) -> None:
        with self._lock:
            self._worker_cached = []
        self._release_gpu()
        if not self._closing:
            self._compute.emit_usage(force=True)
        if self._closing:
            return
        current = self.state()
        if not expected or current.state in ACTIVE:
            self._set_state(
                state="error" if not expected else "stopped",
                error=None if expected else {"code": "internal_error", "message": "The live worker stopped unexpectedly", "detail": {}},
                meter=False,
                passthrough=False,
            )

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

        full = {"enable_asio": self._settings.settings.live.enable_asio, "backend": self.backend, **({"fake": self.fake} if self.fake is not None else {}), **request}
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
        """The audio devices, as the menus and device resolution see them (``live.show_all_devices``
        merges the two directions; the enumeration itself is cached as PortAudio reports it).

        ``refresh`` enumerates again; ``max_age`` enumerates again only when the cached list is
        older than that many seconds (an enumeration is a subprocess, most of a second)."""
        with self._lock:
            cached, age = self._devices, time.monotonic() - self._devices_at
        if cached is not None and not refresh and (max_age is None or age <= max_age):
            return self._view(cached)
        data = self._run_devices({"action": "enumerate"})
        fresh = DeviceList.model_validate(data)
        changed = cached is None or _signature(cached) != _signature(fresh)
        with self._lock:
            self._devices = fresh
            self._devices_at = time.monotonic()
        if changed and cached is not None:
            self._events.publish(DevicesChangedEvent(devices=self._view(fresh)))
        return self._view(fresh)

    def _view(self, listing: DeviceList) -> DeviceList:
        if not self._settings.settings.live.show_all_devices:
            return listing
        return listing.model_copy(update={"inputs": _with_others(listing.inputs, listing.outputs), "outputs": _with_others(listing.outputs, listing.inputs)})

    def _on_settings_keys(self, keys: list[str]) -> None:
        if any(k.startswith(("live.devices", "live.stream")) for k in keys):
            self._sync_latency_test()
        if "live.show_all_devices" in keys:
            with self._lock:
                cached = self._devices
            if cached is not None:
                self._events.publish(DevicesChangedEvent(devices=self._view(cached)))

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

        def sel(role: str, selection: DeviceSelection | None) -> dict[str, Any] | None:
            return _probe_selection(selection, by_role.get(role))

        topology = sample_rate = None
        if not any(p.reason == "missing" for p in problems):
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
                    raise DeviceError(r.message or f"No {r.role} device", "missing", {"role": r.role})
            meter = current.meter and self._supervisor.running
            if meter:
                self._supervisor.send(P.Meter(on=False))
                time.sleep(METER_SETTLE)
            try:
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
                if meter:
                    self.meter(True)
        finally:
            with self._lock:
                self._measuring = False
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
        voice = self._models.get(voice_id)
        ids = ["hubert"]
        if voice.pitch_guidance and params.f0_method == "rmvpe":
            ids.append("rmvpe-onnx" if self._settings.settings.compute.device == "dml" else "rmvpe")
        elif voice.pitch_guidance and params.f0_method == "fcpe":
            ids.append("fcpe")
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
            self._set_state(state="starting", stage="devices", voice_id=config.voice_id, config=config, error=None)
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
        resolved = self.resolve(config.devices, self.devices(max_age=START_LIST_MAX_AGE))
        by_role = {r.role: r for r in resolved}
        for r in resolved:
            if r.status == "missing":
                raise DeviceError(r.message or f"No {r.role} device", "missing", {"role": r.role})
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
            if current.state in ACTIVE and self._supervisor.running:
                resolved = self.resolve(devices, self.devices(refresh=True))
                for r in resolved:
                    if r.status == "missing":
                        raise DeviceError(r.message or f"No {r.role} device", "missing", {"role": r.role})
                self._supervisor.send(P.SetDevices(devices=self._worker_devices(devices, resolved)))
                config = current.config.model_copy(update={"devices": devices}) if current.config else None
                self._set_state(resolved=resolved, config=config)
            self._sync_latency_test()
            return self.state()

    def meter(self, on: bool) -> LiveState:
        current = self.state()
        if not on and not self._supervisor.running:
            return self._set_state(meter=False)
        if current.state in ACTIVE:
            # While a session runs, its own meters are shown.
            return current
        devices = self._settings.settings.live.devices
        resolved = self.resolve(LiveDevices(input=devices.input, output=devices.output), self.devices())
        if on:
            self._supervisor.ensure(self._worker_request(), env=self._worker_env())
        endpoint = self._endpoint(devices.input, next((r for r in resolved if r.role == "input"), None))
        self._supervisor.send(P.Meter(on=on, input=endpoint))
        return self._set_state(meter=on)

    def passthrough(self, on: bool) -> LiveState:
        if not self._supervisor.running:
            if not on:
                return self._set_state(passthrough=False)
            devices = self._settings.settings.live.devices
            self._supervisor.ensure(self._worker_request(), env=self._worker_env())
            resolved = self.resolve(devices, self.devices())
            self._supervisor.send(P.SetDevices(devices=self._worker_devices(devices, resolved)))
        self._supervisor.send(P.Passthrough(on=on))
        return self._set_state(passthrough=on)

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
        if self._supervisor.running:
            self._supervisor.shutdown()
        self._release_gpu()


def _probe_selection(selection: DeviceSelection | None, resolved: ResolvedDevice | None) -> dict[str, Any] | None:
    """A selection as the devices worker opens it: the saved choice plus the resolved device's index."""
    if selection is None or resolved is None or resolved.device is None:
        return None
    d = resolved.device
    return {**selection.model_dump(), "device_id": d.id, "host_api": d.host_api, "portaudio_index": d.portaudio_index}


def _with_others(own: list[PhysicalDevice], others: list[PhysicalDevice]) -> list[PhysicalDevice]:
    """``own`` followed by the devices of the other direction it does not already list; those keep
    their own ``direction``, which the menus show as a badge."""
    keys = {p.key for p in own}
    return [*own, *(p for p in others if p.key not in keys)]


def _signature(listing: DeviceList) -> str:
    data = listing.model_dump()
    data.pop("enumerated_at", None)
    for direction in ("inputs", "outputs"):
        for p in data.get(direction, []):
            for v in p.get("variants", []):
                v.pop("portaudio_index", None)
    return json.dumps(data, sort_keys=True)
