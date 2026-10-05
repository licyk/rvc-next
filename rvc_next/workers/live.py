"""The live conversion worker.

``python -m rvc_next.workers.live <request.json>``. One long-lived process owns the audio devices
and the real-time threads, apart from the server's event loop and its crashes. It connects back to
the core with ``multiprocessing.connection.Client`` and speaks ``rvc_next.protocol.live``.

Threads: device callbacks only move samples (``AudioSession``); a processing thread runs
``StreamEngine.process``; this control thread reads commands; a stats thread reports at 10 Hz.
The voice stays loaded across stream, device and voice changes. After a device loss the worker
reports it and waits in ``reconnecting`` for ``set_devices`` (the reconnect policy is the core's).
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import fields, replace
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import Any

from rvc_next.engine.audio_io.fake import backend_from_request
from rvc_next.engine.audio_io.streams import AudioSession, DeviceOpenError, Endpoint, SessionConfig, engine_rate
from rvc_next.engine.audio_io.tone import chime
from rvc_next.engine.convert.params import VoiceParams
from rvc_next.engine.errors import MissingAssetError, ModelFormatError
from rvc_next.engine.runtime import is_oom
from rvc_next.engine.stream.latency import estimate_latency_ms
from rvc_next.engine.stream.params import StreamParams
from rvc_next.protocol import live as P
from rvc_next.protocol.jsonl import Emitter
from rvc_next.protocol.messages import OUT_OF_MEMORY
from rvc_next.workers.base import run_worker

logger = logging.getLogger("rvc_next.workers.live")

STATS_INTERVAL = 0.1


def _dataclass(cls: Any, data: dict[str, Any] | None, base: Any = None) -> Any:
    names = {f.name for f in fields(cls)}
    values = {k: v for k, v in (data or {}).items() if k in names}
    return replace(base, **values) if base is not None else cls(**values)


def session_config(devices: dict[str, Any] | None) -> SessionConfig:
    """A ``SessionConfig`` from the protocol's ``devices`` dict; missing input/output follow the system defaults."""
    devices = devices or {}
    inp = Endpoint.from_dict(devices.get("input")) or Endpoint()
    out = Endpoint.from_dict(devices.get("output")) or Endpoint()
    return SessionConfig(
        input=inp,
        output=out,
        monitor=Endpoint.from_dict(devices.get("monitor")),
        monitor_source=str(devices.get("monitor_source", "converted")),
        monitor_gain_db=float(devices.get("monitor_gain_db", 0.0)),
        output_gain_db=float(devices.get("output_gain_db", 0.0)),
    )


def _error(code: str, message: str, **detail: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "detail": detail}


class LiveWorker:
    def __init__(self, conn: Connection, request: P.WorkerRequest, backend: Any = None, runtime: Any = None) -> None:
        self.conn = conn
        self.request = request
        self.backend = backend if backend is not None else backend_from_request(request.backend, request.enable_asio, request.fake)
        self.runtime = runtime
        self._send_lock = threading.Lock()
        self._lock = threading.RLock()
        self.state = "stopped"
        self._last_error: dict[str, Any] | None = None
        self.engine: Any = None
        self.session: AudioSession | None = None
        self.aux: AudioSession | None = None
        """A session without the voice: the meter, or passthrough while stopped."""
        self.params = VoiceParams()
        self.stream = StreamParams()
        self.devices: dict[str, Any] = {}
        self.voice_path: str | None = None
        self.index_path: str | None = None
        self.passthrough = False
        self.meter = False
        self._closed = threading.Event()
        self._stats_thread = threading.Thread(target=self._stats_loop, name="rvc-live-stats", daemon=True)

    # -- messaging ------------------------------------------------------------------------------------

    def send(self, message: P.Message) -> None:
        with self._send_lock:
            try:
                self.conn.send_bytes(P.encode(message))
            except (OSError, EOFError, BrokenPipeError):
                self._closed.set()

    def log(self, level: str, message: str) -> None:
        getattr(logger, level, logger.info)(message)
        self.send(P.Log(level=level, message=message))

    def set_state(self, state: str, error: dict[str, Any] | None = None, voice_path: str | None = None) -> None:
        self.state = state
        self._last_error = error
        session = self.session
        running = state == "running" and session is not None
        self.send(
            P.State(
                state=state,
                error=error,
                topology=session.topology if running and session else None,
                sample_rate=session.sample_rate if running and session else None,
                meter=self.meter,
                passthrough=self.passthrough,
                cached=self.runtime.cached() if self.runtime is not None else [],
                voice_path=voice_path,
            )
        )

    # -- runtime ----------------------------------------------------------------------------------------

    def _runtime(self) -> Any:
        if self.runtime is None:
            from rvc_next.engine.runtime import Runtime

            r = self.request
            self.runtime = Runtime(Path(r.assets_dir), device=r.device, precision=r.precision, cuda_graph=r.cuda_graph)
        return self.runtime

    def _load_voice(self, voice_path: str, index_path: str | None) -> tuple[Any, Any]:
        rt = self._runtime()
        voice = rt.voice(voice_path)
        index = rt.index(index_path) if index_path else None
        rt.hubert()
        if voice.if_f0:
            rt.f0(self.params.f0_method)
        return voice, index

    # -- the main loop ------------------------------------------------------------------------------------

    def serve(self) -> None:
        self._stats_thread.start()
        self.set_state("stopped")
        try:
            while not self._closed.is_set():
                try:
                    data = self.conn.recv_bytes()
                except (EOFError, OSError):
                    break
                message = P.decode(data)
                if message is None:
                    self.log("warning", f"Ignored a malformed message: {data[:200]!r}")
                    continue
                if isinstance(message, P.Shutdown):
                    break
                try:
                    self.handle(message)
                except Exception as e:
                    self.log("error", traceback.format_exc())
                    self._stop_sessions()
                    self.set_state("error", self._failure(e))
        finally:
            self._closed.set()
            self._stop_sessions()
            self.state = "stopped"

    def handle(self, message: P.Message) -> None:
        with self._lock:
            if isinstance(message, P.Start):
                self.start(message.config)
            elif isinstance(message, P.Stop):
                self.stop()
            elif isinstance(message, P.UpdateVoice):
                self.update_voice(message.params)
            elif isinstance(message, P.UpdateStream):
                self.update_stream(message.stream)
            elif isinstance(message, P.SetVoice):
                self.set_voice(message.voice_path, message.index_path, message.speaker_id)
            elif isinstance(message, P.SetDevices):
                self.set_devices(message.devices)
            elif isinstance(message, P.Meter):
                self.set_meter(message.on, message.input)
            elif isinstance(message, P.TestTone):
                self.test_tone(message.device)
            elif isinstance(message, P.Passthrough):
                self.set_passthrough(message.on)
            elif isinstance(message, P.ReleaseMemory):
                self.release_memory()

    # -- commands ------------------------------------------------------------------------------------------

    def start(self, config: dict[str, Any]) -> None:
        self._stop_sessions()
        self.voice_path = config["voice_path"]
        self.index_path = config.get("index_path")
        self.params = _dataclass(VoiceParams, config.get("params"))
        self.stream = _dataclass(StreamParams, config.get("stream"))
        self.devices = dict(config.get("devices") or {})
        self.set_state("starting")
        self.set_state("loading")
        try:
            voice, index = self._load_voice(self.voice_path, self.index_path)
        except MissingAssetError as e:
            self.set_state("error", _error("asset_missing", str(e), assets=e.assets))
            return
        except ModelFormatError as e:
            self.set_state("error", _error("invalid_model", str(e)))
            return
        except FileNotFoundError as e:
            self.set_state("error", _error("not_found", str(e)))
            return
        cfg = session_config(self.devices)
        rate = engine_rate(cfg.output, cfg.input)
        try:
            self.engine = self._new_engine(voice, index, rate)
            self.set_state("prewarming")
            self.engine.prewarm()
        except Exception as e:
            if not is_oom(e):
                raise
            self.log("error", traceback.format_exc())
            self.set_state("error", self._failure(e))
            return
        self.engine.passthrough = False
        self._open_session(cfg)

    def _new_engine(self, voice: Any, index: Any, rate: int) -> Any:
        from rvc_next.engine.stream.engine import StreamEngine

        return StreamEngine(self._runtime(), voice, self.params, self.stream, rate, index)

    def _open_session(self, cfg: SessionConfig) -> bool:
        """Open the devices around the loaded engine and go to ``running``; report a refusal as an error."""
        assert self.engine is not None
        session = AudioSession(
            self.backend,
            cfg,
            self.engine.block_size,
            self.engine.sample_rate,
            processor=self.engine.process,
            on_device_lost=self._device_lost,
            on_error=self._processing_failed,
        )
        session.passthrough = self.passthrough
        try:
            session.start()
        except DeviceOpenError as e:
            self.session = None
            self.set_state("error", _error("device_unavailable", e.message, reason=e.reason, role=e.role, device_id=e.device_id))
            return False
        self.session = session
        self.set_state("running")
        return True

    def stop(self) -> None:
        if self.session is not None or self.state not in ("stopped",):
            self.set_state("stopping")
        self._stop_sessions()
        self.set_state("stopped")

    def release_memory(self) -> None:
        """Drop the engine and every loaded model (Free GPU memory); not while a session converts."""
        if self.session is not None or self.state in ("starting", "loading", "prewarming", "running", "reconnecting", "stopping"):
            self.log("info", "Not freeing GPU memory: Live is running")
            return
        self.engine = None
        if self.runtime is not None:
            self.runtime.release()
        self.set_state(self.state, self._last_error)

    def _failure(self, error: BaseException) -> dict[str, Any]:
        """The ``error`` dict for a failure; on out-of-memory, unload everything first (as ComfyUI does)."""
        if not is_oom(error):
            return _error("internal_error", str(error) or type(error).__name__)
        self.log("error", "Out of GPU memory: unloading every loaded model")
        self.engine = None
        if self.runtime is not None:
            self.runtime.release()
        return _error("compute_unavailable", f"Out of GPU memory: {error}", reason=OUT_OF_MEMORY)

    def _stop_sessions(self) -> None:
        for name in ("session", "aux"):
            s = getattr(self, name)
            setattr(self, name, None)
            if s is not None:
                s.stop()

    def update_voice(self, params: dict[str, Any]) -> None:
        new = _dataclass(VoiceParams, params, self.params)
        if new.f0_method != self.params.f0_method and self.engine is not None and self.engine.voice.if_f0:
            try:
                self._runtime().f0(new.f0_method)  # load here, not in the processing thread
            except MissingAssetError as e:
                # Keep converting with the method that is loaded, and say why.
                self.set_state(self.state, _error("asset_missing", str(e), assets=e.assets))
                new = replace(new, f0_method=self.params.f0_method)
        self.params = new
        if self.engine is not None:
            self.engine.update(new)

    def update_stream(self, stream: dict[str, Any]) -> None:
        new = _dataclass(StreamParams, stream, self.stream)
        old, self.stream = self.stream, new
        if self.engine is None:
            return
        if old.same_buffers(new) or self.session is None:
            self.engine.reconfigure(new)
            return
        cfg = self.session.config
        self.session.stop()
        self.session = None
        self.engine.reconfigure(new)
        self._open_session(cfg)

    def set_voice(self, voice_path: str, index_path: str | None, speaker_id: int | None) -> None:
        """Swap the voice of a running session between two blocks; HuBERT and F0 stay loaded.

        The new voice loads while the old one keeps converting. A voice that cannot be loaded is
        reported (``detail.reason == "voice_load"``) and the session goes on with the old voice.
        Either way the state is re-sent with ``voice_path``, the voice now converting, so the core
        follows what is actually loaded; a successful swap also clears an earlier failure.
        """
        if self.engine is None:
            self.voice_path, self.index_path = voice_path, index_path
            if speaker_id is not None:
                self.params = replace(self.params, speaker_id=int(speaker_id))
            return
        try:
            voice, index = self._load_voice(voice_path, index_path)
        except Exception as e:
            if is_oom(e):
                raise  # serve() reports it, unloading everything first
            code, extra = "internal_error", {}
            if isinstance(e, MissingAssetError):
                code, extra = "asset_missing", {"assets": e.assets}
            elif isinstance(e, ModelFormatError):
                code = "invalid_model"
            elif isinstance(e, FileNotFoundError):
                code = "not_found"
            self.log("error", f"Cannot load the voice {voice_path}: {e}")
            self.set_state(self.state, _error(code, str(e) or type(e).__name__, reason="voice_load", voice_path=voice_path, **extra), voice_path=self.voice_path)
            return
        self.voice_path, self.index_path = voice_path, index_path
        if speaker_id is not None:
            self.params = replace(self.params, speaker_id=int(speaker_id))
        self.engine.update(self.params)
        self.engine.set_voice(voice, index)
        self.set_state(self.state, voice_path=voice_path)

    def set_devices(self, devices: dict[str, Any]) -> None:
        self.devices = dict(devices or {})
        if self.engine is None or self.state not in ("running", "reconnecting", "error"):
            return
        cfg = session_config(self.devices)
        if self.session is not None:
            self.session.stop()
            self.session = None
        rate = engine_rate(cfg.output, cfg.input)
        if rate != self.engine.sample_rate:
            self.engine = self._new_engine(self.engine.voice, self.engine.index, rate)
        else:
            self.engine.reset()
        self._open_session(cfg)

    def set_meter(self, on: bool, input: dict[str, Any] | None) -> None:
        self.meter = on
        if self.session is not None:
            self.set_state(self.state)
            return
        if self.aux is not None:
            self.aux.stop()
            self.aux = None
        if on:
            ep = Endpoint.from_dict(input) or session_config(self.devices).input
            aux = AudioSession(self.backend, SessionConfig(input=ep), self._aux_block(ep), on_device_lost=self._device_lost)
            try:
                aux.start()
                self.aux = aux
            except DeviceOpenError as e:
                self.meter = False
                self.send(P.DeviceLost(direction="input", device_id=e.device_id, reason=e.reason, message=e.message))
        self.set_state(self.state)

    def _aux_block(self, ep: Endpoint | None) -> int:
        rate = engine_rate(ep)
        return rate // 20

    def set_passthrough(self, on: bool) -> None:
        self.passthrough = on
        if self.session is not None:
            self.session.passthrough = on
            self.set_state(self.state)
            return
        if self.aux is not None:
            self.aux.stop()
            self.aux = None
        if on:
            cfg = session_config(self.devices)
            if cfg.monitor is not None:
                cfg = replace(cfg, output=cfg.monitor, monitor=None)
            aux = AudioSession(self.backend, cfg, self._aux_block(cfg.output), processor=None, on_device_lost=self._device_lost)
            aux.passthrough = True
            try:
                aux.start()
                self.aux = aux
            except DeviceOpenError as e:
                self.passthrough = False
                self.send(P.DeviceLost(direction=e.role or "output", device_id=e.device_id, reason=e.reason, message=e.message))
        elif self.meter:
            self.set_meter(True, None)
            return
        self.set_state(self.state)

    def test_tone(self, device: dict[str, Any] | None) -> None:
        ep = Endpoint.from_dict(device)
        rate = int((ep.sample_rate if ep else None) or engine_rate(ep))

        def play() -> None:
            try:
                self.backend.play(ep, chime(rate), rate)
            except DeviceOpenError as e:
                self.send(P.DeviceLost(direction="output", device_id=e.device_id, reason=e.reason, message=e.message))

        threading.Thread(target=play, name="rvc-test-tone", daemon=True).start()

    # -- failures ----------------------------------------------------------------------------------------

    def _device_lost(self, role: str, device_id: str | None, reason: str, message: str) -> None:
        self.send(P.DeviceLost(direction=role, device_id=device_id, reason=reason, message=message))

        def settle() -> None:
            with self._lock:
                if self.session is not None and not self.session.running:
                    self.session.stop()
                    self.session = None
                    self.set_state("reconnecting")
                if self.aux is not None and not self.aux.running:
                    self.aux.stop()
                    self.aux = None
                    self.meter = self.passthrough = False
                    self.set_state(self.state)

        threading.Thread(target=settle, daemon=True).start()

    def _processing_failed(self, error: BaseException) -> None:
        self.log("error", "".join(traceback.format_exception(type(error), error, error.__traceback__)))

        def settle() -> None:
            with self._lock:
                if self.session is not None:
                    self.session.stop()
                    self.session = None
                self.set_state("error", self._failure(error))

        threading.Thread(target=settle, daemon=True).start()

    # -- stats ---------------------------------------------------------------------------------------------

    def stats(self) -> P.Stats | None:
        session = self.session or self.aux
        if session is None:
            return None
        p50, p95 = session.infer_percentiles()
        lin, lout = session.latency_ms
        block_ms = 1000.0 * session.block / session.sample_rate
        est = estimate_latency_ms(self.stream, session.topology == "split", lin, lout) if session is self.session else 0.0
        vram = None
        if self.runtime is not None:
            used = self.runtime.vram()
            vram = round(used[0], 1) if used else None
        return P.Stats(
            input_peak_db=round(session.input_levels.peak_db, 2),
            input_rms_db=round(session.input_levels.rms_db, 2),
            output_peak_db=round(session.output_levels.peak_db, 2),
            output_rms_db=round(session.output_levels.rms_db, 2),
            infer_ms_p50=round(p50, 2),
            infer_ms_p95=round(p95, 2),
            block_ms=round(block_ms, 2),
            est_latency_ms=round(est, 1),
            underruns=session.underruns,
            overruns=session.overruns,
            drift_ppm=None if session.drift_ppm is None else round(session.drift_ppm, 1),
            vram_mb=vram,
        )

    def _stats_loop(self) -> None:
        while not self._closed.wait(STATS_INTERVAL):
            try:
                stats = self.stats()
            except Exception:
                stats = None
            if stats is not None:
                self.send(stats)


def main(request: dict[str, Any], emitter: Emitter) -> None:
    del emitter  # everything goes over the connection; logs also reach stdout through run_worker
    req = _dataclass(P.WorkerRequest, request)
    conn = Client(tuple(req.address), authkey=bytes.fromhex(req.authkey))
    try:
        LiveWorker(conn, req).serve()
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(run_worker(main))
