"""Audio streams around the streaming engine.

``AudioSession`` opens the input, the output and an optional monitor on a ``Backend``, joins them
with rings, and runs a processing thread that feeds blocks to a processor (normally
``StreamEngine.process``). Device callbacks only move samples and count; nothing in them touches
torch or logs.

Topology: one duplex stream when input and output share a host API and the engine rate; otherwise
an input stream and an output stream joined by a ring with drift correction. A monitor is always
its own output stream, fed from a tee, with its own drift correction.

A block that takes longer than the block length leaves a backlog: input piles up meanwhile, and
once it is converted the output buffer holds more than its prefill, for good (duplex has no drift
correction, and split's would take minutes). So the processing thread skips input that is a whole
block or more behind, and trims the output buffer back to its prefill when its level after a write
stays above the normal range; the output then plays from the newest audio again.

The backend is pluggable: ``SounddeviceBackend`` drives PortAudio; ``fake.FakeBackend`` simulates
devices with independent clocks for tests.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from rvc_next.engine.audio.dsp import db_to_gain, peak_db, rms_db
from rvc_next.engine.audio_io.devices import accepted_channels, preferred_channels
from rvc_next.engine.audio_io.drift import DriftCorrector
from rvc_next.engine.audio_io.rings import Ring

DEFAULT_RATE = 48000
MIN_TIMINGS = 10
"""Blocks timed before the 95th percentile is reported; with fewer, one slow block would be all of it."""
TRIM_FADE = 64
BACKLOG_WRITES = 3
"""Writes in a row above the normal level before the output buffer is trimmed."""


PROBE_SECONDS = 0.4
"""How long ``probe_stream`` runs a stream to see whether it keeps going."""


class DeviceOpenError(Exception):
    """A device refused to open or was lost. ``reason`` is missing, busy, channels, format, permission
    or stopped (the stream started, then stopped by itself).

    ``role`` (input, output or monitor) names the device at fault; None when a duplex stream failed
    and neither of its devices could be singled out.
    """

    def __init__(self, message: str, reason: str = "busy", role: str | None = None, device_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.role = role
        self.device_id = device_id


def classify_portaudio_error(message: str) -> str:
    """Map a PortAudio error text to a ``DeviceError`` reason.

    ``channels`` (paInvalidChannelCount, -9998) is kept apart from ``format`` (a rate or sample
    format the device refuses): another rate cannot fix it.
    """
    low = message.lower()
    if "-9998" in low or "channel" in low:
        return "channels"
    if any(s in low for s in ("sample rate", "samplerate", "sample format", "format")):
        return "format"
    if any(s in low for s in ("invalid device", "device unavailable", "no such device", "not found", "-9996", "-9985")):
        return "missing" if "unavailable" not in low else "busy"
    if "permission" in low or "access denied" in low:
        return "permission"
    return "busy"


@dataclass
class StatusFlags:
    input_overflow: bool = False
    output_underflow: bool = False


@dataclass
class Endpoint:
    """A concrete device to open: an ``AudioDevice`` dict (None follows the system default) and how to use it."""

    device: dict[str, Any] | None = None
    channels: list[int] | None = None
    """1-based. Input: channels mixed to mono (default the first). Output: channels written (default the first two)."""
    sample_rate: int | None = None
    exclusive: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Endpoint | None:
        if data is None:
            return None
        return cls(device=data.get("device"), channels=data.get("channels"), sample_rate=data.get("sample_rate"), exclusive=bool(data.get("exclusive", False)))

    @property
    def index(self) -> int | None:
        return None if self.device is None else int(self.device["portaudio_index"])

    @property
    def device_id(self) -> str | None:
        return None if self.device is None else self.device.get("id")

    @property
    def host_api(self) -> str | None:
        return None if self.device is None else self.device.get("host_api")

    @property
    def loopback_source(self) -> str | None:
        """The loopback provider's id when this input records an output device through it (``capture.py``)."""
        return None if self.device is None else self.device.get("loopback_source")

    @property
    def max_channels(self) -> int:
        return 2 if self.device is None else int(self.device.get("channels", 2))

    def supports(self, rate: int) -> bool:
        if self.device is None:
            return True
        rates = self.device.get("supported_rates") or [self.device.get("default_sample_rate")]
        return rate in rates

    @property
    def channel_counts(self) -> list[int]:
        """The counts the device opens with, when it refuses some (``devices.probe_device``); empty restricts nothing."""
        return [] if self.device is None else [int(c) for c in self.device.get("channel_counts") or []]

    def _wanted(self, direction: str) -> int:
        if self.channels:
            return max(1, min(max(self.channels), self.max_channels))
        return preferred_channels(direction, self.max_channels)

    def open_channels(self, direction: str) -> int:
        """How many channels to open: enough for the highest selected one, raised to a count the device opens with."""
        return accepted_channels(self._wanted(direction), self.channel_counts)

    def selected(self, direction: str) -> list[int]:
        """0-based channel indexes to mix (input) or write (output); a wider stream leaves the rest out (silent)."""
        n = self.open_channels(direction)
        if self.channels:
            chosen = [c - 1 for c in self.channels if 1 <= c <= n]
            if chosen:
                return chosen
        return [0] if direction == "input" else list(range(min(self._wanted(direction), n)))


def engine_rate(output: Endpoint | None, input: Endpoint | None = None) -> int:
    """The rate the engine runs at: the output's chosen rate, else its default when supported, else 48 kHz."""
    ep = output or input
    if ep is None:
        return DEFAULT_RATE
    if ep.sample_rate:
        return int(ep.sample_rate)
    if ep.device is not None:
        default = int(ep.device.get("default_sample_rate") or DEFAULT_RATE)
        if ep.supports(default):
            return default
    return DEFAULT_RATE


def choose_topology(input: Endpoint, output: Endpoint | None, rate: int) -> str:
    """``duplex`` when one stream can carry both directions at ``rate``; else ``split``."""
    if output is None:
        return "input"
    if input.device is None and output.device is None:
        return "duplex"
    if input.device is None or output.device is None or input.loopback_source is not None:
        return "split"  # a loopback recorder is its own stream, whatever it records
    if input.host_api != output.host_api:
        return "split"
    if not (input.supports(rate) and output.supports(rate)):
        return "split"
    return "duplex"


class StreamHandle(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    @property
    def latency(self) -> tuple[float, float]:
        """(input, output) latency in seconds; 0 for a missing direction."""
        ...


InputCallback = Callable[[np.ndarray, int, StatusFlags], None]
OutputCallback = Callable[[np.ndarray, int, StatusFlags], None]
DuplexCallback = Callable[[np.ndarray, np.ndarray, int, StatusFlags], None]
FinishedCallback = Callable[[], None]


class Backend(Protocol):
    name: str

    def query(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]: ...
    def check(self, ep: Endpoint, direction: str, channels: int, rate: int) -> None: ...
    def open_duplex(self, inp: Endpoint, out: Endpoint, rate: int, block: int, in_ch: int, out_ch: int, callback: DuplexCallback, finished: FinishedCallback) -> StreamHandle: ...
    def open_input(self, ep: Endpoint, rate: int, block: int, channels: int, callback: InputCallback, finished: FinishedCallback) -> StreamHandle: ...
    def open_output(self, ep: Endpoint, rate: int, block: int, channels: int, callback: OutputCallback, finished: FinishedCallback) -> StreamHandle: ...
    def play(self, ep: Endpoint | None, audio: np.ndarray, rate: int) -> None: ...


# -- PortAudio ----------------------------------------------------------------------------------------


class _SdHandle:
    def __init__(self, stream: Any, duplex: bool, direction: str) -> None:
        self.stream = stream
        self.duplex = duplex
        self.direction = direction

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        try:
            self.stream.abort()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.stream.close()
        except Exception:
            pass

    @property
    def latency(self) -> tuple[float, float]:
        lat = self.stream.latency
        if self.duplex:
            return float(lat[0]), float(lat[1])
        return (float(lat), 0.0) if self.direction == "input" else (0.0, float(lat))


class SounddeviceBackend:
    """PortAudio through ``sounddevice``. ``enable_asio`` selects the ASIO-enabled DLL on Windows (0.5.1+).

    An input with a ``loopback_source`` records an output device through the loopback provider
    (``capture.py``) instead; ``list_loopback`` adds the provider's sources to ``query``, and
    ``loopback_error`` then says why there are none.
    """

    name = "sounddevice"

    def __init__(self, enable_asio: bool = False, list_loopback: bool = False) -> None:
        if enable_asio:
            os.environ["SD_ENABLE_ASIO"] = "1"
        import sounddevice as sd

        self.sd = sd
        self.list_loopback = list_loopback
        self.loopback_error: str | None = None
        self._provider: Any = None

    def query(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        from rvc_next.engine.audio_io.devices import enumerate_raw, loopback_target

        hostapis, devices, default_api = enumerate_raw()
        self.loopback_error = None
        native = any(int(d.get("max_input_channels", 0) or 0) > 0 and loopback_target(str(d.get("name", ""))) for d in devices)
        if self.list_loopback and not native:
            from rvc_next.engine.audio_io.capture import add_loopback_sources

            try:
                add_loopback_sources(hostapis, devices, self._loopback().sources())
            except DeviceOpenError as e:
                self.loopback_error = e.message
            except Exception as e:
                self.loopback_error = f"Output devices cannot be listed for recording: {e}"
        return hostapis, devices, default_api

    def _loopback(self) -> Any:
        """The loopback provider, made once; ``DeviceOpenError`` when this system has none."""
        if self._provider is None:
            from rvc_next.engine.audio_io.capture import loopback_provider

            provider, why = loopback_provider()
            if provider is None:
                raise DeviceOpenError(why or "Output devices cannot be recorded on this system", "missing", "input")
            self._provider = provider
        return self._provider

    def _extra(self, ep: Endpoint | None) -> Any:
        """WASAPI: exclusive when chosen; otherwise shared with Windows converting channels and rate, so
        a stream need not match the device's mix format (PortAudio alone converts only mono ↔ stereo)."""
        if ep is None or ep.host_api != "wasapi":
            return None
        return self.sd.WasapiSettings(exclusive=True) if ep.exclusive else self.sd.WasapiSettings(auto_convert=True)

    def _error(self, e: Exception, role: str, ep: Endpoint | None) -> DeviceOpenError:
        message = str(e)
        return DeviceOpenError(message, classify_portaudio_error(message), role, ep.device_id if ep else None)

    @staticmethod
    def _flags(status: Any) -> StatusFlags:
        return StatusFlags(bool(getattr(status, "input_overflow", False)), bool(getattr(status, "output_underflow", False)))

    def check(self, ep: Endpoint, direction: str, channels: int, rate: int) -> None:
        if ep.loopback_source is not None:
            from rvc_next.engine.audio_io.capture import check_capture

            try:
                provider = self._loopback()
            except DeviceOpenError as e:
                raise DeviceOpenError(e.message, e.reason, "input", ep.device_id) from e
            check_capture(provider, ep.loopback_source, ep.device_id)
            return
        try:
            if direction == "input":
                self.sd.check_input_settings(device=ep.index, channels=channels, samplerate=rate, dtype="float32", extra_settings=self._extra(ep))
            else:
                self.sd.check_output_settings(device=ep.index, channels=channels, samplerate=rate, dtype="float32", extra_settings=self._extra(ep))
        except Exception as e:
            raise self._error(e, direction, ep) from e

    def open_duplex(self, inp: Endpoint, out: Endpoint, rate: int, block: int, in_ch: int, out_ch: int, callback: DuplexCallback, finished: FinishedCallback) -> StreamHandle:
        flags = self._flags

        def cb(indata: Any, outdata: Any, frames: int, _time: Any, status: Any) -> None:
            callback(indata, outdata, frames, flags(status))

        try:
            stream = self.sd.Stream(
                device=(inp.index, out.index),
                samplerate=rate,
                blocksize=block,
                channels=(in_ch, out_ch),
                dtype="float32",
                latency="low",
                extra_settings=(self._extra(inp), self._extra(out)),
                callback=cb,
                finished_callback=finished,
            )
        except Exception as e:
            raise self._duplex_error(e, inp, out, rate, in_ch, out_ch) from e
        return _SdHandle(stream, True, "duplex")

    def _duplex_error(self, e: Exception, inp: Endpoint, out: Endpoint, rate: int, in_ch: int, out_ch: int) -> DeviceOpenError:
        """PortAudio's error for a duplex stream names neither side: check each alone to find the one
        that refuses. When both pass alone, the error stays the pair's (``role`` None)."""
        for ep, direction, channels in ((inp, "input", in_ch), (out, "output", out_ch)):
            try:
                self.check(ep, direction, channels, rate)
            except DeviceOpenError as side:
                return DeviceOpenError(
                    f"{side.message} (opening it with the {'output' if direction == 'input' else 'input'} as one stream: {e})", side.reason, direction, ep.device_id
                )
        message = str(e)
        return DeviceOpenError(message, classify_portaudio_error(message), None, None)

    def open_input(self, ep: Endpoint, rate: int, block: int, channels: int, callback: InputCallback, finished: FinishedCallback) -> StreamHandle:
        if ep.loopback_source is not None:
            from rvc_next.engine.audio_io.capture import open_capture

            try:
                provider = self._loopback()
            except DeviceOpenError as e:
                raise DeviceOpenError(e.message, e.reason, "input", ep.device_id) from e
            return open_capture(provider, ep.loopback_source, rate, channels, callback, finished, ep.device_id)
        flags = self._flags

        def cb(indata: Any, frames: int, _time: Any, status: Any) -> None:
            callback(indata, frames, flags(status))

        try:
            stream = self.sd.InputStream(
                device=ep.index,
                samplerate=rate,
                blocksize=block,
                channels=channels,
                dtype="float32",
                latency="low",
                extra_settings=self._extra(ep),
                callback=cb,
                finished_callback=finished,
            )
        except Exception as e:
            raise self._error(e, "input", ep) from e
        return _SdHandle(stream, False, "input")

    def open_output(self, ep: Endpoint, rate: int, block: int, channels: int, callback: OutputCallback, finished: FinishedCallback) -> StreamHandle:
        flags = self._flags

        def cb(outdata: Any, frames: int, _time: Any, status: Any) -> None:
            callback(outdata, frames, flags(status))

        try:
            stream = self.sd.OutputStream(
                device=ep.index,
                samplerate=rate,
                blocksize=block,
                channels=channels,
                dtype="float32",
                latency="low",
                extra_settings=self._extra(ep),
                callback=cb,
                finished_callback=finished,
            )
        except Exception as e:
            raise self._error(e, "output", ep) from e
        return _SdHandle(stream, False, "output")

    def play(self, ep: Endpoint | None, audio: np.ndarray, rate: int) -> None:
        channels = ep.open_channels("output") if ep else 2
        data = np.repeat(audio[:, None], channels, axis=1)
        try:
            self.sd.play(data, samplerate=rate, device=ep.index if ep else None, blocking=True)
        except Exception as e:
            raise self._error(e, "output", ep) from e


# -- the session ----------------------------------------------------------------------------------------


class LinearResampler:
    """A stateful linear-interpolation resampler for an input device that cannot run at the engine rate.

    The engine resamples to 16 kHz for features anyway, so linear interpolation here costs little.
    """

    def __init__(self, src: int, dst: int) -> None:
        self.step = src / dst
        self.pos = 0.0
        """Position of the next output sample, in input samples from the start of ``buf``."""
        self.buf = np.zeros(0, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        data = np.concatenate([self.buf, np.asarray(x, dtype=np.float32)])
        n = int(np.floor((len(data) - 1 - self.pos) / self.step)) + 1 if len(data) >= 2 else 0
        if n <= 0:
            self.buf = data
            return np.zeros(0, dtype=np.float32)
        t = self.pos + self.step * np.arange(n)
        out = np.interp(t, np.arange(len(data)), data).astype(np.float32)
        following = t[-1] + self.step
        keep = min(int(np.floor(following)), len(data) - 1)
        self.buf = data[keep:]
        self.pos = following - keep
        return out


@dataclass
class _Levels:
    peak_db: float = -120.0
    rms_db: float = -120.0

    def update(self, x: np.ndarray) -> None:
        self.peak_db = peak_db(x)
        self.rms_db = rms_db(x)


Processor = Callable[[np.ndarray], np.ndarray]
DeviceLostHandler = Callable[[str, "str | None", str, str], None]


@dataclass
class SessionConfig:
    input: Endpoint
    output: Endpoint | None = None
    monitor: Endpoint | None = None
    monitor_source: str = "converted"
    monitor_gain_db: float = 0.0
    output_gain_db: float = 0.0
    input_gain_db: float = 0.0
    """Applied to the input before it is metered and converted."""
    extra: dict[str, Any] = field(default_factory=dict)

    def same_endpoints(self, other: SessionConfig) -> bool:
        """Whether ``other`` opens the same devices, so only gains and the monitor's source changed."""
        return (self.input, self.output, self.monitor) == (other.input, other.output, other.monitor)


class AudioSession:
    """Open the devices, run the processing thread, count and measure.

    Without an output the session only meters the input. ``processor`` may be swapped or set to
    None (passthrough) between blocks.
    """

    def __init__(
        self,
        backend: Backend,
        config: SessionConfig,
        block: int,
        sample_rate: int | None = None,
        processor: Processor | None = None,
        on_device_lost: DeviceLostHandler | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        stall_seconds: float | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.block = int(block)
        self.sample_rate = int(sample_rate or engine_rate(config.output, config.input))
        self.processor = processor
        self.passthrough = False
        self.tap: Callable[[np.ndarray, np.ndarray], None] | None = None
        """Called with every processed block's input and output (at the session rate): the recorder."""
        self.on_device_lost = on_device_lost
        self.on_error = on_error
        self.topology = choose_topology(config.input, config.output, self.sample_rate)
        self.input_rate = self.sample_rate
        if self.topology == "split" and not config.input.supports(self.sample_rate) and config.input.device is not None:
            self.input_rate = int(config.input.device.get("default_sample_rate") or self.sample_rate)
        self._resampler = LinearResampler(self.input_rate, self.sample_rate) if self.input_rate != self.sample_rate else None
        self.in_block = self.block if self._resampler is None else max(1, int(round(self.block * self.input_rate / self.sample_rate)))
        self.in_ring = Ring(self.in_block * 8)
        self._pending = np.zeros(0, dtype=np.float32)
        prefill = self.block * (2 if self.topology == "split" else 1)
        self.out_ring = Ring(self.block * 12)
        self.out_reader = DriftCorrector(self.out_ring, self.sample_rate, self.block, prefill, enabled=self.topology == "split")
        self._prefill = prefill
        self.mon_ring = Ring(self.block * 12)
        self.mon_reader = DriftCorrector(self.mon_ring, self.sample_rate, self.block, self.block * 2)
        self.input_levels = _Levels()
        self.output_levels = _Levels()
        self.monitor_levels = _Levels()
        self.infer_ms: collections.deque[float] = collections.deque(maxlen=50)
        self._untimed = 0
        """Blocks still to process without timing them (see ``skip_timing``)."""
        # After a write the output buffer holds the prefill (duplex), or up to a block more (split,
        # where the two clocks' phase moves the reads); more than that for several writes is a backlog.
        self._backlog_limit = prefill + (3 * self.block) // 2 if self.topology == "split" else prefill + self.block // 2
        self._after_write: collections.deque[int] = collections.deque(maxlen=BACKLOG_WRITES)
        self.trimmed = 0
        """Output samples dropped to undo a backlog."""
        self.skipped = 0
        """Input samples skipped to catch up after a stall."""
        self.callback_underruns = 0
        self.callback_overruns = 0
        self.processed_blocks = 0
        self._handles: list[tuple[str, StreamHandle]] = []
        self._running = False
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_cb: dict[str, float] = {}
        self._lost = False
        self._stall = stall_seconds if stall_seconds is not None else max(0.5, 3 * self.block / self.sample_rate)
        in_sel = config.input.selected("input")
        self._in_sel = in_sel
        self._in_ch = config.input.open_channels("input")
        self._out_ch = config.output.open_channels("output") if config.output else 0
        self._out_sel = config.output.selected("output") if config.output else []
        self._mon_ch = config.monitor.open_channels("output") if config.monitor else 0
        self._mon_sel = config.monitor.selected("output") if config.monitor else []

    # -- callbacks: copy and count only ----------------------------------------------------------

    def _mix_in(self, indata: np.ndarray) -> np.ndarray:
        if indata.ndim == 1:
            return indata.astype(np.float32, copy=False)
        if len(self._in_sel) == 1:
            return np.ascontiguousarray(indata[:, self._in_sel[0]], dtype=np.float32)
        return indata[:, self._in_sel].mean(axis=1).astype(np.float32)

    @staticmethod
    def _write_out(outdata: np.ndarray, mono: np.ndarray, selected: list[int]) -> None:
        outdata.fill(0)
        for c in selected:
            outdata[: mono.shape[0], c] = mono

    def _input_cb(self, indata: np.ndarray, frames: int, status: StatusFlags) -> None:
        self._last_cb["input"] = time.monotonic()
        if status.input_overflow:
            self.callback_overruns += 1
        self.in_ring.write(self._mix_in(indata))

    def _output_cb(self, outdata: np.ndarray, frames: int, status: StatusFlags) -> None:
        self._last_cb["output"] = time.monotonic()
        if status.output_underflow:
            self.callback_underruns += 1
        self._write_out(outdata, self.out_reader.read(frames), self._out_sel)

    def _duplex_cb(self, indata: np.ndarray, outdata: np.ndarray, frames: int, status: StatusFlags) -> None:
        now = time.monotonic()
        self._last_cb["input"] = now
        self._last_cb["output"] = now
        if status.input_overflow:
            self.callback_overruns += 1
        if status.output_underflow:
            self.callback_underruns += 1
        self.in_ring.write(self._mix_in(indata))
        self._write_out(outdata, self.out_reader.read(frames), self._out_sel)

    def _monitor_cb(self, outdata: np.ndarray, frames: int, status: StatusFlags) -> None:
        self._last_cb["monitor"] = time.monotonic()
        if status.output_underflow:
            self.callback_underruns += 1
        self._write_out(outdata, self.mon_reader.read(frames), self._mon_sel)

    def _finished(self, role: str) -> FinishedCallback:
        def done() -> None:
            if self._running and not self._stopping:
                self._report_lost(role, "stopped", f"The {role} stream stopped by itself")

        return done

    # -- lifecycle -------------------------------------------------------------------------------------

    def start(self) -> None:
        """Open the streams and start processing. Raises ``DeviceOpenError`` (with ``role``) when a device refuses."""
        cfg = self.config
        self.out_ring.clear()
        self.out_ring.write(np.zeros(self._prefill, dtype=np.float32))
        self._running = True
        self._stopping = False
        role = "duplex" if self.topology == "duplex" else "input"  # the stream being opened or started: a failure names its device
        try:
            if self.topology == "duplex":
                assert cfg.output is not None
                h = self.backend.open_duplex(cfg.input, cfg.output, self.sample_rate, self.block, self._in_ch, self._out_ch, self._duplex_cb, self._finished("output"))
                self._handles.append(("duplex", h))
            else:
                h = self.backend.open_input(cfg.input, self.input_rate, self.in_block, self._in_ch, self._input_cb, self._finished("input"))
                self._handles.append(("input", h))
                if cfg.output is not None:
                    role = "output"
                    h = self.backend.open_output(cfg.output, self.sample_rate, self.block, self._out_ch, self._output_cb, self._finished("output"))
                    self._handles.append(("output", h))
            if cfg.monitor is not None and cfg.output is not None:
                role = "monitor"
                h = self.backend.open_output(cfg.monitor, self.sample_rate, self.block, self._mon_ch, self._monitor_cb, self._finished("monitor"))
                self._handles.append(("monitor", h))
            now = time.monotonic()
            for role, h in self._handles:
                for r in ("input", "output") if role == "duplex" else (role,):
                    self._last_cb[r] = now
                h.start()
        except DeviceOpenError as e:
            self._running = False
            self._close_streams()
            ep = self.endpoint(role)
            if ep is not None:
                # One stream, one device: its role (the backend calls a monitor an output).
                e.role, e.device_id = role, e.device_id or ep.device_id
            raise
        except Exception as e:
            self._running = False
            self._close_streams()
            # A stream that opened but would not start (an unanticipated host error, a device gone
            # between the two): the failing stream's role; a duplex stream's could be either device.
            ep = self.endpoint(role)
            raise DeviceOpenError(str(e), classify_portaudio_error(str(e)), None if role == "duplex" else role, ep.device_id if ep else None) from e
        self._thread = threading.Thread(target=self._loop, name="rvc-live-processing", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping = True
        self._running = False
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        self._thread = None
        self._close_streams()

    def _close_streams(self) -> None:
        handles, self._handles = self._handles, []
        for _, h in handles:
            h.stop()
            h.close()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def latency_ms(self) -> tuple[float, float]:
        lin = lout = 0.0
        for role, h in self._handles:
            a, b = h.latency
            if role in ("duplex", "input"):
                lin = max(lin, a * 1000)
            if role in ("duplex", "output"):
                lout = max(lout, b * 1000)
        return lin, lout

    # -- processing thread -----------------------------------------------------------------------------

    def endpoint(self, role: str | None) -> Endpoint | None:
        """The endpoint a role (input, output or monitor) opens; None for another role."""
        return {"input": self.config.input, "output": self.config.output, "monitor": self.config.monitor}.get(role or "")

    def _report_lost(self, role: str, reason: str, message: str) -> None:
        with self._lock:
            if self._lost:
                return
            self._lost = True
        self._running = False
        ep = self.endpoint(role)
        if self.on_device_lost is not None:
            self.on_device_lost(role, ep.device_id if ep else None, reason, message)

    def _watchdog(self) -> None:
        now = time.monotonic()
        for role, last in list(self._last_cb.items()):
            if now - last > self._stall:
                self._report_lost(role, "missing", f"The {role} device stopped delivering audio")
                return

    def _loop(self) -> None:
        wait = max(0.02, self.in_block / self.input_rate)
        try:
            while self._running:
                if not self.in_ring.wait_for(self.in_block, timeout=wait):
                    self._watchdog()
                    continue
                behind = self.in_ring.available // self.in_block - 1
                if behind > 0:
                    # A whole block or more is waiting behind this one: convert only the newest.
                    self.skipped += self.in_ring.discard(behind * self.in_block)
                x = self.in_ring.read(self.in_block)
                if self._resampler is not None:
                    self._pending = np.concatenate([self._pending, self._resampler(x)])
                    if self._pending.shape[0] < self.block:
                        continue
                    x, self._pending = self._pending[: self.block], self._pending[self.block :]
                self._process_block(x)
                self._watchdog()
        except Exception as e:  # the processor failed: report, never hang the audio
            self._running = False
            if self.on_error is not None and not self._stopping:
                self.on_error(e)
        finally:
            if self._lost:
                self._close_streams()

    def _process_block(self, x: np.ndarray) -> None:
        cfg = self.config
        if cfg.input_gain_db:
            x = x * db_to_gain(cfg.input_gain_db)
        self.input_levels.update(x)
        if cfg.output is None:
            self.processed_blocks += 1
            return
        processor = self.processor
        if processor is not None and not (self.passthrough and cfg.monitor is None):
            t0 = time.perf_counter()
            y = np.asarray(processor(x), dtype=np.float32).reshape(-1)
            if self._untimed > 0:
                self._untimed -= 1
            else:
                self.infer_ms.append((time.perf_counter() - t0) * 1000)
        else:
            y = x
        if y.shape[0] != self.block:
            y = np.resize(y, self.block) if y.shape[0] else np.zeros(self.block, dtype=np.float32)
        tap = self.tap
        if tap is not None:
            tap(x, y)
        out = y * db_to_gain(cfg.output_gain_db) if cfg.output_gain_db else y
        self.output_levels.update(out)
        self.out_ring.write(out)
        self._trim_backlog()
        if cfg.monitor is not None:
            if self.passthrough or cfg.monitor_source == "input":
                mon = x
            elif cfg.monitor_source == "both":
                mon = np.clip(x + y, -1.0, 1.0)
            else:
                mon = y
            if cfg.monitor_gain_db:
                mon = mon * db_to_gain(cfg.monitor_gain_db)
            self.monitor_levels.update(mon)
            self.mon_ring.write(mon.astype(np.float32, copy=False))
        self.processed_blocks += 1

    def _trim_backlog(self) -> None:
        self._after_write.append(self.out_ring.available)
        if len(self._after_write) < BACKLOG_WRITES or min(self._after_write) <= self._backlog_limit:
            return
        self.trimmed += self.out_ring.discard(self.out_ring.available - self._prefill, fade=TRIM_FADE)
        self._after_write.clear()
        self.out_reader.resettle()

    def skip_timing(self, blocks: int) -> None:
        """Leave the next ``blocks`` out of the inference timings: they do one-off work (a new voice's
        first run and CUDA Graph capture) that says nothing about the load."""
        self._untimed = max(self._untimed, int(blocks))

    # -- stats ---------------------------------------------------------------------------------------------

    @property
    def underruns(self) -> int:
        return self.callback_underruns + self.out_reader.underruns + (self.mon_reader.underruns if self.config.monitor else 0)

    @property
    def overruns(self) -> int:
        return self.callback_overruns + self.in_ring.overruns + self.out_ring.overruns

    @property
    def drift_ppm(self) -> float | None:
        return self.out_reader.drift_ppm if self.topology == "split" else None

    def infer_percentiles(self) -> tuple[float, float]:
        """Median and 95th percentile of the last 50 blocks' inference time. Until ``MIN_TIMINGS``
        blocks are in, both are the median: the 95th percentile of a few would be their slowest."""
        if not self.infer_ms:
            return 0.0, 0.0
        data = np.array(self.infer_ms)
        p50 = float(np.percentile(data, 50))
        return p50, float(np.percentile(data, 95)) if data.size >= MIN_TIMINGS else p50


def probe_stream(backend: Backend, ep: Endpoint, direction: str, rate: int, seconds: float = PROBE_SECONDS) -> None:
    """Open ``ep`` for ``seconds`` (an output plays silence) and raise ``DeviceOpenError`` when it
    will not open or start, or stops by itself meanwhile: some drivers accept the settings, then give
    up once the stream runs. A stream that is merely slow to deliver (Bluetooth) is not a failure."""
    stopped = threading.Event()
    channels = ep.open_channels(direction)
    block = max(1, rate // 100)
    try:
        if direction == "input":
            handle = backend.open_input(ep, rate, block, channels, lambda *_: None, stopped.set)
        else:
            handle = backend.open_output(ep, rate, block, channels, lambda outdata, *_: outdata.fill(0), stopped.set)
    except DeviceOpenError as e:
        raise DeviceOpenError(e.message, e.reason, direction, e.device_id or ep.device_id) from e
    try:
        handle.start()
        if stopped.wait(seconds):
            raise DeviceOpenError(f"the driver ended the {direction} stream within {seconds:g} s of starting it", "stopped", direction, ep.device_id)
    except DeviceOpenError:
        raise
    except Exception as e:
        raise DeviceOpenError(str(e), classify_portaudio_error(str(e)), direction, ep.device_id) from e
    finally:
        stopped.set()  # its own stop below is not a failure
        handle.stop()
        handle.close()
