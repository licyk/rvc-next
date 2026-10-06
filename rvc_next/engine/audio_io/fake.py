"""A simulated PortAudio for tests: devices with independent clocks, recorded outputs, device loss.

Each fake stream runs a thread that calls its callback once per block at the device's own rate,
``1 + ppm·1e-6`` times nominal (``time_scale`` speeds everything up). Inputs come from a signal
function; outputs are recorded per device. ``lose(index)`` stalls a device's streams, as an
unplugged USB device does; ``refuse`` makes opening fail with a reason. ``loopback`` wires an
output back into every input, as a cable would: input sample ``n`` is output sample
``n − block − delay`` (in each stream's own count; a block played during one callback is captured
by the next, as on a real device), times ``gain``, plus ``noise``. A device's
``<direction>_channel_counts``, when given, are the only counts it opens with (a WDM-KS pin, say);
``probe_device`` over ``check`` finds them again. ``list_loopback`` lists every output as a
loopback source too, as the real provider does (``capture.add_loopback_sources``); opening one
records what that output has played, sample for sample.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from rvc_next.engine.audio_io.capture import LoopbackSource, add_loopback_sources
from rvc_next.engine.audio_io.streams import DeviceOpenError, DuplexCallback, Endpoint, FinishedCallback, InputCallback, OutputCallback, StatusFlags

Source = Callable[[int, int, int], np.ndarray]
"""(sample rate, first sample index, frames) → mono float32."""


def harmonic_source(freq: float = 220.0, amp: float = 0.3) -> Source:
    def source(rate: int, start: int, n: int) -> np.ndarray:
        t = (start + np.arange(n)) / rate
        x = sum(np.sin(2 * np.pi * k * freq * t) / k for k in range(1, 5))
        return (amp * x / 2.1).astype(np.float32)

    return source


def default_devices() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A small machine: one host API, a microphone, speakers, headphones and a virtual cable."""

    def dev(index: int, name: str, i: int, o: int, rates: list[int] | None = None) -> dict[str, Any]:
        return {
            "name": name,
            "index": index,
            "hostapi": 0,
            "max_input_channels": i,
            "max_output_channels": o,
            "default_samplerate": 48000.0,
            "default_low_input_latency": 0.01,
            "default_high_input_latency": 0.05,
            "default_low_output_latency": 0.01,
            "default_high_output_latency": 0.05,
            "supported_rates": rates or [44100, 48000],
        }

    devices = [dev(0, "Fake Microphone", 2, 0), dev(1, "Fake Speakers", 0, 2), dev(2, "Fake Headphones", 0, 2), dev(3, "CABLE Input (VB-Audio Virtual Cable)", 0, 8)]
    hostapis = [{"name": "ALSA", "devices": [0, 1, 2, 3], "default_input_device": 0, "default_output_device": 1}]
    return hostapis, devices


class FakeStream:
    def __init__(
        self,
        backend: FakeBackend,
        kind: str,
        indexes: tuple[int | None, int | None],
        rate: int,
        block: int,
        channels: tuple[int, int],
        callback: Any,
        finished: FinishedCallback,
        source: Source | None = None,
    ) -> None:
        self.backend = backend
        self.source = source
        self.kind = kind
        self.in_index, self.out_index = indexes
        self.rate = rate
        self.block = block
        self.in_ch, self.out_ch = channels
        self.callback = callback
        self.finished = finished
        self._thread: threading.Thread | None = None
        self._running = False
        self.samples = 0
        self.calls = 0

    @property
    def latency(self) -> tuple[float, float]:
        return self.backend.latency

    @property
    def clock_index(self) -> int:
        idx = self.in_index if self.in_index is not None else self.out_index
        return int(idx or 0)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"fake-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def close(self) -> None:
        self.stop()
        self.backend._streams.discard(self)

    def _run(self) -> None:
        ppm = self.backend.ppm.get(self.clock_index, 0.0)
        period = self.block / (self.rate * (1 + ppm * 1e-6)) / self.backend.time_scale
        next_t = time.perf_counter()
        while self._running:
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            if not self._running:
                break
            if self.in_index in self.backend.lost or self.out_index in self.backend.lost:
                continue
            flags = StatusFlags()
            if self.kind in ("input", "duplex"):
                mono = (self.source or self.backend.source)(self.rate, self.samples, self.block)
                indata = np.repeat(mono[:, None], self.in_ch, axis=1)
            outdata = np.zeros((self.block, self.out_ch), dtype=np.float32) if self.kind in ("output", "duplex") else None
            try:
                if self.kind == "input":
                    self.callback(indata, self.block, flags)
                elif self.kind == "output":
                    self.callback(outdata, self.block, flags)
                else:
                    self.callback(indata, outdata, self.block, flags)
            except Exception:
                self._running = False
                self.finished()
                raise
            self.samples += self.block
            self.calls += 1
            if outdata is not None and self.out_index is not None:
                self.backend.record(self.out_index, outdata[:, 0].copy())


class FakeBackend:
    name = "fake"

    def __init__(
        self,
        hostapis: list[dict[str, Any]] | None = None,
        devices: list[dict[str, Any]] | None = None,
        ppm: dict[int, float] | None = None,
        source: Source | None = None,
        time_scale: float = 1.0,
        latency: tuple[float, float] = (0.01, 0.01),
        refuse: dict[int, str] | None = None,
        loopback: dict[str, Any] | None = None,
        list_loopback: bool = False,
    ) -> None:
        if hostapis is None or devices is None:
            hostapis, devices = default_devices()
        self.hostapis = hostapis
        self.devices = devices
        self.ppm = {int(k): float(v) for k, v in (ppm or {}).items()}
        self.source = source or harmonic_source()
        self.time_scale = time_scale
        self.latency = latency
        self.refuse = {int(k): v for k, v in (refuse or {}).items()}
        self.list_loopback = list_loopback
        self.loopback_error: str | None = None
        if loopback is not None:
            self.source = self._loopback_source(**loopback)
        self.lost: set[int] = set()
        self.recorded: dict[int, list[np.ndarray]] = {}
        self.played: list[tuple[int | None, int]] = []
        self._streams: set[FakeStream] = set()
        self._lock = threading.Lock()

    def _loopback_source(self, delay_ms: float, gain: float = 1.0, noise: float = 0.0, output: int | None = None) -> Source:
        """Input from ``output`` (default: the default output), one input block plus ``delay_ms`` later."""
        rng = np.random.default_rng(0)
        source_index = int(self.hostapis[0]["default_output_device"]) if output is None else int(output)

        def source(rate: int, start: int, n: int) -> np.ndarray:
            played = self.output(source_index)
            first = start - n - int(round(delay_ms * rate / 1000))
            out = np.zeros(n, dtype=np.float32)
            lo, hi = max(first, 0), min(first + n, played.shape[0])
            if hi > lo:
                out[lo - first : hi - first] = played[lo:hi] * gain
            if noise:
                out += (noise * rng.standard_normal(n)).astype(np.float32)
            return out

        return source

    # -- simulation controls --------------------------------------------------------------------

    def lose(self, index: int) -> None:
        self.lost.add(index)

    def restore(self, index: int) -> None:
        self.lost.discard(index)

    def record(self, index: int, data: np.ndarray) -> None:
        with self._lock:
            self.recorded.setdefault(index, []).append(data)

    def output(self, index: int) -> np.ndarray:
        with self._lock:
            chunks = list(self.recorded.get(index, []))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    @property
    def streams(self) -> list[FakeStream]:
        return list(self._streams)

    # -- Backend ---------------------------------------------------------------------------------

    def query(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        hostapis, devices = [dict(h) for h in self.hostapis], [d for d in self.devices if d["index"] not in self.lost]
        if self.list_loopback:
            outputs = list({str(d["name"]): d for d in reversed(devices) if int(d.get("max_output_channels", 0)) > 0}.values())[::-1]
            add_loopback_sources(hostapis, devices, [LoopbackSource(str(d["index"]), str(d["name"]), int(d["max_output_channels"])) for d in outputs])
        return hostapis, devices, 0

    def _recorded_output(self, ep: Endpoint) -> dict[str, Any]:
        """The output device a loopback endpoint records (its source id is the output's index)."""
        return self._device(Endpoint(device={"portaudio_index": int(ep.loopback_source or -1), "id": ep.device_id}), "output")

    def _capture_source(self, index: int) -> Source:
        def source(rate: int, start: int, n: int) -> np.ndarray:
            played = self.output(index)
            out = np.zeros(n, dtype=np.float32)
            hi = min(start + n, played.shape[0])
            if hi > start:
                out[: hi - start] = played[start:hi]
            return out

        return source

    def _device(self, ep: Endpoint | None, direction: str) -> dict[str, Any]:
        index = ep.index if ep is not None else None
        if index is None:
            key = "default_input_device" if direction == "input" else "default_output_device"
            index = int(self.hostapis[0][key])
        for d in self.devices:
            if d["index"] == index:
                if index in self.lost:
                    raise DeviceOpenError(f"{d['name']} is not connected", "missing", direction, ep.device_id if ep else None)
                if index in self.refuse:
                    raise DeviceOpenError(f"{d['name']} refused: {self.refuse[index]}", self.refuse[index], direction, ep.device_id if ep else None)
                return d
        raise DeviceOpenError(f"No device {index}", "missing", direction, ep.device_id if ep else None)

    def check(self, ep: Endpoint, direction: str, channels: int, rate: int) -> None:
        if ep.loopback_source is not None:
            self._recorded_output(ep)  # any rate and channel count: the provider converts
            return
        d = self._device(ep, direction)
        max_ch = int(d["max_input_channels" if direction == "input" else "max_output_channels"])
        counts = d.get(f"{direction}_channel_counts") or range(1, max_ch + 1)
        if channels > max_ch or max_ch == 0 or channels not in counts:
            raise DeviceOpenError(f"{d['name']}: Invalid number of channels ({channels}) [PaErrorCode -9998]", "channels", direction, ep.device_id)
        if rate not in d.get("supported_rates", [int(d["default_samplerate"])]):
            raise DeviceOpenError(f"{d['name']} does not support {rate} Hz", "format", direction, ep.device_id)

    def _open(
        self, kind: str, inp: Endpoint | None, out: Endpoint | None, rate: int, block: int, channels: tuple[int, int], callback: Any, finished: FinishedCallback
    ) -> FakeStream:
        in_dev = self._device(inp, "input") if kind in ("input", "duplex") else None
        out_dev = self._device(out, "output") if kind in ("output", "duplex") else None
        for d, direction, ch in ((in_dev, "input", channels[0]), (out_dev, "output", channels[1])):
            if d is not None:
                self.check(Endpoint(device={"portaudio_index": d["index"], "id": None}), direction, ch, rate)
        stream = FakeStream(self, kind, (in_dev["index"] if in_dev else None, out_dev["index"] if out_dev else None), rate, block, channels, callback, finished)
        self._streams.add(stream)
        return stream

    def open_duplex(self, inp: Endpoint, out: Endpoint, rate: int, block: int, in_ch: int, out_ch: int, callback: DuplexCallback, finished: FinishedCallback) -> FakeStream:
        return self._open("duplex", inp, out, rate, block, (in_ch, out_ch), callback, finished)

    def open_input(self, ep: Endpoint, rate: int, block: int, channels: int, callback: InputCallback, finished: FinishedCallback) -> FakeStream:
        if ep.loopback_source is not None:
            index = int(self._recorded_output(ep)["index"])
            stream = FakeStream(self, "input", (index, None), rate, block, (channels, 0), callback, finished, source=self._capture_source(index))
            self._streams.add(stream)
            return stream
        return self._open("input", ep, None, rate, block, (channels, 0), callback, finished)

    def open_output(self, ep: Endpoint, rate: int, block: int, channels: int, callback: OutputCallback, finished: FinishedCallback) -> FakeStream:
        return self._open("output", None, ep, rate, block, (0, channels), callback, finished)

    def play(self, ep: Endpoint | None, audio: np.ndarray, rate: int) -> None:
        d = self._device(ep, "output")
        self.record(int(d["index"]), audio.astype(np.float32))
        self.played.append((int(d["index"]), int(audio.shape[0])))


def backend_from_request(name: str, enable_asio: bool = False, fake: dict[str, Any] | None = None, list_loopback: bool = False) -> Any:
    """``sounddevice`` (default) or the fake backend configured from a request's ``fake`` dict; ``list_loopback`` lists output devices as loopback inputs."""
    if name == "fake":
        cfg = dict(fake or {})
        return FakeBackend(
            hostapis=cfg.get("hostapis"),
            devices=cfg.get("devices"),
            ppm=cfg.get("ppm"),
            time_scale=float(cfg.get("time_scale", 1.0)),
            refuse=cfg.get("refuse"),
            loopback=cfg.get("loopback"),
            list_loopback=list_loopback,
        )
    from rvc_next.engine.audio_io.streams import SounddeviceBackend

    return SounddeviceBackend(enable_asio=enable_asio, list_loopback=list_loopback)
