"""The browser as an audio device: its microphone in, its speakers out, over the network.

The browser's audio clock drives the stream. Each frame of microphone samples that arrives
(``feed``) runs the duplex callback once, as a sound card's would, and the output samples the
callback wrote go straight back (``send``), as many as came in. So an ``AudioSession`` runs
unchanged on top: duplex topology, no drift between input and output, the processing thread and
its prefill as with any device. The browser buffers what it receives against network jitter.

``BROWSER_HOST_API`` marks the endpoints; ``browser_device`` makes the device dict they carry.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from rvc_next.engine.audio_io.streams import DeviceOpenError, DuplexCallback, Endpoint, FinishedCallback, InputCallback, OutputCallback, StatusFlags

BROWSER_HOST_API = "browser"
BROWSER_INDEX = -1000
STALL_SECONDS = 3.0
"""How long a session waits for the browser's next frame before it counts the browser as gone:
longer than for a sound card, since frames cross a network and a busy tab can pause."""


def browser_device(sample_rate: int) -> dict[str, Any]:
    """The device dict of a browser endpoint at ``sample_rate`` (its AudioContext's)."""
    return {
        "id": "browser",
        "name": "Browser",
        "host_api": BROWSER_HOST_API,
        "portaudio_index": BROWSER_INDEX,
        "channels": 1,
        "default_sample_rate": int(sample_rate),
        "supported_rates": [int(sample_rate)],
    }


def is_browser(ep: Endpoint | None) -> bool:
    return ep is not None and ep.host_api == BROWSER_HOST_API


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class BrowserStream:
    """One duplex stream; ``feed`` runs its callback."""

    def __init__(self, callback: DuplexCallback, out_channels: int, send: Callable[[bytes], None], finished: FinishedCallback) -> None:
        self.callback = callback
        self.out_channels = max(1, out_channels)
        self.send = send
        self.finished = finished
        self.running = False
        self._lock = threading.Lock()

    @property
    def latency(self) -> tuple[float, float]:
        return 0.0, 0.0

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def close(self) -> None:
        self.running = False

    def feed(self, x: np.ndarray) -> None:
        with self._lock:
            if not self.running or x.size == 0:
                return
            out = np.zeros((x.shape[0], self.out_channels), dtype=np.float32)
            self.callback(x.reshape(-1, 1), out, x.shape[0], StatusFlags())
            self.send(float_to_pcm16(out[:, 0]))


class BrowserBackend:
    """The ``Backend`` for browser endpoints; ``send`` carries converted PCM back to the browser."""

    name = "browser"

    def __init__(self, send: Callable[[bytes], None]) -> None:
        self.send = send
        self.stream: BrowserStream | None = None
        self.received = 0
        """Frames fed in, for tests."""

    def feed(self, pcm: bytes) -> None:
        """Microphone samples from the browser (16-bit PCM). Dropped while no stream runs."""
        self.received += 1
        stream = self.stream
        if stream is not None:
            stream.feed(pcm16_to_float(pcm))

    def query(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        return [], [], 0

    def check(self, ep: Endpoint, direction: str, channels: int, rate: int) -> None:
        return None

    def open_duplex(self, inp: Endpoint, out: Endpoint, rate: int, block: int, in_ch: int, out_ch: int, callback: DuplexCallback, finished: FinishedCallback) -> BrowserStream:
        self.stream = BrowserStream(callback, out_ch, self.send, finished)
        return self.stream

    def open_input(self, ep: Endpoint, rate: int, block: int, channels: int, callback: InputCallback, finished: FinishedCallback) -> BrowserStream:
        raise DeviceOpenError("The browser's microphone and speakers go together", "format", "input", ep.device_id)

    def open_output(self, ep: Endpoint, rate: int, block: int, channels: int, callback: OutputCallback, finished: FinishedCallback) -> BrowserStream:
        raise DeviceOpenError("The browser's microphone and speakers go together; it has no monitor", "format", "output", ep.device_id)

    def play(self, ep: Endpoint | None, audio: np.ndarray, rate: int) -> None:
        raise DeviceOpenError("The test tone plays on the server's devices", "format", "output", None)


class NoDevices:
    """The server-device backend when PortAudio cannot load (a headless server without it): every
    use says why, while browser audio still works."""

    name = "none"

    def __init__(self, why: str) -> None:
        self.why = why

    def _refuse(self, role: str | None = None) -> DeviceOpenError:
        return DeviceOpenError(f"The server's audio devices are unavailable ({self.why}); use the browser's", "missing", role, None)

    def query(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        return [], [], 0

    def check(self, ep: Endpoint, direction: str, channels: int, rate: int) -> None:
        raise self._refuse(direction)

    def open_duplex(self, inp: Endpoint, out: Endpoint, rate: int, block: int, in_ch: int, out_ch: int, callback: DuplexCallback, finished: FinishedCallback) -> BrowserStream:
        raise self._refuse("output")

    def open_input(self, ep: Endpoint, rate: int, block: int, channels: int, callback: InputCallback, finished: FinishedCallback) -> BrowserStream:
        raise self._refuse("input")

    def open_output(self, ep: Endpoint, rate: int, block: int, channels: int, callback: OutputCallback, finished: FinishedCallback) -> BrowserStream:
        raise self._refuse("output")

    def play(self, ep: Endpoint | None, audio: np.ndarray, rate: int) -> None:
        raise self._refuse("output")
