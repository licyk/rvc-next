"""Recording an output device as an input ("loopback"): what it plays becomes Live's input.

PortAudio cannot record a render endpoint: an output device has no input channels, so opening it
as an input fails with ``Invalid number of channels [PaErrorCode -9998]``. Each system offers the
capture its own way:

- Windows: WASAPI loopback (``AUDCLNT_STREAMFLAGS_LOOPBACK`` on the render endpoint). PortAudio
  gained it in 2022 (devices named ``<output> [Loopback]``), after the 19.7.0 release whose DLL
  sounddevice bundles, so SoundCard's WASAPI binding records it instead.
- Linux: every PulseAudio/PipeWire sink has a monitor source (``Monitor of <sink>``). PortAudio's
  ALSA host API cannot name one; SoundCard's PulseAudio binding can.
- macOS: Core Audio offers no loopback below its process taps (macOS 14.2, behind a permission
  prompt), which PortAudio does not use; a loopback driver such as BlackHole is an ordinary input.

PortAudio devices that already are loopbacks (a newer PortAudio's ``[Loopback]``, a PulseAudio
host API's ``Monitor of``) are used as they are, and the provider is asked only when there are
none. Provider sources join the raw device list under a ``Loopback`` host API with
``loopback_source`` (the provider's id) and ``loopback_of`` (the output's name);
``SounddeviceBackend.open_input`` opens them through ``open_capture``. ``soundcard`` is imported
only inside functions.
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rvc_next.engine.audio_io.devices import COMMON_RATES, loopback_target, platform_key
from rvc_next.engine.audio_io.streams import DeviceOpenError, FinishedCallback, InputCallback, StatusFlags

LOOPBACK_HOST_API = "Loopback"
OPEN_TIMEOUT = 5.0
"""Seconds to wait for a recorder to open before the open counts as failed."""
CLOSE_TIMEOUT = 2.0
CAPTURE_LATENCY = (0.01, 0.03)
"""What a loopback source reports as its low and high latency: one 10 ms engine period, three."""


@dataclass(frozen=True)
class LoopbackSource:
    id: str
    """The provider's id: a Windows endpoint id, a PulseAudio source name."""
    name: str
    """The name of the output it records, as the system shows it."""
    channels: int


class LoopbackProvider(Protocol):
    def sources(self) -> list[LoopbackSource]: ...
    def recorder(self, source_id: str, rate: int, channels: int) -> AbstractContextManager[Any]:
        """A context manager whose value's ``record(None)`` returns what has arrived, frames × channels float32."""
        ...


class SoundcardLoopback:
    """Loopback through SoundCard: WASAPI loopback on Windows, monitor sources on PulseAudio/PipeWire.

    Its recorders open in shared mode with conversion, so any rate and channel count opens, and on
    Windows they return silence while nothing plays (WASAPI delivers no packets then).
    """

    def __init__(self) -> None:
        # By name: soundcard is not installed on macOS, where an import statement fails type checking
        # (and an ignore comment would be unused everywhere else).
        self.sc: Any = importlib.import_module("soundcard")

    def sources(self) -> list[LoopbackSource]:
        out = []
        for mic in self.sc.all_microphones(include_loopback=True):
            if mic.isloopback:
                name = str(mic.name)
                out.append(LoopbackSource(str(mic.id), loopback_target(name) or name, max(1, int(mic.channels))))
        return out

    def recorder(self, source_id: str, rate: int, channels: int) -> AbstractContextManager[Any]:
        mic = self.sc.get_microphone(source_id, include_loopback=True)
        return mic.recorder(samplerate=rate, channels=channels, blocksize=max(rate // 100, 1))


def loopback_provider(platform: str | None = None) -> tuple[LoopbackProvider | None, str | None]:
    """This system's provider, or None and why (in English, shown under the device list)."""
    if platform_key(platform) == "darwin":
        return None, "macOS cannot record an output device: install a loopback driver such as BlackHole and choose its input"
    try:
        return SoundcardLoopback(), None
    except Exception as e:  # not installed, no PulseAudio server, COM refused
        detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        return None, f"Output devices cannot be recorded on this system ({detail})"


def add_loopback_sources(hostapis: list[dict[str, Any]], devices: list[dict[str, Any]], sources: list[LoopbackSource], rates: tuple[int, ...] = COMMON_RATES) -> None:
    """Append ``sources`` to a raw enumeration as input devices of a ``Loopback`` host API.

    Their indexes are negative (no PortAudio device has them); any rate opens, as the provider converts.
    """
    if not sources:
        return
    api = len(hostapis)
    first = -1 - sum(1 for d in devices if int(d.get("index", 0)) < 0)
    entries = [
        {
            "name": s.name,
            "index": first - i,
            "hostapi": api,
            "max_input_channels": s.channels,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
            "default_low_input_latency": CAPTURE_LATENCY[0],
            "default_high_input_latency": CAPTURE_LATENCY[1],
            "supported_rates": sorted(rates),
            "loopback_source": s.id,
            "loopback_of": s.name,
        }
        for i, s in enumerate(sources)
    ]
    hostapis.append({"name": LOOPBACK_HOST_API, "devices": [e["index"] for e in entries], "default_input_device": -1, "default_output_device": -1})
    devices.extend(entries)


def _com_init() -> Callable[[], None]:
    """Join the COM multithreaded apartment on this thread (Windows); returns the matching cleanup."""
    if sys.platform != "win32":
        return lambda: None
    import ctypes

    ole32 = ctypes.windll.ole32
    if ole32.CoInitializeEx(None, 0) < 0:  # COINIT_MULTITHREADED; S_FALSE (already joined) is fine
        return lambda: None
    return ole32.CoUninitialize


def _reason(error: BaseException) -> str:
    text = str(error).lower()
    return "missing" if isinstance(error, (IndexError, KeyError)) or "no soundcard" in text or "not found" in text else "busy"


class CaptureStream:
    """A loopback recorder read on its own thread, calling an input callback as a PortAudio input stream would.

    The recorder opens, is read and closes on that thread (COM objects stay in its apartment).
    ``open`` waits until it is open, so a refusal surfaces as ``DeviceOpenError`` like a PortAudio
    open; callbacks begin at ``start``. Data that arrives before is dropped.
    """

    def __init__(self, open_recorder: Callable[[], AbstractContextManager[Any]], callback: InputCallback, finished: FinishedCallback, device_id: str | None = None) -> None:
        self._open_recorder = open_recorder
        self._callback = callback
        self._finished = finished
        self._device_id = device_id
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._started = False
        self._closed = False
        self._thread: threading.Thread | None = None

    def open(self, timeout: float = OPEN_TIMEOUT) -> CaptureStream:
        self._thread = threading.Thread(target=self._run, name="rvc-loopback-capture", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self._closed = True
            raise DeviceOpenError("The output device did not start recording", "busy", "input", self._device_id)
        if self._error is not None:
            raise DeviceOpenError(f"Cannot record the output device: {self._error}", _reason(self._error), "input", self._device_id) from self._error
        return self

    def _run(self) -> None:
        cleanup = _com_init()
        try:
            with self._open_recorder() as recorder:
                self._ready.set()
                while not self._closed:
                    data = np.asarray(recorder.record(None), dtype=np.float32)
                    if data.ndim == 1:
                        data = data[:, None]
                    if self._started and not self._closed and data.shape[0]:
                        self._callback(np.ascontiguousarray(data), int(data.shape[0]), StatusFlags())
        except Exception as e:
            self._error = e
        finally:
            ready = self._ready.is_set()
            self._ready.set()
            cleanup()
            # A recorder that dies while running (the device was unplugged, the server went away) is lost.
            if ready and self._started and not self._closed:
                self._finished()

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False
        self._closed = True
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=CLOSE_TIMEOUT)

    def close(self) -> None:
        self.stop()

    @property
    def latency(self) -> tuple[float, float]:
        return (CAPTURE_LATENCY[0], 0.0)


def open_capture(
    provider: LoopbackProvider, source_id: str, rate: int, channels: int, callback: InputCallback, finished: FinishedCallback, device_id: str | None = None
) -> CaptureStream:
    """Open a loopback source at ``rate`` with ``channels`` (the provider converts); raises ``DeviceOpenError``."""
    return CaptureStream(lambda: provider.recorder(source_id, rate, channels), callback, finished, device_id).open()


def check_capture(provider: LoopbackProvider, source_id: str, device_id: str | None = None) -> None:
    """Raise ``DeviceOpenError`` (missing) unless the provider still lists ``source_id``."""
    try:
        ids = {s.id for s in provider.sources()}
    except Exception as e:
        raise DeviceOpenError(f"Cannot list the output devices to record: {e}", "busy", "input", device_id) from e
    if source_id not in ids:
        raise DeviceOpenError("The output device to record is not connected", "missing", "input", device_id)
