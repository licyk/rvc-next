"""Recording an output device as an input: the capture stream, the provider, the backend's routing and a fake session."""

import sys
import threading
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from rvc_next.engine.audio_io import capture
from rvc_next.engine.audio_io.capture import CaptureStream, LoopbackSource, add_loopback_sources, check_capture, loopback_provider, open_capture
from rvc_next.engine.audio_io.devices import group_devices
from rvc_next.engine.audio_io.fake import FakeBackend, default_devices
from rvc_next.engine.audio_io.streams import AudioSession, DeviceOpenError, Endpoint, SessionConfig, SounddeviceBackend, choose_topology


def wait(predicate: Any, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class Recorder:
    """A SoundCard-like recorder: ``record(None)`` returns a 10 ms chunk of a ramp, or raises once ``fail`` is set."""

    def __init__(self, channels: int = 2, rate: int = 48000) -> None:
        self.channels, self.chunk = channels, rate // 100
        self.fail: Exception | None = None
        self.reads = 0
        self.closed = threading.Event()

    def record(self, numframes: Any = None) -> np.ndarray:
        time.sleep(0.002)
        if self.fail is not None:
            raise self.fail
        self.reads += 1
        return np.full((self.chunk, self.channels), 0.25, dtype=np.float32)


class Provider:
    def __init__(self, recorder: Recorder | None = None, refuse: Exception | None = None) -> None:
        self.recorder_ = recorder or Recorder()
        self.refuse = refuse
        self.opened: list[tuple[str, int, int]] = []

    def sources(self) -> list[LoopbackSource]:
        return [LoopbackSource("{endpoint-1}", "Speakers (Realtek(R) Audio)", 2)]

    @contextmanager
    def recorder(self, source_id: str, rate: int, channels: int) -> Any:
        if self.refuse is not None:
            raise self.refuse
        self.opened.append((source_id, rate, channels))
        try:
            yield self.recorder_
        finally:
            self.recorder_.closed.set()


def test_capture_calls_back_only_after_start_and_closes_on_its_thread() -> None:
    provider = Provider()
    got: list[np.ndarray] = []
    finished = threading.Event()
    stream = open_capture(provider, "{endpoint-1}", 48000, 1, lambda data, frames, status: got.append(data), finished.set, "dev")
    assert provider.opened == [("{endpoint-1}", 48000, 1)]
    assert wait(lambda: provider.recorder_.reads >= 3) and got == []  # read and dropped before start
    stream.start()
    assert wait(lambda: len(got) >= 3)
    assert got[0].shape == (480, 2) and got[0].dtype == np.float32 and got[0].flags.c_contiguous
    stream.stop()
    assert provider.recorder_.closed.is_set() and not finished.is_set()
    assert stream.latency == (0.01, 0.0)


@pytest.mark.parametrize(("error", "reason"), [(IndexError("no soundcard with id {x}"), "missing"), (RuntimeError("Error 0x88890004"), "busy")])
def test_a_refused_capture_raises_like_a_portaudio_open(error: Exception, reason: str) -> None:
    with pytest.raises(DeviceOpenError) as info:
        open_capture(Provider(refuse=error), "{x}", 48000, 1, lambda *a: None, lambda: None, "dev")
    assert (info.value.reason, info.value.role, info.value.device_id) == (reason, "input", "dev")


def test_a_capture_that_dies_is_reported_lost() -> None:
    provider = Provider()
    finished = threading.Event()
    stream = open_capture(provider, "{endpoint-1}", 48000, 1, lambda *a: None, finished.set)
    stream.start()
    provider.recorder_.fail = RuntimeError("AUDCLNT_E_DEVICE_INVALIDATED")
    assert finished.wait(5)
    stream.close()


def test_a_capture_that_never_opens_times_out() -> None:
    @contextmanager
    def hang() -> Any:
        time.sleep(0.5)
        yield Recorder()

    with pytest.raises(DeviceOpenError) as info:
        CaptureStream(hang, lambda *a: None, lambda: None).open(timeout=0.05)
    assert info.value.reason == "busy"


def test_check_capture_needs_the_source_listed() -> None:
    check_capture(Provider(), "{endpoint-1}")
    with pytest.raises(DeviceOpenError) as info:
        check_capture(Provider(), "{gone}", "dev")
    assert info.value.reason == "missing"


def test_provider_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, why = loopback_provider("darwin")
    assert provider is None and why is not None and "BlackHole" in why
    monkeypatch.setitem(sys.modules, "soundcard", None)  # not installed, or no PulseAudio server
    provider, why = loopback_provider("linux")
    assert provider is None and why is not None and why.startswith("Output devices cannot be recorded on this system")


def test_sources_join_the_listing_as_loopback_inputs() -> None:
    hostapis = [{"name": "Windows WASAPI", "default_input_device": 0, "default_output_device": 1}]
    devices = [
        {"name": "Microphone (Realtek(R) Audio)", "index": 0, "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speakers (Realtek(R) Audio)", "index": 1, "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
    ]
    add_loopback_sources(hostapis, devices, [LoopbackSource("{1}", "Speakers (Realtek(R) Audio)", 2), LoopbackSource("{2}", "Headphones (USB)", 8)])
    assert hostapis[-1]["name"] == "Loopback" and [d["index"] for d in devices[2:]] == [-1, -2]
    grouped = group_devices(hostapis, devices, "win32")
    assert [p["name"] for p in grouped["outputs"]] == ["Speakers (Realtek(R) Audio)"]
    mic, *loopbacks = grouped["inputs"]
    assert mic["name"] == "Microphone (Realtek(R) Audio)" and mic["is_default"] and not mic["is_loopback"]
    assert [p["key"] for p in loopbacks] == ["loopback:headphones (usb)", "loopback:speakers (realtek(r) audio)"]
    speakers = loopbacks[1]["variants"][0]
    assert speakers["host_api"] == "loopback" and speakers["loopback_source"] == "{1}" and speakers["loopback_of"] == "Speakers (Realtek(R) Audio)"
    assert speakers["supported_rates"] == [44100, 48000, 88200, 96000] and speakers["channels"] == 2 and not speakers["is_default"]


# -- the backend -----------------------------------------------------------------------------------------


def _backend(monkeypatch: pytest.MonkeyPatch, provider: Any, raw: tuple[list, list, int], list_loopback: bool = True) -> SounddeviceBackend:
    monkeypatch.setattr("rvc_next.engine.audio_io.devices.enumerate_raw", lambda: (list(raw[0]), list(raw[1]), raw[2]))
    backend = SounddeviceBackend.__new__(SounddeviceBackend)
    backend.list_loopback, backend.loopback_error, backend._provider = list_loopback, None, provider
    return backend


def test_backend_lists_sources_unless_portaudio_has_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    hostapis = [{"name": "Windows WASAPI"}]
    speakers = {"name": "Speakers (Realtek(R) Audio)", "index": 0, "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2}
    _, devices, _ = _backend(monkeypatch, Provider(), (hostapis, [speakers], 0)).query()
    assert [d.get("loopback_source") for d in devices] == [None, "{endpoint-1}"]
    assert len(_backend(monkeypatch, Provider(), (hostapis, [speakers], 0), list_loopback=False).query()[1]) == 1
    # A PortAudio built after 2022 lists WASAPI loopback devices itself: they are used, the provider is not asked.
    native = {**speakers, "name": "Speakers (Realtek(R) Audio) [Loopback]", "index": 1, "max_input_channels": 2, "max_output_channels": 0}
    assert len(_backend(monkeypatch, Provider(), (hostapis, [speakers, native], 0)).query()[1]) == 2
    # No provider here: the listing says why.
    monkeypatch.setattr(capture, "loopback_provider", lambda platform=None: (None, "macOS cannot record an output device"))
    backend = _backend(monkeypatch, None, (hostapis, [speakers], 0))
    assert len(backend.query()[1]) == 1 and backend.loopback_error == "macOS cannot record an output device"


def test_backend_opens_a_loopback_input_through_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Provider()
    backend = _backend(monkeypatch, provider, ([], [], 0))
    ep = Endpoint({"id": "lb", "portaudio_index": -1, "host_api": "loopback", "channels": 2, "loopback_source": "{endpoint-1}"})
    backend.check(ep, "input", 1, 44100)
    stream = backend.open_input(ep, 44100, 441, 1, lambda *a: None, lambda: None)
    try:
        assert isinstance(stream, CaptureStream) and provider.opened == [("{endpoint-1}", 44100, 1)]
    finally:
        stream.close()
    assert choose_topology(ep, Endpoint({"id": "o", "portaudio_index": 3, "host_api": "loopback", "channels": 2}), 44100) == "split"


# -- a session on the fake backend ---------------------------------------------------------------------


def _fake_endpoints(backend: FakeBackend) -> dict[str, dict[str, Any]]:
    grouped = group_devices(*backend.query()[:2], "linux")
    return {("loopback " if v["loopback_of"] else "") + v["raw_name"]: v for p in grouped["inputs"] + grouped["outputs"] for v in p["variants"]}


def test_a_session_converts_what_an_output_device_plays() -> None:
    backend = FakeBackend(*default_devices(), time_scale=4.0, list_loopback=True)
    eps = _fake_endpoints(backend)
    tone = (0.4 * np.sin(2 * np.pi * 440 * np.arange(48000 * 4) / 48000)).astype(np.float32)
    backend.play(Endpoint(eps["Fake Speakers"]), tone, 48000)  # a video playing on the speakers
    cfg = SessionConfig(input=Endpoint(eps["loopback Fake Speakers"]), output=Endpoint(eps["Fake Headphones"]))
    session = AudioSession(backend, cfg, 4800, 48000, processor=lambda x: 0.5 * x)
    assert session.topology == "split"
    session.start()
    try:
        assert wait(lambda: session.processed_blocks >= 5)
        assert session.input_levels.peak_db > -10
        assert np.abs(backend.output(2)[-4800:]).max() > 0.1  # converted, on the headphones
    finally:
        session.stop()
    # Unplugging the speakers ends their recording too.
    lost: list[tuple] = []
    session = AudioSession(backend, SessionConfig(input=Endpoint(eps["loopback Fake Speakers"])), 4800, 48000, on_device_lost=lambda *a: lost.append(a))
    session.start()
    try:
        backend.lose(1)
        assert wait(lambda: bool(lost))
    finally:
        session.stop()
    assert lost[0][0] == "input"
