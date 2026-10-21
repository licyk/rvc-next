import json
import time
from pathlib import Path
from typing import Any

import pytest

from rvc_next.engine.audio_io.devices import group_devices, resolve_drivers
from rvc_next.engine.audio_io.fake import FakeBackend
from rvc_next.engine.audio_io.fallback import MAX_ATTEMPTS, describe_attempts, open_with_fallback
from rvc_next.engine.audio_io.streams import AudioSession, DeviceOpenError, Endpoint, SessionConfig

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "devices"
HOST_ERROR = "Unanticipated host error [PaErrorCode -9999]: 'WdmSyncIoctl: DeviceIoControl GLE = 0x00000492' [Windows WDM-KS error 0]"


def windows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = json.loads((FIXTURES / "windows.json").read_text())
    return data["hostapis"], data["devices"], group_devices(data["hostapis"], data["devices"], "win32", data["default_hostapi"])


def auto(listing: dict[str, Any], direction: str, name: str) -> Endpoint:
    """An endpoint for ``name`` whose driver is chosen automatically, as the core sends it."""
    physical = next(p for p in listing[f"{direction}s"] if p["name"] == name)
    _, device, _, candidates = resolve_drivers({"physical_key": physical["key"], "name": physical["name"], "driver": "auto"}, listing, direction)
    return Endpoint(device=device, candidates=candidates)


def index(ep: Endpoint) -> int | None:
    return ep.index


@pytest.fixture
def machine() -> tuple[FakeBackend, dict[str, Any]]:
    hostapis, devices, listing = windows()
    return FakeBackend(hostapis, devices, time_scale=4.0), listing


def open_session(backend: FakeBackend) -> Any:
    def start(cfg: SessionConfig) -> AudioSession:
        session = AudioSession(backend, cfg, 480)  # at the output's rate, or the input's alone, as the worker opens them
        session.passthrough = True
        session.start()
        return session

    return start


def test_a_driver_that_will_not_start_gives_way_to_the_next(machine) -> None:
    """The reported failure: WASAPI's stream opens, then will not start; DirectSound opens the microphone."""
    backend, listing = machine
    mic = auto(listing, "input", "Microphone (Realtek(R) Audio)")
    speakers = auto(listing, "output", "Speakers (Realtek(R) Audio)")
    assert (mic.host_api, index(mic)) == ("wasapi", 26)
    backend.refuse_start[26] = HOST_ERROR
    moves: list[tuple[str, str | None, str | None]] = []
    session, used, attempts = open_with_fallback(
        SessionConfig(input=mic, output=speakers), open_session(backend), lambda role, a, e, b: moves.append((role, a.host_api, b.host_api))
    )
    try:
        assert session.running and used.input.host_api == "directsound" and index(used.input) == 13
        assert used.output is not None and used.output.host_api == "wasapi"
        assert session.topology == "split"  # two drivers: two streams
        assert moves == [("input", "wasapi", "directsound")]
        assert list(attempts) == ["input"] and attempts["input"][0]["host_api"] == "wasapi" and attempts["input"][0]["reason"] == "busy"
        assert HOST_ERROR in attempts["input"][0]["error"]
        # The configuration it opened is still the same choice as the one asked for.
        assert used.same_endpoints(SessionConfig(input=mic, output=speakers))
    finally:
        session.stop()


def test_a_driver_that_ends_the_stream_by_itself_gives_way_to_the_next(machine) -> None:
    """WASAPI starts the microphone's stream, then the driver ends it: within the settle window that is a refusal."""
    backend, listing = machine
    mic = auto(listing, "input", "Microphone (Realtek(R) Audio)")
    backend.stops.add(26)
    lost: list[str] = []

    def start(cfg: SessionConfig) -> AudioSession:
        session = AudioSession(backend, cfg, 480, on_device_lost=lambda role, *_: lost.append(role))
        session.start()
        return session

    session, used, attempts = open_with_fallback(SessionConfig(input=mic), start)
    try:
        assert session.running and used.input.host_api == "directsound"
        assert attempts["input"][0]["reason"] == "stopped" and "ended the input stream" in attempts["input"][0]["error"]
        assert lost == []  # a refusal, never a lost device
    finally:
        session.stop()
    # A fixed driver is not watched: its stop is a lost device, as before.
    backend.stops.add(13)
    fixed = AudioSession(backend, SessionConfig(input=Endpoint(device=mic.candidates[1])), 480, on_device_lost=lambda role, *_: lost.append(role))
    fixed.start()
    try:
        deadline = time.monotonic() + 5
        while not lost and time.monotonic() < deadline:
            time.sleep(0.01)
        assert lost == ["input"]
    finally:
        fixed.stop()


def test_every_driver_refusing_names_them_all(machine) -> None:
    backend, listing = machine
    mic = auto(listing, "input", "Microphone (Realtek(R) Audio)")
    for i in (26, 13, 3):
        backend.refuse_start[i] = HOST_ERROR
    with pytest.raises(DeviceOpenError) as raised:
        open_with_fallback(SessionConfig(input=mic), open_session(backend))
    e = raised.value
    assert e.role == "input" and e.reason == "busy"
    assert [a["host_api"] for a in e.attempts["input"]] == ["wasapi", "directsound", "mme"]
    assert describe_attempts(e.attempts["input"]).startswith("WASAPI: Error starting stream: Unanticipated host error")


def test_a_fixed_driver_and_a_refused_permission_are_not_retried(machine) -> None:
    backend, listing = machine
    mic = auto(listing, "input", "Microphone (Realtek(R) Audio)")
    backend.refuse[26] = "permission"
    with pytest.raises(DeviceOpenError) as raised:
        open_with_fallback(SessionConfig(input=mic), open_session(backend))
    assert raised.value.reason == "permission" and len(raised.value.attempts["input"]) == 1
    fixed = Endpoint(device=mic.device)
    backend.refuse[26] = "busy"
    with pytest.raises(DeviceOpenError) as raised:
        open_with_fallback(SessionConfig(input=fixed), open_session(backend))
    assert raised.value.attempts == {}


def test_a_duplex_error_moves_the_input_first_and_the_attempts_are_capped() -> None:
    devices = [{"id": f"in{i}", "host_api": f"api{i}", "portaudio_index": i} for i in range(12)]
    outputs = [{"id": f"out{i}", "host_api": f"api{i}", "portaudio_index": 100 + i} for i in range(12)]
    cfg = SessionConfig(input=Endpoint(device=devices[0], candidates=devices), output=Endpoint(device=outputs[0], candidates=outputs))
    seen: list[tuple[str | None, str | None]] = []

    def start(c: SessionConfig) -> None:
        assert c.output is not None
        seen.append((c.input.device_id, c.output.device_id))
        raise DeviceOpenError("the pair refused", "busy", None)

    with pytest.raises(DeviceOpenError):
        open_with_fallback(cfg, start)
    assert len(seen) == MAX_ATTEMPTS
    assert seen[:2] == [("in0", "out0"), ("in1", "out0")]


def test_same_choice_whichever_driver_opened() -> None:
    a, b = {"id": "a", "host_api": "wasapi"}, {"id": "b", "host_api": "directsound"}
    asked = Endpoint(device=a, candidates=[a, b])
    assert asked.same(Endpoint(device=b, candidates=[b, a]))  # reordered after a fallback: no reopen
    assert not asked.same(Endpoint(device=a))  # the driver fixed now
    assert not asked.same(Endpoint(device=a, candidates=[a, b], channels=[2]))
    assert Endpoint(device=a).same(Endpoint(device=a)) and not Endpoint(device=a).same(Endpoint(device=b))
