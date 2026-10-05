import threading
import time
from typing import Any

import numpy as np
import pytest

from rvc_next.engine.audio.sola import Sola
from rvc_next.engine.audio_io.devices import group_devices
from rvc_next.engine.audio_io.drift import DriftCorrector
from rvc_next.engine.audio_io.fake import FakeBackend
from rvc_next.engine.audio_io.rings import Ring
from rvc_next.engine.audio_io.streams import AudioSession, Endpoint, LinearResampler, SessionConfig, choose_topology, engine_rate

# -- ring ---------------------------------------------------------------------------------------------


def test_ring_wraps_and_counts_overruns() -> None:
    ring = Ring(8)
    assert ring.write(np.arange(6, dtype=np.float32)) == 6
    assert ring.read(4).tolist() == [0, 1, 2, 3]
    assert ring.write(np.arange(10, 16, dtype=np.float32)) == 6
    assert ring.available == 8 and ring.overruns == 0
    assert ring.write(np.ones(3, dtype=np.float32)) == 0 and ring.overruns == 1
    assert ring.read(10).tolist() == [4, 5, 10, 11, 12, 13, 14, 15]


# -- drift ------------------------------------------------------------------------------------------------


def _simulate(writer_ppm: float, reader_ppm: float, seconds: float, rate: int = 48000, block: int = 480) -> tuple[DriftCorrector, list[int]]:
    """A writer and a reader on independent clocks, simulated event by event."""
    ring = Ring(block * 20)
    ring.write(np.zeros(block * 2, dtype=np.float32))
    corrector = DriftCorrector(ring, rate, block, block * 2)
    w_period = block / (rate * (1 + writer_ppm * 1e-6))
    r_period = block / (rate * (1 + reader_ppm * 1e-6))
    t_w, t_r, n = 0.0, r_period / 3, 0
    levels = []
    signal = np.sin(np.arange(block, dtype=np.float32) * 0.05).astype(np.float32)
    while t_r < seconds:
        if t_w <= t_r:
            ring.write(signal)
            t_w += w_period
        else:
            out = corrector.read(block)
            assert out.shape == (block,)
            t_r += r_period
            n += 1
            levels.append(ring.available)
    return corrector, levels


@pytest.mark.parametrize(("w", "r"), [(200, -200), (-200, 200), (0, 0)])
def test_drift_correction_holds_the_fill(w: float, r: float) -> None:
    corrector, levels = _simulate(w, r, seconds=60)
    late = levels[len(levels) // 4 :]
    assert corrector.underruns == 0
    assert max(late) - min(late) <= 3 * 480
    if w == r:
        assert corrector.dropped == corrector.duplicated == 0
    else:
        expected = w - r
        assert corrector.drift_ppm is not None and abs(corrector.drift_ppm - expected) < 0.35 * abs(expected)


def test_drift_underrun_fades_to_silence() -> None:
    ring = Ring(1000)
    ring.write(np.ones(100, dtype=np.float32))
    out = DriftCorrector(ring, 48000, 480, 960, enabled=False).read(480)
    assert out[0] == 1 and out[99] == 0 and np.all(out[100:] == 0)


# -- SOLA ----------------------------------------------------------------------------------------------------


def _signal(n: int) -> Any:
    import torch

    rng = np.random.default_rng(1)
    t = np.arange(n) / 16000
    x = np.sin(2 * np.pi * (200 + 300 * t) * t) + 0.3 * rng.standard_normal(n)
    return torch.from_numpy(x.astype(np.float32))


@pytest.mark.parametrize("shift", [0, 37, 120])
def test_sola_realigns_shifted_blocks(shift: int) -> None:
    block, buf, search = 1600, 640, 160
    s = _signal(block * 12)
    sola = Sola(buf, search, "cpu")
    out = []
    for k in range(1, 10):
        # Each new block starts ``shift`` samples early; SOLA must find the shift and stay continuous.
        start = k * block - (shift if k > 1 else 0)
        infer_wav = s[start : start + block + buf + search].clone()
        out.append(sola.apply(infer_wav, block).numpy())
        if k > 1:
            assert sola.last_offset == shift
    joined = np.concatenate(out[1:])
    expected = s[2 * block : 10 * block].numpy()
    assert np.allclose(joined, expected, atol=1e-5)


# -- resampler and topology ---------------------------------------------------------------------------------


def test_linear_resampler_length_and_continuity() -> None:
    res = LinearResampler(44100, 48000)
    x = np.sin(np.arange(44100) * 2 * np.pi * 440 / 44100).astype(np.float32)
    out = np.concatenate([res(x[i : i + 441]) for i in range(0, 44100, 441)])
    assert abs(out.shape[0] - 48000) <= 2
    ref = np.sin(np.arange(out.shape[0]) * 2 * np.pi * 440 / 48000)
    assert np.max(np.abs(out[10:-10] - ref[10:-10])) < 0.01


def _dev(index: int, name: str, i: int, o: int, rates: list[int], hostapi: int = 0) -> dict[str, Any]:
    return {"name": name, "index": index, "hostapi": hostapi, "max_input_channels": i, "max_output_channels": o, "default_samplerate": 16000.0, "supported_rates": rates}


def fake_machine() -> tuple[list[dict], list[dict]]:
    hostapis = [
        {"name": "ALSA", "devices": [0, 1, 2], "default_input_device": 0, "default_output_device": 1},
        {"name": "JACK Audio Connection Kit", "devices": [3], "default_input_device": -1, "default_output_device": 3},
    ]
    devices = [_dev(0, "Mic", 2, 0, [16000, 48000]), _dev(1, "Speakers", 0, 2, [16000, 48000]), _dev(2, "Headphones", 0, 2, [16000]), _dev(3, "Jack Out", 0, 2, [16000], hostapi=1)]
    return hostapis, devices


def endpoints() -> dict[str, dict]:
    hostapis, devices = fake_machine()
    grouped = group_devices(hostapis, devices, "linux")
    out = {}
    for p in grouped["inputs"] + grouped["outputs"]:
        for v in p["variants"]:
            out[v["raw_name"]] = v
    return out


def test_topology_choice() -> None:
    eps = endpoints()
    mic, spk, jack = Endpoint(eps["Mic"]), Endpoint(eps["Speakers"]), Endpoint(eps["Jack Out"])
    assert choose_topology(mic, spk, 16000) == "duplex"
    assert choose_topology(mic, jack, 16000) == "split"
    assert choose_topology(mic, Endpoint(eps["Headphones"]), 48000) == "split"
    assert choose_topology(mic, None, 16000) == "input"
    assert engine_rate(spk) == 16000
    assert engine_rate(Endpoint(eps["Speakers"], sample_rate=48000)) == 48000
    assert engine_rate(None) == 48000


# -- sessions on the fake backend ------------------------------------------------------------------------------


def _session(backend: FakeBackend, output: str, processor: Any = None, monitor: str | None = None, **kw: Any) -> AudioSession:
    eps = endpoints()
    cfg = SessionConfig(input=Endpoint(eps["Mic"]), output=Endpoint(eps[output]), monitor=Endpoint(eps[monitor]) if monitor else None, **kw)
    return AudioSession(backend, cfg, 1600, 16000, processor=processor or (lambda x: 0.5 * x))


def _wait(predicate: Any, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize(("output", "topology"), [("Speakers", "duplex"), ("Jack Out", "split")])
def test_session_moves_audio(output: str, topology: str) -> None:
    backend = FakeBackend(*fake_machine(), time_scale=4.0)
    session = _session(backend, output)
    assert session.topology == topology
    session.start()
    try:
        assert _wait(lambda: session.processed_blocks >= 10)
    finally:
        session.stop()
    index = 1 if output == "Speakers" else 3
    out = backend.output(index)
    assert out.shape[0] >= 10 * 1600
    assert np.abs(out[:1600]).max() == 0  # the pre-filled block of silence
    assert np.abs(out[-1600:]).max() > 0.05
    assert -8 < session.output_levels.peak_db - session.input_levels.peak_db < -4  # the processor halves it


def test_monitor_tee_and_passthrough() -> None:
    backend = FakeBackend(*fake_machine(), time_scale=4.0)
    session = _session(backend, "Speakers", processor=lambda x: np.zeros_like(x), monitor="Headphones", monitor_source="input")
    session.start()
    try:
        assert _wait(lambda: session.processed_blocks >= 8)
    finally:
        session.stop()
    assert np.abs(backend.output(2)[-1600:]).max() > 0.05  # the monitor hears the input
    assert np.abs(backend.output(1)).max() == 0  # the output carries the (silent) conversion


def test_slow_processing_counts_underruns_and_overruns() -> None:
    backend = FakeBackend(*fake_machine(), time_scale=4.0)

    def slow(x: np.ndarray) -> np.ndarray:
        time.sleep(0.2)  # two blocks at this time scale
        return x

    session = _session(backend, "Speakers", processor=slow)
    session.start()
    try:
        assert _wait(lambda: session.underruns > 2 and session.overruns > 0, timeout=10)
    finally:
        session.stop()


def test_device_loss_is_reported() -> None:
    backend = FakeBackend(*fake_machine(), time_scale=4.0)
    lost: list[tuple] = []
    event = threading.Event()
    session = _session(backend, "Jack Out")

    def on_lost(*args: Any) -> None:
        lost.append(args)
        event.set()

    session.on_device_lost = on_lost
    session.start()
    try:
        assert _wait(lambda: session.processed_blocks >= 3)
        backend.lose(3)
        assert event.wait(5)
    finally:
        session.stop()
    role, device_id, reason, _ = lost[0]
    assert role == "output" and reason == "missing" and device_id == endpoints()["Jack Out"]["id"]
    assert not session.running


def test_refused_format_raises_with_role() -> None:
    from rvc_next.engine.audio_io.streams import DeviceOpenError

    backend = FakeBackend(*fake_machine(), time_scale=4.0)
    eps = endpoints()
    cfg = SessionConfig(input=Endpoint(eps["Mic"]), output=Endpoint(eps["Headphones"], sample_rate=48000))
    with pytest.raises(DeviceOpenError) as info:
        AudioSession(backend, cfg, 4800, 48000).start()
    assert info.value.reason == "format" and info.value.role == "output"


def test_meter_only_session() -> None:
    backend = FakeBackend(*fake_machine(), time_scale=4.0)
    session = AudioSession(backend, SessionConfig(input=Endpoint(endpoints()["Mic"])), 800, 16000)
    assert session.topology == "input"
    session.start()
    try:
        assert _wait(lambda: session.input_levels.peak_db > -20)
    finally:
        session.stop()
