import threading
import time
from typing import Any, cast

import numpy as np
import pytest

from rvc_next.engine.audio.sola import Sola
from rvc_next.engine.audio_io.devices import group_devices
from rvc_next.engine.audio_io.drift import DriftCorrector
from rvc_next.engine.audio_io.fake import FakeBackend, FakeStream
from rvc_next.engine.audio_io.rings import Ring
from rvc_next.engine.audio_io.streams import (
    AudioSession,
    DeviceOpenError,
    Endpoint,
    LinearResampler,
    SessionConfig,
    SounddeviceBackend,
    choose_topology,
    classify_portaudio_error,
    engine_rate,
)

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


def test_slow_processing_underruns_and_skips_stale_input() -> None:
    # Processing that never keeps up: the output runs dry, and the input converted is always the
    # newest block, never a queue that overflows.
    backend = FakeBackend(*fake_machine(), time_scale=4.0)

    def slow(x: np.ndarray) -> np.ndarray:
        time.sleep(0.2)  # two blocks at this time scale
        return x

    session = _session(backend, "Speakers", processor=slow)
    session.start()
    try:
        assert _wait(lambda: session.underruns > 2 and session.skipped > 0, timeout=10)
    finally:
        session.stop()
    assert session.in_ring.overruns == 0


def _stall_first_block(seconds: float) -> Any:
    calls = [0]

    def processor(x: np.ndarray) -> np.ndarray:
        calls[0] += 1
        if calls[0] == 1:
            time.sleep(seconds)
        return x

    return processor


@pytest.mark.parametrize(("output", "normal_max"), [("Speakers", 1), ("Jack Out", 3)])
def test_a_stalled_first_block_leaves_no_backlog(output: str, normal_max: int) -> None:
    """A first block 4.5 blocks late used to stay in the output buffer as latency for the whole session."""
    backend = FakeBackend(*fake_machine(), time_scale=4.0)
    session = _session(backend, output, processor=_stall_first_block(4.5 * 0.1 / 4))
    session.start()
    try:
        assert _wait(lambda: session.processed_blocks >= 40, timeout=10)
        fills = []
        for _ in range(20):
            fills.append(session.out_ring.available)
            time.sleep(0.005)
    finally:
        session.stop()
    assert session.trimmed + session.skipped > 0
    assert max(fills) <= normal_max * 1600


@pytest.mark.parametrize(("output", "ppm"), [("Speakers", 0), ("Jack Out", 0), ("Jack Out", 300)])
def test_steady_sessions_never_trim(output: str, ppm: float) -> None:
    backend = FakeBackend(*fake_machine(), ppm={3: ppm}, time_scale=4.0)
    session = _session(backend, output)
    session.start()
    try:
        assert _wait(lambda: session.processed_blocks >= 60, timeout=10)
    finally:
        session.stop()
    assert (session.trimmed, session.skipped, session.underruns) == (0, 0, 0)


def test_load_timings_need_enough_blocks_and_skip_warmup() -> None:
    session = _session(FakeBackend(*fake_machine()), "Speakers")
    session.infer_ms.extend([500.0, 10.0, 11.0])
    assert session.infer_percentiles() == (11.0, 11.0)  # a few blocks: the median, not the slowest
    session.infer_ms.extend([10.0] * 37)
    assert session.infer_percentiles()[1] == pytest.approx(10.0, abs=1.0)  # one slow block in 40 is not the 95th percentile
    session.infer_ms.clear()
    session.skip_timing(1)
    for _ in range(3):
        session._process_block(np.zeros(1600, dtype=np.float32))
    assert len(session.infer_ms) == 2


def test_ring_discard_fades_in() -> None:
    ring = Ring(8)
    ring.write(np.ones(6, dtype=np.float32))
    assert ring.discard(2, fade=3) == 2 and ring.available == 4
    assert ring.read(4).tolist() == [0.0, 0.5, 1.0, 1.0]
    assert ring.discard(10) == 0


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


# -- channel counts ------------------------------------------------------------------------------------


def test_endpoint_opens_a_count_the_device_accepts() -> None:
    stereo_only = {"portaudio_index": 0, "channels": 2, "channel_counts": [2]}
    mic = Endpoint(stereo_only)
    assert mic.open_channels("input") == 2 and mic.selected("input") == [0]
    assert Endpoint(stereo_only, channels=[2]).selected("input") == [1]
    assert Endpoint(stereo_only, channels=[1, 2]).selected("input") == [0, 1]
    # Opened wider than wanted, an output still writes only the first two channels.
    surround = Endpoint({"portaudio_index": 1, "channels": 8, "channel_counts": [1, 6, 8]})
    assert surround.open_channels("output") == 6 and surround.selected("output") == [0, 1]
    # Unprobed devices open what they always did.
    assert Endpoint({"portaudio_index": 0, "channels": 2}).open_channels("input") == 1
    assert Endpoint({"portaudio_index": 1, "channels": 8}).open_channels("output") == 2
    assert Endpoint().open_channels("output") == 2
    # A saved channel the device lacks falls back to the first instead of mixing nothing.
    assert Endpoint({"portaudio_index": 0, "channels": 2}, channels=[3]).selected("input") == [0]


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("Error opening InputStream: Invalid number of channels [PaErrorCode -9998]", "channels"),
        ("Error opening OutputStream: Invalid sample rate [PaErrorCode -9997]", "format"),
        ("Error opening InputStream: Sample format not supported [PaErrorCode -9994]", "format"),
        ("Error opening InputStream: Device unavailable [PaErrorCode -9985]", "busy"),
        ("Error querying device -1: Invalid device [PaErrorCode -9996]", "missing"),
    ],
)
def test_classify_portaudio_error(message: str, reason: str) -> None:
    assert classify_portaudio_error(message) == reason


def test_a_stereo_only_microphone_opens_in_stereo() -> None:
    hostapis, devices = fake_machine()
    devices[0]["input_channel_counts"] = [2]
    backend = FakeBackend(hostapis, devices, time_scale=4.0)
    mic = next(v for p in group_devices(hostapis, devices, "linux")["inputs"] for v in p["variants"] if v["raw_name"] == "Mic")
    session = AudioSession(backend, SessionConfig(input=Endpoint(mic)), 800, 16000)
    session.start()
    try:
        assert [s.in_ch for s in backend.streams] == [2]
        assert _wait(lambda: session.input_levels.peak_db > -20)
    finally:
        session.stop()
    # Opened in mono, as before, the driver refuses it: a channel problem, which another rate cannot fix.
    mono = Endpoint({k: v for k, v in mic.items() if k != "channel_counts"})
    with pytest.raises(DeviceOpenError) as info:
        AudioSession(backend, SessionConfig(input=mono), 800, 16000).start()
    assert info.value.reason == "channels" and info.value.role == "input"


def test_wasapi_shared_streams_let_windows_convert() -> None:
    class WasapiSettings:
        def __init__(self, exclusive: bool = False, auto_convert: bool = False) -> None:
            self.exclusive, self.auto_convert = exclusive, auto_convert

    backend = SounddeviceBackend.__new__(SounddeviceBackend)
    backend.sd = cast(Any, type("sd", (), {"WasapiSettings": WasapiSettings}))
    wasapi = {"portaudio_index": 3, "host_api": "wasapi", "channels": 2}
    shared, exclusive = backend._extra(Endpoint(wasapi)), backend._extra(Endpoint(wasapi, exclusive=True))
    assert (shared.exclusive, shared.auto_convert) == (False, True)
    assert (exclusive.exclusive, exclusive.auto_convert) == (True, False)
    assert backend._extra(Endpoint({**wasapi, "host_api": "wdm-ks"})) is None and backend._extra(Endpoint()) is None


def test_a_refused_duplex_stream_names_the_device_at_fault() -> None:
    """PortAudio's duplex error names neither device: each side is checked alone to find it."""

    class FakeSd:
        bad: int | None = None

        def Stream(self, **kw: Any) -> Any:
            raise RuntimeError("Error opening Stream: Device unavailable [PaErrorCode -9985]")

        def check_input_settings(self, device: int, **kw: Any) -> None:
            if device == self.bad:
                raise RuntimeError("Error opening InputStream: Device unavailable [PaErrorCode -9985]")

        def check_output_settings(self, device: int, **kw: Any) -> None:
            if device == self.bad:
                raise RuntimeError("Error opening OutputStream: Invalid sample rate [PaErrorCode -9997]")

    sd = FakeSd()
    backend = SounddeviceBackend.__new__(SounddeviceBackend)
    backend.sd = cast(Any, sd)
    mic, speakers = Endpoint({"portaudio_index": 0, "id": "mic"}), Endpoint({"portaudio_index": 1, "id": "spk"})
    for bad, role, reason, device_id in ((0, "input", "busy", "mic"), (1, "output", "format", "spk"), (None, None, "busy", None)):
        sd.bad = bad
        with pytest.raises(DeviceOpenError) as info:
            backend.open_duplex(mic, speakers, 48000, 480, 1, 2, lambda *a: None, lambda: None)
        assert (info.value.role, info.value.reason, info.value.device_id) == (role, reason, device_id)
        assert "[PaErrorCode -9985]" in info.value.message


@pytest.mark.parametrize(("output", "monitor", "role"), [("Jack Out", None, "output"), ("Speakers", "Headphones", "monitor"), ("Speakers", None, None)])
def test_a_stream_that_will_not_start_names_its_device(output: str, monitor: str | None, role: str | None) -> None:
    """A stream that opens and then fails to start: its role (a duplex stream's could be either device)."""
    failing = {"output": 3, "monitor": 2, None: 1}[role]

    class Backend(FakeBackend):
        def _open(self, kind: str, inp: Endpoint | None, out: Endpoint | None, *args: Any, **kw: Any) -> FakeStream:
            stream = super()._open(kind, inp, out, *args, **kw)
            return cast(FakeStream, Unstartable(stream)) if stream.out_index == failing else stream

    class Unstartable:
        def __init__(self, stream: Any) -> None:
            self.stream = stream
            self.latency = stream.latency

        def start(self) -> None:
            raise RuntimeError("Error starting stream: Unanticipated host error [PaErrorCode -9999]")

        def stop(self) -> None:
            self.stream.stop()

        def close(self) -> None:
            self.stream.close()

    backend = Backend(*fake_machine(), time_scale=4.0)
    session = _session(backend, output, monitor=monitor)
    with pytest.raises(DeviceOpenError) as info:
        session.start()
    name = monitor if role == "monitor" and monitor else output
    assert (info.value.role, info.value.device_id) == (role, endpoints()[name]["id"] if role else None)
    assert not backend.streams  # every stream it opened is closed again


def test_probe_stream_finds_a_stream_that_stops() -> None:
    from rvc_next.engine.audio_io.streams import probe_stream

    eps = endpoints()
    backend = FakeBackend(*fake_machine(), time_scale=4.0)
    probe_stream(backend, Endpoint(eps["Speakers"]), "output", 16000)  # runs: no error
    probe_stream(backend, Endpoint(eps["Mic"]), "input", 16000)
    backend.stops.add(1)
    with pytest.raises(DeviceOpenError) as info:
        probe_stream(backend, Endpoint(eps["Speakers"]), "output", 16000)
    assert (info.value.reason, info.value.role, info.value.device_id) == ("stopped", "output", eps["Speakers"]["id"])
    backend.refuse[0] = "busy"
    with pytest.raises(DeviceOpenError) as info:
        probe_stream(backend, Endpoint(eps["Mic"]), "input", 16000)
    assert (info.value.reason, info.value.role) == ("busy", "input")
    assert not backend.streams


def test_recorder_writes_what_it_is_given(tmp_path) -> None:
    import soundfile as sf

    from rvc_next.engine.audio_io.recorder import Recorder

    rec = Recorder(tmp_path / "r.wav", 16000, "both")
    for _ in range(10):
        rec.write(np.full(1600, 0.25, np.float32), np.full(1600, -0.5, np.float32))
    result = rec.close()
    data, rate = sf.read(result.path)
    assert rate == 16000 and data.shape == (16000, 2) and result.seconds == 1.0 and result.dropped_blocks == 0
    assert np.allclose(data[:, 0], 0.25, atol=1e-4) and np.allclose(data[:, 1], -0.5, atol=1e-4)
    mono = Recorder(tmp_path / "m.wav", 16000, "converted")
    mono.write(np.zeros(160, np.float32), np.ones(160, np.float32) * 2)  # clipped to full scale
    data, _ = sf.read(mono.close().path)
    assert data.ndim == 1 and np.allclose(data, 1.0, atol=1e-4)
