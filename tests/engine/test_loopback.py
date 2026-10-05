"""Loopback latency measurement: the burst analysis, then whole measurements on the fake backend."""

import numpy as np
import pytest

from rvc_next.engine.audio_io.fake import FakeBackend
from rvc_next.engine.audio_io.loopback import LoopbackProbe, analyse, measure
from rvc_next.engine.audio_io.streams import Endpoint, SessionConfig

RATE = 48000


def _played_back(probe: LoopbackProbe, delay: int, gain: float = 0.5, noise: float = 0.0, echo: tuple[int, float] | None = None) -> np.ndarray:
    """What the input would record: the probe's signal ``delay`` samples later, with noise and a room echo."""
    rec = np.zeros(probe.total, dtype=np.float32)
    rec[delay:] = gain * probe.signal[: probe.total - delay]
    if echo is not None:
        lag, amount = echo
        rec[delay + lag :] += amount * probe.signal[: probe.total - delay - lag]
    rec += (noise * np.random.default_rng(3).standard_normal(probe.total)).astype(np.float32)
    return rec


def test_finds_the_round_trip_in_noise_and_echo() -> None:
    probe = LoopbackProbe(RATE, max_delay_s=1.0)
    rec = _played_back(probe, delay=7391, gain=0.05, noise=0.02, echo=(900, 0.03))
    result = analyse(rec, probe.sent, probe.starts, RATE, probe.max_delay)
    assert result.ok and result.round_trip == 7391 and result.delays == [7391] * 4
    assert result.round_trip_ms == pytest.approx(7391 / 48) and result.jitter_ms == 0.0
    assert result.snr_db is not None and result.snr_db > 20 and not result.clipped


def test_silence_and_unrelated_sound_are_no_signal() -> None:
    probe = LoopbackProbe(RATE, max_delay_s=1.0)
    for rec in (np.zeros(probe.total, dtype=np.float32), (0.3 * np.sin(np.arange(probe.total) * 0.03)).astype(np.float32)):
        result = analyse(rec, probe.sent, probe.starts, RATE, probe.max_delay)
        assert not result.ok and result.reason == "no_signal" and result.round_trip is None


def test_bursts_that_disagree_are_inconsistent() -> None:
    probe = LoopbackProbe(RATE, max_delay_s=1.0)
    rec = np.zeros(probe.total, dtype=np.float32)
    for k, (burst, start) in enumerate(zip(probe.sent, probe.starts, strict=True)):
        at = start + 4000 + k * 600  # a clock slipping 12.5 ms per burst
        rec[at : at + burst.shape[0]] += burst
    result = analyse(rec, probe.sent, probe.starts, RATE, probe.max_delay)
    assert not result.ok and result.reason == "inconsistent" and result.round_trip == 4900


def test_measures_a_duplex_loop_exactly() -> None:
    # One duplex stream: the one-block prefill, the block the device plays while the next is captured, and the cable's 60 ms.
    backend = FakeBackend(time_scale=4.0, loopback={"delay_ms": 60.0, "gain": 0.5, "noise": 0.001})
    run = measure(backend, SessionConfig(input=Endpoint(), output=Endpoint()), block=2400, timeout_scale=0.25)
    assert run.topology == "duplex" and run.sample_rate == RATE and run.underruns == 0
    assert run.result.ok and run.result.round_trip == 2 * 2400 + 2880
    assert run.reported_ms == (10.0, 10.0)


def test_measures_split_streams() -> None:
    # An input and an output stream: a two-block prefill and the device's block, give or take the drift correction.
    backend = FakeBackend(time_scale=4.0, loopback={"delay_ms": 60.0, "gain": 0.5})
    output = Endpoint(device={"id": "spk", "portaudio_index": 1, "host_api": "alsa", "channels": 2, "default_sample_rate": 48000, "supported_rates": [48000]})
    run = measure(backend, SessionConfig(input=Endpoint(), output=output), block=2400, timeout_scale=0.25)
    assert run.topology == "split"
    assert run.result.ok and run.result.round_trip_ms == pytest.approx(150 + 60, abs=2.0)


def test_no_loopback_is_no_signal() -> None:
    run = measure(FakeBackend(time_scale=4.0), SessionConfig(input=Endpoint(), output=Endpoint()), block=2400, timeout_scale=0.25)
    assert not run.result.ok and run.result.reason == "no_signal"


def test_a_refused_device_raises() -> None:
    from rvc_next.engine.audio_io.streams import DeviceOpenError

    with pytest.raises(DeviceOpenError):
        measure(FakeBackend(refuse={1: "busy"}), SessionConfig(input=Endpoint(), output=Endpoint()), block=2400)
