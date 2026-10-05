"""Measured latency: play test bursts on the output and find them again in the input.

The output has to reach the input: a cable from the output to the input, a virtual cable's own
loopback, or speakers near the microphone. ``LoopbackProbe`` is an ``AudioSession`` processor, so
the bursts go through the same rings, prefill and drift correction as converted audio; what it
measures is the round trip from the processor's output back to its input, in samples. With no
underruns that is the whole session's latency except the engine's own delay (``engine_delay_ms``),
which depends only on the stream parameters.

Each burst is a different band-limited noise sequence, so a burst cannot be mistaken for another
however long the round trip is, and its cross-correlation with the recording has one sharp peak
even in a noisy, reverberant room. A burst counts as found when that peak stands 20 dB above the
correlation's noise floor; the measurement holds when most bursts are found and agree.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from rvc_next.engine.audio.dsp import peak_db
from rvc_next.engine.audio_io.streams import AudioSession, Backend, DeviceOpenError, SessionConfig

BURST_S = 0.1
INTERVAL_S = 0.25
LEAD_S = 0.3
"""Silence before the first burst, while the streams settle."""
MIN_SNR_DB = 20.0
MAX_SPREAD_MS = 2.0
"""Found bursts further apart than this make the measurement inconsistent (dropouts, a drifting clock)."""
LOW_HZ, HIGH_HZ = 300.0, 6000.0


def bursts(sample_rate: int, count: int, level_db: float = -12.0, seed: int = 7) -> list[np.ndarray]:
    """``count`` distinct noise bursts, band-limited to 300 Hz–6 kHz (what small speakers and
    microphones pass), peaking at ``level_db`` dBFS, with 2 ms fades."""
    n = int(round(BURST_S * sample_rate))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    band = (freqs >= LOW_HZ) & (freqs <= min(HIGH_HZ, 0.45 * sample_rate))
    fade = max(1, int(0.002 * sample_rate))
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 2
    out = []
    for k in range(count):
        rng = np.random.default_rng(seed + k)
        spectrum = np.fft.rfft(rng.standard_normal(n))
        x = np.fft.irfft(spectrum * band, n)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
        x *= 10 ** (level_db / 20) / max(float(np.max(np.abs(x))), 1e-9)
        out.append(x.astype(np.float32))
    return out


@dataclass(frozen=True)
class LoopbackResult:
    ok: bool
    sample_rate: int
    round_trip: int | None
    """Samples from the processor's output back to its input (the median over the found bursts)."""
    delays: list[int]
    """Each found burst's round trip, in samples."""
    pings: int
    snr_db: float | None
    """The weakest found burst's correlation peak over the noise floor."""
    input_peak_db: float
    clipped: bool
    reason: str | None = None
    """no_signal (too few bursts came back) or inconsistent (they disagree)."""

    @property
    def round_trip_ms(self) -> float | None:
        return None if self.round_trip is None else 1000.0 * self.round_trip / self.sample_rate

    @property
    def jitter_ms(self) -> float | None:
        return 1000.0 * (max(self.delays) - min(self.delays)) / self.sample_rate if self.delays else None


def find_delay(recording: np.ndarray, burst: np.ndarray, start: int, max_delay: int) -> tuple[int, float]:
    """The lag (0…``max_delay``) at which ``burst``, played at ``start``, best matches ``recording``,
    and the peak's ratio to the correlation's noise floor in dB."""
    n = burst.shape[0]
    region = recording[start : start + max_delay + n]
    lags = region.shape[0] - n + 1
    if lags <= 0:
        return 0, -math.inf
    size = 1 << int(np.ceil(np.log2(region.shape[0] + n)))
    corr = np.fft.irfft(np.fft.rfft(region, size) * np.conj(np.fft.rfft(burst, size)), size)[:lags]
    mag = np.abs(corr)
    lag = int(np.argmax(mag))
    # The floor is the median magnitude away from the peak (reflections near it are signal, not noise).
    guard = max(1, n // 20)
    rest = np.concatenate([mag[: max(0, lag - guard)], mag[lag + guard :]])
    floor = float(np.median(rest)) if rest.size else 0.0
    if floor <= 0.0:
        return lag, math.inf if mag[lag] > 0 else -math.inf
    return lag, 20.0 * math.log10(float(mag[lag]) / floor)


def analyse(recording: np.ndarray, sent: list[np.ndarray], starts: list[int], sample_rate: int, max_delay: int) -> LoopbackResult:
    """Find every burst in ``recording`` (indexed as the processor's input) and judge the result."""
    found: list[tuple[int, float]] = []
    for burst, start in zip(sent, starts, strict=True):
        lag, snr = find_delay(recording, burst, start, max_delay)
        if snr >= MIN_SNR_DB:
            found.append((lag, snr))
    level = round(float(peak_db(recording)), 1) if recording.size else -120.0
    clipped = bool(level >= -0.1)  # plain Python values: the result crosses a process boundary as JSON

    def result(ok: bool, round_trip: int | None, delays: list[int], snr_db: float | None, reason: str | None) -> LoopbackResult:
        return LoopbackResult(ok, sample_rate, round_trip, delays, len(sent), snr_db, level, clipped, reason)

    if len(found) < max(2, (len(sent) + 1) // 2):
        return result(False, None, [d for d, _ in found], None, "no_signal")
    delays = [d for d, _ in found]
    snr = round(float(min(s for _, s in found)), 1)
    median = int(np.median(delays))
    if 1000.0 * (max(delays) - min(delays)) / sample_rate > MAX_SPREAD_MS:
        return result(False, median, delays, snr, "inconsistent")
    return result(True, median, delays, snr, None)


class LoopbackProbe:
    """An ``AudioSession`` processor: records each input block and answers with the next stretch of
    the test signal (silence, ``pings`` bursts ``INTERVAL_S`` apart, then silence while the last
    one comes back). ``done`` once ``max_delay_s`` has passed after the last burst."""

    def __init__(self, sample_rate: int, max_delay_s: float, pings: int = 4, level_db: float = -12.0) -> None:
        self.sample_rate = int(sample_rate)
        self.sent = bursts(self.sample_rate, pings, level_db)
        lead, interval = int(LEAD_S * sample_rate), int(INTERVAL_S * sample_rate)
        self.starts = [lead + k * interval for k in range(pings)]
        self.max_delay = int(max_delay_s * sample_rate)
        self.total = self.starts[-1] + self.sent[-1].shape[0] + self.max_delay
        self.signal = np.zeros(self.total, dtype=np.float32)
        for burst, start in zip(self.sent, self.starts, strict=True):
            self.signal[start : start + burst.shape[0]] = burst
        self.recording = np.zeros(self.total, dtype=np.float32)
        self.position = 0
        self.finished = threading.Event()

    def __call__(self, block: np.ndarray) -> np.ndarray:
        n = int(block.shape[0])
        pos = self.position
        take = max(0, min(n, self.total - pos))
        self.recording[pos : pos + take] = block[:take]
        out = np.zeros(n, dtype=np.float32)
        out[:take] = self.signal[pos : pos + take]
        self.position = pos + n
        if self.position >= self.total:
            self.finished.set()
        return out

    def result(self) -> LoopbackResult:
        return analyse(self.recording, self.sent, self.starts, self.sample_rate, self.max_delay)


@dataclass
class LoopbackRun:
    """What one measurement saw, besides the result: the session's shape and the latency PortAudio reported."""

    result: LoopbackResult
    topology: str
    sample_rate: int
    block: int
    reported_ms: tuple[float, float]
    underruns: int = 0
    overruns: int = 0
    lost: list[str] = field(default_factory=list)


def max_delay_s(block: int, sample_rate: int) -> float:
    """The longest round trip looked for: a second, or more with long blocks (prefill and rings hold several)."""
    return max(1.0, 4.0 * block / sample_rate + 0.5)


def measure(backend: Backend, config: SessionConfig, block: int, pings: int = 4, level_db: float = -12.0, timeout_scale: float = 1.0) -> LoopbackRun:
    """Open ``config``'s input and output as a live session would and measure the round trip.

    Raises ``DeviceOpenError`` when a device refuses, or when one is lost during the measurement.
    ``timeout_scale`` shortens the wall-clock limit for a sped-up fake backend.
    """
    config = SessionConfig(input=config.input, output=config.output)  # no monitor, no gain
    lost: list[str] = []
    session = AudioSession(backend, config, block, on_device_lost=lambda role, _id, _reason, message: lost.append(f"{role}: {message}"))
    probe = LoopbackProbe(session.sample_rate, max_delay_s(session.block, session.sample_rate), pings, level_db)
    session.processor = probe
    session.start()
    try:
        limit = time.monotonic() + (probe.total / session.sample_rate * 2 + 5) * timeout_scale
        while not probe.finished.wait(0.05):
            if lost or not session.running:
                raise DeviceOpenError(lost[0] if lost else "The audio stream stopped", "missing")
            if time.monotonic() > limit:
                raise DeviceOpenError("The audio devices did not deliver audio in time", "busy")
        reported = session.latency_ms
    finally:
        session.stop()
    return LoopbackRun(probe.result(), session.topology, session.sample_rate, session.block, reported, session.underruns, session.overruns, lost)
