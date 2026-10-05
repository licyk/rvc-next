"""Clock-drift correction for a split-stream ring.

Two devices never run at exactly the same rate. The reader of a ring filled by another device's
clock watches the filtered fill level; when it drifts more than half a block from its target, it
drops or duplicates one sample per 10 ms, at the quietest sample of each 10 ms stretch, until the
level is back within an eighth of a block. At real clock drift (tens of ppm) that is a handful of
samples a second, which is inaudible.
"""

from __future__ import annotations

import numpy as np

from rvc_next.engine.audio_io.rings import Ring


class DriftCorrector:
    def __init__(self, ring: Ring, sample_rate: int, block: int, target: int, alpha: float = 0.05, enabled: bool = True, settle_seconds: float = 1.0) -> None:
        self.ring = ring
        self.sample_rate = int(sample_rate)
        self.block = int(block)
        self.target = int(target)
        self.alpha = alpha
        self.level = float(target)
        self._correcting = 0
        """+1 while dropping, -1 while duplicating, 0 when settled."""
        self.dropped = 0
        self.duplicated = 0
        self.consumed = 0
        self.underruns = 0
        self._last = 0.0
        self.enabled = enabled
        self._settle_samples = int(settle_seconds * sample_rate)
        self._settle = self._settle_samples
        """Samples read before the target is taken from the measured level: where the fill settles
        depends on the phase between the two callbacks, which no constant can know. Restarted after
        an underrun or a trim, so a stall never becomes the target."""

    def resettle(self) -> None:
        """Measure the level afresh and take the target from it once it has settled (after the ring was trimmed)."""
        self.level = float(self.ring.available)
        self._settle = self._settle_samples
        self._correcting = 0

    @property
    def drift_ppm(self) -> float | None:
        """Net correction in parts per million of the samples read (positive: the writer's clock is fast)."""
        if self.consumed < self.sample_rate:
            return None
        return (self.dropped - self.duplicated) / self.consumed * 1e6

    def _steps(self, n: int) -> int:
        return max(1, n * 100 // self.sample_rate)

    def read(self, n: int) -> np.ndarray:
        """Return exactly ``n`` samples for the device callback. On an underrun the missing part is silence after a short fade."""
        self.level += self.alpha * (self.ring.available - self.level)
        if self._settle > 0:
            self._settle -= n
            if self._settle <= 0:
                self.target = int(round(self.level))
        error = self.level - self.target
        if not self.enabled or self._settle > 0:
            self._correcting = 0
        elif self._correcting == 0 and abs(error) > self.block / 2:
            self._correcting = 1 if error > 0 else -1
        elif self._correcting and abs(error) < self.block / 8:
            self._correcting = 0
        k = self._steps(n) if self._correcting else 0
        want = n + k * self._correcting
        raw = np.zeros(want, dtype=np.float32)
        got = self.ring.read_into(raw, want)
        self.consumed += got
        if got < want:
            self.underruns += 1
            self._settle = self._settle_samples
            out = np.zeros(n, dtype=np.float32)
            take = min(got, n)
            out[:take] = raw[:take]
            fade = min(64, take)
            if fade:
                out[take - fade : take] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            self._last = 0.0
            return out
        if self._correcting > 0:
            out = _drop(raw, k)
            self.dropped += k
        elif self._correcting < 0:
            out = _duplicate(raw, k)
            self.duplicated += k
        else:
            out = raw
        self._last = float(out[-1]) if out.size else 0.0
        return out[:n]


def _quiet_positions(x: np.ndarray, k: int) -> list[int]:
    """One position per equal stretch of ``x``: its quietest sample."""
    edges = np.linspace(0, x.shape[0], k + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a <= 2:
            out.append(int(a))
            continue
        out.append(int(a + 1 + np.argmin(np.abs(x[a + 1 : b - 1]))))
    return out


def _drop(x: np.ndarray, k: int) -> np.ndarray:
    return np.delete(x, _quiet_positions(x, k))


def _duplicate(x: np.ndarray, k: int) -> np.ndarray:
    positions = _quiet_positions(x, k)
    return np.insert(x, positions, x[positions])
