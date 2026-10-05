"""Test sounds for checking an output before starting."""

from __future__ import annotations

import numpy as np


def chime(sample_rate: int, seconds: float = 0.9, gain_db: float = -12.0) -> np.ndarray:
    """Three soft rising notes (C6, E6, G6) with exponential decay, mono float32."""
    sr = int(sample_rate)
    out = np.zeros(int(seconds * sr), dtype=np.float32)
    amp = 10 ** (gain_db / 20)
    for i, freq in enumerate((1046.5, 1318.5, 1568.0)):
        start = int(i * 0.15 * sr)
        n = out.shape[0] - start
        t = np.arange(n) / sr
        note = np.sin(2 * np.pi * freq * t) * np.exp(-t * 6.0)
        attack = min(n, int(0.005 * sr))
        note[:attack] *= np.linspace(0.0, 1.0, attack)
        out[start:] += (amp / 3 * note).astype(np.float32)
    fade = min(out.shape[0], int(0.02 * sr))
    out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out
