"""Waveform peaks for display."""

from __future__ import annotations

import numpy as np


def compute_peaks(audio: np.ndarray, points: int) -> np.ndarray:
    """``points`` (min, max) pairs over ``audio`` (mono ``[T]`` or ``[C, T]``), interleaved."""
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    points = max(1, int(points))
    n = audio.shape[0]
    out = np.zeros(points * 2, dtype=np.float32)
    if n == 0:
        return out
    edges = np.linspace(0, n, points + 1).astype(np.int64)
    for i in range(points):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        seg = audio[a : min(b, n)]
        if seg.size:
            out[2 * i] = float(seg.min())
            out[2 * i + 1] = float(seg.max())
    return np.clip(out, -1.0, 1.0)
