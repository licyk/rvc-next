"""RMVPE above ~1040 Hz: an octave-class corrector (from Applio's ``rvc/lib/predictors/f0.py``, MIT).

``rmvpe.pt`` was trained on vocal pitch corpora that end near C6, so for higher fundamentals it
confidently reports f/2 or f/3. A second pass on the audio resampled to 32 kHz and read as 16 kHz
plays everything an octave lower, into the range RMVPE tracks; doubled again, that guide fixes the
octave class of the normal pass, whose finer 10 ms contour is kept wherever the two agree. Frames
whose guide is below 460 Hz are never changed, so speech and most singing are untouched.

Modes: ``true_pitch`` writes the true pitch, up to ``ceiling`` (RVC voices trained on stock RMVPE
labels cannot synthesise much above 1250 Hz; above it the stock octave-low drive is kept, and the
vocoder folds its harmonics back up). ``fold`` only repairs wrong pitch classes, with half the true
pitch, so every drive stays in the register such voices were trained on.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MODES = ("off", "true_pitch", "fold")
GATE_CENTS = 150.0


def half_speed_guide(x: np.ndarray, n_frames: int, rmvpe: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """The guide pass: RMVPE on ``x`` upsampled ×2 (so pitched an octave down), doubled back, per normal frame."""
    import librosa

    y2 = librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=16000, target_sr=32000, res_type="soxr_vhq")
    f0h = rmvpe(y2)
    # Normal frame i is guide frame 2i (the upsampled audio has twice the frames).
    j = np.minimum(np.arange(n_frames) * 2, len(f0h) - 1)
    return np.where(f0h[j] > 0, 2.0 * f0h[j], 0.0)


def fix_high_register(normal: np.ndarray, guide: np.ndarray, mode: str = "true_pitch", ceiling: float = 1250.0, gate: float = GATE_CENTS) -> np.ndarray:
    """Correct the octave class of ``normal`` (Hz per frame, 0 unvoiced) from ``guide``."""
    out = np.array(normal, dtype=np.float64, copy=True)
    if mode == "off":
        return out
    nv, gv = normal > 0, guide > 0
    both = nv & gv
    c_keep = np.full(len(normal), 1e9)
    c_double = np.full(len(normal), 1e9)
    c_keep[both] = np.abs(1200.0 * np.log2(normal[both] / guide[both]))
    c_double[both] = np.abs(1200.0 * np.log2(2.0 * normal[both] / guide[both]))
    agree = both & (c_keep < gate)
    doubled = both & ~agree & (c_double < gate)
    other = both & ~agree & ~doubled
    if mode == "fold":
        for frames in (other & (guide >= 990.0), ~nv & gv & (guide >= 900.0)):
            out[frames] = guide[frames] / 2.0
        return out
    if mode != "true_pitch":
        raise ValueError(f"Unknown high-register mode: {mode}")
    dbl = doubled & (guide >= 460.0) & (2.0 * normal <= ceiling)
    other = other & (guide >= 990.0) & (guide <= ceiling)
    filled = ~nv & gv & (guide >= 460.0) & (guide <= ceiling)
    out[dbl] = 2.0 * normal[dbl]
    out[other] = guide[other]
    out[filled] = guide[filled]
    return out
