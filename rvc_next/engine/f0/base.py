"""The pitch-provider interface and the post-processing every path shares.

The original repeated the shift, the unvoiced interpolation and the 256-bin coarse quantisation in
the offline pipeline, the realtime engine and the training extractor; here they exist once.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 160
F0_MIN = 50.0
F0_MAX = 1100.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


class F0Provider(Protocol):
    """Pitch in Hz, one value per 10 ms frame of 16 kHz audio; 0 where unvoiced."""

    name: str

    def compute(self, x: np.ndarray, p_len: int) -> np.ndarray: ...


def interpolate_unvoiced(f0: np.ndarray) -> np.ndarray:
    """Fill unvoiced frames by linear interpolation from voiced neighbours, in place (offline rule)."""
    uv = f0 == 0
    if uv.any() and (~uv).any():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return f0


def shift(f0: np.ndarray, semitones: float) -> np.ndarray:
    """Multiply by ``2^(semitones/12)``, in place."""
    f0 *= pow(2, semitones / 12)
    return f0


def to_coarse(f0: np.ndarray) -> np.ndarray:
    """Quantise to the 1..255 mel bins the synthesizer's pitch embedding reads."""
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int32)


UNVOICED_MODES = ("original", "protect", "zero")


def offline_f0(provider: F0Provider, x: np.ndarray, p_len: int, semitones: float, unvoiced: str = "original") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The offline converter's pitch: compute, interpolate unvoiced frames, shift, quantise.

    Returns ``(coarse, hz, voiced)``: ``coarse`` and ``hz`` as the original's ``Pipeline.get_f0``,
    and the frames the detector found voiced. ``unvoiced = "zero"`` keeps 0 Hz there instead of
    filling them, as classic RVC (and Applio) convert and train.
    """
    f0 = provider.compute(x, p_len)
    voiced = f0 > 0
    if unvoiced != "zero":
        f0 = interpolate_unvoiced(f0)
    f0 = shift(f0, semitones)
    f0bak = f0.copy()
    return to_coarse(f0), f0bak, voiced
