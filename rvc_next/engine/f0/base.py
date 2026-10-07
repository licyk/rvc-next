"""The pitch-provider interface and the post-processing every path shares.

The original repeated the shift, the unvoiced interpolation and the 256-bin coarse quantisation in
the offline pipeline, the realtime engine and the training extractor; here they exist once.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

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
AUTO_PITCH_MODES = ("off", "semitone", "octave")
AUTO_PITCH_LIMIT = 24


def autotune(f0: np.ndarray, voiced: np.ndarray, strength: float, offset: float = 0.0) -> np.ndarray:
    """Pull voiced frames toward the nearest equal-tempered note (A4 = 440 Hz), in place.

    ``strength`` 1 snaps, 0.5 goes half way (in log frequency). ``offset`` (semitones) is added to
    the pitch the listener hears (the formant shift raises it back after synthesis), so the snap
    lands in tune there. Unvoiced frames keep their value, so consonants never get a note.
    """
    if strength <= 0:
        return f0
    sel = voiced & (f0 > 0)
    if not sel.any():
        return f0
    midi = 69 + 12 * np.log2(f0[sel] / 440.0) + offset
    f0[sel] *= np.power(2.0, strength * (np.round(midi) - midi) / 12)
    return f0


def auto_key(f0: np.ndarray, voiced: np.ndarray, target_hz: float, mode: str) -> int:
    """Semitones that move the input's median pitch (over voiced frames, in log frequency) onto
    ``target_hz``: rounded to a semitone, or to whole octaves so the song keeps its key."""
    sel = voiced & (f0 > 0)
    if mode == "off" or target_hz <= 0 or sel.sum() < 2:
        return 0
    median = float(np.exp2(np.median(np.log2(f0[sel]))))
    semitones = 12 * np.log2(target_hz / median)
    key = 12 * round(semitones / 12) if mode == "octave" else round(semitones)
    return int(max(-AUTO_PITCH_LIMIT, min(AUTO_PITCH_LIMIT, key)))


class OfflinePitch(NamedTuple):
    coarse: np.ndarray
    hz: np.ndarray
    voiced: np.ndarray
    """The frames the detector found voiced."""
    key: int = 0
    """Semitones the automatic key added (0 when it is off)."""


def offline_f0(
    provider: F0Provider,
    x: np.ndarray,
    p_len: int,
    semitones: float,
    unvoiced: str = "original",
    *,
    high_register: str = "off",
    ceiling: float = 1250.0,
    autotune_strength: float = 0.0,
    formant: float = 0.0,
    auto_pitch: str = "off",
    auto_pitch_target: float = 155.0,
) -> OfflinePitch:
    """The offline converter's pitch: compute, interpolate unvoiced frames, shift, quantise.

    Returns ``coarse`` and ``hz`` as the original's ``Pipeline.get_f0``, with the frames the
    detector found voiced. ``unvoiced = "zero"`` keeps 0 Hz there instead of filling them, as
    classic RVC (and Applio) convert and train. ``high_register`` corrects RMVPE's octave errors
    above ~1040 Hz (``high_register.py``; other methods ignore it). ``auto_pitch`` adds the key that
    brings the median pitch to ``auto_pitch_target``; ``autotune_strength`` pulls notes to the
    nearest semitone of the pitch heard after the ``formant`` shift. With the defaults the
    arithmetic is the original's.
    """
    f0 = provider.compute(x, p_len)
    corrector = getattr(provider, "high_register", None)
    if high_register != "off" and corrector is not None:
        f0 = corrector(x, f0, high_register, ceiling)
    voiced = f0 > 0
    key = auto_key(f0, voiced, auto_pitch_target, auto_pitch)
    if unvoiced != "zero":
        f0 = interpolate_unvoiced(f0)
    f0 = shift(f0, semitones + key)
    if autotune_strength > 0:
        f0 = autotune(f0, voiced, autotune_strength, formant)
    f0bak = f0.copy()
    return OfflinePitch(to_coarse(f0), f0bak, voiced, key)
