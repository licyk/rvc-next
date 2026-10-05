"""Sample-rate conversion."""

from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """librosa's default (soxr high quality), as the original uses after conversion."""
    if orig_sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, axis=-1).astype(np.float32, copy=False)
