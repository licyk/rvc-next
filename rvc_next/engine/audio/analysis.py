"""What a piece of audio looks like: a mel spectrogram for display, and its pitch curve.

Both are for people judging a result (the high-frequency cut-off of a 32k voice, artefacts,
separation bleed, octave errors, how far apart two voices sit), not for the models. The pitch comes
from SwiftF0 on the CPU, which is fast enough for whole songs and never waits for the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEC_ROWS = 128
SPEC_FLOOR_DB = -80.0
PITCH_HOP = 0.01


@dataclass
class Spectrogram:
    rows: int
    cols: int
    fmax: float
    data: np.ndarray
    """``uint8 [rows, cols]``, row 0 the lowest band: 0 is ``SPEC_FLOOR_DB`` below the loudest, 255 the loudest."""


def spectrogram(audio: np.ndarray, sample_rate: int, columns: int = 800, rows: int = SPEC_ROWS) -> Spectrogram:
    """A mel spectrogram of mono ``audio`` with about ``columns`` frames, scaled to 0–255."""
    import librosa

    n = audio.shape[0]
    columns = max(16, int(columns))
    hop = max(64, int(np.ceil(n / columns)))
    n_fft = 2048 if sample_rate > 24000 else 1024
    mel = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=rows, fmax=sample_rate / 2)
    db = librosa.power_to_db(mel, ref=np.max(mel) if mel.size and np.max(mel) > 0 else 1.0, top_db=-SPEC_FLOOR_DB)
    scaled = np.clip((db - SPEC_FLOOR_DB) / -SPEC_FLOOR_DB * 255.0, 0, 255).astype(np.uint8)
    return Spectrogram(rows=scaled.shape[0], cols=scaled.shape[1], fmax=sample_rate / 2, data=scaled)


def pitch_curve(audio_16k: np.ndarray) -> np.ndarray:
    """Hz per 10 ms frame of 16 kHz mono audio, 0 where unvoiced (SwiftF0 with its repair, on the CPU)."""
    from rvc_next.engine.f0.swift import SwiftProvider

    if audio_16k.size < 160:
        return np.zeros(0)
    return SwiftProvider("cpu").compute(np.asarray(audio_16k, dtype=np.float32), audio_16k.shape[0] // 160 + 1)


def median_pitch(f0: np.ndarray) -> float | None:
    voiced = f0[f0 > 0]
    return float(np.exp2(np.median(np.log2(voiced)))) if voiced.size else None
