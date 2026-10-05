"""Small signal-processing helpers shared by the offline and streaming paths."""

from __future__ import annotations

import numpy as np
from scipy import signal

# 48 Hz fifth-order Butterworth high-pass at 16 kHz, applied before feature extraction.
HIGHPASS_B, HIGHPASS_A = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def highpass_16k(audio: np.ndarray) -> np.ndarray:
    return signal.filtfilt(HIGHPASS_B, HIGHPASS_A, audio)


def change_rms(data1: np.ndarray, sr1: int, data2: np.ndarray, sr2: int, rate: float) -> np.ndarray:
    """Mix the source loudness envelope into the output (the original's ``change_rms``).

    ``data1`` is the input, ``data2`` the output; ``rate`` is the output envelope's share.
    """
    import librosa
    import torch
    import torch.nn.functional as F

    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1_t = F.interpolate(torch.from_numpy(rms1).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2_t = F.interpolate(torch.from_numpy(rms2).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2_t = torch.max(rms2_t, torch.zeros_like(rms2_t) + 1e-6)
    data2 *= (torch.pow(rms1_t, torch.tensor(1 - rate)) * torch.pow(rms2_t, torch.tensor(rate - 1))).numpy()
    return data2


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def peak_db(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return 20 * np.log10(peak) if peak > 1e-6 else -120.0


def rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20 * np.log10(rms) if rms > 1e-6 else -120.0
