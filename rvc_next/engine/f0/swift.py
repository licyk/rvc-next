"""SwiftF0 as an ``F0Provider``: the ported model (``swift_model``, weights shipped in the package)
plus Applio's subharmonic repair, resampled from SwiftF0's 16 ms frames to RVC's 10 ms.

SwiftF0 sometimes reads a note an octave (or a twelfth) low for a few frames with middling
confidence. The repair (Applio, ``rvc/lib/predictors/f0.py``, MIT) lifts such a run back when a
confident frame before it and one after it agree with the lifted contour.
"""

from __future__ import annotations

from typing import Any

import numpy as np

F0_MIN = 50.0
F0_MAX = 1100.0
THRESHOLD = 0.5


def repair_subharmonics(pitch: np.ndarray, confidence: np.ndarray, frame_period: float) -> tuple[np.ndarray, np.ndarray]:
    """Lift runs of doubtful frames sitting at f/2 or f/3 between confident frames; returns
    ``(pitch, repaired)``."""
    pitch = np.asarray(pitch, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    corrected = pitch.copy()
    repaired = np.zeros(len(pitch), dtype=bool)
    n = len(pitch)
    i = 1
    while i < n - 1:
        if confidence[i - 1] < 0.5 or not 0.3 <= confidence[i] < 0.95:
            i += 1
            continue
        ratio = pitch[i - 1] / pitch[i]
        factor = min((2, 3), key=lambda value: abs(np.log2(ratio / value)))
        if abs(1200 * np.log2(ratio / factor)) > 100:
            i += 1
            continue
        j = i
        while j < n and (j - i) * frame_period < 1.0:
            if not 0.3 <= confidence[j] < 0.95:
                break
            previous = pitch[i - 1] if j == i else pitch[j - 1] * factor
            if abs(1200 * np.log2(pitch[j] * factor / previous)) > 100:
                break
            j += 1
        if i < j < n and confidence[j] >= 0.5:
            return_error = abs(1200 * np.log2(pitch[j] / (pitch[j - 1] * factor)))
            anchor_error = abs(1200 * np.log2(pitch[j] / pitch[i - 1]))
            if return_error <= 100 and anchor_error <= 150:
                corrected[i:j] *= factor
                repaired[i:j] = True
                i = j + 1
                continue
        i += 1
    return corrected, repaired


class SwiftProvider:
    name = "swift"

    def __init__(self, device: Any) -> None:
        from rvc_next.engine.f0.swift_model import load_swift_f0

        # DirectML lacks the float64 decoder; the model is tiny, so it runs on the CPU there.
        self.model = load_swift_f0(device="cpu" if "privateuseone" in str(device) else device)

    def compute(self, x: np.ndarray, p_len: int) -> np.ndarray:
        import torch

        from rvc_next.engine.f0.swift_model import FRAME_PERIOD

        n = p_len if p_len > 0 else x.shape[0] // 160 + 1
        with torch.no_grad():
            track = self.model.detect(np.asarray(x, dtype=np.float32), fmin=F0_MIN, fmax=F0_MAX)
        pitch, repaired = repair_subharmonics(track.pitch_hz, track.confidence, FRAME_PERIOD)
        repaired &= (pitch >= F0_MIN) & (pitch <= F0_MAX)
        pitch = np.where(repaired, pitch, track.pitch_hz)
        confidence = np.where(repaired, np.maximum(track.confidence, THRESHOLD), track.confidence)
        voiced = confidence >= THRESHOLD
        if not voiced.any():
            return np.zeros(n, dtype=np.float64)
        # Log-frequency interpolation onto the 10 ms grid, bridging gaps; frames whose
        # interpolated confidence is below the threshold are unvoiced.
        t = np.arange(n) * 0.01
        f0 = np.exp2(np.interp(t, track.timestamps[voiced], np.log2(pitch[voiced])))
        f0[np.interp(t, track.timestamps, confidence) < THRESHOLD] = 0.0
        return f0
