"""Praat autocorrelation pitch (``pm``)."""

from __future__ import annotations

import numpy as np

from rvc_next.engine.f0.base import F0_MAX, F0_MIN, SAMPLE_RATE, WINDOW


class PmProvider:
    name = "pm"

    def compute(self, x: np.ndarray, p_len: int) -> np.ndarray:
        import parselmouth  # ty: ignore[unresolved-import]  # a compiled extension without stubs

        time_step = WINDOW / SAMPLE_RATE
        f0 = parselmouth.Sound(x, SAMPLE_RATE).to_pitch_ac(time_step=time_step, voicing_threshold=0.6, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX).selected_array["frequency"]
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return f0
