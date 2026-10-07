"""RMVPE as an ``F0Provider``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rvc_next.engine.errors import MissingAssetError

# The original runs RMVPE over the whole file at once; past this many frames (320 s) the mel, the
# U-Net and the bidirectional GRU outgrow a typical GPU, so longer audio goes in overlapping chunks.
CHUNK_FRAMES = 32000
OVERLAP_FRAMES = 300
HOP = 160


class RmvpeProvider:
    name = "rmvpe"

    def __init__(self, model_path: Path, device: Any, is_half: bool, onnx_path: Path | None = None) -> None:
        from rvc_next.engine.f0.rmvpe import RMVPE

        directml = "privateuseone" in str(device)
        needed = onnx_path if directml else model_path
        if needed is None or not Path(needed).is_file():
            raise MissingAssetError(f"RMVPE is not installed ({needed})", ["rmvpe-onnx" if directml else "rmvpe"])
        self.model = RMVPE(str(model_path), is_half=is_half, device=device, onnx_path=onnx_path)

    def compute(self, x: np.ndarray, p_len: int) -> np.ndarray:
        return self._infer(x)

    def _infer(self, x: np.ndarray) -> np.ndarray:
        """``infer_from_audio`` in one go (bit-identical to the original) up to ``CHUNK_FRAMES``; in
        chunks beyond, each with ``OVERLAP_FRAMES`` of context on both sides, keeping the middles."""
        n = x.shape[0] // HOP + 1
        if n <= CHUNK_FRAMES:
            return self.model.infer_from_audio(x, thred=0.03)
        out: np.ndarray | None = None
        for start in range(0, n, CHUNK_FRAMES):
            end = min(start + CHUNK_FRAMES, n)
            lo = max(0, start - OVERLAP_FRAMES)
            hi = min(n, end + OVERLAP_FRAMES)
            f0 = self.model.infer_from_audio(x[lo * HOP : hi * HOP], thred=0.03)
            part = f0[start - lo : start - lo + end - start]
            if out is None:
                out = np.zeros(n, dtype=part.dtype)
            out[start : start + len(part)] = part
        assert out is not None
        return out

    def high_register(self, x: np.ndarray, f0: np.ndarray, mode: str, ceiling: float) -> np.ndarray:
        """``f0`` (this provider's result for ``x``) with the octave errors above ~1040 Hz corrected."""
        from rvc_next.engine.f0.high_register import fix_high_register, half_speed_guide

        if mode == "off":
            return f0
        guide = half_speed_guide(x, len(f0), self._infer)
        return fix_high_register(f0, guide, mode, ceiling).astype(f0.dtype)
