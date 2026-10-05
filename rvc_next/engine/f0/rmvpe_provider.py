"""RMVPE as an ``F0Provider``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rvc_next.engine.errors import MissingAssetError


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
        return self.model.infer_from_audio(x, thred=0.03)
