"""CREPE as an ``F0Provider``: the ported network (``crepe_model``) with torchcrepe's weights, the
``crepe`` (full) and ``crepe-tiny`` assets, decoded and filtered as RVC and Applio use it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class CrepeProvider:
    def __init__(self, model_path: Path, capacity: str, device: Any) -> None:
        from rvc_next.engine.f0.crepe_model import load_crepe

        self.name = "crepe" if capacity == "full" else "crepe-tiny"
        self.device = device
        # DirectML lacks pieces the framing needs; the network is small enough for the CPU there.
        self.model_device = "cpu" if "privateuseone" in str(device) else device
        self.model = load_crepe(Path(model_path), capacity, self.model_device)  # ty: ignore[invalid-argument-type]

    def compute(self, x: np.ndarray, p_len: int) -> np.ndarray:
        from rvc_next.engine.f0.crepe_model import crepe_f0

        n = p_len if p_len > 0 else x.shape[0] // 160 + 1
        return crepe_f0(self.model, x, n, self.model_device).astype(np.float64)
