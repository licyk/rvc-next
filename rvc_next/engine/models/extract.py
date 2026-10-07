"""Extract a small model from a G checkpoint (the original's ``extract_small_model``)."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from rvc_next.engine.errors import ModelFormatError
from rvc_next.engine.models.checkpoint import SmallModel, is_g_checkpoint, legacy_weight_norm, normalize_speaker_info, torch_load, write_small_model

_RES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


def _config(n_fft_bins: int, upsample_rates: list[int], kernels: list[int], sr: int) -> list[Any]:
    return [n_fft_bins, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], _RES, upsample_rates, 512, kernels, 109, 256, sr]


def config_for(sample_rate: str, version: str) -> list[Any]:
    """The positional config the original writes for each rate and version.

    v2 at 40k uses the v1 40k layout, as the original's ``configs/v1/40k.json`` does.
    """
    if sample_rate == "40k":
        return _config(1025, [10, 10, 2, 2], [16, 16, 4, 4], 40000)
    if sample_rate == "48k":
        if version == "v1":
            return _config(1025, [10, 6, 2, 2, 2], [16, 16, 4, 4, 4], 48000)
        return _config(1025, [12, 10, 2, 2], [24, 20, 4, 4], 48000)
    if sample_rate == "32k":
        if version == "v1":
            return _config(513, [10, 4, 2, 2, 2], [16, 16, 4, 4, 4], 32000)
        return _config(513, [10, 8, 2, 2], [20, 16, 4, 4], 32000)
    raise ModelFormatError(f"Unknown sample rate {sample_rate!r}")


def extract_small_model(
    checkpoint: Path | str,
    dst: Path | str,
    *,
    sample_rate: str,
    pitch_guidance: bool,
    version: str,
    info: str = "",
    speaker_info: Any = None,
) -> SmallModel:
    data = torch_load(checkpoint)
    if is_g_checkpoint(data):
        state = legacy_weight_norm(data["model"])
    elif isinstance(data, dict) and any(k.startswith("dec.") for k in data):
        state = legacy_weight_norm(data)
    else:
        raise ModelFormatError(f"{Path(checkpoint).name} is not a G checkpoint")
    weight = OrderedDict((k, v.half()) for k, v in state.items() if "enc_q" not in k)
    model = SmallModel(
        weight=weight,
        config=config_for(sample_rate, version),
        info=info or "Extracted from a training checkpoint",
        sr=sample_rate,
        pitch_guidance=pitch_guidance,
        version=version,
        speaker_info=normalize_speaker_info(speaker_info),
    )
    write_small_model(model, dst)
    return model
