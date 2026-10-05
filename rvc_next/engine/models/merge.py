"""Merge two models by weight (the original's ``merge``)."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from rvc_next.engine.errors import ModelFormatError
from rvc_next.engine.models.checkpoint import SmallModel, normalize_speaker_info, torch_load, write_small_model


def _weights(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and "model" in data:
        return OrderedDict((k, v) for k, v in data["model"].items() if "enc_q" not in k)
    if isinstance(data, dict) and "weight" in data:
        return data["weight"]
    raise ModelFormatError("Not an RVC model")


def merge_models(a: Path | str, b: Path | str, dst: Path | str, *, alpha: float, info: str = "") -> SmallModel:
    """Write ``alpha·a + (1−alpha)·b``.

    The speaker embeddings may differ in rows; the merge keeps the shorter, as the original does.
    The sample rate, version and pitch guidance come from ``a``, and both must match.
    """
    data_a, data_b = torch_load(a), torch_load(b)
    if "config" not in data_a:
        raise ModelFormatError(f"{Path(a).name} is not a small model; extract it first")
    for key in ("version", "f0"):
        if key in data_b and data_a.get(key) != data_b.get(key):
            raise ModelFormatError(f"The models differ in {'version' if key == 'version' else 'pitch guidance'}")
    if "config" in data_b and int(data_a["config"][-1]) != int(data_b["config"][-1]):
        raise ModelFormatError("The models have different sample rates")
    wa, wb = _weights(data_a), _weights(data_b)
    if sorted(wa) != sorted(wb):
        raise ModelFormatError("The two models have different structures")
    weight = OrderedDict()
    for key in wa:
        if key == "emb_g.weight" and wa[key].shape != wb[key].shape:
            n = min(wa[key].shape[0], wb[key].shape[0])
            weight[key] = (alpha * wa[key][:n].float() + (1 - alpha) * wb[key][:n].float()).half()
        else:
            weight[key] = (alpha * wa[key].float() + (1 - alpha) * wb[key].float()).half()
    info_a = normalize_speaker_info(data_a.get("speaker_info", []))
    info_b = normalize_speaker_info(data_b.get("speaker_info", []))
    model = SmallModel(
        weight=weight,
        config=list(data_a["config"]),
        info=info,
        sr=data_a.get("sr", ""),
        pitch_guidance=bool(int(data_a.get("f0", 1))),
        version=str(data_a.get("version", "v1")),
        speaker_info=info_a if info_a and info_a == info_b else [],
    )
    write_small_model(model, dst)
    return model
