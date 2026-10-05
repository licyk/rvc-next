"""Feature stage: HuBERT features for every 16 kHz slice (``3_feature256`` for v1, ``3_feature768`` for v2).

Ported from the original ``train/dataset/extract_hubert_feature.py``; the same feature function as
inference, so normalisation follows the asset's ``preprocessor_config.json`` in both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rvc_next.engine.cancel import CancelToken
from rvc_next.engine.train.parallel import Log, PartReporter, Progress, merge_counts, run_parts


def feature_dir_name(version: str) -> str:
    return "3_feature256" if version == "v1" else "3_feature768"


@dataclass(frozen=True)
class FeatureRequest:
    exp_dir: str
    version: str = "v2"
    assets_dir: str = ""
    device: str = "cpu"
    is_half: bool = False
    gpus: tuple[int, ...] = ()
    """With more than one, one process per card."""
    clean: bool = False


def _part(version: str, assets_dir: str, device: str, is_half: bool, items: list[tuple[str, str]], reporter: PartReporter) -> dict[str, Any]:
    import soundfile as sf
    import torch

    from rvc_next.engine.features.hubert import extract_features, load_hubert

    dev: Any = device
    if device == "dml":
        from rvc_next.engine.runtime import directml_device

        dev = directml_device()
    half = is_half and str(device).startswith("cuda")
    model = load_hubert(Path(assets_dir) / "hubert_base", dev, half)
    done = failed = 0
    for wav_path, out_path in items:
        try:
            wav, sr = sf.read(wav_path, dtype="float32")
            if sr != 16000:
                raise ValueError("not 16 kHz")
            feats = torch.from_numpy(wav).float()
            if feats.dim() == 2:
                feats = feats.mean(-1)
            feats = feats.view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
            source = feats.half().to(dev) if half else feats.to(dev)
            with torch.no_grad():
                out = extract_features(model, source, version, padding_mask=padding_mask.to(dev))
            arr = out.squeeze(0).float().cpu().numpy()
            if np.isnan(arr).any():
                failed += 1
                reporter.log(f"Features contain NaN, skipped: {os.path.basename(wav_path)}")
            else:
                np.save(out_path, arr, allow_pickle=False)
                done += 1
        except Exception as e:
            failed += 1
            reporter.log(f"Features failed for {os.path.basename(wav_path)}: {e}")
        reporter.done(os.path.basename(wav_path))
    return {"done": done, "failed": failed}


def run(request: FeatureRequest, progress: Progress | None = None, log: Log | None = None, cancel: CancelToken | None = None) -> dict[str, Any]:
    from rvc_next.engine.errors import MissingAssetError

    exp = Path(request.exp_dir)
    src = exp / "1_16k_wavs"
    out = exp / feature_dir_name(request.version)
    if request.clean and out.is_dir():
        for p in out.glob("*.npy"):
            p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    hubert = Path(request.assets_dir) / "hubert_base"
    if not (hubert / "config.json").is_file():
        raise MissingAssetError(f"HuBERT is not installed in {hubert}", ["hubert"])
    names = sorted(n for n in os.listdir(src) if n.endswith(".wav")) if src.is_dir() else []
    if not names:
        raise ValueError("Slice the dataset first: 1_16k_wavs is empty")
    todo = [(str(src / n), str(out / (os.path.splitext(n)[0] + ".npy"))) for n in names if not (out / (os.path.splitext(n)[0] + ".npy")).exists()]
    if log:
        log(f"Features ({request.version}) for {len(todo)} of {len(names)} slices")
    env: list[dict[str, str] | None] | None = None
    if len(request.gpus) > 1:
        n = len(request.gpus)
        parts = [(request.version, request.assets_dir, "cuda:0", request.is_half, todo[i::n]) for i in range(n)]
        env = [{"CUDA_VISIBLE_DEVICES": str(g)} for g in request.gpus]
    else:
        parts = [(request.version, request.assets_dir, request.device, request.is_half, todo)]
    parts = [p for p in parts if p[-1]]
    counts = merge_counts(run_parts(_part, parts, len(todo), progress, log, cancel, "features", env))
    return {"done": counts.get("done", 0), "skipped": len(names) - len(todo), "failed": counts.get("failed", 0), "total": len(names)}
