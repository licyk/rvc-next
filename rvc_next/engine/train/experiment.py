"""Read-only helpers over an experiment folder, for the core's training service."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rvc_next.engine.train import multispeaker
from rvc_next.engine.train.hparams import read_config

STAGE_DIRS = {"slice": ["0_gt_wavs", "1_16k_wavs"], "f0": ["2a_f0", "2b-f0nsf"]}
LATEST_ONLY_STEP = 2333333
_RATE_NAMES = {32000: "32k", 40000: "40k", 48000: "48k"}


def feature_dir(exp_dir: Path | str, version: str) -> Path:
    return Path(exp_dir) / ("3_feature256" if version == "v1" else "3_feature768")


def pretrained_paths(assets_dir: Path | str, version: str, f0: bool, sample_rate: str) -> tuple[str, str]:
    """The base G and D models for this rate, version and pitch mode; "" for one that is not installed."""
    folder = Path(assets_dir) / ("pretrained_v2" if version == "v2" else "pretrained")
    prefix = "f0" if f0 else ""
    g = folder / f"{prefix}G{sample_rate}.pth"
    d = folder / f"{prefix}D{sample_rate}.pth"
    return (str(g) if g.is_file() else "", str(d) if d.is_file() else "")


def _non_empty(folder: Path, pattern: str = "*") -> bool:
    return folder.is_dir() and any(folder.glob(pattern))


def stage_outputs_exist(exp_dir: Path | str, version: str) -> dict[str, bool]:
    exp = Path(exp_dir)
    return {
        "slice": all(_non_empty(exp / d, "*.wav") for d in STAGE_DIRS["slice"]),
        "f0": all(_non_empty(exp / d, "*.npy") for d in STAGE_DIRS["f0"]),
        "features": _non_empty(feature_dir(exp, version), "*.npy"),
        "fit": _non_empty(exp, "G_*.pth"),
        "index": _non_empty(exp, "added_*.index"),
    }


def _step_of(name: str) -> int | None:
    m = re.match(r"^[GD]_(\d+)\.pth$", name)
    return int(m.group(1)) if m else None


def list_checkpoints(exp_dir: Path | str) -> list[dict[str, Any]]:
    """G and D checkpoints and small models (``weights/``), newest step last."""
    exp = Path(exp_dir)
    out: list[dict[str, Any]] = []
    for p in sorted(exp.glob("[GD]_*.pth")):
        step = _step_of(p.name)
        if step is None:
            continue
        epoch = None
        st = p.stat()
        out.append({"name": p.name, "path": str(p), "kind": p.name[0], "epoch": epoch, "step": step, "size": st.st_size, "mtime": st.st_mtime})
    weights = exp / "weights"
    if weights.is_dir():
        for p in sorted(weights.glob("*.pth")):
            m = re.search(r"_e(\d+)_s(\d+)$", p.stem)
            st = p.stat()
            out.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "kind": "small",
                    "epoch": int(m.group(1)) if m else None,
                    "step": int(m.group(2)) if m else None,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                }
            )
    out.sort(key=lambda c: (c["kind"] == "small", c["step"] if c["step"] is not None else 1 << 60, c["name"]))
    return out


def read_metrics(exp_dir: Path | str, since_step: int = 0) -> list[dict[str, Any]]:
    """Lines of ``metrics.jsonl`` whose step is at least ``since_step``; malformed lines are skipped."""
    path = Path(exp_dir) / "metrics.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and int(item.get("step", 0)) >= since_step:
            out.append(item)
    return out


def read_legacy(exp_dir: Path | str) -> dict[str, Any]:
    """What an original experiment folder (``logs/<exp>``) was trained with, as far as it can be told.

    Returns ``sample_rate`` ("40k" or None), ``version``, ``pitch_guidance``, ``multi_speaker``,
    ``speakers`` (``[{id, name}]``), ``speaker_rows`` (``[{name, id, folder, repeat}]`` when the
    manifest kept them) and ``stages`` (which outputs exist).
    """
    exp = Path(exp_dir)
    config = read_config(exp) or {}
    sr = config.get("data", {}).get("sampling_rate")
    sample_rate = _RATE_NAMES.get(int(sr)) if sr else None
    if _non_empty(exp / "3_feature768", "*.npy"):
        version = "v2"
    elif _non_empty(exp / "3_feature256", "*.npy"):
        version = "v1"
    else:
        # v1 and v2 differ in upsampling at 32k and 48k; 40k uses the same config for both.
        rates = config.get("model", {}).get("upsample_rates")
        version = "v2" if rates in ([10, 8, 2, 2], [12, 10, 2, 2]) else ("v1" if rates else "v2")
    pitch = _non_empty(exp / "2a_f0", "*.npy") or not _non_empty(exp / "3_feature768", "*.npy") and not _non_empty(exp / "3_feature256", "*.npy")
    manifest = None
    try:
        manifest = multispeaker.load_manifest(exp, check_files=False)
    except multispeaker.ManifestError:
        pass
    speakers = config.get("speaker_info") or (manifest or {}).get("speakers") or []
    rows = []
    for row in (manifest or {}).get("rows", []) or []:
        try:
            rows.append({"name": str(row["speaker_name"]), "id": int(row["speaker_id"]), "folder": str(row["path"]), "repeat": int(row.get("repeat", 1))})
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "sample_rate": sample_rate,
        "version": version,
        "pitch_guidance": bool(pitch),
        "multi_speaker": manifest is not None,
        "speakers": [{"id": int(s["id"]), "name": str(s["name"])} for s in speakers],
        "speaker_rows": rows,
        "stages": stage_outputs_exist(exp, version),
    }


def default_batch_size() -> int:
    """The smallest GPU's memory in GiB divided by 2 (the original's default); 1 without a card."""
    try:
        from rvc_next.engine.runtime import gpu_profiles

        mems = [p.memory_gb for p in gpu_profiles() if p.memory_gb]
        return max(1, int(min(mems) // 2)) if mems else 1
    except Exception:
        return 1


def default_cpu_workers() -> int:
    import os

    return max(1, int((os.cpu_count() or 1) / 1.5))
