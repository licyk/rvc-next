"""F0 stage: coarse (``2a_f0``) and continuous (``2b-f0nsf``) pitch for every 16 kHz slice.

Ported from the original ``train/dataset/extract_f0.py``. ``pm`` runs on the CPU split over
processes; ``rmvpe`` runs on one device, or one process per GPU. Training pitch is not shifted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rvc_next.engine.cancel import CancelToken
from rvc_next.engine.f0.base import to_coarse
from rvc_next.engine.train.parallel import Log, PartReporter, Progress, merge_counts, run_parts

F0_DIR = "2a_f0"
F0NSF_DIR = "2b-f0nsf"


@dataclass(frozen=True)
class F0Request:
    exp_dir: str
    method: str = "rmvpe"
    """pm or rmvpe."""
    assets_dir: str = ""
    device: str = "cpu"
    """cpu, cuda[:N], xpu[:N] or dml; ignored when ``gpus`` lists several cards."""
    is_half: bool = False
    n_workers: int = 1
    """CPU processes for pm."""
    gpus: tuple[int, ...] = ()
    """With more than one, rmvpe runs one process per card."""
    clean: bool = False


def _provider(method: str, assets_dir: str, device: str, is_half: bool) -> Any:
    if method == "pm":
        from rvc_next.engine.f0.pm import PmProvider

        return PmProvider()
    if method == "rmvpe":
        from rvc_next.engine.f0.rmvpe_provider import RmvpeProvider

        dev: Any = device
        if device == "dml":
            from rvc_next.engine.runtime import directml_device

            dev = directml_device()
        root = Path(assets_dir)
        return RmvpeProvider(root / "rmvpe" / "rmvpe.pt", dev, is_half, root / "rmvpe" / "rmvpe.onnx")
    raise ValueError(f"F0 method {method!r} is not offered for training (pm or rmvpe)")


def compute_f0(provider: Any, path: str) -> np.ndarray | None:
    """Pitch of one 16 kHz file with unvoiced frames interpolated; None when nothing is voiced."""
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{path} is not 16 kHz")
    f0 = np.asarray(provider.compute(x, x.shape[0] // 160), dtype=np.float64)
    uv = f0 == 0
    if uv.all():
        return None
    if uv.any():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return f0


def _part(method: str, assets_dir: str, device: str, is_half: bool, items: list[tuple[str, str, str]], reporter: PartReporter) -> dict[str, Any]:
    provider = _provider(method, assets_dir, device, is_half)
    done = skipped = failed = 0
    for inp, opt_coarse, opt_nsf in items:
        try:
            f0 = compute_f0(provider, inp)
            if f0 is None:
                skipped += 1
                reporter.log(f"No voiced frames, skipped: {os.path.basename(inp)}")
            else:
                np.save(opt_nsf, f0, allow_pickle=False)
                np.save(opt_coarse, to_coarse(f0.copy()).astype(np.int64), allow_pickle=False)
                done += 1
        except Exception as e:
            failed += 1
            reporter.log(f"F0 failed for {os.path.basename(inp)}: {e}")
        reporter.done(os.path.basename(inp))
    return {"done": done, "skipped": skipped, "failed": failed}


def run(request: F0Request, progress: Progress | None = None, log: Log | None = None, cancel: CancelToken | None = None) -> dict[str, Any]:
    exp = Path(request.exp_dir)
    src = exp / "1_16k_wavs"
    out1, out2 = exp / F0_DIR, exp / F0NSF_DIR
    if request.clean:
        for d in (out1, out2):
            if d.is_dir():
                for p in d.glob("*.npy"):
                    p.unlink()
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)
    names = sorted(n for n in os.listdir(src) if n.endswith(".wav") and "spec" not in n) if src.is_dir() else []
    if not names:
        raise ValueError("Slice the dataset first: 1_16k_wavs is empty")
    todo = [(str(src / n), str(out1 / n), str(out2 / n)) for n in names if not ((out1 / f"{n}.npy").exists() and (out2 / f"{n}.npy").exists())]
    if log:
        log(f"Pitch ({request.method}) for {len(todo)} of {len(names)} slices")
    env: list[dict[str, str] | None] | None = None
    from rvc_next.engine.runtime import gpu_backend, visible_devices_env

    if request.method == "rmvpe" and len(request.gpus) > 1:
        n = len(request.gpus)
        backend = gpu_backend() or "cuda"
        parts = [(request.method, request.assets_dir, f"{backend}:0", request.is_half, todo[i::n]) for i in range(n)]
        env = [visible_devices_env([g], backend) for g in request.gpus]
    elif request.method == "pm":
        n = max(1, min(request.n_workers, len(todo)))
        parts = [(request.method, request.assets_dir, "cpu", False, todo[i::n]) for i in range(n)]
    else:
        parts = [(request.method, request.assets_dir, request.device, request.is_half, todo)]
    parts = [p for p in parts if p[-1]]
    counts = merge_counts(run_parts(_part, parts, len(todo), progress, log, cancel, "f0", env))
    return {"done": counts.get("done", 0), "skipped": len(names) - len(todo) + counts.get("skipped", 0), "failed": counts.get("failed", 0), "total": len(names)}
