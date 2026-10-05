"""Dataset scan: what the user is about to train on, before any slicing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rvc_next.engine.train.multispeaker import audio_files

MIN_SECONDS = 1.0
"""Shorter clips yield no slice longer than the slicer's 1.5 s minimum once trimmed."""
MAX_SECONDS = 30 * 60.0
CLIP_LEVEL = 0.999
CLIP_FRACTION = 0.001
SILENT_PEAK = 10 ** (-50 / 20)


def dataset_files(folder: Path | str, multi: bool) -> list[str]:
    """The files a dataset folder contributes: top-level audio files (single speaker, as the original) or all audio below it (a speaker folder)."""
    return audio_files(folder, recursive=multi)


def scan_dataset(folders: list[tuple[str | None, str]]) -> dict[str, Any]:
    """Report on ``(speaker name or None, folder)`` pairs, shaped like the core's ``DatasetReport``."""
    from rvc_next.engine.audio.io import decode

    clips = 0
    total = 0.0
    rates: dict[str, int] = {}
    issues: list[dict[str, str]] = []
    speakers: dict[str, int] = {}
    multi = any(name is not None for name, _ in folders)
    for name, folder in folders:
        files = dataset_files(folder, multi)
        if not Path(folder).is_dir():
            issues.append({"path": str(folder), "problem": "unreadable", "detail": "Not a folder"})
            continue
        for path in files:
            try:
                audio, rate = decode(path, None, mono=True)
            except Exception as e:
                issues.append({"path": path, "problem": "unreadable", "detail": str(e)})
                continue
            clips += 1
            if name is not None:
                speakers[name] = speakers.get(name, 0) + 1
            seconds = audio.shape[0] / rate if rate else 0.0
            total += seconds
            rates[str(rate)] = rates.get(str(rate), 0) + 1
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if seconds < MIN_SECONDS:
                issues.append({"path": path, "problem": "too_short", "detail": f"{seconds:.2f} s"})
            elif seconds > MAX_SECONDS:
                issues.append({"path": path, "problem": "too_long", "detail": f"{seconds / 60:.1f} min"})
            if peak < SILENT_PEAK:
                issues.append({"path": path, "problem": "silent", "detail": f"peak {20 * np.log10(max(peak, 1e-9)):.0f} dB"})
            elif audio.size and float(np.mean(np.abs(audio) >= CLIP_LEVEL)) > CLIP_FRACTION:
                issues.append({"path": path, "problem": "clipped", "detail": f"{100 * float(np.mean(np.abs(audio) >= CLIP_LEVEL)):.2f}% of samples at full scale"})
    return {"clips": clips, "total_seconds": round(total, 3), "sample_rates": rates, "issues": issues, "speakers": speakers}
