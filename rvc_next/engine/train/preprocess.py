"""Slice stage: cut the dataset into ~3.7 s pieces at the target rate and at 16 kHz.

Ported from the original ``train/preprocess.py``, with the tail write back inside the slice loop
(as at ``5d47da1``): the original at ``81eed5e`` wrote only the last slice's tail, dropping the
tail of every other slice from the training set.

Options after Applio, all at the original's values by default (``SliceOptions``): how files are cut
(``auto``: the silence slicer, then pieces of ``per`` seconds; ``fixed``: pieces of ``per`` seconds
straight through, silences included; ``none``: files as they are, for a dataset already cut; a file
longer than ``per`` + 5 s is still cut into pieces, since training would drop it), the overlap
between pieces, the 48 Hz high-pass, the loudness normalisation (per slice, as the original; per
file; or none), and a noise gate over each file before cutting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

from rvc_next.engine.cancel import CancelToken
from rvc_next.engine.train import multispeaker
from rvc_next.engine.train.parallel import Log, PartReporter, Progress, merge_counts, run_parts
from rvc_next.engine.train.scan import dataset_files
from rvc_next.engine.train.slicer import Slicer

GT_DIR = "0_gt_wavs"
WAV16K_DIR = "1_16k_wavs"
DONE_FILE = "slice_done.txt"
"""Output keys whose slicing finished, one per line; a key not listed is sliced again from scratch."""


@dataclass(frozen=True)
class SpeakerSpec:
    name: str
    id: int
    folder: str
    repeat: int = 1


CUT_MODES = ("auto", "fixed", "none")
NORMALIZE_MODES = ("slice", "file", "none")
LONG_FILE_MARGIN = 5.0
"""With ``cut = "none"``, files longer than ``per`` + this many seconds are still cut."""


@dataclass(frozen=True)
class SliceOptions:
    cut: str = "auto"
    overlap: float = 0.3
    highpass: bool = True
    normalize: str = "slice"
    denoise: float = 0.0
    """0–1: a non-stationary noise gate over each file before cutting (0 off)."""


@dataclass(frozen=True)
class SliceRequest:
    exp_dir: str
    sample_rate: int
    """Target rate in Hz: 32000, 40000 or 48000."""
    folder: str | None = None
    """Single-speaker dataset folder (its top-level audio files)."""
    speakers: tuple[SpeakerSpec, ...] = ()
    """Multi-speaker table; written to ``multispeaker_manifest.json`` before slicing."""
    use_manifest: bool = False
    """Slice from an existing manifest in the experiment folder (legacy experiments)."""
    n_workers: int = 1
    per: float = 3.7
    """Slice length in seconds; the original uses 3.0 on CPU and small GPUs."""
    clean: bool = False
    """Remove earlier slices first (the dataset changed)."""
    options: SliceOptions = field(default_factory=SliceOptions)
    extra: dict[str, Any] = field(default_factory=dict)


class PreProcess:
    def __init__(self, sr: int, exp_dir: str, per: float = 3.7, options: SliceOptions | None = None) -> None:
        self.slicer = Slicer(sr=sr, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.options = options or SliceOptions()
        self.overlap = self.options.overlap
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.gt_wavs_dir = os.path.join(exp_dir, GT_DIR)
        self.wavs16k_dir = os.path.join(exp_dir, WAV16K_DIR)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio: np.ndarray, output_key: str, idx1: int, log: Any) -> bool:
        import librosa

        tmp_max = np.abs(tmp_audio).max() if tmp_audio.size else 0.0
        if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > 2.5:
            log(f"Skipped an invalid slice {output_key}_{idx1} (peak {tmp_max})")
            return False
        if self.options.normalize == "slice":
            tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (1 - self.alpha) * tmp_audio
        wavfile.write(os.path.join(self.gt_wavs_dir, f"{output_key}_{idx1}.wav"), self.sr, tmp_audio.astype(np.float32))
        audio_16k = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000).astype(np.float32)
        wavfile.write(os.path.join(self.wavs16k_dir, f"{output_key}_{idx1}.wav"), 16000, audio_16k)
        return True

    def slices(self, audio: np.ndarray) -> list[np.ndarray]:
        """The pieces one file is cut into: every slicer chunk, split with overlap, tail included."""
        cut = self.options.cut
        if cut == "none" and len(audio) <= (self.per + LONG_FILE_MARGIN) * self.sr:
            return [audio]
        chunks = [audio] if cut in ("fixed", "none") else self.slicer.slice(audio)
        out: list[np.ndarray] = []
        for chunk in chunks:
            i = 0
            while True:
                start = int(self.sr * (self.per - self.overlap) * i)
                i += 1
                if len(chunk[start:]) > self.tail * self.sr:
                    out.append(chunk[start : start + int(self.per * self.sr)])
                else:
                    out.append(chunk[start:])
                    break
        return out

    def pipeline(self, path: str, output_key: str, log: Any) -> int:
        from rvc_next.engine.audio.io import decode

        audio, _ = decode(path, self.sr)
        if self.options.highpass:
            # A zero-phase filter causes pre-ringing; the causal one is kept, as the original.
            audio = signal.lfilter(self.bh, self.ah, audio)
        if self.options.denoise > 0:
            audio = self._denoise(audio)
        if self.options.normalize == "file":
            peak = np.abs(audio).max() if audio.size else 0.0
            if 0 < peak <= 2.5:
                audio = audio / peak * (self.max * self.alpha) + (1 - self.alpha) * audio
        written = 0
        for idx1, piece in enumerate(self.slices(audio)):
            written += self.norm_write(piece, output_key, idx1, log)
        return written

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        import torch

        from rvc_next.engine.audio.denoise import TorchGate

        gate = TorchGate(sr=self.sr, nonstationary=True, prop_decrease=min(max(self.options.denoise, 0.0), 1.0))
        with torch.no_grad():
            return gate(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)).squeeze(0).double().numpy()


def _remove_key(exp_dir: str, key: str) -> None:
    for sub in (GT_DIR, WAV16K_DIR):
        folder = Path(exp_dir) / sub
        for p in folder.glob(f"{key}_*.wav"):
            if p.stem.rsplit("_", 1)[0] == key:
                p.unlink(missing_ok=True)


def _part(sr: int, exp_dir: str, per: float, options: SliceOptions, infos: list[tuple[str, str]], reporter: PartReporter) -> dict[str, Any]:
    pp = PreProcess(sr, exp_dir, per, options)
    files = slices = failed = 0
    for path, key in infos:
        _remove_key(exp_dir, key)
        try:
            slices += pp.pipeline(path, key, reporter.log)
            files += 1
            with open(os.path.join(exp_dir, DONE_FILE), "a", encoding="utf-8") as f:
                f.write(key + "\n")
        except Exception as e:  # one bad file must not stop the dataset
            failed += 1
            reporter.log(f"Cannot slice {path}: {e}")
        reporter.done(os.path.basename(path))
    return {"files": files, "slices": slices, "failed": failed}


def _clean(exp_dir: Path) -> None:
    for sub in (GT_DIR, WAV16K_DIR):
        folder = exp_dir / sub
        if folder.is_dir():
            for p in folder.iterdir():
                if p.is_file():
                    p.unlink()
    (exp_dir / DONE_FILE).unlink(missing_ok=True)


def plan_inputs(request: SliceRequest) -> list[tuple[str, str]]:
    """``(source file, output key)`` for every input file, in order."""
    exp_dir = Path(request.exp_dir)
    if request.speakers:
        manifest = multispeaker.build_manifest([{"name": s.name, "id": s.id, "folder": s.folder, "repeat": s.repeat} for s in request.speakers])
        old = None
        try:
            old = multispeaker.load_manifest(exp_dir, check_files=False)
        except multispeaker.ManifestError:
            pass
        if (
            old is None
            or [(e["path"], e["output_key"]) for e in old["entries"]] != [(e["path"], e["output_key"]) for e in manifest["entries"]]
            or old.get("rows") != manifest["rows"]
        ):
            multispeaker.write_manifest(exp_dir, manifest)
        return [(e["path"], e["output_key"]) for e in manifest["entries"]]
    if request.use_manifest:
        manifest = multispeaker.load_manifest(exp_dir)
        return [(e["path"], e["output_key"]) for e in manifest["entries"]]
    if not request.folder:
        raise ValueError("A dataset folder or a speaker table is required")
    files = dataset_files(request.folder, multi=False)
    if not files:
        raise ValueError(f"No audio files in {request.folder}")
    return [(path, str(idx)) for idx, path in enumerate(files)]


def run(request: SliceRequest, progress: Progress | None = None, log: Log | None = None, cancel: CancelToken | None = None) -> dict[str, Any]:
    exp_dir = Path(request.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    if request.clean:
        _clean(exp_dir)
    infos = plan_inputs(request)
    done_file = exp_dir / DONE_FILE
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.is_file() else set()
    todo = [(p, k) for p, k in infos if k not in done]
    if log:
        log(f"Slicing {len(todo)} of {len(infos)} files ({len(infos) - len(todo)} already done)")
    workers = max(1, min(request.n_workers, len(todo)))
    parts = [(request.sample_rate, str(exp_dir), request.per, request.options, todo[i::workers]) for i in range(workers)] if todo else []
    results = run_parts(_part, parts, len(todo), progress, log, cancel, "slice")
    counts = merge_counts(results)
    total_slices = len(list((exp_dir / GT_DIR).glob("*.wav")))
    return {"files": counts.get("files", 0), "skipped": len(infos) - len(todo), "failed": counts.get("failed", 0), "slices": total_slices}
