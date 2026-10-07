"""Training stages on synthetic audio: slicing, pitch, features, the file list and the fingerprint."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rvc_next.engine.train import f0_extract, feature_extract, filelist, fingerprint, preprocess
from rvc_next.engine.train.preprocess import PreProcess, SliceRequest
from tests.tiny import tone

SR = 40000


def voiced_with_gaps(segments: list[float], sr: int = SR, gap: float = 1.0) -> np.ndarray:
    """Tone segments of the given lengths separated by silence, so the slicer cuts between them."""
    parts = []
    for i, seconds in enumerate(segments):
        parts.append(tone(seconds, sr, freq=180 + 40 * i))
        parts.append(np.zeros(int(gap * sr), dtype=np.float32))
    return np.concatenate(parts)


def test_slicing_keeps_every_tail(tmp_path: Path) -> None:
    """Every slicer chunk is covered to its last sample; the §5.3 regression dropped every tail but the last."""
    pp = PreProcess(SR, str(tmp_path), 3.7)
    chunks = pp.slicer.slice(voiced_with_gaps([9.0, 7.5, 5.2]))
    assert len(chunks) == 3
    step, per, tail = int(SR * (pp.per - pp.overlap)), int(SR * pp.per), pp.tail * SR
    expected: list[np.ndarray] = []
    for chunk in chunks:
        start = 0
        while len(chunk) - start > tail:
            expected.append(chunk[start : start + per])
            start += step
        expected.append(chunk[start:])
    pieces = pp.slices(voiced_with_gaps([9.0, 7.5, 5.2]))
    assert [len(p) for p in pieces] == [len(e) for e in expected]
    assert all(np.array_equal(p, e) for p, e in zip(pieces, expected))
    # Samples in = samples out once the overlaps are taken off: nothing is dropped.
    overlaps = sum(per - step for e, nxt in zip(expected, expected[1:]) if len(e) == per)
    assert sum(len(p) for p in pieces) - overlaps == sum(len(c) for c in chunks)


def test_slice_run_writes_all_pieces_and_resumes(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    sf.write(data / "a.wav", voiced_with_gaps([6.0, 5.0], sr=44100), 44100)
    sf.write(data / "b.flac", voiced_with_gaps([2.5], sr=22050), 22050)
    (data / "notes.txt").write_text("not audio")
    exp = tmp_path / "exp"
    result = preprocess.run(SliceRequest(exp_dir=str(exp), sample_rate=SR, folder=str(data)))
    assert result["files"] == 2 and result["failed"] == 0
    gt = sorted(p.name for p in (exp / "0_gt_wavs").glob("*.wav"))
    k16 = sorted(p.name for p in (exp / "1_16k_wavs").glob("*.wav"))
    assert gt == k16 and len(gt) == result["slices"]
    # Two chunks in a.wav, each with its tail: at least four pieces.
    assert sum(1 for n in gt if n.startswith("0_")) >= 4
    rate, piece = __import__("scipy.io.wavfile", fromlist=["read"]).read(exp / "0_gt_wavs" / gt[0])
    assert rate == SR and piece.dtype == np.float32
    again = preprocess.run(SliceRequest(exp_dir=str(exp), sample_rate=SR, folder=str(data)))
    assert again["files"] == 0 and again["skipped"] == 2
    cleaned = preprocess.run(SliceRequest(exp_dir=str(exp), sample_rate=SR, folder=str(data), clean=True))
    assert cleaned["files"] == 2 and cleaned["slices"] == len(gt)


def test_slice_in_parallel(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        sf.write(data / f"{i}.wav", voiced_with_gaps([2.0]), SR)
    seen: list[float | None] = []
    result = preprocess.run(SliceRequest(exp_dir=str(tmp_path / "exp"), sample_rate=SR, folder=str(data), n_workers=2), progress=lambda v, s: seen.append(v))
    assert result["files"] == 3 and seen and seen[-1] == 1.0


@pytest.fixture
def sliced(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    sf.write(data / "a.wav", voiced_with_gaps([4.0, 2.5]), SR)
    exp = tmp_path / "exp"
    preprocess.run(SliceRequest(exp_dir=str(exp), sample_rate=SR, folder=str(data)))
    return exp


def test_f0_and_features(sliced: Path, tiny_assets_dir: Path) -> None:
    r = f0_extract.run(f0_extract.F0Request(exp_dir=str(sliced), method="pm"))
    n = len(list((sliced / "1_16k_wavs").glob("*.wav")))
    assert r["done"] == n
    coarse = np.load(next((sliced / "2a_f0").glob("*.npy")))
    nsf = np.load(next((sliced / "2b-f0nsf").glob("*.npy")))
    assert coarse.dtype == np.int64 and coarse.min() >= 1 and coarse.max() <= 255
    assert nsf.shape == coarse.shape and nsf.min() > 0
    assert f0_extract.run(f0_extract.F0Request(exp_dir=str(sliced), method="pm"))["skipped"] == n
    r = feature_extract.run(feature_extract.FeatureRequest(exp_dir=str(sliced), version="v2", assets_dir=str(tiny_assets_dir)))
    assert r["done"] == n
    feats = np.load(next((sliced / "3_feature768").glob("*.npy")))
    assert feats.ndim == 2 and feats.shape[1] == 768
    r = feature_extract.run(feature_extract.FeatureRequest(exp_dir=str(sliced), version="v1", assets_dir=str(tiny_assets_dir)))
    assert np.load(next((sliced / "3_feature256").glob("*.npy"))).shape[1] == 256


def test_features_need_hubert(sliced: Path, tmp_path: Path) -> None:
    from rvc_next.engine.errors import MissingAssetError

    with pytest.raises(MissingAssetError):
        feature_extract.run(feature_extract.FeatureRequest(exp_dir=str(sliced), assets_dir=str(tmp_path / "none")))


def test_filelist_single_speaker(sliced: Path, tiny_assets_dir: Path) -> None:
    f0_extract.run(f0_extract.F0Request(exp_dir=str(sliced), method="pm"))
    feature_extract.run(feature_extract.FeatureRequest(exp_dir=str(sliced), version="v2", assets_dir=str(tiny_assets_dir)))
    n = len(list((sliced / "0_gt_wavs").glob("*.wav")))
    out = filelist.write_filelist(sliced, "40k", "v2", True, speaker_id=3, seed=1)
    lines = (sliced / "filelist.txt").read_text().splitlines()
    assert out["lines"] == len(lines) == n + 2
    mute = [line for line in lines if "/mute/" in line]
    assert len(mute) == 2 and all(line.endswith("|3") for line in lines)
    for col in mute[0].split("|")[:4]:
        assert Path(col).is_file()
    assert all(len(line.split("|")) == 5 for line in lines)
    nof0 = filelist.build_lines(sliced, "40k", "v2", False)[0]
    assert all(len(line.split("|")) == 3 for line in nof0)
    config = filelist.prepare_config(sliced, "40k", "v2")
    assert json.loads((sliced / "config.json").read_text())["data"]["sampling_rate"] == 40000 and "speaker_info" not in config


def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.wav").write_bytes(b"1")
    first = fingerprint.of_paths([d])
    assert fingerprint.of_paths([d]) == first
    assert fingerprint.of_paths([d], extra="40k") != first
    (d / "b.wav").write_bytes(b"2")
    second = fingerprint.of_paths([d])
    assert second != first
    os.utime(d / "a.wav", ns=(1, 1))
    assert fingerprint.of_paths([d]) != second
    assert fingerprint.of_paths([tmp_path / "missing"]) != fingerprint.of_paths([tmp_path / "empty"])


def test_slicing_options(tmp_path: Path) -> None:
    from rvc_next.engine.train.preprocess import SliceOptions

    audio = voiced_with_gaps([9.0, 7.5])
    fixed = PreProcess(SR, str(tmp_path / "a"), 3.0, SliceOptions(cut="fixed", overlap=0.0)).slices(audio)
    # Straight through, silences included: back to back pieces of 3 s and the tail.
    assert sum(len(p) for p in fixed) == len(audio) and all(len(p) == 3 * SR for p in fixed[:-1])
    short = tone(4.0, SR)
    assert [len(p) for p in PreProcess(SR, str(tmp_path / "b"), 3.7, SliceOptions(cut="none")).slices(short)] == [len(short)]
    # A long file is still cut, or training would drop it.
    assert len(PreProcess(SR, str(tmp_path / "c"), 3.7, SliceOptions(cut="none")).slices(tone(12.0, SR))) > 1

    data = tmp_path / "data"
    data.mkdir()
    sf.write(data / "q.wav", 0.05 * tone(3.0, SR), SR)
    for normalize, expect_peak in (("slice", 0.9 * 0.75), ("none", None)):
        exp = tmp_path / f"exp-{normalize}"
        preprocess.run(SliceRequest(exp_dir=str(exp), sample_rate=SR, folder=str(data), options=SliceOptions(cut="none", normalize=normalize, highpass=False, denoise=0.3)))
        out, _ = sf.read(next((exp / preprocess.GT_DIR).glob("*.wav")))
        peak = np.abs(out).max()
        assert (expect_peak is None and peak < 0.1) or (expect_peak is not None and peak > 0.25)
