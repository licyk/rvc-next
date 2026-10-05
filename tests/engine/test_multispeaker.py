from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rvc_next.engine.index import build
from rvc_next.engine.train import experiment, f0_extract, feature_extract, filelist, multispeaker, preprocess
from rvc_next.engine.train.preprocess import SliceRequest, SpeakerSpec
from tests.tiny import tone


def speaker_dirs(root: Path) -> Path:
    for name, freq in (("Alice_3_2", 200.0), ("Bob_7_1", 120.0)):
        d = root / name
        (d / "sub").mkdir(parents=True)
        sf.write(d / "x.wav", tone(4.5, 40000, freq), 40000)
        sf.write(d / "sub" / "y.wav", tone(4.0, 40000, freq * 1.1), 40000)
    return root


def test_speakers_from_folders(tmp_path: Path) -> None:
    rows = multispeaker.speakers_from_folders(speaker_dirs(tmp_path / "ds"))
    assert [(r["name"], r["id"], r["repeat"]) for r in rows] == [("Alice", 3, 2), ("Bob", 7, 1)]
    (tmp_path / "ds" / "bad").mkdir()
    with pytest.raises(multispeaker.ManifestError, match="bad"):
        multispeaker.speakers_from_folders(tmp_path / "ds")


def test_manifest_rejects_conflicts(tmp_path: Path) -> None:
    root = speaker_dirs(tmp_path / "ds")
    with pytest.raises(multispeaker.ManifestError):
        multispeaker.build_manifest([{"name": "A", "id": 1, "folder": str(root / "Alice_3_2")}, {"name": "B", "id": 1, "folder": str(root / "Bob_7_1")}])
    with pytest.raises(multispeaker.ManifestError):
        multispeaker.build_manifest([{"name": "A", "id": 200, "folder": str(root / "Alice_3_2")}])


def test_multi_speaker_pipeline(tmp_path: Path, tiny_assets_dir: Path) -> None:
    rows = multispeaker.speakers_from_folders(speaker_dirs(tmp_path / "ds"))
    exp = tmp_path / "exp"
    speakers = tuple(SpeakerSpec(**r) for r in rows)
    preprocess.run(SliceRequest(exp_dir=str(exp), sample_rate=40000, speakers=speakers))
    manifest = json.loads((exp / "multispeaker_manifest.json").read_text())
    assert len(manifest["entries"]) == 4 and manifest["speakers"] == [{"id": 3, "name": "Alice"}, {"id": 7, "name": "Bob"}]
    assert all(p.name.startswith("ms") for p in (exp / "0_gt_wavs").glob("*.wav"))
    f0_extract.run(f0_extract.F0Request(exp_dir=str(exp), method="pm"))
    feature_extract.run(feature_extract.FeatureRequest(exp_dir=str(exp), version="v2", assets_dir=str(tiny_assets_dir)))

    lines, active = filelist.build_lines(exp, "40k", "v2", True, multi_speaker=True)
    assert active == [{"id": 3, "name": "Alice"}, {"id": 7, "name": "Bob"}]
    alice = [line for line in lines if line.endswith("|3|Alice") and "/mute/" not in line]
    bob = [line for line in lines if line.endswith("|7|Bob") and "/mute/" not in line]
    assert len(alice) == 2 * len(set(alice)) and len(bob) == len(set(bob))
    assert sum("/mute/" in line for line in lines) == 4
    config = filelist.prepare_config(exp, "40k", "v2", active)
    assert config["model"]["spk_embed_dim"] == 110 and config["speaker_info"] == active

    result = build.run(build.IndexRequest(exp_dir=str(exp), version="v2", seed=0))
    assert sorted(r["speaker_id"] for r in result["indexes"]) == [3, 7]
    names = sorted(p.name for p in exp.glob("added_*.index"))
    assert len(names) == 2 and names[0].endswith("_exp_v2_spkid3.index") and names[1].endswith("_exp_v2_spkid7.index")
    assert (exp / "total_fea_spkid3.npy").is_file()

    legacy = experiment.read_legacy(exp)
    assert legacy["multi_speaker"] and legacy["version"] == "v2" and legacy["sample_rate"] == "40k" and legacy["pitch_guidance"]
    assert legacy["speakers"] == active and [r["name"] for r in legacy["speaker_rows"]] == ["Alice", "Bob"]
    assert all(legacy["stages"][s] for s in ("slice", "f0", "features", "index")) and not legacy["stages"]["fit"]


def test_index_rebuilds_only_on_change(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    fea = exp / "3_feature256"
    fea.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        np.save(fea / f"0_{i}.npy", rng.standard_normal((60, 256)).astype(np.float32))
    first = build.run(build.IndexRequest(exp_dir=str(exp), version="v1", name="voice", seed=0))
    assert first["built"] == 1
    added = next(exp.glob("added_*_voice_v1.index"))
    second = build.run(build.IndexRequest(exp_dir=str(exp), version="v1", name="voice", seed=0))
    assert second["reused"] == 1 and added.is_file()
    np.save(fea / "0_9.npy", rng.standard_normal((60, 256)).astype(np.float32))
    third = build.run(build.IndexRequest(exp_dir=str(exp), version="v1", name="voice", seed=0))
    assert third["built"] == 1 and third["indexes"][0]["vectors"] == 300
    assert len(list(exp.glob("added_*.index"))) == 1

    from rvc_next.engine.index.retrieval import load_index

    loaded = load_index(next(exp.glob("added_*.index")))
    assert loaded.vectors.shape == (300, 256)
