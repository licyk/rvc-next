"""A two-epoch fit of a tiny model on the CPU, resume, and the training helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from rvc_next.engine.train import experiment, f0_extract, feature_extract, loop, preprocess
from tests.tiny import make_tiny_g_checkpoint, tone

TINY_MODEL = {
    "inter_channels": 8,
    "hidden_channels": 8,
    "filter_channels": 16,
    "n_heads": 2,
    "n_layers": 1,
    "kernel_size": 3,
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 3, 5]],
    "upsample_initial_channel": 16,
    "gin_channels": 8,
    "spk_embed_dim": 1,
}


@pytest.fixture(autouse=True, scope="module")
def few_threads():
    """Tiny models drown in thread overhead on a busy machine; a few threads keep the fit to seconds."""
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(min(4, before))
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory, tiny_assets_dir: Path) -> Path:
    root = tmp_path_factory.mktemp("fit")
    data = root / "data"
    data.mkdir()
    for i in range(3):
        sf.write(data / f"{i}.wav", tone(3.0, 40000, 150 + 30 * i), 40000)
    exp = root / "exp"
    preprocess.run(preprocess.SliceRequest(exp_dir=str(exp), sample_rate=40000, folder=str(data)))
    f0_extract.run(f0_extract.F0Request(exp_dir=str(exp), method="pm"))
    feature_extract.run(feature_extract.FeatureRequest(exp_dir=str(exp), version="v2", assets_dir=str(tiny_assets_dir)))
    return exp


def tiny_request(exp: Path, **kw) -> loop.TrainRequest:
    base = make_tiny_g_checkpoint(exp.parent / "base" / "G.pth")
    values: dict[str, Any] = dict(
        exp_dir=str(exp),
        sample_rate="40k",
        version="v2",
        pitch_guidance=True,
        epochs=2,
        save_every=1,
        batch_size=2,
        pretrained_g=str(base),
        config_override={"model": TINY_MODEL, "train": {"segment_size": 4000}},
        log_interval=1,
        num_workers=0,
        seed=0,
        save_small_every=True,
    )
    values.update(kw)
    return loop.TrainRequest(**values)


def test_two_epoch_fit_and_resume(prepared: Path) -> None:
    metrics: list[dict] = []
    outputs: list[tuple[str, str]] = []
    result = loop.run(tiny_request(prepared), metrics=metrics.append, output=lambda p, k, *_: outputs.append((p, k)))
    assert result["start_epoch"] == 1 and result["epochs"] == 2
    final = Path(result["final_model"])
    assert final.is_file() and final.parent.name == "weights"
    assert {k for _, k in outputs} == {"G", "D", "small", "preview"}
    previews = sorted(p for p, k in outputs if k == "preview")
    assert [Path(p).name for p in previews] == ["exp_e1.wav", "exp_e2.wav"] and sf.info(previews[0]).samplerate == 40000
    steps = result["steps_per_epoch"]
    assert len(list(prepared.glob("G_*.pth"))) == 2
    assert {m["epoch"] for m in metrics} == {1, 2}
    assert set(metrics[0]["losses"]) == {"g_total", "d_total", "mel", "kl", "fm", "gen", "grad_g", "grad_d"}
    lines = (prepared / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == len(metrics) and json.loads(lines[-1])["kind"] == "epoch"

    from rvc_next.engine.models.checkpoint import summarize

    info = summarize(final)
    assert info.kind == "small" and info.sample_rate == 40000 and info.version == "v2" and info.pitch_guidance and info.info == "2epoch"
    assert info.provenance["epoch"] == "2" and int(info.provenance["step"]) == 2 * result["steps_per_epoch"] and "creation_date" in info.provenance

    checkpoints = experiment.list_checkpoints(prepared)
    assert [c["kind"] for c in checkpoints].count("small") == 3
    assert max(c["step"] for c in checkpoints if c["kind"] == "G") == 2 * steps
    assert experiment.read_metrics(prepared, since_step=steps)

    resumed = loop.run(tiny_request(prepared, epochs=3))
    assert resumed["start_epoch"] == 3 and resumed["global_step"] == 3 * steps
    done = loop.run(tiny_request(prepared, epochs=3))
    assert done["start_epoch"] == 4


def test_fit_options(prepared: Path, tmp_path: Path) -> None:
    """Gradient checkpointing, fresh speaker vectors and the author stamp; bf16 without a GPU trains in fp32."""
    import shutil

    import torch

    from rvc_next.engine.models.checkpoint import read_small_model

    exp = tmp_path / "opts"
    shutil.copytree(prepared, exp, ignore=shutil.ignore_patterns("G_*", "D_*", "weights", "metrics.jsonl", "previews"))
    base = make_tiny_g_checkpoint(exp.parent / "base" / "G.pth")
    result = loop.run(tiny_request(exp, epochs=1, checkpointing=True, fresh_speakers=True, precision="bf16", author="Ana", previews=False, pretrained_g=str(base)))
    assert not (exp / "previews").exists()
    small = read_small_model(result["final_model"])
    assert small.extra["author"] == "Ana"
    # Speaker 0 trained from a new vector, not from the base model's row (rows are compared in test_fresh_rows…).
    pre = torch.load(str(base), weights_only=True)["model"]["emb_g.weight"].float()
    assert torch.cosine_similarity(small.weight["emb_g.weight"].float()[0], pre[0], dim=0) < 0.9


def test_fresh_rows_are_the_same_on_every_rank(tmp_path: Path) -> None:
    import torch

    from rvc_next.engine.models.loader import synthesizer_class
    from tests.tiny import tiny_config

    base = make_tiny_g_checkpoint(tmp_path / "G.pth", speakers=4)
    nets = [synthesizer_class("v2", True)(*tiny_config("40k", 4), is_half=False) for _ in range(2)]
    for net in nets:
        loop.load_pretrained_generator(net, str(base), [1, 2], seed=7)
    a, b = (n.emb_g.weight.detach() for n in nets)
    assert torch.equal(a, b)
    pre = torch.load(str(base), weights_only=True)["model"]["emb_g.weight"]
    assert torch.equal(a[0], pre[0]) and torch.equal(a[3], pre[3]) and not torch.equal(a[1], pre[1])
    assert torch.allclose(a[1].norm(), pre.float().norm(dim=1).mean(), rtol=1e-4)


def test_fit_cancel(prepared: Path, tmp_path: Path) -> None:
    import shutil

    from rvc_next.engine.cancel import CancelToken
    from rvc_next.engine.errors import Cancelled

    exp = tmp_path / "copy"
    shutil.copytree(prepared, exp, ignore=shutil.ignore_patterns("G_*", "D_*", "weights", "metrics.jsonl"))
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        loop.run(tiny_request(exp), cancel=token)


def test_pretrained_paths_and_batch(tmp_path: Path) -> None:
    (tmp_path / "pretrained_v2").mkdir()
    (tmp_path / "pretrained_v2" / "f0G40k.pth").write_bytes(b"x")
    assert experiment.pretrained_paths(tmp_path, "v2", True, "40k") == (str(tmp_path / "pretrained_v2" / "f0G40k.pth"), "")
    assert experiment.pretrained_paths(tmp_path, "v1", False, "48k") == ("", "")
    assert experiment.default_batch_size() >= 1


def test_legacy_single_speaker(tmp_path: Path) -> None:
    exp = tmp_path / "logs" / "old"
    for d in ("0_gt_wavs", "1_16k_wavs"):
        (exp / d).mkdir(parents=True)
        sf.write(exp / d / "0_0.wav", np.zeros(1600, dtype=np.float32), 16000)
    (exp / "3_feature256").mkdir()
    np.save(exp / "3_feature256" / "0_0.npy", np.zeros((5, 256), dtype=np.float32))
    from rvc_next.engine.train.hparams import default_config, write_config

    write_config(exp, default_config("v1", "48k"))
    legacy = experiment.read_legacy(exp)
    assert legacy["sample_rate"] == "48k" and legacy["version"] == "v1" and not legacy["pitch_guidance"] and not legacy["multi_speaker"]
    assert legacy["stages"] == {"slice": True, "f0": False, "features": True, "fit": False, "index": False}


def test_ddp_over_gloo_on_cpu(prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The several-GPU path, run as two CPU processes: rank 0 reports through the queue."""
    import shutil

    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    exp = tmp_path / "ddp"
    shutil.copytree(prepared, exp, ignore=shutil.ignore_patterns("G_*", "D_*", "weights", "metrics.jsonl"))
    metrics: list[dict] = []
    outputs: list[str] = []
    result = loop.run(tiny_request(exp, epochs=1, ddp_cpu_processes=2), metrics=metrics.append, output=lambda p, k, *_: outputs.append(k))
    assert result["epochs"] == 1 and Path(result["final_model"]).is_file()
    assert metrics and "small" in outputs and "G" in outputs
