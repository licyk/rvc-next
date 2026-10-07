from pathlib import Path

import pytest

from rvc_next.engine.index.inspect import inspect_index
from rvc_next.engine.models.checkpoint import inspect_checkpoint
from tests.tiny import make_tiny_g_checkpoint, make_tiny_index, make_tiny_voice


def make_index(path: Path, dim: int, add: bool = True) -> Path:
    return make_tiny_index(path, dim, 400, add)


def test_index_dimension_and_empty(tmp_path):
    v2 = inspect_index(make_index(tmp_path / "a.index", 768))
    assert (v2.dim, v2.version, v2.ntotal, v2.empty) == (768, "v2", 400, False)
    trained = inspect_index(make_index(tmp_path / "trained.index", 256, add=False))
    assert trained.version == "v1" and trained.empty
    (tmp_path / "junk.index").write_bytes(b"not an index")
    with pytest.raises(ValueError):
        inspect_index(tmp_path / "junk.index")


def test_checkpoint_kinds(tmp_path):
    small = inspect_checkpoint(make_tiny_voice(tmp_path / "v.pth", version="v1"))
    assert (small.kind, small.version, small.sample_rate, small.pitch_guidance) == ("small", "v1", "40k", True)
    g = inspect_checkpoint(make_tiny_g_checkpoint(tmp_path / "G_10.pth", version="v2", f0=False))
    assert (g.kind, g.version, g.sample_rate, g.pitch_guidance) == ("G", "v2", "40k", False)
    import torch

    torch.save({"model": {"discriminators.0.convs.0.weight": torch.zeros(1)}}, str(tmp_path / "D.pth"))
    assert inspect_checkpoint(tmp_path / "D.pth").kind == "D"
    torch.save({"something": 1}, str(tmp_path / "x.pth"))
    assert inspect_checkpoint(tmp_path / "x.pth").kind == "unknown"


def test_separation_config(tmp_path):
    from rvc_next.engine.separate.presets import CONFIGS_DIR, inspect_config

    info = inspect_config(CONFIGS_DIR / "model_bs_roformer_ep_368_sdr_12.9628.yaml")
    assert info.model_type == "bs_roformer" and info.instruments == ("vocals", "instrumental") and info.target_instrument == "vocals"
    (tmp_path / "bad.yaml").write_text("training: {instruments: [a]}\nmodel: {}\n")
    with pytest.raises(ValueError):
        inspect_config(tmp_path / "bad.yaml")


def test_applio_generators_and_discriminators_are_unsupported(tmp_path):
    import torch

    torch.save({"model": {"dec.mrfs.0.0.layers.0.conv1.weight_v": torch.zeros(1), "dec.conv_pre.weight": torch.zeros(1)}}, str(tmp_path / "G.pth"))
    g = inspect_checkpoint(tmp_path / "G.pth")
    assert g.kind == "unsupported" and "MRF" in g.note
    d_state = {f"discriminators.{i}.convs.0.weight_v": torch.zeros(32, 1, 5, 1) for i in range(6)}
    d_state |= {f"discriminators.{i}.convs.0.weight_v": torch.zeros(32, 1, 3, 9) for i in range(6, 9)}
    torch.save({"model": d_state}, str(tmp_path / "D.pth"))
    d = inspect_checkpoint(tmp_path / "D.pth")
    assert d.kind == "unsupported" and "RefineGAN" in d.note
