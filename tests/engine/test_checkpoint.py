from pathlib import Path

import pytest

from rvc_next.engine.errors import ModelFormatError
from rvc_next.engine.models.checkpoint import change_info, read_small_model, summarize
from rvc_next.engine.models.extract import extract_small_model
from rvc_next.engine.models.merge import merge_models
from tests.tiny import make_tiny_g_checkpoint, make_tiny_voice


def test_round_trip_keeps_fields(tmp_path: Path) -> None:
    src = make_tiny_voice(tmp_path / "a.pth", speakers=3, speaker_info=[{"id": 2, "name": "Bo"}, {"id": 0, "name": "Al"}, {"id": 7, "name": "out of range"}])
    model = read_small_model(src)
    assert model.speaker_slots == 3
    assert model.speaker_info == [{"id": 0, "name": "Al"}, {"id": 2, "name": "Bo"}]
    change_info(src, tmp_path / "b.pth", info="hello", speaker_info=[{"id": 1, "name": "Cy"}])
    b = summarize(tmp_path / "b.pth")
    assert b.kind == "small" and b.info == "hello" and b.speaker_info == [{"id": 1, "name": "Cy"}]
    assert b.sample_rate == 40000 and b.version == "v2" and b.pitch_guidance


def test_merge_truncates_unequal_embeddings(tmp_path: Path) -> None:
    a = make_tiny_voice(tmp_path / "a.pth", speakers=3, seed=1)
    b = make_tiny_voice(tmp_path / "b.pth", speakers=2, seed=2)
    merged = merge_models(a, b, tmp_path / "m.pth", alpha=0.25)
    assert merged.speaker_slots == 2
    wa, wb = read_small_model(a).weight, read_small_model(b).weight
    key = "dec.conv_pre.weight"
    expected = (0.25 * wa[key].float() + 0.75 * wb[key].float()).half()
    assert (merged.weight[key] == expected).all()


def test_merge_refuses_different_versions(tmp_path: Path) -> None:
    a = make_tiny_voice(tmp_path / "a.pth", version="v1")
    b = make_tiny_voice(tmp_path / "b.pth", version="v2")
    with pytest.raises(ModelFormatError):
        merge_models(a, b, tmp_path / "m.pth", alpha=0.5)


def test_extract_from_g_checkpoint(tmp_path: Path) -> None:
    g = make_tiny_g_checkpoint(tmp_path / "G_10.pth")
    info = summarize(g)
    assert info.kind == "checkpoint" and info.pitch_guidance and info.version == "v2" and info.iteration == 10
    small = extract_small_model(g, tmp_path / "x.pth", sample_rate="40k", pitch_guidance=True, version="v2")
    assert not any("enc_q" in k for k in small.weight)
    assert summarize(tmp_path / "x.pth").kind == "small"


def test_not_a_model(tmp_path: Path) -> None:
    import torch

    torch.save({"hello": 1}, str(tmp_path / "n.pth"))
    with pytest.raises(ModelFormatError):
        summarize(tmp_path / "n.pth")
