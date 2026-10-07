from pathlib import Path

import pytest

from rvc_next.engine.separate.presets import MODELS, asset_files, load_presets, output_name, plan_chain, resolve


def test_presets_match_the_core_record_and_models() -> None:
    from rvc_next.core.separation.models import SeparationPreset

    presets = load_presets()
    ids = [p["id"] for p in presets]
    assert ids == ["vocals", "vocals-aggressive", "dereverb", "dereverb-aggressive", "karaoke", "vocals-clean", "lead-clean"]
    for p in presets:
        SeparationPreset.model_validate(p)
        assert all(step in MODELS for step in p["steps"])
        assert p["assets"] == list(dict.fromkeys(MODELS[s].asset_id for s in p["steps"]))
        assert p["primary_stem"] == p["stems"][0]
        assert (p["kind"] == "chain") == (len(p["steps"]) > 1)


def test_asset_table_covers_every_model() -> None:
    table = asset_files()
    assert set(table) == {f"separation-{m}" for m in MODELS}
    for files in table.values():
        assert len(files) == 1 and files[0]["path"].startswith("pymss_weights/") and files[0]["path"].endswith(".ckpt")
        assert len(files[0]["sha256"]) == 64 and files[0]["size"] > 100_000_000


def test_resolve_reports_missing_assets(tmp_path: Path) -> None:
    preset, models, missing = resolve("vocals-clean", tmp_path)
    assert preset["id"] == "vocals-clean"
    assert list(models) == ["vocals", "dereverb"]
    assert missing == ["separation-vocals", "separation-dereverb"]
    # The packaged YAML is used when the assets folder has none.
    assert Path(models["vocals"]["config"]).is_file()
    weights = tmp_path / "pymss_weights"
    weights.mkdir()
    (weights / MODELS["vocals"].ckpt).write_bytes(b"x")
    (weights / MODELS["vocals"].config).write_text("audio: {}")
    _, models, missing = resolve("vocals-clean", tmp_path)
    assert missing == ["separation-dereverb"]
    assert models["vocals"]["config"] == str(weights / MODELS["vocals"].config)
    with pytest.raises(KeyError):
        resolve("nope", tmp_path)


def _plan(preset_id: str) -> list[tuple[str, tuple, str | None]]:
    preset = next(p for p in load_presets() if p["id"] == preset_id)
    return [(s.model_id, s.keep, s.feeds_next) for s in plan_chain(preset)]


def test_plan_single_model() -> None:
    assert _plan("vocals-aggressive") == [("vocals-aggressive", (("vocals", "vocals"), ("other", "instrumental")), None)]
    assert _plan("karaoke") == [("karaoke", (("karaoke", "main_vocal"), ("other", "off_vocal")), None)]


def test_plan_chains() -> None:
    assert _plan("vocals-clean") == [
        ("vocals", (("instrumental", "instrumental"),), "vocals"),
        ("dereverb", (("noreverb", "vocals"), ("reverb", "reverb")), None),
    ]
    assert _plan("lead-clean") == [
        ("vocals", (("instrumental", "instrumental"),), "vocals"),
        ("karaoke", (("other", "off_vocal"),), "karaoke"),
        ("dereverb", (("noreverb", "main_vocal"), ("reverb", "reverb")), None),
    ]


def test_plan_writes_each_label_once() -> None:
    preset = {"id": "x", "steps": ["dereverb", "dereverb-aggressive"], "stems": ["noreverb", "reverb"], "primary_stem": "noreverb", "kind": "chain"}
    plan = plan_chain(preset)
    labels = [label for step in plan for _, label in step.keep]
    assert labels == ["reverb", "noreverb"]


def test_output_name_never_overwrites(tmp_path: Path) -> None:
    assert output_name("song", "vocals", "flac") == "song.vocals.flac"
    (tmp_path / "song.vocals.flac").write_bytes(b"")
    assert output_name("song", "vocals", "flac", directory=tmp_path) == "song.vocals-2.flac"
    assert output_name("song", "vocals", "flac", {"song.vocals-2.flac"}, tmp_path) == "song.vocals-3.flac"


def test_imported_checkpoints_must_be_plain_data(tmp_path: Path) -> None:
    import os

    import torch

    from rvc_next.engine.separate.runner import ensure_plain_checkpoint

    torch.save({"state_dict": {"w": torch.zeros(2)}}, str(tmp_path / "ok.ckpt"))
    ensure_plain_checkpoint(tmp_path / "ok.ckpt", "bs_roformer")

    class Payload:
        def __reduce__(self):
            return (os.getcwd, ())

    torch.save({"state_dict": {}, "hp": Payload()}, str(tmp_path / "evil.ckpt"))
    for kind in ("bs_roformer", "htdemucs"):
        with pytest.raises(ValueError, match="could run code"):
            ensure_plain_checkpoint(tmp_path / "evil.ckpt", kind)
