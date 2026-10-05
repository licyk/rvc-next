"""Import sessions: pairing by content, review decisions, the inbox, base and separation models."""

import json
from pathlib import Path

import pytest

from rvc_next.core.errors import ConflictError, ValidationError
from rvc_next.core.models.models import AssignIndexRequest, ImportDecisions, IndexPlan
from rvc_next.core.training.models import ExperimentCreate, FitSettings
from tests.tiny import make_tiny_g_checkpoint, make_tiny_index, make_tiny_voice


def stage(services, *paths):
    sid = services.imports.create()
    services.imports.add_paths(sid, list(paths))
    return sid


def test_unrelated_names_pair_and_keep_the_original_name(services, tmp_path):
    pth = make_tiny_voice(tmp_path / "up" / "Kiki singing.pth")
    idx = make_tiny_index(tmp_path / "up" / "final_v3_index_DO_NOT_LOSE.index")
    result = services.models.import_paths([pth, idx])
    voice = result.voices[0]
    assert voice.has_index and voice.indexes["default"].endswith("model.index")
    assert [(a.voice_id, a.key, a.name) for a in result.attached] == [(voice.id, "default", idx.name)]
    meta = json.loads((Path(voice.model_path).parent / "voice.json").read_text())
    assert meta["index_sources"]["default"] == idx.name


def test_version_decides_between_two_voices(services, tmp_path):
    a = make_tiny_voice(tmp_path / "x" / "alpha.pth", version="v1")
    b = make_tiny_voice(tmp_path / "x" / "beta.pth", version="v2")
    ia = make_tiny_index(tmp_path / "x" / "one.index", dim=256)
    ib = make_tiny_index(tmp_path / "x" / "two.index", dim=768)
    result = services.models.import_paths([a, b, ia, ib])
    by_name = {v.name: v for v in result.voices}
    assert by_name["alpha"].has_index and by_name["beta"].has_index


def test_ambiguous_import_stops_with_the_plan_and_commits_decisions(services, tmp_path):
    files = [make_tiny_voice(tmp_path / "x" / "alpha.pth"), make_tiny_voice(tmp_path / "x" / "beta.pth", seed=1)]
    files += [make_tiny_index(tmp_path / "x" / "one.index"), make_tiny_index(tmp_path / "x" / "two.index", seed=1)]
    with pytest.raises(ConflictError) as e:
        services.models.import_paths(files)
    assert e.value.detail["plan"]["confident"] is False
    sid = stage(services, *files)
    plan = services.imports.plan(sid)
    assert all(i.status == "choose" and len(i.candidates) == 2 for i in plan.indexes)
    voice_of = {Path(f.name).stem: f.id for f in plan.files if f.kind == "voice"}
    index_of = {Path(f.name).stem: f.id for f in plan.files if f.kind == "index"}
    decisions = ImportDecisions(
        indexes=[IndexPlan(file_id=index_of["one"], target=f"file:{voice_of['beta']}"), IndexPlan(file_id=index_of["two"], target=f"file:{voice_of['alpha']}")]
    )
    result = services.imports.commit(sid, decisions)
    names = {a.name: next(v.name for v in result.voices if v.id == a.voice_id) for a in result.attached}
    assert names == {"one.index": "beta", "two.index": "alpha"}
    with pytest.raises(Exception):
        services.imports.plan(sid)  # the session is gone


def test_decisions_are_validated(services, tmp_path):
    sid = stage(services, make_tiny_voice(tmp_path / "v1.pth", version="v1"), make_tiny_index(tmp_path / "wide.index", dim=768))
    plan = services.imports.plan(sid)
    assert plan.indexes[0].status == "incompatible" and not plan.confident
    voice = next(f for f in plan.files if f.kind == "voice")
    with pytest.raises(ValidationError) as e:
        services.imports.commit(sid, ImportDecisions(indexes=[IndexPlan(file_id=plan.indexes[0].file_id, target=f"file:{voice.id}")]))
    assert e.value.detail["reason"] == "version_mismatch"


def test_lone_index_goes_to_the_inbox_and_is_assigned(services, tmp_path):
    voice = services.models.import_paths([make_tiny_voice(tmp_path / "kiki.pth")]).voices[0]
    other = services.models.import_paths([make_tiny_voice(tmp_path / "v1voice.pth", version="v1")]).voices[0]
    sid = stage(services, make_tiny_index(tmp_path / "mystery.index"), make_tiny_index(tmp_path / "trained_IVF4_Flat_nprobe_1_x_v2.index", add=False))
    result = services.imports.commit(sid)
    assert len(result.inbox) == 1 and any("no features" in s for s in result.skipped)
    item = services.inbox.list_indexes()[0]
    assert item.name == "mystery.index" and item.version == "v2"
    assert [c.target for c in item.candidates] == [f"voice:{voice.id}"]
    with pytest.raises(ValidationError):
        services.inbox.assign(item.id, AssignIndexRequest(voice_id=other.id))
    updated = services.inbox.assign(item.id, AssignIndexRequest(voice_id=voice.id))
    assert updated.has_index and services.inbox.list_indexes() == []


def test_attach_checks_the_index(services, tmp_path):
    voice = services.models.import_paths([make_tiny_voice(tmp_path / "v.pth", version="v1")]).voices[0]
    with pytest.raises(ValidationError):
        services.models.set_index(voice.id, "default", make_tiny_index(tmp_path / "wide.index", dim=768))
    with pytest.raises(ValidationError):
        services.models.set_index(voice.id, "default", make_tiny_index(tmp_path / "empty.index", dim=256, add=False))
    assert services.models.set_index(voice.id, "default", make_tiny_index(tmp_path / "ok-any-name.index", dim=256)).has_index


def test_base_models_import_list_and_fit(services, tmp_path):
    g = make_tiny_g_checkpoint(tmp_path / "base" / "MyBase_G.pth", version="v2", f0=True)
    import torch

    torch.save({"model": {f"discriminators.{i}.convs.0.weight": torch.zeros(1) for i in range(9)}}, str(tmp_path / "base" / "MyBase_D.pth"))
    result = services.models.import_paths([g, tmp_path / "base" / "MyBase_D.pth"])
    base = services.base_models.get(result.base_models[0])
    assert (base.sample_rate, base.version, base.pitch_guidance, base.has_discriminator, base.source) == ("40k", "v2", True, True, "imported")
    assert base.id in [b.id for b in services.base_models.list_base("40k", "v2", True)]
    assert base.id not in [b.id for b in services.base_models.list_base("48k", "v2", True)]
    services.base_models.check_fits(base.id, "40k", "v2", True)
    with pytest.raises(ValidationError):
        services.base_models.check_fits(base.id, "40k", "v1", True)
    exp = services.training.create(ExperimentCreate(name="e", fit=FitSettings(base_model=base.id)))
    services.training._preflight(exp, ["fit"])  # an imported base model needs no download
    services.base_models.delete(base.id)
    assert base.id not in [b.id for b in services.base_models.list_base()]


def test_separation_import_rejects_a_checkpoint_that_does_not_load(services, tmp_path):
    from rvc_next.engine.separate.presets import CONFIGS_DIR

    (tmp_path / "sep").mkdir()
    cfg = tmp_path / "sep" / "config_mine.yaml"
    cfg.write_text((CONFIGS_DIR / "model_bs_roformer_ep_368_sdr_12.9628.yaml").read_text())
    (tmp_path / "sep" / "mine.ckpt").write_bytes(b"not weights")
    sid = stage(services, cfg, tmp_path / "sep" / "mine.ckpt")
    plan = services.imports.plan(sid)
    s = plan.separations[0]
    assert (s.primary, s.secondary, s.checkpoint_id is not None) == ("vocals", "instrumental", True)
    with pytest.raises(ValidationError) as e:
        services.imports.commit(sid)
    assert e.value.detail["reason"] == "separation_load"
    assert services.separation_models.list_models() == []


def test_layout_migration(make_services, tmp_path):
    services = make_services()
    voice = services.models.import_paths([make_tiny_voice(tmp_path / "old.pth")]).voices[0]
    folder = services.models.root / voice.id
    legacy = services.settings.models_dir / voice.id
    folder.rename(legacy)
    services.close()
    again = make_services()
    assert (again.models.root / voice.id / "model.pth").is_file() and not legacy.exists()
    assert again.models.get(voice.id).model_path == str(again.models.root / voice.id / "model.pth")
