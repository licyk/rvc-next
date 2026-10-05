import json
import zipfile
from pathlib import Path

import pytest

from rvc_next.core.errors import ConflictError, ModelFormatError, ValidationError
from rvc_next.core.models.models import Speaker, VoiceUpdate
from tests.tiny import make_tiny_g_checkpoint, make_tiny_index, make_tiny_voice


def test_import_pth_creates_folder_and_default_preset(services, tiny_voice_file):
    result = services.models.import_paths([tiny_voice_file], name="Alto")
    voice = result.voices[0]
    folder = services.models.root / voice.id
    assert (folder / "model.pth").read_bytes() == tiny_voice_file.read_bytes()
    assert json.loads((folder / "voice.json").read_text())["name"] == "Alto"
    assert voice.sample_rate == 40000 and voice.version == "v2" and voice.pitch_guidance and voice.speakers == []
    default = services.presets.default_for(voice.id)
    assert default.is_default and default.params.rms_mix_rate == 0.25
    assert services.models.resolve("alto").id == voice.id


def test_import_zip_with_index_and_checkpoint_detection(services, tmp_path):
    pth = make_tiny_voice(tmp_path / "Singer_e10_s100.pth")
    idx = make_tiny_index(tmp_path / "added_IVF1_Flat_nprobe_1_Singer_v2.index")
    archive = tmp_path / "share.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(pth, "Singer/Singer_e10_s100.pth")
        zf.write(idx, "Singer/" + idx.name)
    voice = services.models.import_paths([archive]).voices[0]
    assert voice.indexes and Path(voice.indexes["default"]).name == "model.index"
    g = make_tiny_g_checkpoint(tmp_path / "G_100.pth")
    result = services.models.import_paths([g])
    assert result.voices == [] and len(result.base_models) == 1
    with pytest.raises(ValidationError):
        services.models.import_upload("x.txt", [b"hi"])


def test_speaker_names_are_stored_beside_the_model(services, tmp_path):
    pth = make_tiny_voice(tmp_path / "multi.pth", speakers=3, speaker_info=[{"id": 0, "name": "A"}, {"id": 2, "name": "C"}])
    voice = services.models.import_paths([pth]).voices[0]
    assert [s.name for s in voice.speakers] == ["A", "C"]
    before = (services.models.root / voice.id / "model.pth").read_bytes()
    updated = services.models.update(voice.id, VoiceUpdate(speakers=[Speaker(id=1, name="B"), Speaker(id=0, name="A")]))
    assert [(s.id, s.name) for s in updated.speakers] == [(0, "A"), (1, "B")]
    assert (services.models.root / voice.id / "model.pth").read_bytes() == before
    with pytest.raises(ValidationError):
        services.models.update(voice.id, VoiceUpdate(speakers=[Speaker(id=5, name="X")]))


def test_legacy_root_suggests_indexes_and_hides(services, tmp_path):
    root = tmp_path / "RVC"
    weights = root / "assets" / "weights"
    weights.mkdir(parents=True)
    make_tiny_voice(weights / "Kiko_e20_s400.pth")
    logs = root / "logs" / "Kiko"
    logs.mkdir(parents=True)
    (logs / "added_IVF12_Flat_nprobe_1_Kiko_v2.index").write_bytes(b"x")
    (logs / "trained_IVF12_Flat_nprobe_1_Kiko_v2.index").write_bytes(b"x")
    root_id = services.models.add_legacy_root(str(root))
    voice = next(v for v in services.models.list_voices() if v.legacy)
    assert voice.location == f"legacy:{root_id}" and Path(voice.indexes["default"]).name.startswith("added_")
    # A scan suggests again only where nothing was chosen: detaching the index is a choice it keeps.
    services.models.set_index(voice.id, "default", None)
    services.models.scan_legacy_root(root_id)
    assert services.models.get(voice.id).indexes == {}
    services.models.delete(voice.id)
    assert voice.id not in [v.id for v in services.models.list_voices()]
    assert voice.id in [v.id for v in services.models.list_voices(include_hidden=True)]
    assert (weights / "Kiko_e20_s400.pth").exists()


def test_rebuild_from_disk_after_database_loss(make_services, tiny_voice_file):
    services = make_services()
    voice = services.models.import_paths([tiny_voice_file], name="Keep").voices[0]
    db_path = services.db.path
    services.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    again = make_services()
    assert [v.id for v in again.models.list_voices()] == [voice.id]
    assert again.models.get(voice.id).name == "Keep"


def test_delete_moves_folder_away_and_refuses_temporary_edit(services, tiny_voice_file, monkeypatch):
    import send2trash

    monkeypatch.setattr(send2trash, "send2trash", lambda p: (_ for _ in ()).throw(OSError("no trash")))
    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    folder = services.models.root / voice.id
    services.models.delete(voice.id)
    assert not folder.exists()
    assert any((services.settings.data_dir / "trash").iterdir())
    tmp = services.models.temporary(tiny_voice_file)
    with pytest.raises(ConflictError):
        services.models.update(tmp.id, VoiceUpdate(name="x"))


def test_merge_and_extract_jobs(services, tmp_path):
    a = services.models.import_paths([make_tiny_voice(tmp_path / "a.pth", seed=1)]).voices[0]
    b = services.models.import_paths([make_tiny_voice(tmp_path / "b.pth", seed=2)]).voices[0]
    from rvc_next.core.models.models import ExtractRequest, MergeRequest

    job = services.models.merge(MergeRequest(a=a.id, b=b.id, alpha=0.3, name="Blend"), foreground=True)
    assert job.state == "completed", job.error
    assert services.models.get(job.result["voice_id"]).name == "Blend"
    g = make_tiny_g_checkpoint(tmp_path / "exp" / "G_10.pth")
    job = services.models.extract(ExtractRequest(checkpoint=str(g), name="From G", sample_rate="40k"), foreground=True)
    assert job.state == "completed", job.error
    with pytest.raises(ModelFormatError):
        services.models.inspect(str(tmp_path / "exp" / "nope.pth")) if False else services.models._summarize(Path(__file__))
