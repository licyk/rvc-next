"""Recorded paths follow the data directory: moved, copied, or written by a version that kept absolute paths."""

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from rvc_next.core.context import build_services
from rvc_next.core.errors import NotFoundError
from rvc_next.core.files import is_absolute_text
from rvc_next.core.storage import PathCheckRequest
from rvc_next.core.training.models import ExperimentCreate, RunRequest, StageState
from rvc_next.core.training.service import EXPERIMENT_FILE, _named_at
from tests.tiny import make_tiny_index, make_tiny_voice


@pytest.fixture
def environ(tiny_assets_dir):
    return {"RVC_NEXT_PATHS__ASSETS_DIR": str(tiny_assets_dir), "RVC_NEXT_COMPUTE__DEVICE": "cpu"}


def _populate(data: Path, environ, wav_file: Path):
    """An upload, an output made from it, and an experiment on an uploaded dataset with two finished stages."""
    s = build_services(data_dir=data, environ=environ)
    upload = s.audio.upload("speech.wav", [wav_file.read_bytes()])
    result = s.audio.job_output_dir("j1") / "speech.converted.wav"
    result.parent.mkdir(parents=True)
    shutil.copyfile(wav_file, result)
    out = s.audio.add_output(None, result, "converted", "converted", source_path=Path(upload.path), publish=False)
    s.training.create(ExperimentCreate(name="e1"))
    s.training.add_dataset_file("e1", "speech.wav", [wav_file.read_bytes()])
    exp = s.training.get("e1")
    for stage in ("slice", "f0"):
        exp.stages[stage] = StageState(status="done", fingerprint=s.training._fingerprint(exp, stage))
    s.training._save(exp)
    s.close()
    return upload, out


def _records(s) -> list[str]:
    return [r[0] for r in s.db.fetchall("SELECT path FROM audio_files UNION ALL SELECT path FROM outputs UNION ALL SELECT source_path FROM outputs")]


def _current_stages(s) -> list[str]:
    exp = s.training.get("e1")
    return [st for st in ("slice", "f0") if st not in s.training.plan(exp, RunRequest())]


def _age_uploads(data: Path) -> None:
    then = time.time() - 2 * 86400
    for folder in (data / "uploads").iterdir():
        os.utime(folder, (then, then))


def test_everything_follows_a_moved_data_directory(tmp_path, environ, wav_file):
    first = tmp_path / "a" / "data"
    upload, out = _populate(first, environ, wav_file)
    data = tmp_path / "b" / "data"
    data.parent.mkdir()
    shutil.move(str(first), str(data))
    _age_uploads(data)
    s = build_services(data_dir=data, environ=environ)
    try:
        assert not any(is_absolute_text(p) for p in _records(s))
        assert s.audio.file_path(upload.id).is_relative_to(s.settings.data_dir)
        assert s.audio.output_file(out.id).is_file() and s.audio.source_file(out.id).is_file()
        assert s.audio.get_output(out.id).exists
        s.audio.cleanup()
        assert s.audio.file_path(upload.id).is_file()  # still referenced by its output
        exp = s.training.get("e1")
        assert Path(exp.dataset.folder or "").is_dir() and Path(exp.dataset.folder or "").is_relative_to(s.settings.data_dir)
        assert _current_stages(s) == ["slice", "f0"]  # nothing to train again
    finally:
        s.close()


def test_a_copied_data_directory_reads_and_deletes_only_its_own_files(tmp_path, environ, wav_file):
    original = tmp_path / "original"
    _, out = _populate(original, environ, wav_file)
    copy = tmp_path / "copy"
    shutil.copytree(original, copy)
    s = build_services(data_dir=copy, environ=environ)
    try:
        path = s.audio.output_file(out.id)
        assert path.is_relative_to(copy.resolve())
        s.audio.delete_output(out.id)
    finally:
        s.close()
    assert not path.exists()
    assert list((original / "outputs").rglob("*.wav")), "the original's file must stay"


def _make_legacy(data: Path, environ) -> None:
    """Rewrite what a version before relative paths kept: absolute rows, no meta, absolute dataset folders and
    fingerprints naming every path absolute."""
    s = build_services(data_dir=data, environ=environ)
    for table, column in (("audio_files", "path"), ("outputs", "path"), ("outputs", "source_path")):
        for row in s.db.fetchall(f"SELECT id, {column} AS value FROM {table} WHERE {column} IS NOT NULL"):
            s.db.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?", (str(s.storage.resolve(row["value"])), row["id"]))
    s.db.execute("DELETE FROM meta")
    folder = s.training.root / "e1"
    exp = s.training.get("e1")
    raw = json.loads((folder / EXPERIMENT_FILE).read_text())
    raw.pop("fingerprints")
    raw["dataset"]["folder"] = exp.dataset.folder
    for stage in ("slice", "f0"):
        raw["stages"][stage]["fingerprint"] = s.training._fingerprint(exp, stage, rename=_named_at(folder, None))
    (folder / EXPERIMENT_FILE).write_text(json.dumps(raw))
    s.close()


def test_a_legacy_data_directory_is_upgraded_after_a_move(tmp_path, environ, wav_file):
    first = tmp_path / "a" / "data"
    upload, out = _populate(first, environ, wav_file)
    _make_legacy(first, environ)
    data = tmp_path / "b" / "data"
    data.parent.mkdir()
    shutil.move(str(first), str(data))
    s = build_services(data_dir=data, environ=environ)
    try:
        assert not any(is_absolute_text(p) for p in _records(s))
        assert s.storage.previous_data_dir() == str(first.resolve())
        assert s.audio.output_file(out.id).is_file() and s.audio.file_path(upload.id).is_file()
        assert _current_stages(s) == ["slice", "f0"]  # the old fingerprints were recognised, not thrown away
        raw = json.loads((s.training.root / "e1" / EXPERIMENT_FILE).read_text())
        assert raw["fingerprints"] == 2 and raw["dataset"]["folder"] == "dataset_upload"
    finally:
        s.close()


def test_a_changed_dataset_still_counts_as_changed_after_the_upgrade(tmp_path, environ, wav_file):
    data = tmp_path / "data"
    _populate(data, environ, wav_file)
    _make_legacy(data, environ)
    shutil.copyfile(wav_file, data / "experiments" / "e1" / "dataset_upload" / "another.wav")
    s = build_services(data_dir=data, environ=environ)
    try:
        assert "slice" not in _current_stages(s)
    finally:
        s.close()


def test_missing_files_are_reported_never_skipped(tmp_path, environ, wav_file):
    data = tmp_path / "data"
    upload, out = _populate(data, environ, wav_file)
    s = build_services(data_dir=data, environ=environ)
    try:
        s.audio.output_file(out.id).unlink()
        assert s.audio.get_output(out.id).exists is False
        with pytest.raises(NotFoundError) as e:
            s.audio.output_file(out.id)
        assert e.value.detail["reason"] == "file_missing" and e.value.detail["kind"] == "output"
        report = s.storage.check()
        assert [m.id for m in report.missing] == [out.id] and not report.fixed
        s.audio.delete_output(out.id)  # nothing to trash, the record goes
        s.audio.file_path(upload.id).unlink()
        assert s.audio.get_file(upload.id).exists is False
        assert s.storage.check(PathCheckRequest(prune_missing=True)).pruned == 1
        with pytest.raises(NotFoundError):
            s.audio.get_file(upload.id)
    finally:
        s.close()


def test_a_missing_models_folder_keeps_voices_and_presets(services, tiny_voice_file):
    voice = services.models.import_paths([tiny_voice_file], name="Alto").voices[0]
    preset_ids = {p.id for p in services.presets.list_presets(voice.id)}
    models = services.settings.models_dir
    away = models.with_name("models-away")
    models.rename(away)
    services.models.rescan()
    kept = services.models.get(voice.id)
    assert kept.missing and {p.id for p in services.presets.list_presets(voice.id)} == preset_ids
    away.rename(models)
    services.models.rescan()
    assert not services.models.get(voice.id).missing


def test_a_legacy_voices_attached_index_follows_the_data_directory(tmp_path, environ):
    root = tmp_path / "RVC"
    weights = root / "assets" / "weights"
    weights.mkdir(parents=True)
    make_tiny_voice(weights / "Kiko.pth")
    first = tmp_path / "a" / "data"
    s = build_services(data_dir=first, environ=environ)
    s.models.add_legacy_root(str(root))
    voice = next(v for v in s.models.list_voices() if v.legacy)
    index = make_tiny_index(tmp_path / "upload" / "kiko.index")
    s.models.set_index(voice.id, "default", index, original_name="kiko.index", persist=True)
    stored = json.loads((first / "legacy-voices.json").read_text())[voice.id]["indexes"]["default"]
    assert stored.startswith("attached-indexes/")
    s.close()
    data = tmp_path / "b" / "data"
    data.parent.mkdir()
    shutil.move(str(first), str(data))
    s = build_services(data_dir=data, environ=environ)
    try:
        moved = s.models.get(voice.id)
        assert moved.has_index and Path(moved.indexes["default"]).is_relative_to(s.settings.data_dir)
    finally:
        s.close()
