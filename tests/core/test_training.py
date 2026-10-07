from pathlib import Path

import pytest
import soundfile as sf

from rvc_next.core.errors import BusyError, ConflictError, InvalidPathError, ValidationError
from rvc_next.core.training.models import Dataset, ExperimentCreate, ExperimentUpdate, ExportRequest, FitSettings, ImportExperimentRequest, RunRequest
from tests.tiny import make_tiny_g_checkpoint, tone

TINY_FIT = {
    "config_override": {
        "model": {
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
        },
        "train": {"segment_size": 4000},
    },
    "log_interval": 1,
    "num_workers": 0,
    "seed": 0,
}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    folder = tmp_path / "dataset"
    folder.mkdir()
    for i in range(3):
        sf.write(folder / f"{i}.wav", tone(3.0, 40000, 150 + 30 * i), 40000)
    return folder


def test_full_run_export_and_staleness(services, dataset, tmp_path, no_asset_checks, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    base = make_tiny_g_checkpoint(tmp_path / "base" / "G.pth")
    exp = services.training.create(
        ExperimentCreate(
            name="tiny",
            dataset=Dataset(folder=str(dataset)),
            f0_method="pm",
            fit=FitSettings(epochs=1, save_every=1, batch_size=2, pretrained_g=str(base), pretrained_d="", save_small_every=True),
        )
    )
    assert exp.sample_rate == "40k" and exp.version == "v2"
    report = services.training.scan_dataset("tiny")
    assert report.clips == 3 and abs(report.total_seconds - 9.0) < 0.1

    assert services.training.plan(exp, RunRequest()) == ["slice", "f0", "features", "fit", "index"]
    job = services.training.run("tiny", RunRequest(), foreground=True, fit_override=TINY_FIT)
    assert job.state == "completed", (job.error, services.jobs.read_log(job.id).lines[-20:])
    exp = services.training.get("tiny")
    assert all(exp.stages[s].status == "done" for s in ("slice", "f0", "features", "fit", "index"))
    assert services.training.metrics("tiny")
    kinds = {c.kind for c in services.training.checkpoints("tiny")}
    assert {"G", "small"} <= kinds
    samples = services.training.samples("tiny")
    assert [s.label for s in samples] == ["epoch 1"] and samples[0].kind == "sample" and samples[0].source_path

    # Unchanged inputs: nothing to run. Changing the dataset marks slice and later stale.
    with pytest.raises(ConflictError):
        services.training.run("tiny", RunRequest())
    # A file of the dataset changed on disk: its fingerprint no longer matches, so Run all redoes
    # slice and every stage after it; growing the epochs alone only continues fit.
    exp = services.training.get("tiny")
    services.training.update("tiny", ExperimentUpdate(fit=exp.fit.model_copy(update={"epochs": 2})))
    assert services.training.plan(services.training.get("tiny"), RunRequest()) == ["fit"]
    sf.write(dataset / "0.wav", tone(2.0, 40000, 220), 40000)
    assert services.training.plan(services.training.get("tiny"), RunRequest()) == ["slice", "f0", "features", "fit", "index"]
    sf.write(dataset / "0.wav", tone(3.0, 40000, 150), 40000)
    services.training.update("tiny", ExperimentUpdate(fit=exp.fit))
    services.training.update("tiny", ExperimentUpdate(sample_rate="48k"))
    exp = services.training.get("tiny")
    assert exp.stages["slice"].status == "stale" and exp.stages["index"].status == "stale"
    services.training.update("tiny", ExperimentUpdate(sample_rate="40k"))
    # Slicing options: a change marks slicing (and what follows) out of date; the defaults keep old fingerprints.
    from rvc_next.core.training.models import SliceSettings

    assert services.training.plan(services.training.get("tiny"), RunRequest())[0] == "slice"
    services.training.update("tiny", ExperimentUpdate(slicing=SliceSettings(cut="fixed", highpass=False)))
    assert services.training.get("tiny").slicing.cut == "fixed"
    services.training.update("tiny", ExperimentUpdate(slicing=SliceSettings()))

    tried = services.training.try_checkpoint("tiny", ExportRequest())
    assert tried.location == "temporary" and tried.has_index
    # A G checkpoint can be tried too: extracted to a small model once, under .try/.
    g = next(c for c in services.training.checkpoints("tiny") if c.kind == "G")
    tried_g = services.training.try_checkpoint("tiny", ExportRequest(checkpoint=g.name))
    small = services.training._dir("tiny") / ".try" / f"{Path(g.path).stem}.pth"
    assert tried_g.location == "temporary" and tried_g.has_index and small.is_file()
    mtime = small.stat().st_mtime_ns
    services.training.try_checkpoint("tiny", ExportRequest(checkpoint=g.name))
    assert small.stat().st_mtime_ns == mtime
    d = next((c for c in services.training.checkpoints("tiny") if c.kind == "D"), None)
    if d is not None:
        with pytest.raises(ConflictError):
            services.training.try_checkpoint("tiny", ExportRequest(checkpoint=d.name))
    job = services.training.export("tiny", ExportRequest(voice_name="Tiny trained"), foreground=True)
    assert job.state == "completed", job.error
    voice = services.models.get(job.result["voice_id"])
    assert voice.name == "Tiny trained" and voice.has_index
    assert services.training.get("tiny").voice_id == voice.id


def test_validation_and_legacy_import(services, tmp_path, dataset):
    with pytest.raises(InvalidPathError):
        services.training.create(ExperimentCreate(name="bad/name"))
    services.training.create(ExperimentCreate(name="empty"))
    with pytest.raises(ValidationError):
        services.training.run("empty", RunRequest(stages=["slice"]))
    legacy = tmp_path / "logs" / "old"
    (legacy / "0_gt_wavs").mkdir(parents=True)
    (legacy / "1_16k_wavs").mkdir(parents=True)
    sf.write(legacy / "0_gt_wavs" / "0_0.wav", tone(1.0, 40000), 40000)
    sf.write(legacy / "1_16k_wavs" / "0_0.wav", tone(1.0, 16000), 16000)
    exp = services.training.import_legacy(ImportExperimentRequest(path=str(legacy)))
    assert exp.legacy and exp.stages["slice"].status == "done"
    assert [e.name for e in services.training.list_experiments()] == ["empty", "old"]


def test_busy_when_running(services):
    services.training.create(ExperimentCreate(name="busy"))
    services.training._running["busy"] = "job"
    with pytest.raises(BusyError):
        services.training.run("busy", RunRequest())
    with pytest.raises(BusyError):
        services.training.delete("busy")


def test_dataset_upload(services, tmp_path):
    import io
    import zipfile

    import soundfile as sf

    from tests.tiny import tone

    services.training.create(ExperimentCreate(name="up"))
    wav = io.BytesIO()
    sf.write(wav, tone(1.0), 16000, format="WAV")
    exp = services.training.add_dataset_file("up", "a.wav", [wav.getvalue()])
    folder = Path(exp.dataset.folder)
    assert folder.name == "dataset_upload" and (folder / "a.wav").is_file()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("set/b.wav", wav.getvalue())
        zf.writestr("../evil.wav", wav.getvalue())
        zf.writestr("__MACOSX/set/._b.wav", b"x")
        zf.writestr("notes.txt", b"hi")
    services.training.add_dataset_file("up", "more.zip", [archive.getvalue()])
    assert sorted(p.name for p in folder.iterdir()) == ["a.wav", "b.wav", "evil.wav"]
    assert not (folder.parent / "evil.wav").exists()
    with pytest.raises(ValidationError):
        services.training.add_dataset_file("up", "x.txt", [b"hello"])
    assert services.training.scan_dataset("up").clips == 3
    services.training.clear_dataset_uploads("up")
    assert not folder.exists()

    # Multi-speaker: the zip's folders become the speaker table.
    services.training.create(ExperimentCreate(name="duo", dataset=Dataset(mode="multi")))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Ann_0_1/1.wav", wav.getvalue())
        zf.writestr("Bo_1_2/1.wav", wav.getvalue())
    exp = services.training.add_dataset_file("duo", "speakers.zip", [archive.getvalue()])
    assert [(s.name, s.id, s.repeat) for s in exp.dataset.speakers] == [("Ann", 0, 1), ("Bo", 1, 2)]
