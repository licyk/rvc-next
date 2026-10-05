"""train list | new | dataset scan | dataset speakers | run | status | export | import."""

from pathlib import Path
from typing import Annotated

import typer

from rvc_next.cli.output import console, human_size, open_services, print_json, print_table
from rvc_next.cli.progress import finish, job_progress

JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON")]


def _stages_line(stages: dict) -> str:
    marks = {"done": "✓", "stale": "~", "running": "…", "failed": "✗", "pending": "·", "skipped": "-"}
    return " ".join(f"{name}{marks.get(st.status, '?')}" for name, st in stages.items())


def train_list(json_output: JsonOpt = False) -> None:
    """List experiments."""
    with open_services() as services:
        items = services.training.list_experiments()
        if json_output:
            print_json(items)
            return
        print_table(
            None,
            ["Name", "Rate", "Ver", "Pitch", "Mode", "Stages", "Last activity"],
            [(e.name, e.sample_rate, e.version, "yes" if e.pitch_guidance else "no", e.mode, _stages_line(e.stages), (e.last_activity or "")[:19]) for e in items],
        )


def train_new(
    name: Annotated[str, typer.Argument(help="Experiment name")],
    dataset: Annotated[
        Path | None, typer.Option(help="Dataset folder (single speaker), or the root of Name_ID_Repeat folders with --multi-speaker", exists=True, file_okay=False)
    ] = None,
    multi_speaker: Annotated[bool, typer.Option("--multi-speaker", help="Read speakers from Name_ID_Repeat subfolders of --dataset")] = False,
    sr: Annotated[str | None, typer.Option("--sr", help="32k, 40k or 48k")] = None,
    version: Annotated[str | None, typer.Option(help="v1 or v2")] = None,
    no_pitch: Annotated[bool, typer.Option("--no-pitch", help="Train without pitch guidance")] = False,
    f0: Annotated[str | None, typer.Option("--f0", help="pm or rmvpe")] = None,
    clean: Annotated[str | None, typer.Option(help="Separate the dataset with this preset before slicing, e.g. vocals-clean")] = None,
    json_output: JsonOpt = False,
) -> None:
    """Create an experiment."""
    from rvc_next.core.training.models import Dataset, ExperimentCreate

    with open_services() as services:
        ds = Dataset(folder=str(dataset.resolve()) if dataset and not multi_speaker else None, clean_preset=clean)
        exp = services.training.create(
            ExperimentCreate(name=name, dataset=ds, sample_rate=sr, version=version, pitch_guidance=False if no_pitch else None, f0_method=f0)  # ty: ignore[invalid-argument-type]
        )
        if multi_speaker and dataset:
            exp = services.training.speakers_from_folders(name, str(dataset.resolve()))
        if json_output:
            print_json(exp)
            return
        console.print(f"Created {exp.name} in {exp.path}")
        if exp.dataset.mode == "multi":
            console.print(f"{len(exp.dataset.speakers)} speakers: " + ", ".join(f"{s.name} ({s.id})" for s in exp.dataset.speakers))


def train_dataset_scan(name: Annotated[str, typer.Argument(help="Experiment name")], json_output: JsonOpt = False) -> None:
    """Report clip count, duration, sample rates and clips that are too short, too long, clipped or silent."""
    with open_services() as services:
        report = services.training.scan_dataset(name)
        if json_output:
            print_json(report)
            return
        console.print(f"{report.clips} clips, {report.total_seconds / 60:.1f} minutes; rates: " + ", ".join(f"{k} Hz × {v}" for k, v in report.sample_rates.items()))
        for spk, n in report.speakers.items():
            console.print(f"  {spk}: {n} clips")
        if report.issues:
            print_table("Problems", ["File", "Problem", "Detail"], [(Path(i.path).name, i.problem, i.detail) for i in report.issues])


def train_dataset_speakers(
    name: Annotated[str, typer.Argument(help="Experiment name")],
    from_folders: Annotated[Path | None, typer.Option(help="Fill the table from Name_ID_Repeat subfolders", exists=True, file_okay=False)] = None,
    json_output: JsonOpt = False,
) -> None:
    """Show the speaker table, or fill it from folders."""
    with open_services() as services:
        exp = services.training.speakers_from_folders(name, str(from_folders.resolve())) if from_folders else services.training.get(name)
        if json_output:
            print_json(exp.dataset.speakers)
            return
        print_table(None, ["Id", "Name", "Repeat", "Folder"], [(s.id, s.name, s.repeat, s.folder) for s in exp.dataset.speakers])


def train_run(
    name: Annotated[str, typer.Argument(help="Experiment name")],
    stage: Annotated[list[str] | None, typer.Option("--stage", help="clean, slice, f0, features, fit, index or all; repeat for several")] = None,
    epochs: Annotated[int | None, typer.Option(min=1, help="Total epochs")] = None,
    batch_size: Annotated[int | None, typer.Option(min=1, help="Batch size")] = None,
    save_every: Annotated[int | None, typer.Option(min=1, help="Save every N epochs")] = None,
    gpus: Annotated[str | None, typer.Option(help="auto, cpu, or ids like 0-1; saved in the experiment")] = None,
    force: Annotated[bool, typer.Option(help="Run the chosen stages even when their inputs are unchanged")] = False,
    base: Annotated[str | None, typer.Option("--base", help="Base model id (rvc-next model list --type base); saved in the experiment")] = None,
) -> None:
    """Run stages: every one that is not done, or the chosen ones."""
    from rvc_next.core.training.models import ExperimentUpdate, RunRequest

    with open_services() as services:
        exp = services.training.get(name)
        if epochs or batch_size or save_every or base or gpus:
            changes = {"epochs": epochs, "batch_size": batch_size, "save_every": save_every, "base_model": base, "gpus": gpus}
            fit = exp.fit.model_copy(update={k: v for k, v in changes.items() if v})
            exp = services.training.update(name, ExperimentUpdate(fit=fit))
        stages = None if not stage or "all" in stage else stage
        with job_progress(services):
            job = services.training.run(name, RunRequest(stages=stages, force=force), foreground=True)  # ty: ignore[invalid-argument-type]
        finish(job)
        console.print(f"Done: {', '.join((job.result or {}).get('stages', {}))}")


def train_status(name: Annotated[str, typer.Argument(help="Experiment name")], json_output: JsonOpt = False) -> None:
    """Show stages, checkpoints and the last metrics."""
    with open_services() as services:
        exp = services.training.get(name)
        cps = services.training.checkpoints(name)
        metrics = services.training.metrics(name)[-1:]
        if json_output:
            print_json({"experiment": exp, "checkpoints": cps, "last_metric": metrics[0] if metrics else None})
            return
        print_table(
            f"{exp.name} · {exp.sample_rate} {exp.version} · {'pitch' if exp.pitch_guidance else 'no pitch'} · {exp.f0_method}",
            ["Stage", "State", "Finished", "Counts"],
            [(k, v.status, (v.finished_at or "")[:19], " ".join(f"{a}={b}" for a, b in v.counts.items())) for k, v in exp.stages.items()],
        )
        if cps:
            print_table("Checkpoints", ["Name", "Kind", "Epoch", "Step", "Size"], [(c.name, c.kind, c.epoch or "", c.step or "", human_size(c.size)) for c in cps])
        if metrics:
            m = metrics[0]
            console.print(f"Last: epoch {m.epoch}, step {m.step}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.losses.items()))


def train_export(
    name: Annotated[str, typer.Argument(help="Experiment name")],
    checkpoint: Annotated[str | None, typer.Option("--checkpoint", help="A small model or G checkpoint name; default the latest")] = None,
    epoch: Annotated[int | None, typer.Option(help="The small model or checkpoint of this epoch")] = None,
    voice_name: Annotated[str | None, typer.Option(help="Name of the library voice")] = None,
) -> None:
    """Export a trained model and its index into the library."""
    from rvc_next.core.training.models import ExportRequest

    with open_services() as services:
        if epoch is not None and not checkpoint:
            match = [c for c in services.training.checkpoints(name) if c.epoch == epoch and c.kind in ("small", "G")]
            if not match:
                raise typer.BadParameter(f"No saved model for epoch {epoch}")
            checkpoint = sorted(match, key=lambda c: c.kind != "small")[0].name
        with job_progress(services):
            job = services.training.export(name, ExportRequest(checkpoint=checkpoint, voice_name=voice_name), foreground=True)
        finish(job)
        console.print(f"Exported to the library as {(job.result or {}).get('voice_id')}")


def train_import(
    path: Annotated[Path, typer.Argument(help="An original experiment folder (logs/<name>)", exists=True, file_okay=False)],
    name: Annotated[str | None, typer.Option(help="New name")] = None,
) -> None:
    """Copy an original RVC experiment folder in; stages whose outputs exist count as done."""
    from rvc_next.core.training.models import ImportExperimentRequest

    with open_services() as services:
        exp = services.training.import_legacy(ImportExperimentRequest(path=str(path.resolve()), name=name))
        console.print(f"Imported {exp.name}: {_stages_line(exp.stages)}")
