"""model list | info | import | edit | remove | merge | extract | index attach | index build."""

from pathlib import Path
from typing import Annotated, Any

import typer

from rvc_next.cli.output import console, human_size, open_services, print_json, print_table, short
from rvc_next.cli.progress import finish, job_progress

JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON")]


def model_list(
    json_output: JsonOpt = False,
    all_: Annotated[bool, typer.Option("--all", help="Include hidden legacy voices")] = False,
    type_: Annotated[str, typer.Option("--type", help="voice, index (the unassigned ones), base or separation")] = "voice",
) -> None:
    """List models of one type."""
    with open_services() as services:
        if type_ != "voice":
            _list_other(services, type_, json_output)
            return
        voices = services.models.list_voices(include_hidden=all_)
        if json_output:
            print_json(voices)
            return
        print_table(
            None,
            ["Id", "Name", "Rate", "Ver", "Pitch", "Speakers", "Index", "Where"],
            [
                (v.id, v.name, f"{v.sample_rate // 1000}k", v.version, "yes" if v.pitch_guidance else "no", len(v.speakers) or "-", "yes" if v.has_index else "no", v.location)
                for v in voices
            ],
        )


def model_info(
    voice: Annotated[str, typer.Argument(help="Voice id, name or .pth path")],
    speakers: Annotated[bool, typer.Option("--speakers", help="List the speakers only")] = False,
    json_output: JsonOpt = False,
) -> None:
    """Show a voice: rate, version, pitch guidance, speakers, indexes and its default preset."""
    with open_services() as services:
        v = services.models.resolve(voice)
        if speakers:
            if json_output:
                print_json(v.speakers)
            elif v.speakers:
                for s in v.speakers:
                    typer.echo(f"{s.id}\t{s.name}")
            else:
                typer.echo("Single speaker")
            return
        preset = services.presets.default_for(v.id) if v.location != "temporary" else None
        if json_output:
            print_json({"voice": v, "default_preset": preset})
            return
        rows = [
            ("Name", v.name),
            ("Id", v.id),
            ("File", v.model_path),
            ("Size", human_size(v.size)),
            ("Sample rate", str(v.sample_rate)),
            ("Version", v.version),
            ("Pitch guidance", "yes" if v.pitch_guidance else "no"),
            ("Speakers", ", ".join(f"{s.id}: {s.name}" for s in v.speakers) or f"single ({v.speaker_slots} slots)"),
            ("Indexes", "; ".join(f"{k}: {p}" for k, p in v.indexes.items()) or "none"),
            ("Info", short(v.info, 100)),
        ]
        if preset:
            rows.append(("Default preset", ", ".join(f"{k}={val}" for k, val in preset.params.model_dump().items())))
        print_table(None, ["Field", "Value"], rows)


def model_import(
    paths: Annotated[list[Path], typer.Argument(help="Voices (.pth), indexes, .zip archives, base models (G/D), separation models (.ckpt + .yaml), folders or an RVC install")],
    name: Annotated[str | None, typer.Option(help="Name for a single imported voice")] = None,
    index: Annotated[Path | None, typer.Option(help="An .index to attach to a single voice, whatever its name")] = None,
    pair: Annotated[list[str] | None, typer.Option("--pair", help="VOICE.pth=FILE.index[:spkN] by file name; repeat for several")] = None,
    keep_unassigned: Annotated[bool, typer.Option("--keep-unassigned", help="Put indexes that cannot be paired for sure into the index inbox")] = False,
    extract: Annotated[bool, typer.Option("--extract", help="Turn training generators (G) into voices instead of base models")] = False,
    json_output: JsonOpt = False,
) -> None:
    """Import models; each file is identified by its content, and indexes are paired with voices whatever their names."""
    from rvc_next.core.models.models import ImportDecisions, IndexPlan

    with open_services() as services:
        imports = services.imports
        sid = imports.create()
        try:
            imports.add_paths(sid, [p.resolve() for p in paths + ([index] if index else [])])
            plan = imports.plan(sid)
            if not (plan.voices or plan.indexes or plan.generators or plan.separations):
                notes = "; ".join(f"{f.name}: {f.note or f.kind}" for f in plan.files) or "no model, index or separation files found"
                raise typer.BadParameter(f"Nothing to import ({notes})")
            decisions = ImportDecisions()
            by_name = {Path(f.name).name: f for f in plan.files}
            indexes = {i.file_id: i for i in plan.indexes}
            if index is not None:
                if len(plan.voices) != 1:
                    raise typer.BadParameter("--index pairs with a single voice; use --pair for several")
                fid = by_name[index.name].id
                indexes[fid] = IndexPlan(file_id=fid, target=f"file:{plan.voices[0].file_id}", key="default")
            for item in pair or []:
                voice_name, sep, rest = item.partition("=")
                index_name, _, key = rest.partition(":")
                voice_file, index_file = by_name.get(Path(voice_name).name), by_name.get(Path(index_name).name)
                if not sep or voice_file is None or index_file is None or voice_file.kind != "voice" or index_file.kind != "index":
                    raise typer.BadParameter(f"--pair {item!r}: give a voice .pth and an .index of this import by file name")
                indexes[index_file.id] = IndexPlan(file_id=index_file.id, target=f"file:{voice_file.id}", key=key or "default")
            if keep_unassigned:
                for fid, i in indexes.items():
                    if not i.confident and i.status != "manual" and i.target is None:
                        indexes[fid] = i.model_copy(update={"status": "manual"})
            decisions.indexes = list(indexes.values())
            if name and len(plan.voices) == 1:
                decisions.voices = [plan.voices[0].model_copy(update={"name": name})]
            if extract:
                decisions.generators = [g.model_copy(update={"action": "extract"}) for g in plan.generators]
            undecided = [i for i in decisions.indexes if not i.confident and i.status not in ("manual", "empty")]
            unpaired = [s for s in plan.separations if s.checkpoint_id is None]
            if undecided or unpaired:
                _print_plan(plan)
                console.print("[yellow]Some files could not be paired for sure. Pair them with --pair VOICE.pth=FILE.index, or use --keep-unassigned.[/yellow]")
                raise typer.Exit(3)
            result = imports.commit(sid, decisions)
        finally:
            try:
                imports.delete(sid)
            except Exception:
                pass
        if json_output:
            print_json(result)
            return
        for v in result.voices:
            console.print(f"Voice [bold]{v.name}[/bold] ({v.id}){' with index' if v.has_index else ''}")
        for a in result.attached:
            console.print(f"  {a.name} → {a.voice_id} ({a.key})", markup=False)
        for b in result.base_models:
            console.print(f"Base model [bold]{b}[/bold]")
        for m in result.separation_models:
            console.print(f"Separation model [bold]{m}[/bold] (preset id {m})")
        for i in result.inbox:
            console.print(f"[yellow]Index kept unassigned: {i} (rvc-next model index assign {i} <voice>)[/yellow]")
        for s in result.skipped:
            console.print(f"[yellow]Skipped {s}[/yellow]")


def _print_plan(plan: Any) -> None:
    files = {f.id: f for f in plan.files}
    print_table(
        "Files",
        ["File", "Kind", "Details"],
        [
            (
                f.name,
                f.kind,
                ", ".join(
                    x
                    for x in (
                        f.version or "",
                        f.sample_rate or "",
                        f"{f.dim}-dim, {f.vectors} vectors" if f.dim else "",
                        f.note,
                    )
                    if x
                ),
            )
            for f in plan.files
        ],
    )
    rows = []
    for i in plan.indexes:
        target = i.target or "-"
        if target.startswith("file:") and target[5:] in files:
            target = Path(files[target[5:]].name).name
        rows.append((Path(files[i.file_id].name).name, i.status, target, ", ".join(c.name for c in i.candidates) or "-"))
    if rows:
        print_table("Indexes", ["Index", "Status", "Proposed voice", "Compatible voices"], rows)


def model_edit(
    voice: Annotated[str, typer.Argument(help="Voice id or name")],
    name: Annotated[str | None, typer.Option(help="New name")] = None,
    description: Annotated[str | None, typer.Option(help="New description")] = None,
    speaker_name: Annotated[list[str] | None, typer.Option("--speaker-name", help="ID=NAME; repeat for several")] = None,
) -> None:
    """Rename a voice, describe it, or name its speakers (stored beside the .pth, which is not changed)."""
    from rvc_next.core.models.models import Speaker, VoiceUpdate

    with open_services() as services:
        v = services.models.resolve(voice, allow_path=False)
        speakers = None
        if speaker_name:
            current = {s.id: s.name for s in v.speakers}
            for item in speaker_name:
                sid, sep, sname = item.partition("=")
                if not sep or not sid.strip().isdigit():
                    raise typer.BadParameter(f"Expected ID=NAME, got {item!r}")
                current[int(sid)] = sname.strip()
            speakers = [Speaker(id=k, name=n) for k, n in sorted(current.items()) if n]
        updated = services.models.update(v.id, VoiceUpdate(name=name, description=description, speakers=speakers))
        console.print(f"Updated {updated.name}")


def model_export(
    voice: Annotated[str, typer.Argument(help="Voice id or name")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="The zip to write (default: <name>.zip here)")] = None,
) -> None:
    """Write a voice as a zip (its .pth with the speaker names, and its indexes) to import elsewhere."""
    with open_services() as services:
        v = services.models.resolve(voice, allow_path=False)
        name, stream = services.models.export_archive(v.id)
        target = output or Path(name)
        tmp = target.with_name(target.name + ".part")
        with open(tmp, "wb") as f:
            for chunk in stream:
                f.write(chunk)
        tmp.replace(target)
        console.print(f"Wrote {target}")


def model_remove(voice: Annotated[str, typer.Argument(help="Voice id or name")], yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask")] = False) -> None:
    """Move a library voice to the trash, or hide a legacy one."""
    with open_services() as services:
        v = services.models.resolve(voice, allow_path=False)
        if not yes and not typer.confirm(f"Remove {v.name}?"):
            raise typer.Exit(1)
        services.models.delete(v.id)
        console.print(f"{'Hid' if v.legacy else 'Removed'} {v.name}")


def model_merge(
    a: Annotated[str, typer.Argument(help="First voice")],
    b: Annotated[str, typer.Argument(help="Second voice")],
    name: Annotated[str, typer.Option(help="Name of the new voice")],
    alpha: Annotated[float, typer.Option(min=0, max=1, help="Weight of the first voice")] = 0.5,
    info: Annotated[str, typer.Option(help="Text stored in the model")] = "",
) -> None:
    """Merge two voices by weight into a new library voice."""
    from rvc_next.core.models.models import MergeRequest

    with open_services() as services:
        va, vb = services.models.resolve(a), services.models.resolve(b)
        with job_progress(services):
            job = services.models.merge(MergeRequest(a=va.id, b=vb.id, alpha=alpha, name=name, info=info), foreground=True)
        finish(job)
        console.print(f"Created {name} ({(job.result or {}).get('voice_id')})")


def model_extract(
    checkpoint: Annotated[str, typer.Argument(help="A G_*.pth checkpoint, or <experiment>/<file>")],
    name: Annotated[str, typer.Option(help="Name of the new voice")],
    sample_rate: Annotated[str | None, typer.Option("--sr", help="32k, 40k or 48k when it cannot be read from the experiment")] = None,
    version: Annotated[str | None, typer.Option(help="v1 or v2")] = None,
    pitch: Annotated[bool | None, typer.Option("--pitch/--no-pitch", help="Pitch guidance")] = None,
    info: Annotated[str, typer.Option(help="Text stored in the model")] = "",
) -> None:
    """Extract a small model from a training checkpoint into the library."""
    from rvc_next.core.models.models import ExtractRequest

    with open_services() as services:
        p = Path(checkpoint).expanduser()
        ref = str(p.resolve()) if p.exists() else checkpoint
        with job_progress(services):
            job = services.models.extract(ExtractRequest(checkpoint=ref, name=name, sample_rate=sample_rate, version=version, pitch_guidance=pitch, info=info), foreground=True)  # ty: ignore[invalid-argument-type]
        finish(job)
        console.print(f"Created {name} ({(job.result or {}).get('voice_id')})")


def index_attach(
    voice: Annotated[str, typer.Argument(help="Voice id or name")],
    file: Annotated[Path, typer.Argument(help="The added_*.index file", exists=True, dir_okay=False)],
    speaker: Annotated[int | None, typer.Option(help="Attach as this speaker's own index")] = None,
) -> None:
    """Attach an index to a voice."""
    with open_services() as services:
        v = services.models.resolve(voice, allow_path=False)
        services.models.set_index(v.id, "default" if speaker is None else f"spk{speaker}", file)
        console.print(f"Attached {file.name} to {v.name}")


def index_build(voice: Annotated[str, typer.Argument(help="Voice id or name")], experiment: Annotated[str, typer.Option(help="The experiment whose features to use")]) -> None:
    """Build a voice's index from an experiment's extracted features."""
    with open_services() as services:
        v = services.models.resolve(voice, allow_path=False)
        with job_progress(services):
            job = services.training.build_index_for_voice(v.id, experiment, foreground=True)
        finish(job)
        console.print(f"Built the index for {v.name}")


def model_catalog(json_output: JsonOpt = False) -> None:
    """List the ready-made voices that can be downloaded (the official RVC demo voices)."""
    with open_services() as services:
        items = services.assets.voice_catalog()
        if json_output:
            print_json(items)
            return
        print_table(
            None,
            ["Id", "Name", "Rate", "Ver", "Index", "Size", "In library"],
            [(v.id, v.name, f"{v.sample_rate // 1000}k", v.version, "yes" if v.has_index else "no", human_size(v.size), v.installed_voice_id or "") for v in items],
        )


def model_download(
    ids: Annotated[list[str] | None, typer.Argument(help="Voice ids from 'rvc-next model catalog'")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Every catalog voice not yet in the library")] = False,
) -> None:
    """Download ready-made voices into the library, with their indexes."""
    if not ids and not all_:
        raise typer.BadParameter("Give voice ids or --all")
    with open_services() as services:
        with job_progress(services):
            job = services.assets.download_voices(ids or [], all_missing=all_, foreground=True)
        finish(job)
        for catalog_id, voice_id in (job.result or {}).get("voices", {}).items():
            console.print(f"Added {catalog_id} as {voice_id}")


def _list_other(services: Any, type_: str, json_output: bool) -> None:
    if type_ == "index":
        items: Any = services.inbox.list_indexes()
        rows = [(i.id, i.name, i.version or f"{i.dim}-dim", i.vectors, human_size(i.size), ", ".join(c.name for c in i.candidates[:3])) for i in items]
        columns = ["Id", "Imported as", "Fits", "Vectors", "Size", "Compatible voices"]
    elif type_ == "base":
        items = services.base_models.list_base()
        rows = [
            (b.id, b.name, b.version, b.sample_rate, "yes" if b.pitch_guidance else "no", "yes" if b.has_discriminator else "no", "yes" if b.installed else "no") for b in items
        ]
        columns = ["Id", "Name", "Ver", "Rate", "Pitch", "D", "Installed"]
    elif type_ == "separation":
        items = services.separation_models.list_models()
        rows = [(m.id, m.name, m.model_type, f"{m.primary_label} / {m.secondary_label}", human_size(m.size)) for m in items]
        columns = ["Id (preset)", "Name", "Type", "Stems", "Size"]
    else:
        raise typer.BadParameter("--type is voice, index, base or separation")
    if json_output:
        print_json(items)
    else:
        print_table(None, columns, rows)


def index_assign(
    index_id: Annotated[str, typer.Argument(help="An unassigned index (rvc-next model list --type index)")],
    voice: Annotated[str, typer.Argument(help="Voice id or name")],
    speaker: Annotated[int | None, typer.Option(help="Attach as this speaker's own index")] = None,
) -> None:
    """Attach an unassigned index to a voice; it must fit the voice's version."""
    from rvc_next.core.models.models import AssignIndexRequest

    with open_services() as services:
        v = services.models.resolve(voice, allow_path=False)
        services.inbox.assign(index_id, AssignIndexRequest(voice_id=v.id, key="default" if speaker is None else f"spk{speaker}"))
        console.print(f"Attached {index_id} to {v.name}")
