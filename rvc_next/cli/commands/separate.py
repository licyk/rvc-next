"""separate: split vocals, accompaniment and reverb with a preset."""

from pathlib import Path
from typing import Annotated

import typer

from rvc_next.cli.output import console, open_services, print_json, print_table
from rvc_next.cli.progress import finish, job_progress


def separate(
    inputs: Annotated[list[Path] | None, typer.Argument(help="Audio files or folders", exists=True)] = None,
    preset: Annotated[str | None, typer.Option("--preset", "-p", help="Preset id; see --list")] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output folder", file_okay=False)] = None,
    fmt: Annotated[str | None, typer.Option("--format", help="wav, flac, mp3, m4a or ogg (Opus)")] = None,
    recursive: Annotated[bool, typer.Option(help="Look into sub-folders")] = False,
    list_presets: Annotated[bool, typer.Option("--list", help="List the presets and exit")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Separate audio with a preset or a chain of presets."""
    from rvc_next.core.audio.models import AudioRef
    from rvc_next.core.separation.models import SeparateRequest

    with open_services() as services:
        if list_presets:
            items = services.separation.presets()
            if json_output:
                print_json(items)
            else:
                print_table(None, ["Id", "Title", "Stems", "Description"], [(p.id, p.title, ", ".join(p.stems), p.description) for p in items])
            return
        if not inputs:
            raise typer.BadParameter("Give audio files or folders")
        preset_id = preset or services.settings.settings.separation.default_preset
        files = services.audio.expand_inputs([AudioRef(kind="path", path=str(p.resolve())) for p in inputs], recursive)
        refs = [AudioRef(kind="path", path=str(p)) for p, _ in files]
        request = SeparateRequest(inputs=refs, preset=preset_id, output_format=fmt, output_dir=str(output.resolve()) if output else None)  # ty: ignore[invalid-argument-type]
        with job_progress(services, enabled=not json_output):
            job = services.separation.separate(request, foreground=True)
        if json_output:
            print_json(job)
        finish(job)
        if not json_output:
            for src, stems in (job.result or {}).get("outputs", {}).items():
                console.print(Path(src).name, style="bold", markup=False)
                for label, path in stems.items():
                    console.print(f"  {label}: {path}", markup=False, highlight=False)
