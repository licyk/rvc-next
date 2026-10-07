"""convert: offline conversion of files or folders."""

from pathlib import Path
from typing import Annotated

import typer

from rvc_next.cli.output import console, open_services, print_json
from rvc_next.cli.params import ParamOptions, build_params
from rvc_next.cli.progress import finish, job_progress


def convert(
    inputs: Annotated[list[Path], typer.Argument(help="Audio files or folders", exists=True)],
    voice: Annotated[str, typer.Option("--voice", "-v", help="Voice id, name or .pth path")],
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output folder (default: the job's folder in the data directory)", file_okay=False)] = None,
    preset: Annotated[str | None, typer.Option(help="Start from this preset instead of the voice's default")] = None,
    index: Annotated[Path | None, typer.Option(help="Index for a .pth voice given by path", exists=True, dir_okay=False)] = None,
    speaker: ParamOptions.speaker = None,
    pitch: ParamOptions.pitch = None,
    formant: ParamOptions.formant = None,
    f0: ParamOptions.f0 = None,
    index_rate: ParamOptions.index_rate = None,
    protect: ParamOptions.protect = None,
    rms_mix: ParamOptions.rms_mix = None,
    unvoiced: ParamOptions.unvoiced = None,
    autotune: ParamOptions.autotune = None,
    auto_pitch: ParamOptions.auto_pitch = None,
    auto_pitch_target: ParamOptions.auto_pitch_target = None,
    high_register: ParamOptions.high_register = None,
    f0_ceiling: ParamOptions.f0_ceiling = None,
    separate: Annotated[str | None, typer.Option("--separate", help="Separate first with this preset and convert the vocals")] = None,
    remix: Annotated[bool, typer.Option("--remix", help="Add the accompaniment back (needs --separate)")] = False,
    remix_gain: Annotated[float, typer.Option(help="Accompaniment gain in dB")] = 0.0,
    fmt: Annotated[str | None, typer.Option("--format", help="wav, flac, mp3, m4a or ogg (Opus)")] = None,
    resample: Annotated[int | None, typer.Option(min=16000, help="Resample the result to this rate")] = None,
    denoise: Annotated[float, typer.Option(min=0, max=1, help="Noise reduction over the converted voice; 0 turns it off")] = 0.0,
    recursive: Annotated[bool, typer.Option(help="Look into sub-folders of folder inputs")] = False,
    overwrite: Annotated[bool, typer.Option(help="Replace existing files in the output folder")] = False,
    preview: Annotated[float | None, typer.Option(min=1, max=120, help="Convert only the first N seconds")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the finished job as JSON")] = False,
) -> None:
    """Convert audio with a voice. Parameters start from the voice's default preset."""
    from rvc_next.core.audio.models import AudioRef
    from rvc_next.core.conversion.models import ConvertRequest, RemixStep, SeparationStep

    if remix and not separate:
        raise typer.BadParameter("--remix needs --separate")
    with open_services() as services:
        v = (
            services.models.temporary(Path(voice).expanduser(), index)
            if Path(voice).suffix.lower() == ".pth" and Path(voice).expanduser().is_file()
            else services.models.resolve(voice)
        )
        if v.location == "temporary":
            base = services.settings.settings.convert.default_params
        else:
            base = services.presets.resolve(preset, v.id).params if preset else services.presets.default_for(v.id).params
        params = build_params(
            base,
            speaker=speaker,
            pitch=pitch,
            formant=formant,
            f0=f0,
            index_rate=index_rate,
            protect=protect,
            rms_mix=rms_mix,
            unvoiced=unvoiced,
            autotune=autotune,
            auto_pitch=auto_pitch,
            auto_pitch_target=auto_pitch_target,
            high_register=high_register,
            f0_ceiling=f0_ceiling,
        )
        request = ConvertRequest(
            inputs=[AudioRef(kind="path", path=str(p.resolve())) for p in inputs],
            voice_id=v.id,
            params=params,
            separate=SeparationStep(preset=separate) if separate else None,
            remix=RemixStep(gain_db=remix_gain) if remix else None,
            output_format=fmt,  # ty: ignore[invalid-argument-type]
            resample_to=resample,
            output_denoise=denoise,
            preview_seconds=preview,
            output_dir=str(output.resolve()) if output else None,
            overwrite=overwrite,
            recursive=recursive,
        )
        with job_progress(services, enabled=not json_output):
            job = services.conversion.convert(request, foreground=True)
        if json_output:
            print_json(job)
        finish(job)
        if not json_output:
            for path in (job.result or {}).get("outputs", []):
                console.print(path, markup=False, highlight=False)
