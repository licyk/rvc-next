"""preset list | save | remove."""

from typing import Annotated

import typer

from rvc_next.cli.output import console, open_services, print_json, print_table
from rvc_next.cli.params import ParamOptions, build_params


def preset_list(
    voice: Annotated[str | None, typer.Option(help="Only this voice's presets and the global ones")] = None, json_output: Annotated[bool, typer.Option("--json")] = False
) -> None:
    """List presets."""
    with open_services() as services:
        voice_id = services.models.resolve(voice, allow_path=False).id if voice else None
        items = services.presets.list_presets(voice_id)
        if json_output:
            print_json(items)
            return
        print_table(
            None,
            ["Id", "Name", "Voice", "Default", "Parameters"],
            [(p.id, p.name, p.voice_id or "(global)", "yes" if p.is_default else "", " ".join(f"{k}={v}" for k, v in p.params.model_dump().items())) for p in items],
        )


def preset_save(
    name: Annotated[str, typer.Argument(help="Preset name")],
    voice: Annotated[str | None, typer.Option(help="The voice this preset belongs to; omit for a global preset")] = None,
    default: Annotated[bool, typer.Option("--default", help="Make it the voice's default")] = False,
    speaker: ParamOptions.speaker = None,
    pitch: ParamOptions.pitch = None,
    formant: ParamOptions.formant = None,
    f0: ParamOptions.f0 = None,
    index_rate: ParamOptions.index_rate = None,
    protect: ParamOptions.protect = None,
    rms_mix: ParamOptions.rms_mix = None,
) -> None:
    """Save voice parameters as a preset, starting from the voice's default preset."""
    from rvc_next.core.presets.models import PresetCreate

    with open_services() as services:
        v = services.models.resolve(voice, allow_path=False) if voice else None
        base = services.presets.default_for(v.id).params if v else services.settings.settings.convert.default_params
        params = build_params(base, speaker=speaker, pitch=pitch, formant=formant, f0=f0, index_rate=index_rate, protect=protect, rms_mix=rms_mix)
        p = services.presets.create(PresetCreate(name=name, voice_id=v.id if v else None, params=params, is_default=default))
        console.print(f"Saved preset {p.name} ({p.id})")


def preset_remove(name: Annotated[str, typer.Argument(help="Preset id or name")], voice: Annotated[str | None, typer.Option(help="Voice the preset belongs to")] = None) -> None:
    """Remove a preset."""
    with open_services() as services:
        voice_id = services.models.resolve(voice, allow_path=False).id if voice else None
        p = services.presets.resolve(name, voice_id)
        services.presets.delete(p.id)
        console.print(f"Removed preset {p.name}")
