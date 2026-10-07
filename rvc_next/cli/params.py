"""The voice-parameter options shared by convert, live run and preset save."""

from __future__ import annotations

from typing import Annotated, Any, get_args

import typer

from rvc_next.core.params import F0Method, UnvoicedMode, VoiceParamsModel

F0_METHODS = get_args(F0Method)


class ParamOptions:
    speaker = Annotated[int | None, typer.Option("--speaker", min=0, max=109, help="Speaker id of a multi-speaker voice")]
    pitch = Annotated[float | None, typer.Option("--pitch", min=-24, max=24, help="Pitch shift in semitones")]
    formant = Annotated[float | None, typer.Option("--formant", min=-2, max=2, help="Formant shift in semitones")]
    f0 = Annotated[str | None, typer.Option("--f0", help=f"Pitch method: {', '.join(F0_METHODS)}")]
    index_rate = Annotated[float | None, typer.Option("--index-rate", min=0, max=1, help="Index strength")]
    protect = Annotated[float | None, typer.Option("--protect", min=0, max=0.5, help="Consonant protection; 0.5 turns it off")]
    rms_mix = Annotated[float | None, typer.Option("--rms-mix", min=0, max=1, help="Loudness match; 1 keeps the converted loudness")]
    unvoiced = Annotated[
        str | None,
        typer.Option("--unvoiced", help="Frames without pitch: protect (protect applies), zero (no pitch there, as classic RVC) or original (RVC 2026: protect has no effect)"),
    ]


def build_params(base: VoiceParamsModel, **overrides: Any) -> VoiceParamsModel:
    """Apply the given options on top of ``base`` (a preset), keeping its other values."""
    names = {"speaker": "speaker_id", "f0": "f0_method", "rms_mix": "rms_mix_rate"}
    patch = {names.get(k, k): v for k, v in overrides.items() if v is not None}
    if "f0_method" in patch and patch["f0_method"] not in F0_METHODS:
        raise typer.BadParameter(f"--f0 is one of {', '.join(F0_METHODS)}")
    if "unvoiced" in patch and patch["unvoiced"] not in get_args(UnvoicedMode):
        raise typer.BadParameter(f"--unvoiced is one of {', '.join(get_args(UnvoicedMode))}")
    return VoiceParamsModel.model_validate({**base.model_dump(), **patch})
