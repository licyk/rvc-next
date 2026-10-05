"""The voice-parameter options shared by convert, live run and preset save."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from rvc_next.core.params import VoiceParamsModel


class ParamOptions:
    speaker = Annotated[int | None, typer.Option("--speaker", min=0, max=109, help="Speaker id of a multi-speaker voice")]
    pitch = Annotated[float | None, typer.Option("--pitch", min=-24, max=24, help="Pitch shift in semitones")]
    formant = Annotated[float | None, typer.Option("--formant", min=-2, max=2, help="Formant shift in semitones")]
    f0 = Annotated[str | None, typer.Option("--f0", help="Pitch method: pm, rmvpe or fcpe")]
    index_rate = Annotated[float | None, typer.Option("--index-rate", min=0, max=1, help="Index strength")]
    protect = Annotated[float | None, typer.Option("--protect", min=0, max=0.5, help="Consonant protection; 0.5 turns it off")]
    rms_mix = Annotated[float | None, typer.Option("--rms-mix", min=0, max=1, help="Loudness match; 1 keeps the converted loudness")]


def build_params(base: VoiceParamsModel, **overrides: Any) -> VoiceParamsModel:
    """Apply the given options on top of ``base`` (a preset), keeping its other values."""
    names = {"speaker": "speaker_id", "f0": "f0_method", "rms_mix": "rms_mix_rate"}
    patch = {names.get(k, k): v for k, v in overrides.items() if v is not None}
    if "f0_method" in patch and patch["f0_method"] not in ("pm", "rmvpe", "fcpe"):
        raise typer.BadParameter("--f0 is pm, rmvpe or fcpe")
    return VoiceParamsModel.model_validate({**base.model_dump(), **patch})
