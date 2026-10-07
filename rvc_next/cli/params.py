"""The voice-parameter options shared by convert, live run and preset save, and the effects option."""

from __future__ import annotations

from typing import Annotated, Any, get_args

import typer

from rvc_next.core.params import AutoPitchMode, EffectModel, F0Method, HighRegisterMode, UnvoicedMode, VoiceParamsModel

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
    high_register = Annotated[
        str | None,
        typer.Option("--high-register", help="RMVPE above ~1040 Hz: off, true_pitch (correct its octave errors) or fold (correct, an octave down); offline only"),
    ]
    f0_ceiling = Annotated[float | None, typer.Option("--f0-ceiling", min=1000, max=2000, help="Highest pitch --high-register true_pitch writes, in Hz")]
    autotune = Annotated[float | None, typer.Option("--autotune", min=0, max=1, help="Pull notes to the nearest semitone: 0 off, 1 snap")]
    auto_pitch = Annotated[
        str | None, typer.Option("--auto-pitch", help="off, semitone or octave: add the key that brings the input's median pitch to --auto-pitch-target (offline)")
    ]
    auto_pitch_target = Annotated[float | None, typer.Option("--auto-pitch-target", min=0, max=1000, help="Hz the automatic key aims at; 0: the voice's own, else 155")]


def build_params(base: VoiceParamsModel, **overrides: Any) -> VoiceParamsModel:
    """Apply the given options on top of ``base`` (a preset), keeping its other values."""
    names = {"speaker": "speaker_id", "f0": "f0_method", "rms_mix": "rms_mix_rate", "high_register": "f0_high_register"}
    patch = {names.get(k, k): v for k, v in overrides.items() if v is not None}
    if "f0_method" in patch and patch["f0_method"] not in F0_METHODS:
        raise typer.BadParameter(f"--f0 is one of {', '.join(F0_METHODS)}")
    if "unvoiced" in patch and patch["unvoiced"] not in get_args(UnvoicedMode):
        raise typer.BadParameter(f"--unvoiced is one of {', '.join(get_args(UnvoicedMode))}")
    if "auto_pitch" in patch and patch["auto_pitch"] not in get_args(AutoPitchMode):
        raise typer.BadParameter(f"--auto-pitch is one of {', '.join(get_args(AutoPitchMode))}")
    if "f0_high_register" in patch and patch["f0_high_register"] not in get_args(HighRegisterMode):
        raise typer.BadParameter(f"--high-register is one of {', '.join(get_args(HighRegisterMode))}")
    return VoiceParamsModel.model_validate({**base.model_dump(), **patch})


EffectOption = Annotated[
    list[str] | None,
    typer.Option("--effect", help="An effect over the converted voice, repeatable, applied in order: KIND or KIND:NAME=VALUE,... ('rvc-next effects' lists them)"),
]


def parse_effects(values: list[str]) -> list[EffectModel]:
    """``--effect`` values as a chain: ``reverb`` or ``reverb:room_size=0.6,wet_level=0.3``; ``none`` alone is an empty chain."""
    from pydantic import ValidationError

    if values == ["none"]:
        return []
    chain = []
    for value in values:
        kind, _, rest = value.partition(":")
        params: dict[str, float] = {}
        for item in filter(None, (s.strip() for s in rest.split(","))):
            name, eq, number = item.partition("=")
            try:
                params[name.strip()] = float(number)
            except ValueError:
                raise typer.BadParameter(f"--effect {value}: write NAME=VALUE, not {item!r}") from None
            if not eq:
                raise typer.BadParameter(f"--effect {value}: write NAME=VALUE, not {item!r}")
        try:
            chain.append(EffectModel.model_validate({"kind": kind.strip(), "params": params}))
        except ValidationError as e:
            raise typer.BadParameter(f"--effect {value}: {e.errors()[0]['msg']}") from None
    return chain
