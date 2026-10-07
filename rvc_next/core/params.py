"""Voice and stream parameters, as the API and settings see them.

They mirror the engine's frozen dataclasses (``engine/convert/params.py``,
``engine/stream/params.py``) and carry the ranges as field constraints, so every client shows the
same limits. ``to_engine()`` converts without importing torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from rvc_next.core.record import Record

if TYPE_CHECKING:
    from rvc_next.engine.convert.params import VoiceParams
    from rvc_next.engine.stream.params import StreamParams

F0Method = Literal["pm", "rmvpe", "fcpe", "crepe", "crepe-tiny", "swift"]
UnvoicedMode = Literal["protect", "zero", "original"]
HighRegisterMode = Literal["off", "true_pitch", "fold"]
AutoPitchMode = Literal["off", "semitone", "octave"]
DEFAULT_PITCH_TARGET = 155.0
"""Hz the automatic key aims at when the voice does not record its own pitch (Applio's male target)."""


def f0_assets(method: str, directml: bool = False) -> list[str]:
    """The assets a pitch method needs (pm and SwiftF0, which ships with rvc-next, need none)."""
    if method == "rmvpe":
        return ["rmvpe-onnx" if directml else "rmvpe"]
    return {"fcpe": ["fcpe"], "crepe": ["crepe"], "crepe-tiny": ["crepe-tiny"]}.get(method, [])


class VoiceParamsModel(Record):
    """The one parameter set for Convert, Live, the command line and the API."""

    speaker_id: int = Field(default=0, ge=0, le=109, description="Speaker of a multi-speaker voice")
    pitch: float = Field(default=0.0, ge=-24, le=24, description="Pitch shift in semitones")
    formant: float = Field(default=0.0, ge=-2, le=2, description="Formant shift in semitones")
    f0_method: F0Method = Field(default="rmvpe", description="Pitch extraction method")
    index_rate: float = Field(default=0.75, ge=0, le=1, description="Index strength: how much of the voice's own timbre is drawn from the index")
    protect: float = Field(default=0.33, ge=0, le=0.5, description="Consonant protection; 0.5 turns it off")
    rms_mix_rate: float = Field(default=0.25, ge=0, le=1, description="Loudness match; 1 keeps the converted loudness unchanged")
    unvoiced: UnvoicedMode = Field(
        default="protect",
        description="Frames without pitch: protect applies to them (protect); they also get no pitch, as classic RVC and Applio (zero); or RVC 2026's rule, where protect has no effect (original)",
    )
    f0_high_register: HighRegisterMode = Field(
        default="off",
        description="RMVPE above about 1040 Hz (offline): a second pass corrects its octave errors and writes the true pitch up to the ceiling (true_pitch), or half of it (fold)",
    )
    f0_ceiling: float = Field(default=1250.0, ge=1000, le=2000, description="Highest pitch the high-register correction writes, in Hz")
    autotune: float = Field(default=0.0, ge=0, le=1, description="Pull notes to the nearest semitone; 0 turns it off, 1 snaps")
    auto_pitch: AutoPitchMode = Field(
        default="off", description="Offline: add the key that brings the input's median pitch to the target, in semitones or in whole octaves (which keeps a song's key)"
    )
    auto_pitch_target: float = Field(default=0.0, ge=0, le=1000, description="Pitch the automatic key aims at, in Hz; 0: the voice's own (from its training), else 155 Hz")

    def to_engine(self, pitch_median: float | None = None) -> VoiceParams:
        """The engine's parameters; ``pitch_median`` (the voice's own, in Hz) resolves an automatic-key target of 0."""
        from rvc_next.engine.convert.params import VoiceParams

        values = self.model_dump()
        if not values["auto_pitch_target"]:
            values["auto_pitch_target"] = pitch_median or DEFAULT_PITCH_TARGET
        return VoiceParams(**values)


class StreamParamsModel(Record):
    """Live buffering and clean-up; the defaults are the original realtime GUI's."""

    block_ms: int = Field(default=250, ge=20, le=1500, description="Audio processed per step")
    crossfade_ms: int = Field(default=50, ge=10, le=150, description="Crossfade between blocks")
    context_ms: int = Field(default=2500, ge=50, le=5000, description="Past audio the model sees with each block")
    threshold_db: float = Field(default=-60.0, ge=-60, le=0, description="Input gate; -60 turns it off")
    input_denoise: bool = False
    output_denoise: bool = False
    denoise_strength: float = Field(default=0.9, ge=0, le=1, description="How much input and output noise reduction lowers the noise")
    phase_vocoder: bool = Field(default=False, description="Phase-vocoder crossfade: joins blocks without the dip a plain crossfade can leave")

    def to_engine(self) -> StreamParams:
        from rvc_next.engine.stream.params import StreamParams

        return StreamParams(**self.model_dump())
