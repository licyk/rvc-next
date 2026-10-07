"""Conversion requests."""

from typing import Literal

from pydantic import Field

from rvc_next.core.audio.models import AudioRef
from rvc_next.core.params import EffectModel, VoiceParamsModel
from rvc_next.core.record import Record


class SeparationStep(Record):
    preset: str = "vocals"


class RemixStep(Record):
    gain_db: float = Field(default=0.0, ge=-24, le=12)
    """Gain applied to the accompaniment."""
    vocal_gain_db: float = Field(default=0.0, ge=-24, le=12)


class ConvertRequest(Record):
    inputs: list[AudioRef] = Field(min_length=1)
    voice_id: str
    params: VoiceParamsModel = Field(default_factory=VoiceParamsModel)
    separate: SeparationStep | None = None
    remix: RemixStep | None = None
    output_format: Literal["wav", "flac", "mp3", "m4a", "ogg"] | None = None
    """None: ``convert.output_format``."""
    output_denoise: float = Field(default=0.0, ge=0, le=1)
    """Noise reduction over the converted voice (a non-stationary spectral gate); 0 turns it off."""
    effects: list[EffectModel] = Field(default_factory=list, max_length=16)
    """Effects over the converted voice, in order, before it is mixed back (needs pedalboard)."""
    resample_to: int | None = Field(default=None, ge=16000, le=192000)
    preview_seconds: float | None = Field(default=None, gt=0, le=120)
    output_dir: str | None = None
    """Command line only: write here instead of the job's output folder."""
    overwrite: bool = False
    """With ``output_dir``: replace files that exist instead of adding a numeric suffix."""
    recursive: bool = False
    """Look into sub-folders of a folder input."""


class EffectParamSpec(Record):
    name: str
    default: float
    min: float
    max: float
    unit: str


class EffectSpec(Record):
    kind: str
    params: list[EffectParamSpec]
    live: bool
    """Whether Live offers it."""


class EffectsCatalog(Record):
    """The effects a chain can hold, with their parameters' ranges."""

    available: bool
    """Whether pedalboard is installed (``pip install rvc-next[effects]``)."""
    effects: list[EffectSpec]
