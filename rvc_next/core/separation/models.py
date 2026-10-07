"""Separation records."""

from typing import Literal

from pydantic import Field

from rvc_next.core.audio.models import AudioRef
from rvc_next.core.record import Record


class SeparationPreset(Record):
    id: str
    title: str
    description: str
    kind: Literal["model", "chain"]
    steps: list[str]
    """Model preset ids, in order; one for a model preset."""
    stems: list[str]
    """The stems the preset writes."""
    primary_stem: str
    """The stem a conversion pre-step converts."""
    assets: list[str]
    source: Literal["builtin", "imported"] = "builtin"


class SeparateRequest(Record):
    inputs: list[AudioRef] = Field(min_length=1)
    preset: str
    output_format: Literal["wav", "flac", "mp3", "m4a", "ogg"] | None = None
    output_dir: str | None = None
