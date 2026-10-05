"""Preset records."""

from rvc_next.core.params import StreamParamsModel, VoiceParamsModel
from rvc_next.core.record import Record


class Preset(Record):
    id: str
    name: str
    voice_id: str | None = None
    """None for a global preset."""
    is_default: bool = False
    params: VoiceParamsModel
    stream: StreamParamsModel | None = None
    updated_at: str


class PresetCreate(Record):
    name: str
    voice_id: str | None = None
    params: VoiceParamsModel
    stream: StreamParamsModel | None = None
    is_default: bool = False


class PresetUpdate(Record):
    name: str | None = None
    params: VoiceParamsModel | None = None
    stream: StreamParamsModel | None = None
    is_default: bool | None = None
