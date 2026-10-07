"""Text to speech: the voices and a request."""

from pydantic import Field

from rvc_next.core.record import Record


class TtsVoice(Record):
    id: str
    """The service's voice name, as a request names it (``en-US-AriaNeural``)."""
    name: str
    locale: str
    language: str
    gender: str


class TtsVoices(Record):
    available: bool
    """Whether edge-tts is installed (``pip install rvc-next[tts]``)."""
    voices: list[TtsVoice]


class TtsRequest(Record):
    text: str = Field(min_length=1, max_length=5000)
    voice: str = Field(min_length=1)
    rate: int = Field(default=0, ge=-50, le=100, description="Speaking rate change in percent")
    pitch: int = Field(default=0, ge=-50, le=50, description="Pitch change in Hz")
