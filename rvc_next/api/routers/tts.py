"""Text to speech: an audio source."""

from fastapi import APIRouter

from rvc_next.api.deps import ServicesDep
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.core.audio.models import AudioFile
from rvc_next.core.tts.models import TtsRequest, TtsVoices

router = APIRouter(prefix="/v1/tts", tags=["tts"], responses=ERROR_RESPONSES)


@router.get("/voices", operation_id="list_tts_voices")
def list_voices(services: ServicesDep, refresh: bool = False) -> TtsVoices:
    return services.tts.voices(refresh)


@router.post("", operation_id="synthesize_speech")
def synthesize(services: ServicesDep, body: TtsRequest) -> AudioFile:
    """Speak the text and store it as an uploaded audio file, an input like any other."""
    return services.tts.synthesize(body)
