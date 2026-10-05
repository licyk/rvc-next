"""Voice parameters: the one set shared by offline conversion, live conversion and every client."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceParams:
    speaker_id: int = 0
    pitch: float = 0.0
    """Semitones; screens step by 1, the API accepts fractions."""
    formant: float = 0.0
    """Semitones."""
    f0_method: str = "rmvpe"
    """pm, rmvpe or fcpe."""
    index_rate: float = 0.75
    protect: float = 0.33
    """0.5 turns protection off."""
    rms_mix_rate: float = 0.25
    """1 keeps the converted envelope unchanged."""
