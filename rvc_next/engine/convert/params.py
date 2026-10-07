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
    """``engine.f0.METHODS``: pm, rmvpe, fcpe, crepe, crepe-tiny or swift."""
    index_rate: float = 0.75
    protect: float = 0.33
    """0.5 turns protection off."""
    rms_mix_rate: float = 0.25
    """1 keeps the converted envelope unchanged."""
    unvoiced: str = "original"
    """Frames the pitch detector finds unvoiced. "original": filled by interpolation, and ``protect``
    has no effect (the 2026 package's arithmetic, which the golden runs hold). "protect": filled, and
    ``protect`` applies to them. "zero": kept at 0 Hz with ``protect`` applied, as classic RVC and
    Applio convert, and as most shared voices were trained."""
    f0_high_register: str = "off"
    """RMVPE only, offline: "true_pitch" or "fold" corrects its octave errors above ~1040 Hz with a
    second pass (``engine/f0/high_register.py``); "off" is the original."""
    f0_ceiling: float = 1250.0
    """Highest pitch "true_pitch" writes, in Hz."""
    autotune: float = 0.0
    """0–1: how far voiced frames are pulled to the nearest semitone (0 off); offline and Live."""
    auto_pitch: str = "off"
    """Offline: "semitone" or "octave" adds the key that brings the input's median pitch to
    ``auto_pitch_target`` (on top of ``pitch``); "off" is the original."""
    auto_pitch_target: float = 155.0
    """Hz the automatic key aims at (the core resolves its "voice's own" choice to a number)."""
