"""Streaming parameters; the defaults are the original realtime GUI's fresh-install values."""

from dataclasses import dataclass

from rvc_next.engine.audio.effects import Effect


@dataclass(frozen=True)
class StreamParams:
    block_ms: int = 250
    crossfade_ms: int = 50
    context_ms: int = 2500
    """The original's ``extra_time``."""
    threshold_db: float = -60.0
    """-60 turns the gate off."""
    input_denoise: bool = False
    output_denoise: bool = False
    denoise_strength: float = 0.9
    """How much the noise gate lowers what it takes for noise (TorchGate's ``prop_decrease``)."""
    phase_vocoder: bool = False
    """Crossfade blocks with the phase vocoder (``audio/sola.py``) instead of the plain sin² fade."""
    effects: tuple[Effect, ...] = ()
    """A pedalboard chain over the converted voice (``audio/effects.py``); empty: none."""

    def same_buffers(self, other: "StreamParams") -> bool:
        """Whether ``other`` keeps the buffer layout, so only hot fields changed."""
        return (self.block_ms, self.crossfade_ms, self.context_ms) == (other.block_ms, other.crossfade_ms, other.context_ms)
