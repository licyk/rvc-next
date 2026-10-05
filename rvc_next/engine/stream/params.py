"""Streaming parameters; the defaults are the original realtime GUI's fresh-install values."""

from dataclasses import dataclass


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

    def same_buffers(self, other: "StreamParams") -> bool:
        """Whether ``other`` keeps the buffer layout, so only hot fields changed."""
        return (self.block_ms, self.crossfade_ms, self.context_ms) == (other.block_ms, other.crossfade_ms, other.context_ms)
