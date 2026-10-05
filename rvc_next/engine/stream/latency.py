"""Latency: the estimate shown while running, and the engine's own delay a measurement adds.

The original's display formula (``realtime_gui.py``: output latency + block + crossfade + 10 ms,
plus the input denoise term), extended with the output prefill, the split-stream ring and the input
device latency.
"""

from rvc_next.engine.stream.params import StreamParams

SOLA_SEARCH_MS = 10.0


def block_frames(block_ms: float, sample_rate: int) -> int:
    """A block in samples, rounded to whole 10 ms frames, as ``StreamEngine`` sizes it."""
    zc = sample_rate // 100
    return max(zc, int(round(block_ms / 10)) * zc)


def engine_delay_ms(stream: StreamParams) -> float:
    """How far ``StreamEngine``'s output trails its input, at most: the original's crossfade + 10 ms.

    The engine converts the newest ``crossfade + 10 ms + block`` of its buffer and SOLA starts the
    output block 0–10 ms into it, so the output trails by ``crossfade + 10 ms − offset``; input
    denoise delays the input by ``min(crossfade, 40 ms)`` more. ``test_engine_delay`` measures it.
    """
    crossfade = float(stream.crossfade_ms)
    return crossfade + SOLA_SEARCH_MS + (min(crossfade, 40.0) if stream.input_denoise else 0.0)


def estimate_latency_ms(stream: StreamParams, split: bool, input_latency_ms: float, output_latency_ms: float) -> float:
    block = float(stream.block_ms)
    latency = block + engine_delay_ms(stream)
    latency += block  # the output ring is pre-filled with one block of silence
    if split:
        latency += block  # the split-stream ring's target fill
    latency += max(0.0, input_latency_ms) + max(0.0, output_latency_ms)
    return latency
