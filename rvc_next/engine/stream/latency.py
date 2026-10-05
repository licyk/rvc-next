"""Estimated end-to-end latency.

The original's display formula (``realtime_gui.py``: output latency + block + crossfade + 10 ms,
plus the input denoise term), extended with the output prefill, the split-stream ring and the input
device latency.
"""

from rvc_next.engine.stream.params import StreamParams

SOLA_SEARCH_MS = 10.0


def estimate_latency_ms(stream: StreamParams, split: bool, input_latency_ms: float, output_latency_ms: float) -> float:
    block = float(stream.block_ms)
    latency = block + stream.crossfade_ms + SOLA_SEARCH_MS
    latency += block  # the output ring is pre-filled with one block of silence
    if split:
        latency += block  # the split-stream ring's target fill
    latency += max(0.0, input_latency_ms) + max(0.0, output_latency_ms)
    if stream.input_denoise:
        latency += min(stream.crossfade_ms, 40)
    return latency
