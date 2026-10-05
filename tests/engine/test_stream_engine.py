from pathlib import Path

import numpy as np
import pytest

from rvc_next.engine.convert.params import VoiceParams
from rvc_next.engine.stream.engine import StreamEngine
from rvc_next.engine.stream.latency import estimate_latency_ms
from rvc_next.engine.stream.params import StreamParams
from tests.tiny import make_tiny_voice, tone

STREAM = StreamParams(block_ms=100, crossfade_ms=50, context_ms=300)


@pytest.fixture
def voice(tmp_path: Path, tiny_runtime):
    return tiny_runtime.voice(make_tiny_voice(tmp_path / "a.pth", speakers=3))


def _run(engine: StreamEngine, x: np.ndarray, blocks: int) -> list[np.ndarray]:
    n = engine.block_size
    return [engine.process(x[i * n : (i + 1) * n]) for i in range(blocks)]


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_block_in_block_out(tiny_runtime, voice, rate: int) -> None:
    engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="pm"), STREAM, rate, None)
    assert engine.block_size == rate // 10
    engine.prewarm()
    outs = _run(engine, tone(1.0, sr=rate), 4)
    assert all(o.shape == (engine.block_size,) and o.dtype == np.float32 for o in outs)
    assert np.isfinite(np.concatenate(outs)).all()


def test_fcpe_in_the_stream(tiny_runtime, voice) -> None:
    engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="fcpe"), STREAM, 48000, None)
    engine.prewarm()
    outs = _run(engine, tone(1.0, sr=48000), 3)
    assert all(o.shape == (engine.block_size,) for o in outs) and np.isfinite(np.concatenate(outs)).all()


def test_gate_and_denoise_keep_block_size(tiny_runtime, voice) -> None:
    stream = StreamParams(block_ms=100, crossfade_ms=50, context_ms=300, threshold_db=-40, input_denoise=True, output_denoise=True)
    engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="pm", rms_mix_rate=0.5, formant=1.0), stream, 48000, None)
    outs = _run(engine, tone(1.0, sr=48000), 3)
    assert all(o.shape == (4800,) for o in outs)


def test_silence_under_the_gate_is_silent(tiny_runtime, voice) -> None:
    stream = StreamParams(block_ms=100, crossfade_ms=50, context_ms=300, threshold_db=-30)
    engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="pm", rms_mix_rate=0.0), stream, 16000, None)
    quiet = (0.001 * np.random.default_rng(0).standard_normal(16000)).astype(np.float32)
    outs = _run(engine, quiet, 5)
    assert np.abs(outs[-1]).max() < 1e-3


def test_hot_update_and_reconfigure_keep_the_voice(tiny_runtime, voice) -> None:
    engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="pm"), STREAM, 16000, None)
    net_g = engine.voice.net_g
    engine.update(VoiceParams(f0_method="pm", pitch=5, speaker_id=2))
    assert engine.params.pitch == 5
    engine.reconfigure(StreamParams(block_ms=100, crossfade_ms=50, context_ms=300, threshold_db=-50))
    assert engine.block_size == 1600
    engine.reconfigure(StreamParams(block_ms=200, crossfade_ms=30, context_ms=500))
    assert engine.block_size == 3200 and engine.voice.net_g is net_g
    assert engine.process(tone(0.2, sr=16000)).shape == (3200,)


def test_voice_swap_between_blocks(tmp_path: Path, tiny_runtime, voice) -> None:
    other = tiny_runtime.voice(make_tiny_voice(tmp_path / "b.pth", version="v1", f0=False, sample_rate="48k", seed=3))
    engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="pm"), STREAM, 16000, None)
    engine.process(tone(0.1, sr=16000))
    engine.set_voice(other)
    assert engine.voice is other and engine.tgt_sr == 48000
    assert engine.process(tone(0.1, sr=16000)).shape == (1600,)


def test_speaker_id_reaches_the_synthesizer(tiny_runtime, voice) -> None:
    import torch

    x = tone(0.6, sr=16000)
    outs = []
    for speaker in (0, 2):
        engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="pm", speaker_id=speaker, rms_mix_rate=1.0), STREAM, 16000, None)
        torch.manual_seed(0)
        outs.append(np.concatenate(_run(engine, x, 3)))
    assert not np.allclose(outs[0], outs[1])


def test_passthrough_returns_the_input(tiny_runtime, voice) -> None:
    engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="pm"), STREAM, 16000, None)
    engine.passthrough = True
    x = tone(1.0, sr=16000)
    outs = np.concatenate(_run(engine, x, 6))
    # The input comes back delayed by the context the engine keeps; it is not silence.
    assert np.abs(outs[-1600:]).max() > 0.05


def test_latency_estimate() -> None:
    s = StreamParams(block_ms=250, crossfade_ms=50)
    assert estimate_latency_ms(s, split=False, input_latency_ms=10, output_latency_ms=20) == 250 + 50 + 10 + 250 + 30
    assert estimate_latency_ms(s, split=True, input_latency_ms=0, output_latency_ms=0) == 250 + 50 + 10 + 250 + 250
    denoise = StreamParams(block_ms=250, crossfade_ms=50, input_denoise=True)
    assert estimate_latency_ms(denoise, False, 0, 0) == 250 + 50 + 10 + 250 + 40
