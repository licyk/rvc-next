from pathlib import Path

import numpy as np
import pytest

from rvc_next.engine.convert.params import VoiceParams
from rvc_next.engine.stream.engine import StreamEngine
from rvc_next.engine.stream.latency import engine_delay_ms, estimate_latency_ms
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


@pytest.mark.parametrize(("crossfade_ms", "denoise"), [(20, False), (50, False), (100, False), (50, True)])
def test_engine_delay(tiny_runtime, voice, crossfade_ms: int, denoise: bool) -> None:
    """Passthrough keeps the engine's geometry (buffer, SOLA, denoise); its delay is what a latency
    measurement adds to the measured round trip."""
    sr = 16000
    stream = StreamParams(block_ms=100, crossfade_ms=crossfade_ms, context_ms=300, input_denoise=denoise)
    engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="pm"), stream, sr, None)
    engine.passthrough = True
    t = np.arange(4 * sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.1 * np.random.default_rng(1).standard_normal(t.size)).astype(np.float32)
    n = engine.block_size
    out = np.concatenate(_run(engine, x, len(x) // n))
    seg = out[20 * n : 30 * n]
    lags = np.arange(0, 2000)
    score = [np.dot(seg, x[20 * n - d : 30 * n - d]) / np.linalg.norm(x[20 * n - d : 30 * n - d]) for d in lags]
    delay_ms = 1000 * int(lags[int(np.argmax(score))]) / sr
    assert engine_delay_ms(stream) - 10 <= delay_ms <= engine_delay_ms(stream)


def test_prewarm_takes_every_path_and_leaves_nothing_behind(tiny_runtime, voice, monkeypatch) -> None:
    """Prewarm runs the gate, the denoisers and the loudness match (librosa's rms), then the engine
    converts exactly as one that never prewarmed."""
    import librosa
    import torch

    stream = StreamParams(block_ms=100, crossfade_ms=50, context_ms=300)
    params = VoiceParams(f0_method="pm", rms_mix_rate=1.0)
    x = tone(0.6, sr=16000)
    outs = []
    # The first run in a process differs slightly from all later ones (a one-off in the shared
    # models, prewarm or not), so a throwaway run goes first and the last two are compared.
    for warm in (False, False, True):
        engine = StreamEngine(tiny_runtime, voice, params, stream, 16000, None)
        if warm:
            calls = []
            real = librosa.feature.rms
            monkeypatch.setattr(librosa.feature, "rms", lambda *a, calls=calls, real=real, **k: calls.append(1) or real(*a, **k))
            engine.prewarm()
            monkeypatch.setattr(librosa.feature, "rms", real)
            assert len(calls) >= 6  # three passes, input and output, with the loudness match forced on
            assert (engine.stream, engine.params, engine.passthrough) == (stream, params, False)
        torch.manual_seed(0)
        outs.append(np.concatenate(_run(engine, x, 3)))
    np.testing.assert_array_equal(outs[1], outs[2])


def test_protect_in_the_stream(tmp_path: Path, tiny_runtime, voice) -> None:
    import torch

    from tests.tiny import make_tiny_index

    index = tiny_runtime.index(make_tiny_index(tmp_path / "v.index", 768))
    rng = np.random.default_rng(2)
    x = np.concatenate([tone(0.4, sr=48000), 0.05 * rng.standard_normal(9600).astype(np.float32), tone(0.4, sr=48000)])

    def run(**kw) -> np.ndarray:
        torch.manual_seed(0)
        engine = StreamEngine(tiny_runtime, voice, VoiceParams(f0_method="pm", index_rate=1.0, rms_mix_rate=1, **kw), STREAM, 48000, index)
        return np.concatenate(_run(engine, x, 8))

    run()  # loads HuBERT and pm, whose construction draws from the RNG
    assert np.array_equal(run(protect=0.5, unvoiced="original"), run(protect=0.5, unvoiced="protect"))
    assert not np.array_equal(run(protect=0.0, unvoiced="original"), run(protect=0.0, unvoiced="protect"))
    assert run(protect=0.0, unvoiced="zero").shape[0] == 8 * int(0.1 * 48000)
