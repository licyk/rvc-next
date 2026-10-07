import numpy as np
import pytest

from rvc_next.engine.audio import effects as fx

pytest.importorskip("pedalboard")

SR = 40000


def _tone(seconds: float = 1.0) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * SR)) / SR)).astype(np.float32)


def test_every_effect_runs_with_its_defaults() -> None:
    x = _tone()
    for kind, spec in fx.EFFECTS.items():
        y = fx.apply(x, SR, (fx.effect(kind),))
        assert y.ndim == 1 and np.isfinite(y).all(), kind
        assert (len(y) > len(x)) if spec.tail else (len(y) == len(x)), kind
        for p in spec.params:
            assert p.min <= p.default <= p.max, (kind, p.name)


def test_tail_rings_out_then_stops() -> None:
    x = np.zeros(SR, dtype=np.float32)
    x[-400:] = 0.5
    y = fx.apply(x, SR, (fx.effect("delay", {"delay_seconds": 0.5, "feedback": 0.0, "mix": 0.5}),))
    assert len(y) == SR + SR // 2  # the echo of the last 10 ms, half a second on, then nothing
    assert np.abs(y[-400:]).min() > 0.2
    stereo = fx.apply(np.stack([x, x]), SR, (fx.effect("reverb"),))
    assert stereo.shape[0] == 2 and stereo.shape[1] > SR


def test_blocks_match_the_whole() -> None:
    x = _tone()
    chain = (fx.effect("compressor"), fx.effect("chorus"), fx.effect("reverb", {"room_size": 0.7}))
    whole = fx.board(chain)(x.reshape(1, -1), SR)[0]
    stream = fx.EffectsChain(chain, SR)
    blocks = np.concatenate([stream.process(x[i : i + 4000]) for i in range(0, len(x), 4000)])
    np.testing.assert_allclose(blocks, whole, atol=1e-5)
    stream.reset()
    again = np.concatenate([stream.process(x[i : i + 4000]) for i in range(0, len(x), 4000)])
    np.testing.assert_allclose(again, blocks, atol=1e-6)


def test_checks() -> None:
    assert fx.effect("gain", {"gain_db": -6}).values() == {"gain_db": -6.0}
    assert fx.from_dicts([{"kind": "limiter"}, {"kind": "gain", "params": {"gain_db": 3}}])[1].values() == {"gain_db": 3.0}
    for kind, params in (("echo", {}), ("gain", {"volume": 1.0}), ("gain", {"gain_db": 99.0})):
        with pytest.raises(ValueError):
            fx.effect(kind, params)


def test_a_stream_keeps_its_block_size_through_latency() -> None:
    """Pitch shift returns nothing for the first second of a stream (and, in pedalboard 0.9, silence after): Live leaves it out."""
    x = np.tile(_tone(), 4)
    chain = fx.EffectsChain((fx.effect("pitch_shift", {"semitones": 2.0}), fx.effect("gain")), SR)
    blocks = [chain.process(x[i : i + 4000]) for i in range(0, len(x), 4000)]
    assert all(b.shape == (4000,) for b in blocks)
    assert not fx.EFFECTS["pitch_shift"].streams and all(s.streams for k, s in fx.EFFECTS.items() if k != "pitch_shift")
