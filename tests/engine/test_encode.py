import numpy as np
import pytest

from rvc_next.engine.audio.io import decode, encode, probe


@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3", "m4a"])
@pytest.mark.parametrize("channels", [1, 2])
def test_encode_any_model_rate(tmp_path, fmt, channels):
    """A 40 kHz voice writes as mp3/m4a too: the encoder's nearest rate is used (the original failed here)."""
    x = (0.3 * np.sin(np.arange(40000) / 40000 * 2 * np.pi * 220)).astype(np.float32)
    audio = x if channels == 1 else np.stack([x, x])
    path = encode(tmp_path / f"out.{fmt}", audio, 40000, fmt)
    info = probe(path)
    assert info.channels == channels
    assert info.sample_rate == (40000 if fmt in ("wav", "flac") else 44100)
    assert abs(info.duration - 1.0) < 0.05
    decoded, sr = decode(path, 16000)
    assert sr == 16000 and abs(decoded.shape[0] - 16000) < 1200


def test_failed_encode_leaves_nothing(tmp_path):
    with pytest.raises(Exception):
        encode(tmp_path / "x.aiff", np.zeros(100, dtype=np.float32), 16000, "aiff")
    assert list(tmp_path.iterdir()) == []


def test_ogg_is_opus_at_48k(tmp_path):
    from rvc_next.engine.audio.io import probe

    path = encode(tmp_path / "x.ogg", (0.1 * np.sin(np.arange(40000) / 10)).astype(np.float32), 40000, "ogg")
    info = probe(path)
    assert info.codec == "opus" and info.sample_rate == 48000 and abs(info.duration - 1.0) < 0.05
