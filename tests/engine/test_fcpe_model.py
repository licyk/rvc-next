"""The ported FCPE: bit-identical to torchfcpe where it is installed, and loadable from the ``fcpe`` asset."""

from pathlib import Path

import numpy as np
import pytest
import torch

from rvc_next.engine.errors import MissingAssetError, ModelFormatError
from rvc_next.engine.f0.fcpe_model import load_fcpe
from tests.tiny import TINY_FCPE_CONFIG, make_tiny_fcpe, tone


def same(a: torch.Tensor, b: torch.Tensor) -> None:
    # Bit for bit; unvoiced frames are 0 · -inf = NaN in both.
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def wav(seconds: float, sr: int, seed: int = 0) -> torch.Tensor:
    x = tone(seconds, sr=sr, freq=220.0, amp=0.3) + 0.01 * np.random.default_rng(seed).standard_normal(int(seconds * sr)).astype(np.float32)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


def test_loads_and_infers(tmp_path):
    fcpe = load_fcpe(make_tiny_fcpe(tmp_path / "fcpe.pt"))
    f0 = fcpe.infer(wav(1.0, 16000), sr=16000)
    assert f0.shape == (1, 16000 // 160 + 1, 1)
    # Another rate is resampled to 16 kHz first. As in torchfcpe, the frames then follow the
    # resampled length (it pads at most one frame towards the input's); rvc-next always gives 16 kHz.
    assert fcpe.infer(wav(0.5, 44100), sr=44100).shape == (1, 8000 // 160 + 1, 1)


@pytest.mark.parametrize("decoder", ["local_argmax", "argmax"])
def test_bit_identical_to_torchfcpe(tmp_path, decoder):
    torchfcpe = pytest.importorskip("torchfcpe")
    path = make_tiny_fcpe(tmp_path / "fcpe.pt")
    ours, theirs = load_fcpe(path), torchfcpe.spawn_infer_model_from_pt(str(path), "cpu")
    for sr, x in ((16000, wav(1.0, 16000)), (44100, wav(0.5, 44100, seed=1))):
        mel = ours.wav2mel(x, sr)
        same(mel, theirs.wav2mel(x, sr))
        same(ours.model(mel), theirs.model(mel))
        same(ours.infer(x, sr=sr, decoder_mode=decoder, threshold=0.006), theirs.infer(x, sr=sr, decoder_mode=decoder, threshold=0.006))


def test_bundled_weights_match_torchfcpe():
    """The real model: the ``fcpe`` asset is the file torchfcpe bundles; both give the same f0."""
    torchfcpe = pytest.importorskip("torchfcpe")
    bundled = Path(torchfcpe.__file__).parent / "assets" / "fcpe_c_v001.pt"
    if not bundled.is_file():
        pytest.skip("torchfcpe without its bundled model")
    ours, theirs = load_fcpe(bundled), torchfcpe.spawn_bundled_infer_model("cpu")
    x = wav(1.0, 16000)
    same(ours.infer(x, sr=16000), theirs.infer(x, sr=16000, decoder_mode="local_argmax", threshold=0.006))


def test_missing_and_unsupported_checkpoints(tmp_path):
    with pytest.raises(MissingAssetError) as missing:
        load_fcpe(tmp_path / "absent.pt")
    assert missing.value.assets == ["fcpe"]
    attention = {"config_dict": {**TINY_FCPE_CONFIG, "model": {**TINY_FCPE_CONFIG["model"], "conv_only": False}}, "model": {}}
    torch.save(attention, tmp_path / "attention.pt")
    with pytest.raises(ModelFormatError, match="convolution-only"):
        load_fcpe(tmp_path / "attention.pt")
    torch.save({"weights": {}}, tmp_path / "other.pt")
    with pytest.raises(ModelFormatError, match="not an FCPE checkpoint"):
        load_fcpe(tmp_path / "other.pt")


def test_provider_from_the_runtime(tiny_runtime):
    f0 = tiny_runtime.f0("fcpe").compute(tone(1.0, sr=16000), 101)
    assert f0.shape == (101,) and not np.isnan(f0).any()
