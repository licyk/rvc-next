"""The ported resampler: bit-identical to torchaudio's where it is installed, and correct on its own."""

import math

import numpy as np
import pytest
import torch

from rvc_next.engine.audio.sinc_resample import Resample, apply_sinc_resample_kernel, sinc_resample_kernel

# The rate pairs rvc-next resamples between: Live's input to 16 kHz and the voice's rate to the
# device's, offline formant shift (``upp_res`` → hop), and the output resampling choices.
PAIRS = [(48000, 16000), (44100, 16000), (40000, 48000), (32000, 44100), (48000, 40000), (471, 400), (504, 480), (16000, 22050)]


def signal(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


@pytest.mark.parametrize(("orig", "new"), PAIRS)
@pytest.mark.parametrize("dtype", [torch.float32, None])
def test_bit_identical_to_torchaudio(orig, new, dtype):
    tat = pytest.importorskip("torchaudio.transforms")
    x = signal(2, 3, 4801)
    ours, theirs = Resample(orig, new, dtype=dtype), tat.Resample(orig, new, dtype=dtype)
    assert ours.width == theirs.width
    assert torch.equal(ours.kernel, theirs.kernel)
    assert torch.equal(ours(x), theirs(x))


def test_kaiser_matches_torchaudio():
    tat = pytest.importorskip("torchaudio.transforms")
    x = signal(1, 9000, seed=1)
    for beta in (None, 8.0):
        ours = Resample(44100, 16000, "sinc_interp_kaiser", lowpass_filter_width=16, rolloff=0.95, beta=beta)
        theirs = tat.Resample(44100, 16000, "sinc_interp_kaiser", lowpass_filter_width=16, rolloff=0.95, beta=beta)
        assert torch.equal(ours(x), theirs(x))


def test_lengths_identity_and_device_move():
    x = signal(1, 4800)
    same = Resample(16000, 16000)
    assert same(x) is x and not hasattr(same, "kernel")
    down = Resample(48000, 16000, dtype=torch.float32)
    assert down(x).shape == (1, 1600)
    assert Resample(16000, 22050)(signal(3, 1000)).shape == (3, math.ceil(22050 * 1000 / 16000))
    # The kernel is a buffer: it follows the module to a device and into its state dict.
    assert "kernel" in down.state_dict() and down.to("cpu").kernel.device.type == "cpu"


def test_a_tone_keeps_its_frequency():
    sr, f = 48000, 440.0
    t = torch.arange(sr, dtype=torch.float32) / sr
    y = Resample(sr, 16000)(torch.sin(2 * math.pi * f * t)[None])[0].numpy()
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    assert abs(np.argmax(spectrum) * 16000 / len(y) - f) < 2.0
    # Above the new Nyquist is filtered out (away from the ends, where the zero padding rings).
    alias = Resample(sr, 16000)(torch.sin(2 * math.pi * 12000.0 * t)[None])[0]
    assert float(alias[200:-200].abs().max()) < 0.01


def test_invalid_arguments():
    with pytest.raises(ValueError, match="integers"):
        sinc_resample_kernel(44100.5, 16000, 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Invalid resampling method"):
        Resample(48000, 16000, "linear")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="positive"):
        Resample(48000, 16000, lowpass_filter_width=0)
    kernel, width = sinc_resample_kernel(48000, 16000, 16000)
    with pytest.raises(TypeError, match="floating point"):
        apply_sinc_resample_kernel(torch.zeros(1, 100, dtype=torch.int16), 48000, 16000, 16000, kernel, width)
