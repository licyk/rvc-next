"""The ported CREPE: bit-identical to torchcrepe where it is installed, for RVC's call and filters.

torchcrepe adds random triangular dither (±20 cents) to every decoded pitch; the port does not by
default. The comparisons therefore replace ``torchcrepe.convert.dither`` with the identity (it is
looked up at call time) and pin ``weighted_argmax``'s cached bin weights to the undithered ones.
``test_dither_matches_seeded_torchcrepe`` checks the dithered path too: both draw the noise from
numpy's global RNG in the same order, so seeding it gives the same numbers.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from rvc_next.engine.errors import MissingAssetError, ModelFormatError
from rvc_next.engine.f0 import crepe_model as cm
from tests.tiny import tone

SR = 16000


def harmonic(seconds: float, freq: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return np.sum([amp / k * np.sin(2 * math.pi * freq * k * t) for k in range(1, 5)], axis=0).astype(np.float32)


def chirp(seconds: float, f_start: float = 80.0, f_end: float = 900.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * math.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * seconds)))).astype(np.float32)


def with_gaps(seconds: float) -> np.ndarray:
    x = tone(seconds, freq=180.0)
    n = x.size
    for a, b in ((0.2, 0.35), (0.6, 0.65), (0.85, 1.0)):
        x[int(a * n) : int(b * n)] = 0
    return x


SIGNALS = {
    "tone-110": lambda: harmonic(0.3, 110.0),
    "tone-220": lambda: harmonic(0.3, 220.0),
    "tone-440": lambda: harmonic(0.3, 440.0),
    "tone-880": lambda: harmonic(0.3, 880.0),
    "chirp": lambda: chirp(0.8),
    "noise": lambda: (0.1 * np.random.default_rng(0).standard_normal(int(0.3 * SR))).astype(np.float32),
    "silence": lambda: np.zeros(int(0.3 * SR), np.float32),
    "gaps": lambda: with_gaps(0.8),
    "short": lambda: harmonic(0.03, 300.0),  # 480 samples, 4 frames
}

_models: dict[str, cm.Crepe] = {}


@pytest.fixture
def torchcrepe(monkeypatch):
    tc = pytest.importorskip("torchcrepe")
    monkeypatch.setattr(tc.convert, "dither", lambda cents: cents)
    monkeypatch.setattr(tc.decode.weighted_argmax, "weights", (tc.CENTS_PER_BIN * torch.arange(360) + 1997.3794084376191)[None, :, None], raising=False)
    return tc


def ours(tc, capacity: str) -> cm.Crepe:
    if capacity not in _models:
        _models[capacity] = cm.load_crepe(Path(tc.__file__).parent / "assets" / f"{capacity}.pth")
    assert _models[capacity].capacity == capacity
    return _models[capacity]


def theirs_rvc(tc, x: np.ndarray, capacity: str, batch_size: int = 512) -> np.ndarray:
    """RVC's / Applio's sequence, verbatim."""
    f0, pd = tc.predict(torch.from_numpy(x)[None].float(), SR, 160, 50, 1100, model=capacity, batch_size=batch_size, device="cpu", return_periodicity=True)
    pd = tc.filter.median(pd, 3)
    f0 = tc.filter.mean(f0, 3)
    f0[pd < 0.1] = 0
    return f0[0].cpu().numpy()


def same(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("capacity", ["tiny", "full"])
@pytest.mark.parametrize("signal", list(SIGNALS))
def test_rvc_sequence_bit_identical(torchcrepe, capacity, signal):
    x = SIGNALS[signal]()
    model = ours(torchcrepe, capacity)
    xt = torch.from_numpy(x)[None]
    f0, pd = torchcrepe.predict(xt, SR, 160, 50, 1100, model=capacity, batch_size=512, device="cpu", return_periodicity=True)
    f0_ours, pd_ours = cm.predict(model, xt, 160, 50, 1100, "viterbi", 512)
    same(f0_ours, f0)
    same(pd_ours, pd)
    pd = torchcrepe.filter.median(pd, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    same(cm.median(pd_ours, 3), pd)
    same(cm.mean(f0_ours, 3), f0)
    f0[pd < 0.1] = 0  # the rest of RVC's sequence (theirs_rvc)
    expected = f0[0].numpy()

    p_len = x.size // 160
    got = cm.crepe_f0(model, x, p_len, "cpu")
    assert got.dtype == np.float32 and got.shape == (p_len,)
    np.testing.assert_array_equal(got, expected[:p_len])
    if signal.startswith("tone") and signal != "tone-880":
        # The tones are found (880 Hz too, but its 4th harmonic sits above fmax; just check voicing there).
        freq = float(signal.split("-")[1])
        voiced = got[got > 0]
        assert voiced.size > 0.8 * p_len and abs(np.median(voiced) / freq - 1) < 0.02
    if signal == "silence" and capacity == "tiny":
        # Digital silence normalises to all-zero frames; tiny calls them unvoiced, but full is confident
        # of one pitch on them (about 544 Hz), in torchcrepe as here.
        assert not got.any()


def test_chunks_and_weighted_argmax(torchcrepe):
    """Viterbi runs per ``batch_size`` chunk, as in torchcrepe; ``weighted_argmax`` and ``argmax`` match too."""
    x = torch.from_numpy(with_gaps(1.0))[None]
    model = ours(torchcrepe, "tiny")
    decoders: list[tuple[cm.DecoderName, Any]] = [
        ("viterbi", torchcrepe.decode.viterbi),
        ("weighted_argmax", torchcrepe.decode.weighted_argmax),
        ("argmax", torchcrepe.decode.argmax),
    ]
    for decoder, theirs in decoders:
        for batch_size in (512, 37):
            f0, pd = torchcrepe.predict(x, SR, 160, 50, 1100, model="tiny", decoder=theirs, batch_size=batch_size, device="cpu", return_periodicity=True)
            f0_ours, pd_ours = cm.predict(model, x, 160, 50, 1100, decoder, batch_size)
            same(f0_ours, f0)
            same(pd_ours, pd)
    np.testing.assert_array_equal(cm.crepe_f0(model, x[0].numpy(), 101, "cpu", batch_size=37), theirs_rvc(torchcrepe, x[0].numpy(), "tiny", batch_size=37))


def test_weighted_argmax_full(torchcrepe):
    x = torch.from_numpy(chirp(0.5))[None]
    f0, pd = torchcrepe.predict(x, SR, 160, 50, 1100, model="full", decoder=torchcrepe.decode.weighted_argmax, batch_size=512, device="cpu", return_periodicity=True)
    f0_ours, pd_ours = cm.predict(ours(torchcrepe, "full"), x, 160, 50, 1100, "weighted_argmax", 512)
    same(f0_ours, f0)
    same(pd_ours, pd)


def test_dither_matches_seeded_torchcrepe():
    tc = pytest.importorskip("torchcrepe")
    x = torch.from_numpy(tone(0.5, freq=200.0))[None]
    model = ours(tc, "tiny")
    np.random.seed(1234)
    f0, pd = tc.predict(x, SR, 160, 50, 1100, model="tiny", batch_size=20, device="cpu", return_periodicity=True)
    np.random.seed(1234)
    f0_ours, pd_ours = cm.predict(model, x, 160, 50, 1100, "viterbi", 20, dither=True)
    same(f0_ours, f0)
    same(pd_ours, pd)
    plain, _ = cm.predict(model, x, 160, 50, 1100, "viterbi", 20)
    cents = 1200 * torch.log2(f0_ours / plain)
    assert cents.abs().max() <= cm.CENTS_PER_BIN and cents.abs().max() > 0


def test_filters_match_torchcrepe_with_nans(torchcrepe):
    rng = np.random.default_rng(3)
    signals = torch.from_numpy(rng.uniform(50, 500, (2, 57)).astype(np.float32))
    signals[0, [0, 5, 6, 7, 30]] = float("nan")
    signals[1, 10:20] = float("nan")
    for win in (3, 5, 9):
        same(cm.median(signals, win), torchcrepe.filter.median(signals, win))
        same(cm.mean(signals, win), torchcrepe.filter.mean(signals, win))


def test_conversions_match_torchcrepe(torchcrepe):
    bins = torch.arange(360)
    same(cm.bins_to_frequency(bins), torchcrepe.convert.bins_to_frequency(bins))
    hz = torch.tensor([50.0, 110.0, 1100.0, 2006.0])
    same(cm.frequency_to_bins(hz), torchcrepe.convert.frequency_to_bins(hz))
    same(cm.frequency_to_bins(hz, torch.ceil), torchcrepe.convert.frequency_to_bins(hz, torch.ceil))
    torchcrepe.decode.viterbi(torch.rand(1, 360, 3))  # builds its transition matrix
    np.testing.assert_array_equal(cm.viterbi_transition(), torchcrepe.decode.viterbi.transition)


# -- without torchcrepe ------------------------------------------------------------------------


def test_one_frame_median():
    # torchcrepe's reflect padding raises on one frame; the port returns the value.
    same(cm.median(torch.tensor([[0.7]]), 3), torch.tensor([[0.7]]))


def test_crepe_f0_shape_and_padding():
    torch.manual_seed(0)
    model = cm.Crepe("tiny").eval()
    x = tone(0.25)
    frames = 1 + x.size // 160
    assert cm.crepe_f0(model, x, 10, "cpu").shape == (10,)
    padded = cm.crepe_f0(model, x, frames + 7, "cpu")
    assert padded.shape == (frames + 7,) and not padded[frames:].any()
    f0, pd = cm.predict(model, torch.from_numpy(x)[None], decoder="weighted_argmax")
    assert f0.shape == pd.shape == (1, frames)


def test_loader(tmp_path):
    with pytest.raises(MissingAssetError) as missing:
        cm.load_crepe(tmp_path / "absent.pth", "tiny")
    assert missing.value.assets == ["crepe-tiny"]

    path = tmp_path / "tiny.pth"
    torch.save(cm.Crepe("tiny").state_dict(), path)
    assert cm.load_crepe(path).capacity == "tiny"
    assert not cm.load_crepe(path, "tiny").training
    with pytest.raises(ModelFormatError, match="tiny"):
        cm.load_crepe(path, "full")

    torch.save({"weight": torch.zeros(3)}, tmp_path / "other.pth")
    with pytest.raises(ModelFormatError):
        cm.load_crepe(tmp_path / "other.pth")
    broken = cm.Crepe("tiny").state_dict()
    del broken["classifier.weight"]
    torch.save(broken, tmp_path / "broken.pth")
    with pytest.raises(ModelFormatError):
        cm.load_crepe(tmp_path / "broken.pth")
