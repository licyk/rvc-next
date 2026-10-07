"""The torch SwiftF0 against swift-f0's ONNX graph run by onnxruntime, where both are installed.

The harmonic tones carry a -80 dBFS noise floor, as any recording does: without one, most
spectrogram bins of a pure tone hold only float32 rounding noise, whose log the network reads, and
then the result follows the exact rounding of the DFT's sums. The module reproduces that rounding
too (onnxruntime's trig tables, the same products), which ``test_pure_tones`` checks; the noise
floor keeps the main comparisons independent of the BLAS a machine uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rvc_next.engine.errors import MissingAssetError
from rvc_next.engine.f0.swift_model import DEFAULT_WEIGHTS, FMAX, FMIN, HOP, WINDOW_FRAMES, SwiftF0Model, band, load_swift_f0

SR = 16000
CONFIDENCE_ATOL = 5e-5  # measured: 6e-6 (silence), under 1e-6 elsewhere
PITCH_RTOL = 1e-5  # measured: 4e-8


def harmonic(freq: float, seconds: float = 2.0, floor: float = 1e-4, seed: int = 0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = sum(0.3 / k * np.sin(2 * np.pi * freq * k * t) for k in range(1, 8) if freq * k < SR / 2)
    return (x + floor * np.random.default_rng(seed).standard_normal(len(t))).astype(np.float32)


def chirp(seconds: float = 3.0, f0: float = 60.0, f1: float = 1800.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t**2)
    return (0.3 * np.sin(phase) + 1e-4 * np.random.default_rng(1).standard_normal(len(t))).astype(np.float32)


def gaps() -> np.ndarray:
    # Tone, digital silence, tone, silence: the silence rule and the frames at the edges.
    x = harmonic(196.0, 3.0, seed=2)
    x[int(0.7 * SR) : int(1.3 * SR)] = 0
    x[int(2.2 * SR) :] = 0
    return x


def long_signal() -> np.ndarray:
    # Longer than one detect window (1875 frames, 30 s): a gliding voice-like tone over noise.
    seconds = 33.0
    t = np.arange(int(seconds * SR)) / SR
    freq = 220 * 2 ** (np.sin(2 * np.pi * t / 7.0))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    x = sum(0.25 / k * np.sin(k * phase) for k in range(1, 6)) * (np.sin(2 * np.pi * t / 3.0) > -0.3)
    return (x + 1e-3 * np.random.default_rng(3).standard_normal(len(t))).astype(np.float32)


SIGNALS = {
    "tone-80": lambda: harmonic(80.0),
    "tone-220": lambda: harmonic(220.0),
    "tone-440": lambda: harmonic(440.0),
    "tone-1500": lambda: harmonic(1500.0),
    "chirp": chirp,
    "white-noise": lambda: (0.1 * np.random.default_rng(4).standard_normal(2 * SR)).astype(np.float32),
    "silence": lambda: np.zeros(SR, dtype=np.float32),
    "gaps": gaps,
    "long": long_signal,
    "short": lambda: harmonic(300.0, 0.01),  # 160 samples: less than one hop, one frame
    "one-hop-and-a-bit": lambda: harmonic(300.0, 300 / SR),
}


@pytest.fixture(scope="module")
def model() -> SwiftF0Model:
    return load_swift_f0()


@pytest.fixture(scope="module")
def detector():
    pytest.importorskip("onnxruntime")
    swift_f0 = pytest.importorskip("swift_f0")
    return swift_f0.SwiftF0()


def run_graph(detector, audio: np.ndarray, fmin: float = FMIN, fmax: float = FMAX) -> tuple[np.ndarray, np.ndarray]:
    """``model.onnx`` itself, without the library's windows or silence rule."""
    feed = {"audio": audio[None, :], "fmin": np.asarray(fmin, dtype=np.float32), "fmax": np.asarray(fmax, dtype=np.float32)}
    pitch, confidence = detector.session.run(["pitch", "confidence"], feed)
    return pitch[0], confidence[0]


def run_module(model: SwiftF0Model, audio: np.ndarray, fmin: float = FMIN, fmax: float = FMAX) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        pitch, confidence = model(torch.from_numpy(audio).unsqueeze(0), fmin, fmax)
    return pitch[0].numpy(), confidence[0].numpy()


def differences(theirs: tuple[np.ndarray, np.ndarray], ours: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    """Max |Δconfidence| over all frames and max relative Δpitch over frames voiced in both."""
    (p0, c0), (p1, c1) = theirs, ours
    assert p0.shape == p1.shape == c0.shape == c1.shape
    voiced = (c0 >= 0.5) & (c1 >= 0.5)
    pitch = float((np.abs(p1 - p0) / p0)[voiced].max()) if voiced.any() else 0.0
    return float(np.abs(c1.astype(np.float64) - c0).max()), pitch


def assert_close(theirs: tuple[np.ndarray, np.ndarray], ours: tuple[np.ndarray, np.ndarray]) -> None:
    confidence, pitch = differences(theirs, ours)
    assert confidence < CONFIDENCE_ATOL, confidence
    assert pitch < PITCH_RTOL, pitch


@pytest.mark.parametrize("name", list(SIGNALS))
def test_graph_matches_onnxruntime(model, detector, name):
    audio = SIGNALS[name]()
    theirs, ours = run_graph(detector, audio), run_module(model, audio)
    assert ours[0].dtype == np.float64 and ours[1].dtype == np.float32  # as the graph's outputs
    assert len(ours[0]) == max(1, len(audio) // HOP)
    assert_close(theirs, ours)


@pytest.mark.parametrize("name", list(SIGNALS))
def test_detect_matches_swift_f0(model, detector, name):
    audio = SIGNALS[name]()
    theirs = detector.detect(audio, SR)
    ours = model.detect(audio)
    np.testing.assert_array_equal(ours.timestamps, theirs.timestamps)
    assert ours.pitch_hz.dtype == ours.confidence.dtype == np.float64
    assert_close((theirs.pitch_hz, theirs.confidence), (ours.pitch_hz, ours.confidence))
    # The silence rule zeroes exactly the same frames.
    np.testing.assert_array_equal(ours.confidence == 0, theirs.confidence == 0)
    if name == "long":
        assert len(audio) > WINDOW_FRAMES * HOP


@pytest.mark.parametrize("name", ["tone-220", "chirp", "long"])
def test_narrowed_band(model, detector, name):
    audio = SIGNALS[name]()
    fmin, fmax = 50.0, 1100.0
    theirs, ours = run_graph(detector, audio, fmin, fmax), run_module(model, audio, fmin, fmax)
    assert_close(theirs, ours)
    # Frames whose best bin lies outside the band have confidence 0; pitches stay inside it.
    np.testing.assert_array_equal(ours[1] == 0, theirs[1] == 0)
    voiced = ours[1] >= 0.5
    assert np.all((ours[0][voiced] > fmin * 0.95) & (ours[0][voiced] < fmax * 1.05))
    found, expected = model.detect(audio, fmin, fmax), detector.detect(audio, SR, fmin, fmax)
    assert_close((expected.pitch_hz, expected.confidence), (found.pitch_hz, found.confidence))


@pytest.mark.parametrize("freq", [80.0, 220.0, 1500.0])
def test_pure_tones(model, detector, freq):
    # No noise floor: the near-empty bins are float32 rounding noise (see the module docstring).
    audio = harmonic(freq, floor=0.0)
    assert_close(run_graph(detector, audio), run_module(model, audio))


def test_band_is_clamped_and_checked():
    assert band(None, None) == (FMIN, FMAX)
    assert band(10.0, 5000.0) == (FMIN, FMAX)
    assert band(100.0, 800.0) == (100.0, 800.0)
    for fmin, fmax in ((500.0, 400.0), (200.0, 200.0), (200.0, 201.0), (float("nan"), 500.0)):
        with pytest.raises(ValueError):
            band(fmin, fmax)


def test_weights_load_with_weights_only(tmp_path):
    weights = torch.load(DEFAULT_WEIGHTS, map_location="cpu", weights_only=True)
    assert all(isinstance(v, torch.Tensor) for v in weights.values())
    assert DEFAULT_WEIGHTS.stat().st_size < 200_000
    model = load_swift_f0()
    assert not model.training
    pitch, confidence = model(torch.zeros(1, 4 * HOP))
    assert pitch.shape == confidence.shape == (1, 4)
    with pytest.raises(MissingAssetError):
        load_swift_f0(Path(tmp_path / "missing.pt"))


def test_detect_rejects_bad_audio(model):
    with pytest.raises(ValueError):
        model.detect(np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError):
        model.detect(np.array([0.0, np.nan], dtype=np.float32))
