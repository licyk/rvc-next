"""SwiftF0 0.3.0, the pitch detector of the ``swift-f0`` package, as a PyTorch module.

``swift-f0`` ships one ONNX graph (``model.onnx``) and runs it with onnxruntime. This module is that
graph written out in torch, stage by stage, so it runs wherever torch does (CPU, CUDA, ROCm, XPU,
MPS) without onnxruntime:

1. **Framing** (``_frames``): the audio is zero-padded to whole 256-sample hops (at least one);
   frame ``t`` of each resolution is the ``n_fft`` samples centred on sample ``256·t``, with zeros
   beyond either end. There are ``max(1, samples // 256)`` frames.
2. **Spectrograms** (``LogSpectrogram``, n_fft 1024, 2048 and 512, in the graph's order): window,
   then the DFT as matrix products (a ``rows``-point DFT, twiddle factors, a ``cols``-point DFT,
   with onnxruntime's float32 cos and sin tables); magnitude, a sparse projection onto 128
   log-frequency rows, ``log(x + 1e-8)``. Stacked: ``(B, 3, 128, T)``.
3. **Network** (``SwiftF0Model``): a 3×3 stem, six residual blocks (the middle four dilated along
   frequency), the first 96 rows kept, a pitch head of 4 channels over 95 pitch bins → logits; a
   voicing head (frequency max of 4 more channels plus the input's mean) → confidence.
4. **Decoder** (``SwiftF0Model.decode``): bins outside ``fmin..fmax`` masked to -inf, the argmax and
   its two neighbours softmaxed in float64 to weight the bins' log frequencies → pitch; confidence
   is the sigmoid, or 0 where the unmasked argmax lies outside the band.

``SwiftF0Model.detect`` is ``swift_f0.SwiftF0.detect`` for 16 kHz mono float32 audio: windows of
``WINDOW_FRAMES`` frames with left and look-ahead context, the silence rule and the band check.
The weights and the graph's constant tensors (windows, trig tables, projections, bin frequencies) are
``data/swift_f0.pt``, extracted from ``model.onnx`` by ``scripts/extract_swift_f0.py``. The
results match onnxruntime's to float32 rounding (``tests/engine/test_swift_model.py``).

Ported from SwiftF0 (https://github.com/lars76/swift-f0), MIT License,
Copyright (c) 2025 Lars Nieradzik.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rvc_next.engine.errors import MissingAssetError, ModelFormatError

SAMPLE_RATE = 16000
HOP = 256
FRAME_PERIOD = HOP / SAMPLE_RATE
FMIN = 46.875
FMAX = 2093.75
# 95 log-spaced pitch bins from FMIN to FMAX; a band narrower than one step holds no candidate.
N_BINS = 95
BIN_RATIO = (FMAX / FMIN) ** (1 / (N_BINS - 1))
# The receptive field reaches 2815 samples each way, so 11 frames of left context and 10 of
# look-ahead make a window's frames equal to a whole-signal run (swift_f0.core).
LOOKAHEAD_FRAMES = 10
LEFT_FRAMES = 11
WINDOW_FRAMES = 1875
# Frames whose own hop peaks below this get confidence 0: digital silence makes random voicing.
SILENCE_PEAK = 1e-3

N_ROWS = 128  # log-frequency rows of each spectrogram
LOG_EPS = 1e-8
CHANNELS = 12
PITCH_ROWS = 96  # spectrogram rows the pitch head sees
# n_fft → (rows, cols) of the two-stage DFT, in the graph's channel order.
RESOLUTIONS: tuple[tuple[int, int, int], ...] = ((1024, 32, 32), (2048, 64, 32), (512, 32, 16))
# (kernel, dilation) of the residual blocks, (frequency, time).
BLOCKS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((7, 3), (1, 1)),
    ((7, 1), (17, 1)),
    ((5, 1), (27, 1)),
    ((3, 1), (40, 1)),
    ((3, 1), (48, 1)),
    ((7, 3), (2, 1)),
)
DEFAULT_WEIGHTS = Path(__file__).with_name("data") / "swift_f0.pt"


class SwiftF0Track(NamedTuple):
    """What ``detect`` returns: one value per 16 ms frame, float64."""

    pitch_hz: np.ndarray
    confidence: np.ndarray
    timestamps: np.ndarray


# -- spectrograms -------------------------------------------------------------------------------


def _frames(audio: Tensor, n_fft: int, n_frames: int) -> Tensor:
    """``(B, L)`` (``L`` whole hops) → ``(B, n_frames, n_fft)``: frame ``t`` spans ``256·t ± n_fft/2``."""
    half = n_fft // 2
    padded = F.pad(audio, (half, half))
    return padded.unfold(-1, n_fft, HOP)[:, :n_frames]


class LogSpectrogram(nn.Module):
    """One resolution of the front end: frames → window → DFT → magnitude → projection → log.

    The DFT is the graph's factorisation: ``n = cols·n1 + n2`` and ``k = k1 + rows·k2``; a
    ``rows``-point DFT over ``n1``, the twiddles ``W_N^(k1·n2)``, a ``cols``-point DFT over ``n2``,
    each a matrix product with ``cos`` and ``-sin`` of ``2π·(k·n mod size)/size``. Those come from
    ``*_trig`` (``(2, size)``: cos and sin of ``m · float32(2π/size)``), the values onnxruntime's
    Cos and Sin give for the graph's float32 angles: torch's differ by an ulp in about a third of
    the entries, which is enough to move the log of near-empty bins.
    """

    window: Tensor
    projection: Tensor
    dft_a_trig: Tensor
    twiddle_trig: Tensor
    dft_b_trig: Tensor
    dft_a_index: Tensor
    twiddle_index: Tensor
    dft_b_index: Tensor

    def __init__(self, n_fft: int, rows: int, cols: int) -> None:
        super().__init__()
        self.n_fft, self.rows, self.cols = n_fft, rows, cols
        self.register_buffer("window", torch.zeros(n_fft))
        self.register_buffer("projection", torch.zeros(n_fft // 2 + 1, N_ROWS))
        for stage, size, k, n in (("dft_a", rows, rows, rows), ("twiddle", n_fft, rows, cols), ("dft_b", cols, cols, cols)):
            self.register_buffer(f"{stage}_trig", torch.zeros(2, size))
            self.register_buffer(f"{stage}_index", (torch.arange(k).unsqueeze(1) * torch.arange(n).unsqueeze(0)) % size, persistent=False)

    def _matrix(self, stage: str) -> tuple[Tensor, Tensor]:
        """``cos`` and ``-sin`` of the angle ``2π·(k·n mod size)/size`` for one DFT stage."""
        trig, index = getattr(self, f"{stage}_trig"), getattr(self, f"{stage}_index")
        return trig[0][index], -trig[1][index]

    def power(self, frames: Tensor) -> Tensor:
        """``(B, T, n_fft)`` windowed frames → ``|DFT|²`` ``(B, T, n_fft // 2 + 1)``."""
        b, t = frames.shape[:2]
        rows, cols = self.rows, self.cols
        a_cos, a_nsin = self._matrix("dft_a")
        w_cos, w_nsin = self._matrix("twiddle")
        b_cos, b_nsin = self._matrix("dft_b")
        # Each matrix product takes cos and -sin side by side (one larger product, same sums), and
        # runs over all frames at once on contiguous rows (x^T @ A^T rather than a batched A @ x):
        # batched products of 32×32 matrices are several times slower.
        xt = frames.reshape(b, t, rows, cols).transpose(-1, -2).contiguous()  # (B, T, n2, n1)
        a = (xt @ torch.cat([a_cos.T, a_nsin.T], dim=1)).transpose(-1, -2)
        re, im = a[..., :rows, :].contiguous(), a[..., rows:, :].contiguous()  # (B, T, k1, n2)
        re, im = re * w_cos - im * w_nsin, re * w_nsin + im * w_cos
        b_matrix = torch.cat([b_cos, b_nsin], dim=1)
        re_b, im_b = re @ b_matrix, im @ b_matrix
        y_re = re_b[..., :cols] - im_b[..., cols:]  # re @ b_cos - im @ b_nsin: (B, T, k1, k2)
        y_im = re_b[..., cols:] + im_b[..., :cols]  # re @ b_nsin + im @ b_cos
        power = y_re * y_re + y_im * y_im
        # k = k1 + rows·k2: k2-major, then the bins up to n_fft/2. (Computing only those columns
        # of the last stage would halve it, but the narrower products round differently.)
        return power.transpose(-1, -2).reshape(b, t, self.n_fft)[..., : self.n_fft // 2 + 1]

    def forward(self, audio: Tensor, n_frames: int) -> Tensor:
        """Hop-padded audio ``(B, L)`` → log spectrogram ``(B, 128, n_frames)``."""
        magnitude = torch.sqrt(self.power(_frames(audio, self.n_fft, n_frames) * self.window))
        return torch.log(magnitude @ self.projection + LOG_EPS).transpose(-1, -2)


# -- network ------------------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """``relu(x + conv(relu(conv(x))))``, two same-shape convolutions, padded to keep the size."""

    def __init__(self, kernel: tuple[int, int], dilation: tuple[int, int]) -> None:
        super().__init__()
        padding = ((kernel[0] - 1) * dilation[0] // 2, (kernel[1] - 1) * dilation[1] // 2)
        self.conv1 = nn.Conv2d(CHANNELS, CHANNELS, kernel, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, kernel, padding=padding, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class SwiftF0Model(nn.Module):
    """The ``model.onnx`` graph: audio ``(B, N)`` at 16 kHz → pitch (Hz, float64) and confidence per 256 samples."""

    bin_hz: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.spectrograms = nn.ModuleList(LogSpectrogram(n_fft, rows, cols) for n_fft, rows, cols in RESOLUTIONS)
        self.stem = nn.Conv2d(len(RESOLUTIONS), CHANNELS, 3, padding=1)
        self.blocks = nn.ModuleList(ResidualBlock(kernel, dilation) for kernel, dilation in BLOCKS)
        self.pitch_conv = nn.Conv2d(CHANNELS, 4, 3)  # padded (1, 0) in frequency, 1 in time: 96 rows → 95 bins
        self.pitch_logits = nn.Conv2d(4, 1, 1)
        self.voicing_conv = nn.Conv2d(4, 4, (5, 1), padding=(2, 0))
        self.voicing_hidden = nn.Conv1d(5, 16, 3, padding=1)
        self.voicing_out = nn.Conv1d(16, 1, 1)
        self.register_buffer("bin_hz", torch.zeros(N_BINS))

    @staticmethod
    def n_frames(samples: int) -> int:
        """Frames the graph returns for ``samples`` samples: one per whole hop, at least one."""
        return max(1, samples // HOP)

    def features(self, audio: Tensor) -> Tensor:
        """Audio ``(B, N)`` → stacked log spectrograms ``(B, 3, 128, T)``."""
        samples = audio.shape[-1]
        n_frames = self.n_frames(samples)
        # Zero-pad to whole hops, at least one (the graph's two Pads before framing).
        padded = F.pad(audio, (0, -(-max(samples, HOP) // HOP) * HOP - samples))
        return torch.stack([spectrogram(padded, n_frames) for spectrogram in self.spectrograms], dim=1)

    def network(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Spectrograms ``(B, 3, 128, T)`` → pitch logits ``(B, T, 95)`` and voicing logits ``(B, T)``."""
        # Channels-last: twice as fast on the CPU for these 12-channel convolutions.
        x = F.relu(self.stem(features.contiguous(memory_format=torch.channels_last)))
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.pitch_conv(F.pad(x[:, :, :PITCH_ROWS], (1, 1, 1, 0))))  # (B, 4, 95, T)
        logits = self.pitch_logits(x).squeeze(1).transpose(1, 2)
        voicing = torch.cat([self.voicing_conv(x).amax(dim=2), features.mean(dim=(1, 2), keepdim=True).squeeze(1)], dim=1)
        voicing = self.voicing_out(F.relu(self.voicing_hidden(voicing))).squeeze(1)
        return logits, voicing

    def decode(self, logits: Tensor, voicing: Tensor, fmin: float | Tensor, fmax: float | Tensor) -> tuple[Tensor, Tensor]:
        """Logits → pitch ``(B, T)`` in Hz (float64) and confidence ``(B, T)`` (float32), restricted to ``fmin..fmax``."""
        # The graph compares the float32 bin frequencies with float32 fmin and fmax.
        low = torch.as_tensor(fmin, dtype=torch.float32, device=logits.device)
        high = torch.as_tensor(fmax, dtype=torch.float32, device=logits.device)
        in_band = (self.bin_hz >= low) & (self.bin_hz <= high)
        masked = torch.where(in_band, logits, -math.inf)
        neighbours = masked.argmax(dim=-1, keepdim=True) + torch.tensor([-1, 0, 1], device=logits.device)
        near = torch.gather(F.pad(masked, (1, 1), value=-math.inf), -1, neighbours + 1)
        # Softmax and the weighted log frequency in float64, as the graph (on the CPU for MPS).
        exact = torch.device("cpu") if logits.device.type == "mps" else logits.device
        weights = torch.softmax(near.to(exact, torch.float64), dim=-1)
        log_hz = torch.log(self.bin_hz.to(exact, torch.float64))[neighbours.clamp(0, N_BINS - 1).to(exact)]
        pitch = torch.exp((weights * log_hz).sum(dim=-1))
        confidence = torch.where(in_band[logits.argmax(dim=-1)], torch.sigmoid(voicing), 0.0)
        return pitch, confidence

    def forward(self, audio: Tensor, fmin: float | Tensor = FMIN, fmax: float | Tensor = FMAX) -> tuple[Tensor, Tensor]:
        """Audio ``(B, N)`` float32 at 16 kHz → ``pitch`` and ``confidence`` ``(B, max(1, N // 256))``, as ``model.onnx``."""
        logits, voicing = self.network(self.features(audio))
        return self.decode(logits, voicing, fmin, fmax)

    @torch.inference_mode()
    def detect(self, audio: np.ndarray, fmin: float | None = None, fmax: float | None = None) -> SwiftF0Track:
        """``swift_f0.SwiftF0.detect`` for 16 kHz mono audio: pitch, confidence and frame times (frame ``i`` at ``i · 16 ms``)."""
        fmin, fmax = band(fmin, fmax)
        signal = np.asarray(audio, dtype=np.float32).reshape(-1)
        if signal.size == 0:
            raise ValueError("audio must not be empty")
        if not np.isfinite(signal).all():
            raise ValueError("audio must be finite")
        n = self.n_frames(len(signal))
        pitch_parts, confidence_parts = [], []
        for start in range(0, n, WINDOW_FRAMES):
            end = min(start + WINDOW_FRAMES, n)
            left = max(0, start - LEFT_FRAMES)
            window = signal[left * HOP : (end + LOOKAHEAD_FRAMES) * HOP] if end < n else signal[left * HOP :]
            pitch, confidence = self._run(window, fmin, fmax)
            pitch_parts.append(pitch[start - left : end - left])
            confidence_parts.append(confidence[start - left : end - left])
        pitch, confidence = np.concatenate(pitch_parts), np.concatenate(confidence_parts)
        return SwiftF0Track(pitch, confidence, np.arange(len(pitch)) * FRAME_PERIOD)

    def _run(self, audio: np.ndarray, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
        """One graph run on ``audio``, plus the silence rule (``swift_f0.core.SwiftF0._run``)."""
        device = self.bin_hz.device
        pitch, confidence = self(torch.from_numpy(np.ascontiguousarray(audio)).to(device).unsqueeze(0), fmin, fmax)
        pitch = pitch[0].cpu().numpy().astype(np.float64)
        confidence = confidence[0].cpu().numpy().astype(np.float64)
        n = len(confidence)
        hops = audio[: n * HOP].reshape(n, HOP) if len(audio) >= HOP else audio[None, :]
        confidence[np.abs(hops).max(axis=1) < SILENCE_PEAK] = 0.0
        return pitch, confidence


def band(fmin: float | None, fmax: float | None) -> tuple[float, float]:
    """``fmin``/``fmax`` clamped to the model's range and checked (``swift_f0.core._range``)."""
    if any(f is not None and math.isnan(f) for f in (fmin, fmax)):
        raise ValueError("fmin and fmax must be numbers")
    low = FMIN if fmin is None else max(FMIN, float(fmin))
    high = FMAX if fmax is None else min(FMAX, float(fmax))
    if not low < high:
        raise ValueError(f"require fmin < fmax within the model range {FMIN} to {FMAX} Hz, got fmin={fmin}, fmax={fmax}")
    if high < low * BIN_RATIO:
        raise ValueError(f"fmax must be at least {BIN_RATIO:.5f} times fmin so the band holds a pitch candidate")
    return low, high


def dense_state(weights: dict[str, Tensor]) -> dict[str, Tensor]:
    """The file's tensors as a ``SwiftF0Model`` state dict: each sparse projection made dense (the graph's ScatterElements)."""
    state = {}
    for key, value in weights.items():
        if key.endswith(".projection_index"):
            prefix = key.removesuffix("_index")
            rows = int(weights[f"{prefix}_rows"])
            dense = torch.zeros(rows * N_ROWS, dtype=torch.float32).scatter_(0, value.long(), weights[f"{prefix}_value"])
            state[prefix] = dense.reshape(rows, N_ROWS)
        elif not key.endswith((".projection_value", ".projection_rows")):
            state[key] = value
    return state


def load_swift_f0(path: Path | None = None, device: Any = "cpu") -> SwiftF0Model:
    """The SwiftF0 model from ``path`` (``data/swift_f0.pt`` by default), on ``device``, in eval mode."""
    path = DEFAULT_WEIGHTS if path is None else Path(path)
    if not path.is_file():
        raise MissingAssetError(f"SwiftF0 weights are missing ({path})", [])
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model = SwiftF0Model()
    try:
        model.load_state_dict(dense_state(weights))
    except (KeyError, RuntimeError, TypeError, AttributeError) as e:
        raise ModelFormatError(f"{path.name} is not a SwiftF0 weights file") from e
    return model.to(device).eval()
