"""CREPE pitch estimation as ``torchcrepe`` 0.0.24 runs it, ported for RVC's use of it.

RVC and Applio call ``torchcrepe.predict(x, 16000, 160, 50, 1100, model="full" | "tiny",
batch_size=512, return_periodicity=True)`` (Viterbi decoding), then ``filter.median(pd, 3)``,
``filter.mean(f0, 3)`` and zero the frames whose periodicity is under 0.1. This module is that path:
the network (``torchcrepe.model.Crepe``, same layer names, so torchcrepe's ``full.pth`` and
``tiny.pth`` load as they are), the framing and per-frame normalisation (``core.preprocess``), the
post-processing and periodicity (``core.postprocess``, ``core.periodicity``), the ``viterbi``,
``weighted_argmax`` and ``argmax`` decoders (``decode``), the unit conversions (``convert``) and the
NaN-aware ``median`` and ``mean`` filters (``filter``). torchcrepe itself is not imported: it imports
torchaudio (and resampy for other rates) at import time. Input must already be at 16 kHz.

Differences from torchcrepe, none of which changes RVC's numbers:

- **No dither by default.** torchcrepe's ``bins_to_cents`` adds triangular noise of ±20 cents
  (``scipy.stats.triang``, numpy's global RNG) to every decoded pitch, so its output changes from
  run to run. Here ``dither=False`` is the default and the output is deterministic; ``dither=True``
  draws the same noise the same way (seed numpy's global RNG for torchcrepe's exact numbers).
  ``weighted_argmax`` dithers its bin weights on every call when asked, where torchcrepe dithers
  them once per process and caches them.
- Viterbi runs on each ``batch_size`` chunk of frames separately, as torchcrepe's does (RVC's 512
  frames are 5.12 s); the network also sees the same chunks, so the arithmetic is unchanged.
- ``median`` accepts a one-frame signal (torchcrepe's reflect padding raises there); the padded
  values are masked out either way, so longer signals are unaffected.
- ``weighted_argmax`` masks the window without Python loops and without modifying its input.

Ported from https://github.com/maxrmorrison/torchcrepe, MIT License, Copyright (c) 2020 Max Morrison.
CREPE: Kim, Salamon, Li and Bello, "CREPE: A Convolutional Representation for Pitch Estimation",
ICASSP 2018 (https://github.com/marl/crepe, MIT License).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rvc_next.engine.errors import MissingAssetError, ModelFormatError

Capacity = Literal["full", "tiny"]
DecoderName = Literal["viterbi", "weighted_argmax", "argmax"]

CENTS_PER_BIN = 20  # cents
MAX_FMAX = 2006.0  # Hz
PITCH_BINS = 360
SAMPLE_RATE = 16000  # Hz
WINDOW_SIZE = 1024  # samples

# conv1's output channels tell the capacities apart in a state dict.
_CAPACITY_BY_WIDTH: dict[int, Capacity] = {1024: "full", 128: "tiny"}


# -- network -----------------------------------------------------------------------------------


class Crepe(nn.Module):
    """``torchcrepe.model.Crepe``: six conv/ReLU/BatchNorm/max-pool layers and a sigmoid classifier.

    Input ``(frames, 1024)`` normalised windows at 16 kHz; output ``(frames, 360)`` bin probabilities.
    """

    def __init__(self, capacity: Capacity = "full") -> None:
        super().__init__()
        if capacity == "full":
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif capacity == "tiny":
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f"CREPE capacity {capacity!r} is not supported")
        self.capacity: Capacity = capacity
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # eps and momentum as MMdnn converted them from the Keras model.
        def batch_norm(features: int) -> nn.BatchNorm2d:
            return nn.BatchNorm2d(num_features=features, eps=0.0010000000474974513, momentum=0.0)

        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_sizes[0], strides[0])
        self.conv1_BN = batch_norm(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_sizes[1], strides[1])
        self.conv2_BN = batch_norm(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels[2], out_channels[2], kernel_sizes[2], strides[2])
        self.conv3_BN = batch_norm(out_channels[2])
        self.conv4 = nn.Conv2d(in_channels[3], out_channels[3], kernel_sizes[3], strides[3])
        self.conv4_BN = batch_norm(out_channels[3])
        self.conv5 = nn.Conv2d(in_channels[4], out_channels[4], kernel_sizes[4], strides[4])
        self.conv5_BN = batch_norm(out_channels[4])
        self.conv6 = nn.Conv2d(in_channels[5], out_channels[5], kernel_sizes[5], strides[5])
        self.conv6_BN = batch_norm(out_channels[5])
        self.classifier = nn.Linear(in_features=self.in_features, out_features=PITCH_BINS)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = self.layer(x, self.conv6, self.conv6_BN)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)
        return torch.sigmoid(self.classifier(x))

    def embed(self, x: Tensor) -> Tensor:
        """The first five layers: ``(frames, 1024)`` → ``(frames, channels, 32, 1)``."""
        x = x[:, None, :, None]
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        return self.layer(x, self.conv5, self.conv5_BN)

    @staticmethod
    def layer(x: Tensor, conv: nn.Conv2d, batch_norm: nn.BatchNorm2d, padding: tuple[int, int, int, int] = (0, 0, 31, 32)) -> Tensor:
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return F.max_pool2d(x, (2, 1), (2, 1))


def load_crepe(path: Path, capacity: Capacity | None = None, device: Any = "cpu") -> Crepe:
    """The CREPE weights at ``path`` (torchcrepe's ``full.pth`` or ``tiny.pth``: a plain state dict),
    in eval mode on ``device``. ``capacity`` is read from the weights when not given."""
    asset = "crepe-tiny" if capacity == "tiny" else "crepe"
    if not path.is_file():
        raise MissingAssetError(f"CREPE is not installed ({path})", [asset])
    state = torch.load(path, map_location="cpu", weights_only=True)
    weight = state.get("conv1.weight") if isinstance(state, dict) else None
    found = _CAPACITY_BY_WIDTH.get(int(weight.shape[0])) if isinstance(weight, Tensor) and weight.dim() == 4 else None
    if found is None:
        raise ModelFormatError(f"{path.name} is not a CREPE checkpoint (torchcrepe's full.pth or tiny.pth)")
    if capacity is not None and capacity != found:
        raise ModelFormatError(f"{path.name} holds the {found} CREPE model, not {capacity}")
    model = Crepe(found)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise ModelFormatError(f"{path.name} is not a CREPE checkpoint: {e}") from e
    return model.to(device).eval()


# -- unit conversions (torchcrepe.convert) -----------------------------------------------------


def dither(cents: Tensor) -> Tensor:
    """Add torchcrepe's triangular noise of ±``CENTS_PER_BIN`` (numpy's global RNG) to ``cents``."""
    from scipy import stats

    noise = stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size())
    return cents + cents.new_tensor(noise)


def bins_to_cents(bins: Tensor, dither_cents: bool = False) -> Tensor:
    cents = CENTS_PER_BIN * bins + 1997.3794084376191
    return dither(cents) if dither_cents else cents


def cents_to_frequency(cents: Tensor) -> Tensor:
    return 10 * 2 ** (cents / 1200)


def bins_to_frequency(bins: Tensor, dither_cents: bool = False) -> Tensor:
    return cents_to_frequency(bins_to_cents(bins, dither_cents))


def frequency_to_cents(frequency: Tensor) -> Tensor:
    return 1200 * torch.log2(frequency / 10.0)


def cents_to_bins(cents: Tensor, quantize_fn: Callable[[Tensor], Tensor] = torch.floor) -> Tensor:
    bins = (cents - 1997.3794084376191) / CENTS_PER_BIN
    return quantize_fn(bins).int()


def frequency_to_bins(frequency: Tensor, quantize_fn: Callable[[Tensor], Tensor] = torch.floor) -> Tensor:
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


# -- decoders (torchcrepe.decode) --------------------------------------------------------------

_transition: np.ndarray | None = None


def viterbi_transition() -> np.ndarray:
    """torchcrepe's 360 × 360 transition matrix: a triangle 12 bins wide, rows summing to 1."""
    global _transition
    if _transition is None:
        xx, yy = np.meshgrid(range(PITCH_BINS), range(PITCH_BINS))
        transition = np.maximum(12 - abs(xx - yy), 0)
        _transition = transition / transition.sum(axis=1, keepdims=True)
    return _transition


def argmax(probabilities: Tensor, dither_cents: bool = False) -> tuple[Tensor, Tensor]:
    """``(batch, 360, frames)`` → bins and Hz ``(batch, frames)`` of the most probable bin."""
    bins = probabilities.argmax(dim=1)
    return bins, bins_to_frequency(bins, dither_cents)


def weighted_argmax(probabilities: Tensor, dither_cents: bool = False) -> tuple[Tensor, Tensor]:
    """The probability-weighted mean of the cents within 4 bins of the argmax (sigmoid of the input, as torchcrepe)."""
    bins = probabilities.argmax(dim=1)
    start = torch.clamp(bins - 4, min=0)
    end = torch.clamp(bins + 5, max=probabilities.size(1))
    index = torch.arange(probabilities.size(1), device=probabilities.device)[None, :, None]
    outside = (index < start[:, None, :]) | (index >= end[:, None, :])
    # In place on a copy: it keeps the input's (transposed) strides, so the sums below add in torchcrepe's order.
    logits = probabilities.clone().masked_fill_(outside, -float("inf"))
    weights = bins_to_cents(torch.arange(PITCH_BINS), dither_cents)[None, :, None].to(probabilities.device)
    probs = torch.sigmoid(logits)
    cents = (weights * probs).sum(dim=1) / probs.sum(dim=1)
    return bins, cents_to_frequency(cents)


def viterbi(probabilities: Tensor, dither_cents: bool = False) -> tuple[Tensor, Tensor]:
    """Viterbi decoding of the softmaxed bins with ``librosa.sequence.viterbi``, as torchcrepe."""
    import librosa

    probs = F.softmax(probabilities, dim=1)
    sequences = probs.cpu().numpy()
    transition = viterbi_transition()
    bins = np.array([librosa.sequence.viterbi(sequence, transition).astype(np.int64) for sequence in sequences])
    bins_t = torch.tensor(bins, device=probs.device)
    return bins_t, bins_to_frequency(bins_t, dither_cents)


DECODERS: dict[str, Callable[[Tensor, bool], tuple[Tensor, Tensor]]] = {"viterbi": viterbi, "weighted_argmax": weighted_argmax, "argmax": argmax}


# -- pre- and post-processing (torchcrepe.core) ------------------------------------------------


def frames(audio: Tensor, hop: int, batch_size: int | None, device: Any) -> Iterator[Tensor]:
    """``core.preprocess`` with ``pad=True`` at 16 kHz: zero-pad half a window each side, cut
    ``1 + len // hop`` windows of 1024, and yield them ``batch_size`` at a time on ``device``,
    each centred and scaled to unit (unbiased) standard deviation."""
    total_frames = 1 + int(audio.size(1) // hop)
    audio = F.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    batch_size = total_frames if batch_size is None else batch_size
    for i in range(0, total_frames, batch_size):
        start = max(0, i * hop)
        end = min(audio.size(1), (i + batch_size - 1) * hop + WINDOW_SIZE)
        chunk = F.unfold(audio[:, None, None, start:end], kernel_size=(1, WINDOW_SIZE), stride=(1, hop))
        chunk = chunk.transpose(1, 2).reshape(-1, WINDOW_SIZE).to(device)
        chunk -= chunk.mean(dim=1, keepdim=True)
        # Silent frames become very large values; that is what the network expects.
        chunk /= torch.max(torch.tensor(1e-10, device=chunk.device), chunk.std(dim=1, keepdim=True))
        yield chunk


def periodicity(probabilities: Tensor, bins: Tensor) -> Tensor:
    """The (masked) probability of each frame's decoded bin, ``(batch, frames)``."""
    probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)
    bins_stacked = bins.reshape(-1, 1).to(torch.int64)
    return probs_stacked.gather(1, bins_stacked).reshape(probabilities.size(0), probabilities.size(2))


def postprocess(probabilities: Tensor, fmin: float, fmax: float, decoder: DecoderName = "viterbi", dither: bool = False) -> tuple[Tensor, Tensor]:
    """``(batch, 360, frames)`` → pitch in Hz and periodicity, both ``(batch, frames)``; bins outside
    ``[fmin, fmax]`` are excluded."""
    probabilities = probabilities.detach()
    minidx = frequency_to_bins(torch.tensor(float(fmin)))
    maxidx = frequency_to_bins(torch.tensor(float(fmax)), torch.ceil)
    probabilities[:, :minidx] = -float("inf")
    probabilities[:, maxidx:] = -float("inf")
    if decoder not in DECODERS:
        raise ValueError(f"Unknown CREPE decoder: {decoder}")
    bins, pitch = DECODERS[decoder](probabilities, dither)
    return pitch, periodicity(probabilities, bins)


def predict(
    model: Crepe,
    audio_16k: Tensor,
    hop: int = 160,
    fmin: float = 50.0,
    fmax: float = MAX_FMAX,
    decoder: DecoderName = "viterbi",
    batch_size: int | None = 512,
    dither: bool = False,
) -> tuple[Tensor, Tensor]:
    """``torchcrepe.predict(audio, 16000, hop, fmin, fmax, decoder=…, batch_size=…, return_periodicity=True)``.

    ``audio_16k`` is ``(batch, samples)`` at 16 kHz; returns pitch in Hz and periodicity, each
    ``(batch, 1 + samples // hop)``, on the audio's device. The network runs on the model's device.
    Deterministic unless ``dither`` (torchcrepe's random ±20 cent noise on the decoded pitch).
    """
    device = next(model.parameters()).device
    pitches: list[Tensor] = []
    periodicities: list[Tensor] = []
    with torch.no_grad():
        for chunk in frames(audio_16k, hop, batch_size, device):
            probabilities = model(chunk).reshape(audio_16k.size(0), -1, PITCH_BINS).transpose(1, 2)
            pitch, period = postprocess(probabilities, fmin, fmax, decoder, dither)
            pitches.append(pitch.to(audio_16k.device))
            periodicities.append(period.to(audio_16k.device))
    return torch.cat(pitches, 1), torch.cat(periodicities, 1)


# -- filters (torchcrepe.filter) ---------------------------------------------------------------


def mean(signals: Tensor, win_length: int = 9) -> Tensor:
    """Moving average over ``win_length`` frames of ``(batch, frames)``, ignoring NaNs; a zero mean becomes NaN."""
    assert signals.dim() == 2, "Input tensor must have 2 dimensions (batch_size, width)"
    signals = signals.unsqueeze(1)
    mask = ~torch.isnan(signals)
    masked_x = torch.where(mask, signals, torch.zeros_like(signals))
    ones_kernel = torch.ones(signals.size(1), 1, win_length, device=signals.device, dtype=signals.dtype)
    sum_pooled = F.conv1d(masked_x, ones_kernel, stride=1, padding=win_length // 2)
    valid_count = F.conv1d(mask.to(signals.dtype), ones_kernel, stride=1, padding=win_length // 2).clamp(min=1)
    avg_pooled = sum_pooled / valid_count
    avg_pooled[avg_pooled == 0] = float("nan")
    return avg_pooled.squeeze(1)


def median(signals: Tensor, win_length: int) -> Tensor:
    """Moving median over ``win_length`` frames of ``(batch, frames)``, ignoring NaNs (the lower middle of an even count)."""
    assert signals.dim() == 2, "Input tensor must have 2 dimensions (batch_size, width)"
    signals = signals.unsqueeze(1)
    mask = ~torch.isnan(signals)
    masked_x = torch.where(mask, signals, torch.zeros_like(signals))
    padding = win_length // 2
    # The padded values are masked out, so the mode only matters where reflect cannot pad.
    x = F.pad(masked_x, (padding, padding), mode="reflect" if signals.size(-1) > padding else "constant")
    mask = F.pad(mask.float(), (padding, padding), mode="constant", value=0)
    x = x.unfold(2, win_length, 1)
    mask = mask.unfold(2, win_length, 1)
    x = x.contiguous().view(x.size()[:3] + (-1,))
    mask = mask.contiguous().view(mask.size()[:3] + (-1,))
    x_masked = torch.where(mask.bool(), x.float(), float("inf")).to(x)
    x_sorted, _ = torch.sort(x_masked, dim=-1)
    valid_count = mask.sum(dim=-1)
    median_idx = ((valid_count - 1) // 2).clamp(min=0)
    median_pooled = x_sorted.gather(-1, median_idx.unsqueeze(-1).long()).squeeze(-1)
    median_pooled[torch.isinf(median_pooled)] = float("nan")
    return median_pooled.squeeze(1)


# -- RVC's use ---------------------------------------------------------------------------------


def crepe_f0(model: Crepe, audio_16k: np.ndarray, p_len: int, device: Any, threshold: float = 0.1, fmin: float = 50.0, fmax: float = 1100.0, batch_size: int = 512) -> np.ndarray:
    """RVC's (and Applio's) CREPE pitch: Viterbi-decoded f0 at 10 ms, periodicity median-filtered
    over 3 frames, f0 mean-filtered over 3, and 0 where the periodicity is under ``threshold``.

    Returns float32 Hz per frame, cut or zero-padded to ``p_len`` frames; NaN (only possible from
    NaN input) becomes 0. Deterministic (no dither).
    """
    x = torch.from_numpy(np.asarray(audio_16k)).float().to(device)
    f0, pd = predict(model, x[None], 160, fmin, fmax, "viterbi", batch_size)
    pd = median(pd, 3)
    f0 = mean(f0, 3)
    f0[pd < threshold] = 0
    out = np.nan_to_num(f0[0].cpu().numpy(), nan=0.0)
    if out.shape[0] >= p_len:
        return out[:p_len]
    return np.pad(out, (0, p_len - out.shape[0]))
