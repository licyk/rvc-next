"""FCPE, the pitch estimator RVC uses through ``torchfcpe``, ported for the one model it runs.

RVC calls ``torchfcpe.spawn_bundled_infer_model``: the bundled ``fcpe_c_v001.pt``, a convolution-only
conformer (``conv_only: true``) over a 128-bin log-mel spectrogram at 16 kHz. This module is that
path and nothing else: ``Wav2Mel`` (``torchfcpe.mel_extractor.Wav2MelModule`` and ``MelModule``
with no key shift), ``CFNaiveMelPE`` and its conv-only ``ConformerNaiveEncoder``
(``torchfcpe.models``, ``torchfcpe.model_conformer_naive``) and the loader. The attention blocks
(``einops``, ``local_attention``) never run for this model and are not ported; input at another
rate is resampled with the ported ``sinc_resample.Resample`` (torchaudio's, which torchfcpe used).
The arithmetic is unchanged, so the output equals torchfcpe's bit for bit
(``tests/engine/test_fcpe_model.py``). The weights are the ``fcpe`` asset.

Ported from https://github.com/CNChTu/FCPE (the conformer from https://github.com/CNChTu/Diffusion-SVC),
MIT License, Copyright (c) 2023 CN_ChiTu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import librosa
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from rvc_next.engine.audio.sinc_resample import Resample
from rvc_next.engine.errors import MissingAssetError, ModelFormatError

Decoder = Literal["argmax", "local_argmax"]


# -- mel spectrogram ---------------------------------------------------------------------------


class Wav2Mel(nn.Module):
    """Waveform ``(B, T)`` at ``sample_rate`` → log-mel ``(B, T // hop + 1, n_mels)`` at ``sr``."""

    window: Tensor
    mel_basis: Tensor

    def __init__(
        self, sr: int = 16000, n_mels: int = 128, n_fft: int = 1024, win_size: int = 1024, hop_length: int = 160, fmin: float = 0, fmax: float | None = None, clip_val: float = 1e-5
    ) -> None:
        super().__init__()
        self.sampling_rate = sr
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_length
        self.clip_val = clip_val
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=sr / 2 if fmax is None else fmax)
        self.register_buffer("mel_basis", torch.tensor(mel).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(win_size), persistent=False)
        self.resamplers = nn.ModuleDict()

    def _resample(self, audio: Tensor, sample_rate: int) -> Tensor:
        key = str(sample_rate)
        if key not in self.resamplers:
            if len(self.resamplers) > 8:
                self.resamplers.clear()
            self.resamplers[key] = Resample(sample_rate, self.sampling_rate, lowpass_filter_width=128).to(self.mel_basis.device)
        return self.resamplers[key](audio.squeeze(-1)).unsqueeze(-1)

    def mel(self, y: Tensor) -> Tensor:
        """``torchfcpe.mel_extractor.MelModule`` with key shift 0, speed 1 and ``center=False``."""
        y = y.squeeze(-1)
        win, hop = self.win_size, self.hop_size
        pad_left = (win - hop) // 2
        pad_right = max((win - hop + 1) // 2, win - y.size(-1) - pad_left)
        mode = "reflect" if pad_right < y.size(-1) else "constant"
        y = F.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)
        spec = torch.stft(y, self.n_fft, hop_length=hop, win_length=win, window=self.window, center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log(torch.clamp(spec, min=self.clip_val))
        return spec.transpose(-1, -2)

    @torch.no_grad()
    def forward(self, audio: Tensor, sample_rate: int) -> Tensor:
        audio_res = audio if sample_rate == self.sampling_rate else self._resample(audio, sample_rate)
        mel = self.mel(audio_res)
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]
        return mel


# -- the network -------------------------------------------------------------------------------


class Transpose(nn.Module):
    def __init__(self, dims: tuple[int, int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.dims)


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in: int, chan_out: int, kernel_size: int, padding: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size=kernel_size, padding=padding, groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ConformerConvModule(nn.Module):
    """LayerNorm → pointwise conv → GLU → depthwise conv (31) → SiLU → pointwise conv."""

    def __init__(self, dim: int, expansion_factor: int = 2, kernel_size: int = 31) -> None:
        super().__init__()
        inner_dim = dim * expansion_factor
        pad = kernel_size // 2  # "same" padding for the odd kernel
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            nn.GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=pad, groups=inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(0.0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvOnlyLayer(nn.Module):
    """``CFNEncoderLayer`` with ``conv_only``: a residual conv module. ``norm`` belongs to the
    attention branch, which this model does not have; it stays so the checkpoint loads strictly."""

    def __init__(self, dim_model: int) -> None:
        super().__init__()
        self.conformer = ConformerConvModule(dim_model)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conformer(x)


class ConvOnlyEncoder(nn.Module):
    def __init__(self, num_layers: int, dim_model: int) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList([ConvOnlyLayer(dim_model) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class CFNaiveMelPE(nn.Module):
    """Mel ``(B, T, n_mels)`` → a 360-bin pitch posterior over cents, decoded to f0 in Hz."""

    cent_table: Tensor
    gaussian_blurred_cent_mask: Tensor

    def __init__(self, input_channels: int, out_dims: int, hidden_dims: int = 512, n_layers: int = 6, f0_max: float = 1975.5, f0_min: float = 32.70) -> None:
        super().__init__()
        self.out_dims = out_dims
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.input_stack = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dims, 3, 1, 1),
            nn.GroupNorm(4, hidden_dims),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims, hidden_dims, 3, 1, 1),
        )
        self.net = ConvOnlyEncoder(n_layers, hidden_dims)
        self.norm = nn.LayerNorm(hidden_dims)
        self.output_proj = weight_norm(nn.Linear(hidden_dims, out_dims))
        # Overwritten by the checkpoint; built as torchfcpe builds them.
        cents = torch.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0], self.f0_to_cent(torch.Tensor([f0_max]))[0], out_dims).detach()
        self.register_buffer("cent_table", cents)
        self.register_buffer("gaussian_blurred_cent_mask", (1200.0 * torch.log2(torch.Tensor([f0_max / 10.0])))[0].detach())

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_stack(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.net(x)
        x = self.norm(x)
        x = self.output_proj(x)
        return torch.sigmoid(x)  # latent (B, T, out_dims)

    @torch.no_grad()
    def latent2cents_decoder(self, y: Tensor, threshold: float = 0.05, mask: bool = True) -> Tensor:
        """The posterior-weighted mean over every bin."""
        b, n, _ = y.size()
        ci = self.cent_table[None, None, :].expand(b, n, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= threshold] = float("-INF")
            rtn = rtn * confident_mask
        return rtn

    @torch.no_grad()
    def latent2cents_local_decoder(self, y: Tensor, threshold: float = 0.05, mask: bool = True) -> Tensor:
        """The posterior-weighted mean over the 9 bins around the peak."""
        b, n, _ = y.size()
        ci = self.cent_table[None, None, :].expand(b, n, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
        local_argmax_index[local_argmax_index < 0] = 0
        local_argmax_index[local_argmax_index >= self.out_dims] = self.out_dims - 1
        ci_l = torch.gather(ci, -1, local_argmax_index)
        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)
        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= threshold] = float("-INF")
            rtn = rtn * confident_mask
        return rtn

    @torch.no_grad()
    def infer(self, mel: Tensor, decoder: Decoder = "local_argmax", threshold: float = 0.05) -> Tensor:
        latent = self.forward(mel)
        if decoder == "argmax":
            cents = self.latent2cents_decoder(latent, threshold=threshold)
        elif decoder == "local_argmax":
            cents = self.latent2cents_local_decoder(latent, threshold=threshold)
        else:
            raise ValueError(f"Unknown FCPE decoder: {decoder}")
        return self.cent_to_f0(cents)  # (B, T, 1)

    @staticmethod
    def cent_to_f0(cent: Tensor) -> Tensor:
        return 10.0 * 2 ** (cent / 1200.0)

    @staticmethod
    def f0_to_cent(f0: Tensor) -> Tensor:
        return 1200.0 * torch.log2(f0 / 10.0)


class FCPE(nn.Module):
    """``torchfcpe.InferCFNaiveMelPE``: ``wav2mel`` and ``model``, run together by ``infer``."""

    def __init__(self, wav2mel: Wav2Mel, model: CFNaiveMelPE) -> None:
        super().__init__()
        self.wav2mel = wav2mel
        self.model = model

    @torch.no_grad()
    def infer(self, wav: Tensor, sr: int, decoder_mode: Decoder = "local_argmax", threshold: float = 0.006) -> Tensor:
        """F0 in Hz, ``(B, T // 160 + 1, 1)``, for ``wav`` ``(B, T)`` at ``sr``; unvoiced frames are 0."""
        wav = wav.to(self.model.cent_table.device)
        mel = self.wav2mel(wav, sr)
        return self.model.infer(mel, decoder=decoder_mode, threshold=threshold)

    forward = infer


def load_fcpe(path: Path, device: Any = "cpu") -> FCPE:
    """The FCPE checkpoint at ``path`` (torchfcpe's ``.pt``: ``config_dict`` and ``model``), on ``device``."""
    if not path.is_file():
        raise MissingAssetError(f"FCPE is not installed ({path})", ["fcpe"])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    try:
        config, state = ckpt["config_dict"], ckpt["model"]
        mel, net = config["mel"], config["model"]
    except (KeyError, TypeError) as e:
        raise ModelFormatError(f"{path.name} is not an FCPE checkpoint") from e
    if net.get("type") != "CFNaiveMelPE" or not net.get("conv_only") or net.get("use_harmonic_emb") or mel.get("type") not in (None, "none", "default"):
        raise ModelFormatError(f"{path.name}: only the convolution-only CFNaiveMelPE with the default mel, as bundled with torchfcpe, is supported")
    wav2mel = Wav2Mel(
        sr=mel.get("sr") or 16000,
        n_mels=mel.get("num_mels") or 128,
        n_fft=mel.get("n_fft") or 1024,
        win_size=mel.get("win_size") or 1024,
        hop_length=mel.get("hop_size") or 160,
        fmin=mel.get("fmin") or 0,
        fmax=mel.get("fmax") or 8000,
    )
    model = CFNaiveMelPE(mel["num_mels"], net["out_dims"], net["hidden_dims"], net["n_layers"], net["f0_max"], net["f0_min"])
    model.load_state_dict(state)
    return FCPE(wav2mel, model.eval()).to(device).eval()
