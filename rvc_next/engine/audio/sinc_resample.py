"""Band-limited (windowed-sinc) resampling on a torch device, ported from torchaudio 2.11.

``Resample`` is ``torchaudio.transforms.Resample`` and the two kernel functions are
``torchaudio.functional.functional._get_sinc_resample_kernel`` and ``_apply_sinc_resample_kernel``.
torchaudio's last release is 2.11, so the class rvc-next uses lives here. The arithmetic is
unchanged, step for step, so the output is bit-identical to torchaudio's
(``tests/engine/test_sinc_resample.py``); the changes are types, ``ValueError``/``TypeError`` in
place of a bare ``Exception``, and no deprecated method names.

Ported from https://github.com/pytorch/audio under its licence:

    BSD 2-Clause License

    Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

ResamplingMethod = Literal["sinc_interp_hann", "sinc_interp_kaiser"]

KAISER_BETA = 14.769656459379492
"""torchaudio's default Kaiser window shape."""


def sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: ResamplingMethod = "sinc_interp_hann",
    beta: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, int]:
    """The polyphase filter bank for ``orig_freq → new_freq`` and its half width, in input samples.

    ``kernel`` has shape ``(new_freq // gcd, 1, 2 * width + orig_freq // gcd)``: one FIR filter per
    output phase, each a Hann- (or Kaiser-) windowed sinc that stops after ``lowpass_filter_width``
    zero crossings of the low-pass at ``rolloff`` of the lower Nyquist. Built in float64 and returned
    in float32 unless ``dtype`` is given, which is then used throughout.
    """
    if int(orig_freq) != orig_freq or int(new_freq) != new_freq:
        raise ValueError(
            "Frequencies must be integers to resample accurately; scale both to integers that keep their ratio "
            "(to downsample 44100 Hz by 8, use orig_freq=8 and new_freq=1, not 44100 and 5512.5)."
        )
    if resampling_method not in ("sinc_interp_hann", "sinc_interp_kaiser"):
        raise ValueError(f"Invalid resampling method: {resampling_method}")
    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    # Antialiasing: drop the highest frequencies. Upsampling needs it too, since the edges are
    # equivalent to zero padding, which adds high-frequency artefacts.
    base_freq = min(orig_freq, new_freq) * rolloff

    # x(t) is reconstructed from x[i] by sinc interpolation, x(t) = Σ x[i] sinc(π·orig·(i/orig − t)),
    # and sampled at the new rate, y[j] = x(j/new). That is a convolution of x with one filter per
    # output phase j (mod new); y[j + new] reuses y[j]'s filter on x shifted by orig, hence the
    # conv1d with stride orig. Each filter is cut after ``lowpass_filter_width`` zero crossings.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq

    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    # The window is evaluated at these exact positions, not on a regular grid, so no torch window.
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        beta_tensor = torch.tensor(float(KAISER_BETA if beta is None else beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)

    return kernels, width


def apply_sinc_resample_kernel(waveform: Tensor, orig_freq: int, new_freq: int, gcd: int, kernel: Tensor, width: int) -> Tensor:
    """Resample ``waveform`` (``(..., time)``, floating point) with a kernel from ``sinc_resample_kernel``."""
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = F.pad(waveform, (width, width + orig_freq))
    resampled = F.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    # As torchaudio: the length is rounded up in float32 (``as_tensor`` of a Python float), which can
    # differ from a float64 ``math.ceil`` on very long inputs; kept for identical output lengths.
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]

    return resampled.view(shape[:-1] + resampled.shape[-1:])


class Resample(torch.nn.Module):
    """Resample ``(..., time)`` audio from ``orig_freq`` to ``new_freq`` (``torchaudio.transforms.Resample``).

    The kernel is computed once and kept as a buffer, so ``.to(device)`` moves it with the module.
    With ``dtype`` None it is built in float64 and cached in float32; pass ``torch.float32`` to build
    it in float32 too, as RVC's realtime and formant paths do.
    """

    kernel: Tensor

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        resampling_method: ResamplingMethod = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: float | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.gcd = math.gcd(int(orig_freq), int(new_freq))
        self.resampling_method: ResamplingMethod = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta
        self.width = 0
        if orig_freq != new_freq:
            kernel, self.width = sinc_resample_kernel(orig_freq, new_freq, self.gcd, lowpass_filter_width, rolloff, resampling_method, beta, dtype=dtype)
            self.register_buffer("kernel", kernel)

    def forward(self, waveform: Tensor) -> Tensor:
        if self.orig_freq == self.new_freq:
            return waveform
        return apply_sinc_resample_kernel(waveform, self.orig_freq, self.new_freq, self.gcd, self.kernel, self.width)
