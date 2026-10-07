"""SOLA crossfade between blocks, from DDSP-SVC as the original realtime GUI uses it.

The start of each new block is aligned to the previous block's tail by normalised
cross-correlation over one 10 ms search window, then crossfaded with sin² windows.

Optionally the crossfade is a phase-vocoder one (RVC-Project's ``gui_v1.py`` ``phase_vocoder``, as
Applio uses it, MIT): squared windows on both sides leave out the cross term of an
amplitude-complementary fade, and that gap is filled with sinusoids whose phase glides from the
old tail's phase to the new block's, bin by bin, so two renderings of the same sound that differ in
phase do not partly cancel in the middle of the fade. With identical inputs it returns them.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def fade_windows(frames: int, device: Any) -> tuple[Any, Any]:
    import torch

    fade_in = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=frames, device=device, dtype=torch.float32)) ** 2
    return fade_in, 1 - fade_in


def phase_vocoder(a: Any, b: Any, fade_out: Any, fade_in: Any) -> Any:
    """Crossfade ``a`` (the old tail) into ``b`` (the new block's head), both ``n`` samples."""
    import torch

    window = (fade_out * fade_in).sqrt()
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = fa.abs() + fb.abs()
    n = a.shape[0]
    # One-sided spectrum to a real signal: every bin but DC (and Nyquist, for even n) counts twice.
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = fa.angle()
    deltaphase = fb.angle() - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)  # wrapped to [-π, π)
    bins = 2 * np.pi * torch.arange(n // 2 + 1, device=a.device, dtype=a.dtype)
    t = torch.arange(n, device=a.device, dtype=a.dtype).unsqueeze(-1) / n
    glide = (absab * ((bins + deltaphase) * t + phia).cos()).sum(-1) * window / n
    return a * fade_out**2 + b * fade_in**2 + glide


class Sola:
    def __init__(self, buffer_frame: int, search_frame: int, device: Any) -> None:
        import torch

        self.buffer_frame = buffer_frame
        self.search_frame = search_frame
        self.buffer = torch.zeros(buffer_frame, device=device, dtype=torch.float32)
        self.den_kernel = torch.ones(1, 1, buffer_frame, device=device, dtype=torch.float32)
        self.fade_in, self.fade_out = fade_windows(buffer_frame, device)
        self.last_offset = 0

    def reset(self) -> None:
        self.buffer.zero_()

    def offset(self, infer_wav: Any) -> int:
        """The alignment offset of ``infer_wav`` against the stored tail."""
        import torch
        import torch.nn.functional as F

        conv_input = infer_wav[None, None, : self.buffer_frame + self.search_frame]
        cor_nom = F.conv1d(conv_input, self.buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, self.den_kernel) + 1e-8)
        return int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))

    def apply(self, infer_wav: Any, block_frame: int, use_phase_vocoder: bool = False) -> Any:
        """Align, crossfade and return ``block_frame`` samples; keep the next tail. ``infer_wav`` is modified."""
        offset = self.offset(infer_wav)
        self.last_offset = offset
        infer_wav = infer_wav[offset:]
        if use_phase_vocoder:
            infer_wav[: self.buffer_frame] = phase_vocoder(self.buffer, infer_wav[: self.buffer_frame], self.fade_out, self.fade_in)
        else:
            infer_wav[: self.buffer_frame] *= self.fade_in
            infer_wav[: self.buffer_frame] += self.buffer * self.fade_out
        self.buffer[:] = infer_wav[block_frame : block_frame + self.buffer_frame]
        return infer_wav[:block_frame]
