"""SOLA crossfade between blocks, from DDSP-SVC as the original realtime GUI uses it.

The start of each new block is aligned to the previous block's tail by normalised
cross-correlation over one 10 ms search window, then crossfaded with sin² windows.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def fade_windows(frames: int, device: Any) -> tuple[Any, Any]:
    import torch

    fade_in = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=frames, device=device, dtype=torch.float32)) ** 2
    return fade_in, 1 - fade_in


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

    def apply(self, infer_wav: Any, block_frame: int) -> Any:
        """Align, crossfade and return ``block_frame`` samples; keep the next tail. ``infer_wav`` is modified."""
        offset = self.offset(infer_wav)
        self.last_offset = offset
        infer_wav = infer_wav[offset:]
        infer_wav[: self.buffer_frame] *= self.fade_in
        infer_wav[: self.buffer_frame] += self.buffer * self.fade_out
        self.buffer[:] = infer_wav[block_frame : block_frame + self.buffer_frame]
        return infer_wav[:block_frame]
