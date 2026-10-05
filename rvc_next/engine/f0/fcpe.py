"""FCPE pitch (the original's ``infer/fcpe.py``), on the ported model (``fcpe_model``) and the
``fcpe`` asset's weights; torchfcpe is not imported.

DirectML lacks the complex STFT the mel stage uses, and its ``gather`` returns wrong bins, so on
DirectML the mel stage and the decoder run on the CPU and only the network runs on the GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rvc_next.engine.graph import cuda_graph_enabled, run_cuda_graph


def _is_directml(device: Any) -> bool:
    return getattr(device, "type", None) == "privateuseone" or "privateuseone" in str(device).lower()


class FCPEInfer:
    def __init__(self, model_path: Path, device: Any) -> None:
        import torch

        from rvc_next.engine.f0.fcpe_model import load_fcpe

        self.device = device
        self.is_directml = _is_directml(device)
        if self.is_directml:
            self.infer_model: Any = load_fcpe(model_path, "cpu")
            self.infer_model.wav2mel.eval()
            self.cent_table_cpu = self.infer_model.model.cent_table.detach().float().cpu().clone()
            self.out_dims = int(self.infer_model.model.out_dims)
            self.infer_model.model.to(device).eval()
        else:
            self.infer_model = load_fcpe(model_path, device)
            if getattr(device, "type", None) == "cuda" or str(device).startswith("cuda"):
                # Graph capture forbids host-to-device copies; keep the decoder offsets on the device.
                self.local_offsets = torch.arange(9, device=device, dtype=torch.long).view(1, 1, 9)

    def _graphable_model_infer(self, mel: Any, decoder_mode: str, threshold: float) -> Any:
        import torch

        model = self.infer_model.model
        latent = model(mel)
        batch, frames, _ = latent.shape
        cents = model.cent_table[None, None, :].expand(batch, frames, -1)
        if decoder_mode == "argmax":
            confidence = torch.max(latent, dim=-1, keepdim=True).values
            decoded = torch.sum(cents * latent, dim=-1, keepdim=True) / torch.sum(latent, dim=-1, keepdim=True)
        elif decoder_mode == "local_argmax":
            confidence, max_index = torch.max(latent, dim=-1, keepdim=True)
            local_index = (self.local_offsets + (max_index - 4)).clamp(0, model.out_dims - 1)
            local_cents = torch.gather(cents, -1, local_index)
            local_latent = torch.gather(latent, -1, local_index)
            decoded = torch.sum(local_cents * local_latent, dim=-1, keepdim=True) / torch.sum(local_latent, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown FCPE decoder mode: {decoder_mode}")
        confidence_mask = torch.ones_like(confidence)
        confidence_mask.masked_fill_(confidence <= threshold, float("-inf"))
        decoded = decoded * confidence_mask
        return 10.0 * torch.pow(2.0, decoded / 1200.0)

    def _decode_on_cpu(self, latent: Any, decoder_mode: str, threshold: float) -> Any:
        import torch

        latent = latent.detach().float().cpu()
        batch, frames, _ = latent.shape
        cents = self.cent_table_cpu[None, None, :].expand(batch, frames, -1)
        if decoder_mode == "argmax":
            confidence = torch.max(latent, dim=-1, keepdim=True).values
            decoded = torch.sum(cents * latent, dim=-1, keepdim=True) / torch.sum(latent, dim=-1, keepdim=True)
        elif decoder_mode == "local_argmax":
            confidence, max_index = torch.max(latent, dim=-1, keepdim=True)
            local_index = torch.arange(9, dtype=torch.long) + (max_index - 4)
            local_index.clamp_(0, self.out_dims - 1)
            local_cents = torch.gather(cents, -1, local_index)
            local_latent = torch.gather(latent, -1, local_index)
            decoded = torch.sum(local_cents * local_latent, dim=-1, keepdim=True) / torch.sum(local_latent, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown FCPE decoder mode: {decoder_mode}")
        decoded = decoded.masked_fill(confidence <= threshold, float("-inf"))
        return 10.0 * torch.pow(2.0, decoded / 1200.0)

    def infer(self, wav: Any, sr: int, decoder_mode: str = "local_argmax", threshold: float = 0.006) -> Any:
        import torch

        with torch.no_grad():
            if not self.is_directml:
                wav = wav.to(self.device)
                if cuda_graph_enabled(wav.device):
                    mel = self.infer_model.wav2mel(wav, sr)
                    return run_cuda_graph(
                        self.infer_model.model, f"fcpe-core-{decoder_mode}-{threshold}", lambda input_mel: self._graphable_model_infer(input_mel, decoder_mode, threshold), mel
                    )
                return run_cuda_graph(
                    self.infer_model,
                    f"fcpe-{sr}-{decoder_mode}-{threshold}",
                    lambda input_wav: self.infer_model.infer(input_wav, sr=sr, decoder_mode=decoder_mode, threshold=threshold),
                    wav,
                )
            wav_cpu = wav.detach().to(device="cpu", dtype=torch.float32)
            mel_cpu = self.infer_model.wav2mel(wav_cpu, sr)
            latent = self.infer_model.model(mel_cpu.to(device=self.device, dtype=torch.float32))
            return self._decode_on_cpu(latent, decoder_mode=decoder_mode, threshold=threshold)


class FcpeProvider:
    name = "fcpe"

    def __init__(self, model_path: Path, device: Any) -> None:
        self.model = FCPEInfer(model_path, device)

    def compute(self, x: np.ndarray, p_len: int) -> np.ndarray:
        import torch

        return self.model.infer(torch.from_numpy(x).unsqueeze(0).float(), sr=16000, decoder_mode="local_argmax", threshold=0.006).squeeze().detach().cpu().numpy()
