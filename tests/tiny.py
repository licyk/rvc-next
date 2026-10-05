"""Tiny models: a randomly initialised synthesizer and HuBERT that run every code path in seconds on a CPU.

They need no downloaded assets. The synthesizer keeps the real hop size per sample rate, so lengths
and padding behave as with real voices; only the widths are small.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

UPSAMPLE = {"32k": ([10, 8, 2, 2], [20, 16, 4, 4], 32000, 513), "40k": ([10, 10, 2, 2], [16, 16, 4, 4], 40000, 1025), "48k": ([12, 10, 2, 2], [24, 20, 4, 4], 48000, 1025)}


def tiny_config(sample_rate: str = "40k", speakers: int = 1) -> list[Any]:
    rates, kernels, sr, bins = UPSAMPLE[sample_rate]
    # spec, segment, inter, hidden, filter, heads, layers, kernel, dropout, resblock, rb kernels, rb dilations, ups, up channels, up kernels, speakers, gin, sr
    return [bins, 32, 8, 8, 16, 2, 1, 3, 0, "1", [3], [[1, 3, 5]], rates, 16, kernels, speakers, 8, sr]


def make_tiny_voice(path: Path, version: str = "v2", f0: bool = True, sample_rate: str = "40k", speakers: int = 1, speaker_info: list[dict] | None = None, seed: int = 0) -> Path:
    """Write a small-model ``.pth`` with random weights."""
    import torch

    from rvc_next.engine.models.checkpoint import SmallModel, write_small_model
    from rvc_next.engine.models.loader import synthesizer_class

    torch.manual_seed(seed)
    config = tiny_config(sample_rate, speakers)
    net = synthesizer_class(version, f0)(*config, is_half=False)
    weight = {k: v.half() for k, v in net.state_dict().items() if "enc_q" not in k}
    write_small_model(SmallModel(weight=weight, config=config, info="tiny", sr=sample_rate, pitch_guidance=f0, version=version, speaker_info=speaker_info or []), path)
    return path


def make_tiny_g_checkpoint(path: Path, version: str = "v2", f0: bool = True, sample_rate: str = "40k", speakers: int = 1, iteration: int = 10) -> Path:
    import torch

    from rvc_next.engine.models.loader import synthesizer_class

    net = synthesizer_class(version, f0)(*tiny_config(sample_rate, speakers), is_half=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "iteration": iteration, "optimizer": {}, "learning_rate": 1e-4}, str(path))
    return path


def make_tiny_hubert(assets: Path) -> Path:
    """A HuBERT with RVC's output shapes (layer 9 + final_proj 256, last layer 768) and tiny internals."""
    import torch
    from transformers import HubertConfig

    from rvc_next.engine.features.hubert import _model_class

    target = assets / "hubert_base"
    if (target / "config.json").is_file():
        return target
    torch.manual_seed(0)
    config = HubertConfig(
        hidden_size=768,
        num_hidden_layers=9,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(16, 16, 16, 16, 16, 16, 16),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        classifier_proj_size=256,
        do_stable_layer_norm=False,
        feat_extract_norm="group",
    )
    model = _model_class()(config)
    model.save_pretrained(str(target))
    (target / "preprocessor_config.json").write_text(json.dumps({"do_normalize": False, "feature_size": 1, "sampling_rate": 16000}))
    return target


TINY_FCPE_CONFIG = {
    "mel": {"sr": 16000, "num_mels": 128, "n_fft": 1024, "win_size": 1024, "hop_size": 160, "fmin": 0, "fmax": 8000},
    "model": {"type": "CFNaiveMelPE", "out_dims": 360, "hidden_dims": 16, "n_layers": 2, "n_heads": 2, "f0_min": 32.7, "f0_max": 1975.5, "use_fa_norm": True, "conv_only": True},
}


def make_tiny_fcpe(path: Path, seed: int = 0) -> Path:
    """An FCPE checkpoint in torchfcpe's format (``config_dict``, ``model`` with ``weight_g``/``weight_v``), with the real
    mel front end and a narrow two-layer network."""
    import copy

    import torch

    from rvc_next.engine.f0.fcpe_model import CFNaiveMelPE

    if path.is_file():
        return path
    torch.manual_seed(seed)
    model = CFNaiveMelPE(128, 360, hidden_dims=16, n_layers=2)  # as TINY_FCPE_CONFIG
    state = model.state_dict()
    # As the bundled checkpoint: the output layer's weight norm in the old keys.
    state["output_proj.weight_g"] = state.pop("output_proj.parametrizations.weight.original0")
    state["output_proj.weight_v"] = state.pop("output_proj.parametrizations.weight.original1")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config_dict": copy.deepcopy(TINY_FCPE_CONFIG), "model": state, "global_step": 0}, path)
    return path


def tiny_assets(root: Path) -> Path:
    """An assets folder with the tiny HuBERT and the tiny FCPE; pitch runs with ``pm`` or ``fcpe``."""
    root.mkdir(parents=True, exist_ok=True)
    make_tiny_hubert(root)
    make_tiny_fcpe(root / "fcpe" / "fcpe_c_v001.pt")
    return root


def tone(seconds: float = 2.0, sr: int = 16000, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    """A voiced-sounding test signal: a harmonic tone with a slow vibrato and a little noise."""
    t = np.arange(int(seconds * sr)) / sr
    f = freq * (1 + 0.02 * np.sin(2 * math.pi * 5 * t))
    phase = 2 * math.pi * np.cumsum(f) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 6))
    rng = np.random.default_rng(0)
    return (amp * x / np.max(np.abs(x)) + 0.003 * rng.standard_normal(t.shape)).astype(np.float32)


def make_tiny_index(path: Path, dim: int = 768, vectors: int = 200, add: bool = True, seed: int = 0) -> Path:
    """A faiss IVF index with ``dim``-wide random vectors (``add=False``: trained but empty, like ``trained_*``)."""
    import faiss

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((vectors, dim)).astype("float32")
    index = faiss.index_factory(dim, "IVF4,Flat")
    index.train(data)
    if add:
        index.add(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    return path
