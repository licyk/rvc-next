"""Training hyperparameters (the original's ``configs/v1|v2/*.json``) and a small attribute tree over them."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

# Keyed "<version>/<rate>"; v2 has no 40k file, so v2 at 40k uses "v1/40k", as the original does.
CONFIGS: dict[str, dict[str, Any]] = {
    "v1/32k": {
        "train": {
            "log_interval": 200,
            "seed": 1234,
            "epochs": 20000,
            "learning_rate": 0.0001,
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": 4,
            "lr_decay": 0.999875,
            "segment_size": 12800,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
        },
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": 32000,
            "filter_length": 1024,
            "hop_length": 320,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 4, 2, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4, 4],
            "use_spectral_norm": False,
            "gin_channels": 256,
            "spk_embed_dim": 109,
        },
    },
    "v1/40k": {
        "train": {
            "log_interval": 200,
            "seed": 1234,
            "epochs": 20000,
            "learning_rate": 0.0001,
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": 4,
            "lr_decay": 0.999875,
            "segment_size": 12800,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
        },
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": 40000,
            "filter_length": 2048,
            "hop_length": 400,
            "win_length": 2048,
            "n_mel_channels": 125,
            "mel_fmin": 0.0,
            "mel_fmax": None,
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 10, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "use_spectral_norm": False,
            "gin_channels": 256,
            "spk_embed_dim": 109,
        },
    },
    "v1/48k": {
        "train": {
            "log_interval": 200,
            "seed": 1234,
            "epochs": 20000,
            "learning_rate": 0.0001,
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": 4,
            "lr_decay": 0.999875,
            "segment_size": 11520,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
        },
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": 48000,
            "filter_length": 2048,
            "hop_length": 480,
            "win_length": 2048,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 6, 2, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4, 4],
            "use_spectral_norm": False,
            "gin_channels": 256,
            "spk_embed_dim": 109,
        },
    },
    "v2/32k": {
        "train": {
            "log_interval": 200,
            "seed": 1234,
            "epochs": 20000,
            "learning_rate": 0.0001,
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": 4,
            "lr_decay": 0.999875,
            "segment_size": 12800,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
        },
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": 32000,
            "filter_length": 1024,
            "hop_length": 320,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [20, 16, 4, 4],
            "use_spectral_norm": False,
            "gin_channels": 256,
            "spk_embed_dim": 109,
        },
    },
    "v2/48k": {
        "train": {
            "log_interval": 200,
            "seed": 1234,
            "epochs": 20000,
            "learning_rate": 0.0001,
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": 4,
            "lr_decay": 0.999875,
            "segment_size": 17280,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
        },
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": 48000,
            "filter_length": 2048,
            "hop_length": 480,
            "win_length": 2048,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [12, 10, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [24, 20, 4, 4],
            "use_spectral_norm": False,
            "gin_channels": 256,
            "spk_embed_dim": 109,
        },
    },
}


def config_key(version: str, sample_rate: str) -> str:
    return f"v1/{sample_rate}" if version == "v1" or sample_rate == "40k" else f"v2/{sample_rate}"


def default_config(version: str, sample_rate: str) -> dict[str, Any]:
    """A fresh copy of the training config for this version and rate."""
    return copy.deepcopy(CONFIGS[config_key(version, sample_rate)])


class HParams:
    """Nested attribute access over a config dict (the original's ``train/utils.py:HParams``)."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getattr__(self, name: str) -> Any:
        # Only reached for missing keys; declared so type checkers accept config fields as attributes.
        raise AttributeError(name)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        return self.__dict__.__repr__()

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() if isinstance(v, HParams) else v for k, v in self.__dict__.items()}


def read_config(exp_dir: Path) -> dict[str, Any] | None:
    path = Path(exp_dir) / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_config(exp_dir: Path, config: dict[str, Any]) -> Path:
    """Write ``config.json`` as the original does (sorted keys, 4-space indent)."""
    path = Path(exp_dir) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    return path
