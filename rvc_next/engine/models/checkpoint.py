"""The small model ``.pth`` format and G checkpoints (from the original ``train/process_ckpt.py``).

A small model is ``{weight: fp16 state_dict without enc_q, config: [18 positional hyperparameters],
info, sr, f0, version, speaker_info?}``. Every model users already have is in this format, so it is
read and written unchanged. A G checkpoint is ``{model, optimizer, learning_rate, iteration}``.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rvc_next.engine.errors import ModelFormatError

SAMPLE_RATES = {"32k": 32000, "40k": 40000, "48k": 48000}
MAX_SPEAKER_ID = 109


def normalize_speaker_info(speaker_info: Any, slots: int = MAX_SPEAKER_ID + 1) -> list[dict[str, Any]]:
    """Keep well-formed ``{id, name}`` entries with unique ids below ``slots``, sorted by id."""
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in speaker_info or []:
        try:
            speaker_id = int(item["id"])
            speaker_name = str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if speaker_id < 0 or speaker_id >= min(slots, MAX_SPEAKER_ID + 1) or not speaker_name or speaker_id in seen:
            continue
        seen.add(speaker_id)
        result.append({"id": speaker_id, "name": speaker_name})
    return sorted(result, key=lambda item: item["id"])


def torch_load(path: Path | str) -> Any:
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        raise ModelFormatError(f"Cannot read {Path(path).name}: {e}") from e


def torch_save(data: Any, path: Path | str) -> None:
    """Write through a temporary file, so a crash never leaves half a model."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, str(tmp))
    os.replace(tmp, path)


@dataclass
class SmallModel:
    """A parsed small model."""

    weight: dict[str, Any]
    config: list[Any]
    info: str
    sr: str
    """The ``sr`` field as stored: "40k", or occasionally an integer in old models."""
    pitch_guidance: bool
    version: str
    speaker_info: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def target_sample_rate(self) -> int:
        return int(self.config[-1])

    @property
    def speaker_slots(self) -> int:
        return int(self.weight["emb_g.weight"].shape[0])

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = OrderedDict()
        out["weight"] = self.weight
        out["config"] = self.config
        out["info"] = self.info
        out["sr"] = self.sr
        out["f0"] = int(self.pitch_guidance)
        out["version"] = self.version
        if self.speaker_info:
            out["speaker_info"] = self.speaker_info
        for key, value in self.extra.items():
            out.setdefault(key, value)
        return out


@dataclass
class CheckpointSummary:
    """What a ``.pth`` holds, read without building a model."""

    kind: str
    """small or checkpoint"""
    sample_rate: int | None
    version: str | None
    pitch_guidance: bool | None
    speaker_slots: int
    speaker_info: list[dict[str, Any]]
    info: str
    iteration: int | None = None


def _parse_small(data: Any, name: str) -> SmallModel:
    if not isinstance(data, dict) or "weight" not in data or "config" not in data:
        raise ModelFormatError(f"{name} is not an RVC small model")
    weight = data["weight"]
    if not isinstance(weight, dict) or "emb_g.weight" not in weight:
        raise ModelFormatError(f"{name} has no speaker embedding (weight/emb_g.weight)")
    config = list(data["config"])
    if len(config) != 18:
        raise ModelFormatError(f"{name} has a config of {len(config)} values, not 18")
    slots = int(weight["emb_g.weight"].shape[0])
    known = {"weight", "config", "info", "sr", "f0", "version", "speaker_info"}
    return SmallModel(
        weight=weight,
        config=config,
        info=str(data.get("info", "")),
        sr=data.get("sr", ""),
        pitch_guidance=bool(int(data.get("f0", 1))),
        version=str(data.get("version", "v1")),
        speaker_info=normalize_speaker_info(data.get("speaker_info", []), slots),
        extra={k: v for k, v in data.items() if k not in known},
    )


def read_small_model(path: Path | str) -> SmallModel:
    path = Path(path)
    return _parse_small(torch_load(path), path.name)


def is_g_checkpoint(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("model"), dict) and any(k.startswith("dec.") for k in data["model"])


def summarize(path: Path | str) -> CheckpointSummary:
    """Describe a small model or a G checkpoint; raise ``ModelFormatError`` for anything else."""
    path = Path(path)
    data = torch_load(path)
    if is_g_checkpoint(data):
        model = data["model"]
        emb = model.get("emb_g.weight")
        has_f0 = any(k.startswith("dec.m_source") for k in model)
        enc_dim = model.get("enc_p.emb_phone.weight")
        version = None if enc_dim is None else ("v2" if enc_dim.shape[1] == 768 else "v1")
        return CheckpointSummary(
            kind="checkpoint",
            sample_rate=None,
            version=version,
            pitch_guidance=has_f0,
            speaker_slots=int(emb.shape[0]) if emb is not None else 1,
            speaker_info=[],
            info="",
            iteration=int(data["iteration"]) if "iteration" in data else None,
        )
    small = _parse_small(data, path.name)
    return CheckpointSummary(
        kind="small",
        sample_rate=small.target_sample_rate,
        version=small.version,
        pitch_guidance=small.pitch_guidance,
        speaker_slots=small.speaker_slots,
        speaker_info=small.speaker_info,
        info=small.info,
    )


# Generator geometry → sample rate: the spectrogram width (512 or 1024 FFT bins + 1) and the
# upsampler kernel sizes, as the official base models and every trained G checkpoint have them.
_G_RATES: dict[tuple[int, tuple[int, ...]], tuple[str, str | None]] = {
    (513, (16, 16, 4, 4, 4)): ("32k", "v1"),
    (513, (20, 16, 4, 4)): ("32k", "v2"),
    (1025, (16, 16, 4, 4)): ("40k", None),
    (1025, (16, 16, 4, 4, 4)): ("48k", "v1"),
    (1025, (24, 20, 4, 4)): ("48k", "v2"),
}
_D_VERSION = {7: "v1", 9: "v2"}


@dataclass
class CheckpointInspection:
    """What a ``.pth`` is: a voice, a training generator (G) or discriminator (D), or neither."""

    kind: str
    """small, G, D or unknown."""
    version: str | None = None
    sample_rate: str | None = None
    """"32k", "40k" or "48k"; None when the file does not tell (a D)."""
    pitch_guidance: bool | None = None
    summary: CheckpointSummary | None = None
    note: str = ""


def _state(data: Any) -> dict[str, Any] | None:
    if isinstance(data, dict) and isinstance(data.get("model"), dict):
        return data["model"]
    if isinstance(data, dict) and data and all(isinstance(k, str) for k in data) and any(k.startswith(("dec.", "discriminators.")) for k in data):
        return data
    return None


def inspect_generator(state: dict[str, Any]) -> tuple[str | None, str | None, bool]:
    """Version, sample rate and pitch guidance of a generator state dict."""
    emb = state.get("enc_p.emb_phone.weight")
    version = None if emb is None else ("v2" if emb.shape[1] == 768 else "v1")
    pre = state.get("enc_q.pre.weight")
    ups = sorted((k for k in state if k.startswith("dec.ups.") and k.endswith((".weight_v", ".weight"))), key=lambda k: int(k.split(".")[2]))
    seen: dict[int, Any] = {}
    for k in ups:
        seen.setdefault(int(k.split(".")[2]), state[k])
    kernels = tuple(int(w.shape[2]) for _, w in sorted(seen.items()))
    rate = None
    if pre is not None:
        match = _G_RATES.get((int(pre.shape[1]), kernels))
        if match and (match[1] is None or match[1] == version):
            rate = match[0]
    return version, rate, any(k.startswith("dec.m_source") for k in state)


def inspect_checkpoint(path: Path | str) -> CheckpointInspection:
    """Identify a ``.pth`` by its contents; never raises for a readable file of another kind."""
    path = Path(path)
    data = torch_load(path)
    if isinstance(data, dict) and "weight" in data and "config" in data:
        try:
            small = _parse_small(data, path.name)
        except ModelFormatError as e:
            return CheckpointInspection("unknown", note=str(e))
        summary = summarize_small(small)
        rate = {32000: "32k", 40000: "40k", 48000: "48k"}.get(small.target_sample_rate)
        return CheckpointInspection("small", small.version, rate, small.pitch_guidance, summary)
    state = _state(data)
    if state is None:
        return CheckpointInspection("unknown", note=f"{path.name} is not an RVC model or checkpoint")
    if any(k.startswith("discriminators.") for k in state):
        count = len({k.split(".")[1] for k in state if k.startswith("discriminators.")})
        return CheckpointInspection("D", _D_VERSION.get(count))
    if any(k.startswith("dec.") for k in state):
        version, rate, f0 = inspect_generator(state)
        return CheckpointInspection("G", version, rate, f0)
    return CheckpointInspection("unknown", note=f"{path.name} is not an RVC model or checkpoint")


def summarize_small(small: SmallModel) -> CheckpointSummary:
    return CheckpointSummary(
        kind="small",
        sample_rate=small.target_sample_rate,
        version=small.version,
        pitch_guidance=small.pitch_guidance,
        speaker_slots=small.speaker_slots,
        speaker_info=small.speaker_info,
        info=small.info,
    )


def write_small_model(model: SmallModel, path: Path | str) -> None:
    torch_save(model.to_dict(), path)


def change_info(src: Path | str, dst: Path | str, info: str | None = None, speaker_info: list[dict[str, Any]] | None = None) -> None:
    """Copy a small model with a new ``info`` string and/or speaker names (the original's 修改信息)."""
    model = read_small_model(src)
    if info is not None:
        model.info = info
    if speaker_info is not None:
        model.speaker_info = normalize_speaker_info(speaker_info, model.speaker_slots)
    write_small_model(model, dst)


def save_small_from_state(
    state_dict: dict[str, Any], config: list[Any], path: Path | str, *, sr: str, pitch_guidance: bool, version: str, info: str, speaker_info: Any = None
) -> None:
    """Write a small model from a training state dict (the original's ``savee``)."""
    weight = OrderedDict((k, v.half()) for k, v in state_dict.items() if "enc_q" not in k)
    model = SmallModel(weight=weight, config=list(config), info=info, sr=sr, pitch_guidance=pitch_guidance, version=version, speaker_info=normalize_speaker_info(speaker_info))
    write_small_model(model, path)


def small_config_from_hparams(hps: Any) -> list[Any]:
    """The 18-value config list from a training hyperparameter tree."""
    return [
        hps.data.filter_length // 2 + 1,
        32,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.n_heads,
        hps.model.n_layers,
        hps.model.kernel_size,
        hps.model.p_dropout,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        hps.model.spk_embed_dim,
        hps.model.gin_channels,
        hps.data.sampling_rate,
    ]
