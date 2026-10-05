"""Separation models, presets and chain planning (the model table of the original ``tools/pymss_webui.py``).

The presets themselves are data (``core/separation/presets.json``, shipped with the package); the
model table is here, since it is technical detail no client needs. Nothing in this module imports
pymss or torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

WEIGHTS_DIR = "pymss_weights"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
# The preset file is package data of ``rvc_next.core.separation``; it is read as a file, not imported.
PRESETS_FILE = Path(__file__).resolve().parents[2] / "core" / "separation" / "presets.json"
DEFAULT_PRESET = "vocals"
MODEL_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class ModelSpec:
    """One separation model: its files, its two output stems, and their labels in rvc-next."""

    id: str
    title: str
    model_type: str
    ckpt: str
    config: str
    primary: str
    """Stem name in the model's output that is kept or fed to the next chain step."""
    secondary: str
    primary_label: str
    secondary_label: str
    batch_size: int
    overlap_size: int
    size: int
    sha256: str

    @property
    def asset_id(self) -> str:
        return f"separation-{self.id}"


MODELS: dict[str, ModelSpec] = {
    m.id: m
    for m in (
        ModelSpec(
            "dereverb",
            "Dereverb (Mel-RoFormer, anvuew, less aggressive)",
            "mel_band_roformer",
            "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
            "dereverb_mel_band_roformer_anvuew.yaml",
            "noreverb",
            "reverb",
            "noreverb",
            "reverb",
            1,
            176400,
            456754215,
            "9399701da33d179e328a436e882fe2c79ae2f118a3d49603dbc7210bfeb428ea",
        ),
        ModelSpec(
            "dereverb-aggressive",
            "Dereverb (Mel-RoFormer, anvuew)",
            "mel_band_roformer",
            "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
            "dereverb_mel_band_roformer_anvuew.yaml",
            "noreverb",
            "reverb",
            "noreverb",
            "reverb",
            1,
            176400,
            456706759,
            "4fcbfc9d419abfd1f07a24f920d83f80d51727ca3cde3e9d84c99398418d35f3",
        ),
        ModelSpec(
            "vocals",
            "Vocals (BS-RoFormer ep 368)",
            "bs_roformer",
            "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "model_bs_roformer_ep_368_sdr_12.9628.yaml",
            "vocals",
            "instrumental",
            "vocals",
            "instrumental",
            1,
            264600,
            319821424,
            "97c9e1011924ff647d147b474638c161b80586fd5b9e83fab59cbbb70014bf62",
        ),
        ModelSpec(
            "vocals-aggressive",
            "Vocals (BS-RoFormer ep 317)",
            "bs_roformer",
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "model_bs_roformer_ep_317_sdr_12.9755.yaml",
            "vocals",
            "other",
            "vocals",
            "instrumental",
            4,
            176400,
            319821424,
            "4deed31719b7b92195530a9d0e86ea1c8d5c4aa9f0e2dd967c1f05419ebf96d5",
        ),
        ModelSpec(
            "karaoke",
            "Lead vocal (Mel-RoFormer karaoke, aufr33 & viperx)",
            "mel_band_roformer",
            "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            "config_mel_band_roformer_karaoke.yaml",
            "karaoke",
            "other",
            "main_vocal",
            "off_vocal",
            1,
            264600,
            456715871,
            "6603c56e16e85f0eab78d471e43341b789ddc654230b9bcc81a25a8d1457a8c1",
        ),
    )
}


def asset_files() -> dict[str, list[dict[str, Any]]]:
    """The catalog entries for every separation asset: only the checkpoints; the YAML ships with the package."""
    return {m.asset_id: [{"path": f"{WEIGHTS_DIR}/{m.ckpt}", "source": f"{WEIGHTS_DIR}/{m.ckpt}", "size": m.size, "sha256": m.sha256}] for m in MODELS.values()}


@lru_cache(maxsize=1)
def _load_presets_cached() -> tuple[dict[str, Any], ...]:
    data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    return tuple(data)


def load_presets() -> list[dict[str, Any]]:
    """Every preset, as plain dicts shaped like ``core.separation.models.SeparationPreset``."""
    return [json.loads(json.dumps(p)) for p in _load_presets_cached()]


def get_preset(preset_id: str) -> dict[str, Any]:
    for preset in load_presets():
        if preset["id"] == preset_id:
            return preset
    raise KeyError(preset_id)


def resolve_model(model_id: str, assets_dir: Path | str) -> dict[str, Any]:
    """A model spec with absolute file paths, as the worker request carries it.

    The YAML is taken from the assets folder when one sits beside the checkpoint (as in an
    original RVC install), else from the package.
    """
    spec = MODELS[model_id]
    weights = Path(assets_dir) / WEIGHTS_DIR
    config = weights / spec.config
    if not config.is_file():
        config = CONFIGS_DIR / spec.config
    out = asdict(spec)
    out.update(ckpt=str(weights / spec.ckpt), config=str(config), asset_id=spec.asset_id)
    return out


def resolve(preset_id: str, assets_dir: Path | str) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    """Return ``(preset, models, missing_asset_ids)`` for a preset; raises ``KeyError`` for an unknown id."""
    preset = get_preset(preset_id)
    models = {model_id: resolve_model(model_id, assets_dir) for model_id in preset["steps"]}
    missing: list[str] = []
    for model in models.values():
        if not Path(model["ckpt"]).is_file() and model["asset_id"] not in missing:
            missing.append(model["asset_id"])
    return preset, models, missing


@dataclass(frozen=True)
class PlanStep:
    """One model run in a preset: which model, and which of its two stems are written under which label.

    ``feeds_next`` names the stem passed on to the next step (the primary), or None for the last.
    """

    index: int
    model_id: str
    keep: tuple[tuple[str, str], ...]
    """(stem name in the model output, output label) pairs to write."""
    feeds_next: str | None


def plan_chain(preset: dict[str, Any], models: dict[str, Any] | None = None) -> list[PlanStep]:
    """The step order and the stems each step keeps.

    Each step separates the previous step's primary stem (the first separates the input). The last
    step's primary is written under the preset's ``primary_stem`` label; every other stem is written
    under its model label when the preset lists that label in ``stems``. A label is written once.
    """
    steps = list(preset["steps"])
    wanted = set(preset["stems"])
    written: set[str] = set()
    plan: list[PlanStep] = []
    for i, model_id in enumerate(steps):
        spec = models[model_id] if models else asdict(MODELS[model_id])
        last = i == len(steps) - 1
        keep: list[tuple[str, str]] = []
        primary_label = preset["primary_stem"] if last else spec["primary_label"]
        if last and primary_label not in written:
            keep.append((spec["primary"], primary_label))
            written.add(primary_label)
        secondary_label = spec["secondary_label"]
        if secondary_label in wanted and secondary_label not in written:
            keep.append((spec["secondary"], secondary_label))
            written.add(secondary_label)
        plan.append(PlanStep(i, model_id, tuple(keep), None if last else spec["primary"]))
    return plan


def output_name(input_stem: str, label: str, fmt: str, existing: set[str] | None = None, directory: Path | None = None) -> str:
    """``<input stem>.<label>.<format>``, with ``-2``, ``-3``… before the extension when the name is taken."""
    taken = set(existing or ())

    def free(name: str) -> bool:
        return name not in taken and not (directory is not None and (directory / name).exists())

    name = f"{input_stem}.{label}.{fmt}"
    n = 2
    while not free(name):
        name = f"{input_stem}.{label}-{n}.{fmt}"
        n += 1
    return name


@dataclass(frozen=True)
class ConfigInspection:
    """What a separation model's YAML declares."""

    model_type: str
    instruments: tuple[str, ...]
    target_instrument: str | None
    sample_rate: int | None
    chunk_size: int | None = None
    batch_size: int | None = None


def inspect_config(path: Path | str) -> ConfigInspection:
    """Parse a separation YAML and detect its architecture; raises ``ValueError`` when it is not usable."""
    import yaml
    from pymss_core.model_detection import ModelTypeDetectionError, detect_model_type

    try:
        config = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    except Exception as e:
        raise ValueError(f"{Path(path).name} is not a readable YAML file: {e}") from e
    try:
        model_type = detect_model_type(config)
    except ModelTypeDetectionError as e:
        raise ValueError(str(e)) from e
    training = config.get("training", {}) if isinstance(config, dict) else {}
    instruments = tuple(str(i) for i in (training.get("instruments") or []))
    if len(instruments) < 2:
        raise ValueError(f"{Path(path).name} lists fewer than two stems (training.instruments)")
    target = training.get("target_instrument")
    audio = config.get("audio", {}) if isinstance(config, dict) else {}
    rate = audio.get("sample_rate")
    inference = config.get("inference", {}) if isinstance(config, dict) else {}
    chunk = audio.get("chunk_size") or inference.get("chunk_size")
    batch = inference.get("batch_size")
    return ConfigInspection(model_type, instruments, str(target) if target else None, int(rate) if rate else None, int(chunk) if chunk else None, int(batch) if batch else None)


def custom_model(
    model_id: str,
    title: str,
    model_type: str,
    ckpt: Path | str,
    config: Path | str,
    primary: str,
    secondary: str,
    primary_label: str,
    secondary_label: str,
    chunk_size: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """A model spec for an imported model, shaped like ``resolve_model``'s output; chunking follows its YAML."""
    chunk = chunk_size or 352800
    return {
        "id": model_id,
        "title": title,
        "model_type": model_type,
        "ckpt": str(ckpt),
        "config": str(config),
        "primary": primary,
        "secondary": secondary,
        "primary_label": primary_label,
        "secondary_label": secondary_label,
        "batch_size": batch_size or 1,
        "overlap_size": chunk // 2,
        "chunk_size": chunk,
        "size": Path(ckpt).stat().st_size if Path(ckpt).is_file() else 0,
        "sha256": "",
        "asset_id": "",
    }
