"""``filelist.txt`` and ``config.json``, written before the fit as the original's ``run_train_model`` does.

Each line is ``gt.wav|feature.npy|f0.npy|f0nsf.npy|speaker_id`` (without the two pitch columns for
voices without pitch guidance), plus ``|speaker_name`` in multi-speaker mode. Lines repeat per the
speaker's repeat count, and two mute lines per speaker are added. The mute samples ship with the
package and are copied into ``<experiment>/mute/``, since the loader caches spectrograms beside them.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

from rvc_next.engine.train import multispeaker
from rvc_next.engine.train.feature_extract import feature_dir_name
from rvc_next.engine.train.hparams import default_config, read_config, write_config

MUTE_SOURCE = Path(__file__).parent / "data" / "mute"


def copy_mute(exp_dir: Path, sample_rate: str, version: str) -> Path:
    """Copy the mute samples this experiment needs into ``<exp>/mute``; return that folder."""
    dst = Path(exp_dir) / "mute"
    fea = "3_feature256" if version == "v1" else "3_feature768"
    for rel in (f"0_gt_wavs/mute{sample_rate}.wav", f"{fea}/mute.npy", "2a_f0/mute.wav.npy", "2b-f0nsf/mute.wav.npy"):
        target = dst / rel
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(MUTE_SOURCE / rel, target)
    return dst


def _stems(folder: Path) -> set[str]:
    return {name.split(".")[0] for name in (p.name for p in folder.iterdir())} if folder.is_dir() else set()


def build_lines(
    exp_dir: Path | str, sample_rate: str, version: str, pitch_guidance: bool, speaker_id: int = 0, multi_speaker: bool = False
) -> tuple[list[str], list[dict[str, Any]]]:
    """The file-list lines (unshuffled) and the active speakers ``[{id, name}]`` (multi-speaker only)."""
    exp = Path(exp_dir)
    gt = exp / "0_gt_wavs"
    fea = exp / feature_dir_name(version)
    f0 = exp / "2a_f0"
    f0nsf = exp / "2b-f0nsf"
    names = {n for n in _stems(gt) if not n.endswith(".spec")} & _stems(fea)
    if pitch_guidance:
        names &= _stems(f0) & _stems(f0nsf)
    manifest = multispeaker.load_manifest(exp, check_files=False) if multi_speaker else None
    if manifest is not None:
        names = {n for n in names if multispeaker.speaker_of(n, manifest) is not None}
    if not names:
        raise ValueError("No slices have every stage's output: run slicing, pitch and features first")
    lines: list[str] = []
    active: dict[int, str] = {}
    for name in sorted(names):
        entry = multispeaker.speaker_of(name, manifest) if manifest is not None else None
        sid = int(entry["speaker_id"]) if entry else int(speaker_id)
        repeat = int(entry["repeat"]) if entry else 1
        if entry:
            active[sid] = str(entry["speaker_name"])
        cols = [(gt / f"{name}.wav").as_posix(), (fea / f"{name}.npy").as_posix()]
        if pitch_guidance:
            cols += [(f0 / f"{name}.wav.npy").as_posix(), (f0nsf / f"{name}.wav.npy").as_posix()]
        cols.append(str(sid))
        if entry:
            cols.append(str(entry["speaker_name"]))
        lines.extend(["|".join(cols)] * repeat)
    mute = copy_mute(exp, sample_rate, version)
    fea_name = "3_feature256" if version == "v1" else "3_feature768"
    for sid in sorted(active) if manifest is not None else [int(speaker_id)]:
        cols = [(mute / "0_gt_wavs" / f"mute{sample_rate}.wav").as_posix(), (mute / fea_name / "mute.npy").as_posix()]
        if pitch_guidance:
            cols += [(mute / "2a_f0" / "mute.wav.npy").as_posix(), (mute / "2b-f0nsf" / "mute.wav.npy").as_posix()]
        cols.append(str(sid))
        if manifest is not None:
            cols.append(active[sid])
        lines.extend(["|".join(cols)] * 2)
    speakers = [{"id": i, "name": active[i]} for i in sorted(active)]
    return lines, speakers


def write_filelist(
    exp_dir: Path | str,
    sample_rate: str,
    version: str,
    pitch_guidance: bool,
    speaker_id: int = 0,
    multi_speaker: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Write a shuffled ``filelist.txt``; return ``{"lines", "speakers", "path"}``."""
    lines, speakers = build_lines(exp_dir, sample_rate, version, pitch_guidance, speaker_id, multi_speaker)
    random.Random(seed).shuffle(lines)
    path = Path(exp_dir) / "filelist.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"lines": len(lines), "speakers": speakers, "path": str(path)}


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def prepare_config(exp_dir: Path | str, sample_rate: str, version: str, speakers: list[dict[str, Any]] | None = None, override: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write ``config.json``: an existing one is kept when its rate matches (users edit it), else the default.

    In multi-speaker mode the embedding has 110 rows and ``speaker_info`` lists the speakers.
    """
    from rvc_next.engine.models.checkpoint import SAMPLE_RATES

    existing = read_config(Path(exp_dir))
    if existing and existing.get("data", {}).get("sampling_rate") == SAMPLE_RATES[sample_rate]:
        config = existing
    else:
        config = default_config(version, sample_rate)
    if override:
        config = _merge(config, override)
    if speakers:
        config["model"]["spk_embed_dim"] = multispeaker.SPEAKER_EMBED_DIM
        config["speaker_info"] = [{"id": int(s["id"]), "name": str(s["name"])} for s in speakers]
    else:
        config.pop("speaker_info", None)
    write_config(Path(exp_dir), config)
    return config
