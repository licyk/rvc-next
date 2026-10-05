"""Finding voices on disk: library folders, legacy installs, zip archives, and index pairing."""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

from rvc_next.core.errors import ValidationError

SPK_INDEX = re.compile(r"_spkid(\d+)$", re.IGNORECASE)
MAX_ZIP_MEMBER = 4 * 1024**3


def suggest_index(model_path: Path, roots: list[Path], speaker_id: int | None = None) -> Path | None:
    """The original's ``get_index_path_from_model`` scoring, used once as a suggestion.

    Candidates are ``added`` indexes under ``roots`` whose name matches the experiment name
    (the model stem without ``_e<N>_s<N>``) or the model stem; the matching speaker, a standard
    name, the first root and the newest file win.
    """
    model_stem = model_path.stem
    experiment = re.sub(r"_e\d+_s\d+$", "", model_stem, flags=re.IGNORECASE)
    if not experiment:
        return None
    lower_exp = experiment.lower()
    candidates: list[tuple[tuple, Path]] = []
    for root_index, root in enumerate(roots):
        if not root.is_dir():
            continue
        for dirpath, _, files in os.walk(root, topdown=False):
            for name in files:
                if not name.lower().endswith(".index") or "trained" in name.lower():
                    continue
                stem = os.path.splitext(name)[0]
                lower = stem.lower()
                m = SPK_INDEX.search(stem)
                indexed_speaker = int(m.group(1)) if m else None
                if speaker_id is None and indexed_speaker is not None:
                    continue
                if speaker_id is not None and indexed_speaker is not None and indexed_speaker != speaker_id:
                    continue
                standard = lower.startswith(lower_exp + "_added_") or f"_{lower_exp}_v1" in lower or f"_{lower_exp}_v2" in lower
                exact = model_stem.lower() in lower
                if standard or exact:
                    path = Path(dirpath, name).resolve()
                    score = (0 if indexed_speaker == speaker_id else 1, 0 if standard else 1, root_index, -path.stat().st_mtime, str(path).lower())
                    candidates.append((score, path))
    return min(candidates, key=lambda c: c[0])[1] if candidates else None


def index_key(path: Path) -> str:
    """``spk<N>`` for a per-speaker index, else ``default``."""
    m = SPK_INDEX.search(path.stem)
    return f"spk{int(m.group(1))}" if m else "default"


def extract_zip(archive: Path, dest: Path) -> list[Path]:
    """Extract the ``.pth`` and ``.index`` members of a shared voice archive, flattening folders."""
    out: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = os.path.basename(info.filename.replace("\\", "/"))
                if not name.lower().endswith((".pth", ".index")) or name.startswith("."):
                    continue
                if info.file_size > MAX_ZIP_MEMBER:
                    raise ValidationError(f"{name} in the archive is too large")
                target = dest / name
                n = 1
                while target.exists():
                    target = dest / f"{Path(name).stem}_{n}{Path(name).suffix}"
                    n += 1
                with zf.open(info) as src, open(target, "wb") as dst:
                    while True:
                        block = src.read(1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
                out.append(target)
    except zipfile.BadZipFile as e:
        raise ValidationError(f"{archive.name} is not a zip archive") from e
    return out


def legacy_weights(root: Path) -> list[Path]:
    folder = root / "assets" / "weights"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pth")


def legacy_index_roots(root: Path) -> list[Path]:
    """Where the original looks for indexes: ``assets/indices`` (outside) first, then ``logs``."""
    return [root / "assets" / "indices", root / "logs"]
