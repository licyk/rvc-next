"""Multi-speaker datasets: the speaker table, the ``Name_ID_Repeat`` folder convention and the manifest.

Ported from the original ``tools/multispeaker.py``. Sliced files are named after each entry's
``output_key``, which carries the speaker through F0, features, the file list and the index.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from rvc_next.engine.audio.io import AUDIO_EXTENSIONS

SPEAKER_ID_MIN = 0
SPEAKER_ID_MAX = 109
SPEAKER_EMBED_DIM = 110
MANIFEST_VERSION = 1
MANIFEST_NAME = "multispeaker_manifest.json"
SPEAKER_DIR_RE = re.compile(r"^(.+)_(\d+)_(\d+)$")


class ManifestError(ValueError):
    """The speaker table or folders are not a valid multi-speaker dataset."""


def audio_files(folder: Path | str, recursive: bool = True) -> list[str]:
    """Audio files under ``folder``, sorted, as absolute paths."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    if not recursive:
        return [str(p.resolve()) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    out: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                out.append(os.path.abspath(os.path.join(root, name)))
    return out


def _valid_name(name: str) -> bool:
    return bool(name) and not any(c in name for c in "|\n\r")


def speakers_from_folders(root: Path | str) -> list[dict[str, Any]]:
    """Read a folder of ``Name_ID_Repeat`` subfolders into speaker-table rows ``{name, id, folder, repeat}``."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise ManifestError(f"Not a folder: {root}")
    rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    names_by_id: dict[int, str] = {}
    children = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not children:
        raise ManifestError(f"{root} has no speaker subfolders")
    for child in children:
        match = SPEAKER_DIR_RE.match(child.name)
        if not match:
            invalid.append(child.name)
            continue
        name, speaker_id, repeat = match.group(1).strip(), int(match.group(2)), int(match.group(3))
        if not _valid_name(name) or not SPEAKER_ID_MIN <= speaker_id <= SPEAKER_ID_MAX or repeat < 1 or not audio_files(child):
            invalid.append(child.name)
            continue
        if names_by_id.get(speaker_id, name) != name:
            invalid.append(child.name)
            continue
        names_by_id[speaker_id] = name
        rows.append({"name": name, "id": speaker_id, "folder": str(child.resolve()), "repeat": repeat})
    if invalid:
        raise ManifestError("Invalid speaker folders (expected Name_ID_Repeat with an ID of 0–109, a repeat of 1 or more, one name per ID and audio inside): " + ", ".join(invalid))
    return rows


def _entry(path: str, name: str, speaker_id: int, repeat: int, index: int) -> dict[str, Any]:
    digest = hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode("utf8")).hexdigest()[:10]
    return {
        "path": os.path.abspath(path),
        "speaker_name": name,
        "speaker_id": int(speaker_id),
        "repeat": int(repeat),
        "output_key": f"ms{index:04d}_s{int(speaker_id):03d}_{digest}",
    }


def build_manifest(speakers: list[dict[str, Any]], root: str = "") -> dict[str, Any]:
    """A manifest from speaker-table rows ``{name, id, folder, repeat}`` (the original's helper rows)."""
    entries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    names_by_id: dict[int, str] = {}
    problems: list[str] = []
    for n, row in enumerate(speakers, 1):
        name = str(row.get("name", "")).strip()
        try:
            speaker_id = int(row["id"])
            repeat = int(row.get("repeat", 1))
        except (KeyError, TypeError, ValueError):
            problems.append(f"row {n}: bad id or repeat")
            continue
        folder = os.path.abspath(str(row.get("folder", "")))
        files = audio_files(folder)
        if not _valid_name(name):
            problems.append(f"row {n}: bad name")
        elif not SPEAKER_ID_MIN <= speaker_id <= SPEAKER_ID_MAX or repeat < 1:
            problems.append(f"row {n}: id must be 0–109 and repeat at least 1")
        elif names_by_id.get(speaker_id, name) != name:
            problems.append(f"row {n}: id {speaker_id} already belongs to {names_by_id[speaker_id]}")
        elif not files:
            problems.append(f"row {n}: no audio in {folder}")
        else:
            names_by_id[speaker_id] = name
            rows.append({"path": folder, "speaker_name": name, "speaker_id": speaker_id, "repeat": repeat})
            for path in files:
                entries.append(_entry(path, name, speaker_id, repeat, len(entries)))
    if problems:
        raise ManifestError("; ".join(problems))
    if not entries:
        raise ManifestError("The speaker table holds no audio")
    return {
        "version": MANIFEST_VERSION,
        "source": "helper",
        "root": os.path.abspath(root) if root else "",
        "rows": rows,
        "speakers": [{"id": i, "name": names_by_id[i]} for i in sorted(names_by_id)],
        "entries": entries,
    }


def write_manifest(exp_dir: Path | str, manifest: dict[str, Any]) -> Path:
    path = Path(exp_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(exp_dir: Path | str, check_files: bool = True) -> dict[str, Any]:
    """Read and validate ``multispeaker_manifest.json``; ``check_files`` also requires every source file to exist."""
    path = Path(exp_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ManifestError("This experiment has no multi-speaker manifest")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ManifestError(f"The multi-speaker manifest is not valid JSON: {e}") from e
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ManifestError("The multi-speaker manifest lists no audio")
    seen: set[str] = set()
    names_by_id: dict[int, str] = {}
    for entry in entries:
        try:
            source = os.path.abspath(str(entry["path"]))
            name = str(entry["speaker_name"]).strip()
            speaker_id = int(entry["speaker_id"])
            repeat = int(entry["repeat"])
            key = str(entry["output_key"])
        except (KeyError, TypeError, ValueError) as e:
            raise ManifestError("The multi-speaker manifest is malformed") from e
        if (check_files and not os.path.isfile(source)) or not _valid_name(name) or not SPEAKER_ID_MIN <= speaker_id <= SPEAKER_ID_MAX or repeat < 1 or not key:
            raise ManifestError(f"The multi-speaker manifest has an invalid entry: {source}")
        if names_by_id.get(speaker_id, name) != name:
            raise ManifestError(f"Speaker id {speaker_id} has two names in the manifest")
        if key in seen:
            raise ManifestError(f"Duplicate output key in the manifest: {key}")
        seen.add(key)
        names_by_id[speaker_id] = name
    manifest["speakers"] = [{"id": i, "name": names_by_id[i]} for i in sorted(names_by_id)]
    return manifest


def speaker_of(name: str, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """The manifest entry a sliced file belongs to (``<output_key>_<n>``)."""
    by_key = {e["output_key"]: e for e in manifest.get("entries", [])}
    stem = name.split(".")[0]
    return by_key.get(stem) or by_key.get(stem.rsplit("_", 1)[0])
