"""Input fingerprints: a stage reruns only when what it reads changed."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable
from pathlib import Path


def _walk(path: Path) -> Iterable[tuple[str, int, int]]:
    if path.is_file():
        st = path.stat()
        yield path.name, st.st_size, st.st_mtime_ns
        return
    if not path.is_dir():
        return
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = Path(root) / name
            try:
                st = full.stat()
            except OSError:
                continue
            yield full.relative_to(path).as_posix(), st.st_size, st.st_mtime_ns


def _label(path: Path, base: Path | None) -> str:
    """How a path is named in a fingerprint: relative to ``base`` when it lies inside it, so the fingerprint survives
    the folder being moved, renamed or copied; else absolute."""
    if base is not None:
        for candidate, root in ((path, base), (path.resolve(), base.resolve())):
            try:
                return candidate.relative_to(root).as_posix() or "."
            except ValueError:
                continue
    return str(path.resolve())


def of_paths(paths: Iterable[Path | str], extra: object = None, base: Path | None = None, rename: Callable[[Path], str] | None = None) -> str:
    """A SHA-1 over the names, sizes and modification times of ``paths`` (folders are walked).

    ``extra`` is mixed in as text, for settings that also decide a stage's output (the rate, the F0
    method). A missing path hashes differently from an empty folder. A path inside ``base`` (the
    experiment's folder) is named relative to it. ``rename`` names each path instead: the fingerprints
    written before ``base`` existed named every path absolute, and a caller recognising one passes the
    names the paths had then.
    """
    h = hashlib.sha1()
    named = sorted((rename(Path(p)) if rename else _label(Path(p), base), Path(p)) for p in paths)
    for label, path in named:
        h.update(f"@{label}\n".encode())
        if not path.exists():
            h.update(b"missing\n")
            continue
        for name, size, mtime in _walk(path):
            h.update(f"{name}\t{size}\t{mtime}\n".encode())
    if extra is not None:
        h.update(repr(extra).encode())
    return h.hexdigest()
