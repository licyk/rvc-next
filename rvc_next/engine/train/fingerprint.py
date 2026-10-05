"""Input fingerprints: a stage reruns only when what it reads changed."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
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


def of_paths(paths: Iterable[Path | str], extra: object = None) -> str:
    """A SHA-1 over the names, sizes and modification times of ``paths`` (folders are walked).

    ``extra`` is mixed in as text, for settings that also decide a stage's output (the rate, the F0
    method). A missing path hashes differently from an empty folder.
    """
    h = hashlib.sha1()
    for raw in sorted(str(Path(p)) for p in paths):
        path = Path(raw)
        h.update(f"@{path.resolve()}\n".encode())
        if not path.exists():
            h.update(b"missing\n")
            continue
        for name, size, mtime in _walk(path):
            h.update(f"{name}\t{size}\t{mtime}\n".encode())
    if extra is not None:
        h.update(repr(extra).encode())
    return h.hexdigest()
