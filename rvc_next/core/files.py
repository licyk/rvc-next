"""File helpers: trash, names, and roots."""

from __future__ import annotations

import getpass
import logging
import os
import re
import sys
import unicodedata
import zipfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, cast

from rvc_next.core.clock import new_id
from rvc_next.core.fsops import move_path, remove_path

logger = logging.getLogger(__name__)


def trash(path: Path, fallback_dir: Path) -> None:
    """Move to the operating system's trash; without one, into ``fallback_dir``."""
    if not path.exists() and not path.is_symlink():
        return
    try:
        from send2trash import send2trash

        send2trash(str(path))
        return
    except Exception as e:
        logger.info("OS trash unavailable for %s (%s); using %s", path.name, e, fallback_dir)
    fallback_dir.mkdir(parents=True, exist_ok=True)
    try:
        move_path(path, fallback_dir / f"{new_id()}-{path.name}")
    except Exception:
        remove_path(path)


def slugify(name: str, fallback: str = "voice") -> str:
    """A folder-safe id: lower case letters, digits and dashes, keeping CJK and other letters."""
    text = unicodedata.normalize("NFKC", name).strip().lower()
    text = re.sub(r"[^\w\-]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"[-_]{2,}", "-", text).strip("-_")
    return text[:64] or fallback


def is_within(path: Path, roots: list[Path]) -> bool:
    """Whether ``path`` (resolved) lies inside one of ``roots``."""
    resolved = path.resolve()
    for root in roots:
        r = root.resolve()
        if resolved == r or r in resolved.parents:
            return True
    return False


def windows_drives(listdrives: Callable[[], list[str]] | None = None) -> list[Path]:
    """Every drive letter (C:\\, D:\\, …), read from the drive list without touching the disks."""
    if listdrives is None:
        listdrives = getattr(os, "listdrives", None)  # Python 3.12+
    if listdrives is not None:
        return [Path(d) for d in listdrives()]
    if sys.platform != "win32":
        return []
    import ctypes

    mask = ctypes.windll.kernel32.GetLogicalDrives()
    return [Path(f"{chr(ord('A') + i)}:\\") for i in range(26) if mask >> i & 1]


def volume_roots() -> list[Path]:
    """The machine's drives (Windows) or mounted volumes (macOS, Linux), for the server file browser.

    The boot volume's alias (``/Volumes/Macintosh HD`` → ``/``) is left out: the file system root
    is not offered by default.
    """
    if sys.platform == "win32":
        try:
            return windows_drives()
        except (OSError, AttributeError) as e:
            logger.warning("Cannot list the drives: %s", e)
            return []
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = ""
    bases = [Path("/Volumes")] if sys.platform == "darwin" else [p for p in (Path("/media") / user, Path("/run/media") / user, Path("/mnt")) if user or p.name == "mnt"]
    roots: list[Path] = []
    for base in bases:
        try:
            children = sorted(base.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                if child.is_dir() and child.resolve() != Path("/"):
                    roots.append(child)
            except OSError:
                continue
    return roots


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def reveal(path: Path) -> None:
    """Show ``path`` in the operating system's file manager."""
    import subprocess
    import sys

    target = path if path.exists() else path.parent
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(target)] if target.is_file() else ["explorer", str(target)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(target)] if target.is_file() else ["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target.parent if target.is_file() else target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def zip_stream(entries: Iterable[tuple[str, Path]]) -> Iterator[bytes]:
    """A stored (uncompressed) zip of ``(name in the archive, file)`` pairs, produced in pieces, so a
    response can stream it without building it first. Repeated names get ``_1``, ``_2``…"""

    class _Sink:
        def __init__(self) -> None:
            self.buf = bytearray()
            self.pos = 0

        def write(self, b: bytes) -> int:
            self.buf += b
            self.pos += len(b)
            return len(b)

        def tell(self) -> int:
            return self.pos

        def flush(self) -> None:
            pass

    sink = _Sink()
    used: set[str] = set()
    with zipfile.ZipFile(cast(IO[bytes], sink), "w", compression=zipfile.ZIP_STORED) as zf:
        for arcname, path in entries:
            name, n = arcname, 1
            while name in used:
                stem, dot, ext = arcname.rpartition(".")
                name = f"{stem}_{n}.{ext}" if dot else f"{arcname}_{n}"
                n += 1
            used.add(name)
            with zf.open(name, "w", force_zip64=True) as dst, open(path, "rb") as src:
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    dst.write(block)
                    if sink.buf:
                        yield bytes(sink.buf)
                        sink.buf.clear()
    if sink.buf:
        yield bytes(sink.buf)
