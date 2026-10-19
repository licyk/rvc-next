"""Where the database's file paths point, and finding a file again after the data directory moved.

``audio_files.path``, ``outputs.path`` and ``outputs.source_path`` record a file inside the data directory relative
to it (``outputs/2026-10-19/<job>/a.wav``), so the records follow the directory when it is moved, renamed, copied or
opened through a link; a file elsewhere (a server path, an outputs folder set outside the data directory) is
recorded absolute.

Databases from before kept absolute paths everywhere. ``reconcile`` (run whenever the data directory is opened at a
place other than the last one) rewrites the records of files inside it as relative ones, and ``find`` looks for a
file that is not where its record says: under the data directory and the outputs and experiments folders, by the
last parts of the recorded path. A candidate counts only with the recorded size (where the row keeps one), and a
record found again is rewritten. A file found nowhere is reported missing, never skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field

from rvc_next.core.db import Database
from rvc_next.core.errors import NotFoundError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import OutputsRemovedEvent
from rvc_next.core.files import from_stored, is_absolute_text, relocations, to_stored
from rvc_next.core.record import Record
from rvc_next.core.settings import SettingsService

logger = logging.getLogger(__name__)

FileKind = Literal["output", "source", "audio"]

FILE_MISSING = "file_missing"
"""``detail["reason"]`` of a ``not_found`` whose record's file is gone."""

DATA_DIR_KEY = "data_dir"
"""``meta``: where the data directory was opened last."""
PREVIOUS_DATA_DIR_KEY = "data_dir_previous"
"""``meta``: where it was before that, when known (recorded, or read off the records rewritten), for the experiments'
fingerprints written before they became relative."""


@dataclass(frozen=True)
class _Field:
    kind: FileKind
    table: str
    column: str
    size: str | None
    """The column holding the file's size, which a found file must match."""


FIELDS: dict[FileKind, _Field] = {
    "output": _Field("output", "outputs", "path", "size"),
    "source": _Field("source", "outputs", "source_path", None),
    "audio": _Field("audio", "audio_files", "path", "size"),
}


class MissingFile(Record):
    kind: FileKind
    """``output``: a result's file; ``source``: the input a result was made from; ``audio``: an upload or server file."""
    id: str
    name: str
    recorded: str
    """The path as the database records it."""


class PathCheckReport(Record):
    """What ``StorageService.check`` found among the recorded files."""

    data_dir: str
    relative: int = 0
    """Files inside the data directory, where their records say (relative, or absolute the old way until ``fixed``)."""
    external: int = 0
    """Files outside the data directory, where their records say."""
    relocated: int = 0
    """Files found away from their recorded place (their records rewritten when ``fixed``)."""
    rebased: int = 0
    """Records whose folder was replaced by ``rebase``."""
    missing: list[MissingFile] = Field(default_factory=list)
    """Files found nowhere."""
    pruned: int = 0
    """Records of missing outputs and audio files deleted (``prune_missing``); a missing source only stays reported."""
    fixed: bool = False
    """Whether the records were rewritten, or only checked."""


class PathCheckRequest(Record):
    fix: bool = False
    """Rewrite records: relative for files inside the data directory, the new place for files found elsewhere."""
    prune_missing: bool = False
    """Delete the records of outputs and audio files found nowhere."""
    rebase_from: str | None = None
    """With ``rebase_to``: records under this folder point under ``rebase_to`` instead (an outputs folder that moved
    with its disk)."""
    rebase_to: str | None = None


class StorageService:
    def __init__(self, settings: SettingsService, db: Database, events: EventBus) -> None:
        self._settings = settings
        self._db = db
        self._events = events

    @property
    def base(self) -> Path:
        return self._settings.data_dir

    def store(self, path: Path) -> str:
        """``path`` as a record keeps it: relative inside the data directory, absolute elsewhere."""
        return to_stored(path, self.base)

    def resolve(self, value: str) -> Path:
        """A recorded path on disk, where the record says (whether or not the file is there)."""
        return from_stored(value, self.base)

    def roots(self) -> list[Path]:
        """Folders a recorded file may be found under: the data directory, and the outputs and experiments folders
        when they are set elsewhere."""
        found: list[Path] = []
        for root in (self._settings.data_dir, self._settings.outputs_dir, self._settings.experiments_dir):
            if root not in found:
                found.append(root)
        return found

    def previous_data_dir(self) -> str | None:
        """Where the data directory was before it was opened here, when known."""
        return self._db.get_meta(PREVIOUS_DATA_DIR_KEY)

    def relocate(self, value: str, size: int | None, roots: list[Path] | None = None) -> Path | None:
        """The file recorded at ``value``, found by the last parts of its path under ``roots`` (default: ``roots()``)
        with the recorded size; None when none matches."""
        recorded = self.resolve(value)
        for candidate in relocations(value, self.roots() if roots is None else roots):
            if candidate != recorded and _matches(candidate, size):
                return candidate
        return None

    def find(self, kind: FileKind, row: Any) -> Path | None:
        """A row's file: where its record says, else found again (and the record rewritten); None when missing."""
        field = FIELDS[kind]
        value = row[field.column]
        if not value:
            return None
        path = self.resolve(value)
        if path.is_file():
            return path
        found = self.relocate(value, row[field.size] if field.size else None)
        if found is not None:
            self._db.execute(f"UPDATE {field.table} SET {field.column} = ? WHERE id = ?", (self.store(found), row["id"]))
            logger.info("Found %s again at %s (it was recorded at %s)", found.name, found, value)
        return found

    def missing_error(self, kind: FileKind, row: Any) -> NotFoundError:
        """The error for a row whose file is found nowhere."""
        value = row[FIELDS[kind].column]
        path = self.resolve(value)
        what = {"output": "The output file", "source": "The input this output was made from", "audio": "The file"}[kind]
        return NotFoundError(
            f"{what} is missing: {path.name} is not at {path}, nor under the data directory ({self.base})",
            {"kind": kind, "id": row["id"], "reason": FILE_MISSING, "path": str(path), "recorded": value, "hint": "rvc-next output check"},
        )

    # -- the whole database --------------------------------------------------------------

    def reconcile(self) -> PathCheckReport | None:
        """Run ``check(fix=True)`` when the data directory is opened at a new place, or for the first time since records
        became relative; None when it was opened here last. Safe while other processes open the same directory."""
        here = str(self.base)
        if self._db.get_meta(DATA_DIR_KEY) == here:
            return None
        with self._db.transaction(immediate=True):
            last = self._db.get_meta(DATA_DIR_KEY)
            if last == here:  # another process opening the directory did it meanwhile
                return None
            inferred: list[str] = []
            report = self._scan(fix=True, prune=False, previous=inferred)
            previous = last or (inferred[0] if inferred else None)
            if previous and previous != here:
                self._db.set_meta(PREVIOUS_DATA_DIR_KEY, previous)
            self._db.set_meta(DATA_DIR_KEY, here)
        if previous and previous != here:
            logger.info("The data directory is at %s now (it was at %s)", here, previous)
        if report.relocated or report.missing:
            logger.info(
                "Recorded files: %d found again, %d missing%s",
                report.relocated,
                len(report.missing),
                " (rvc-next output check lists them)" if report.missing else "",
            )
        return report

    def check(self, request: PathCheckRequest | None = None) -> PathCheckReport:
        """Every recorded output, source and audio file: where it is, rewritten (``fix``), moved (``rebase_*``) or
        pruned."""
        req = request or PathCheckRequest()
        if (req.rebase_from is None) != (req.rebase_to is None):
            raise ValidationError("A rebase needs both the old folder and the new one")
        if not (req.fix or req.prune_missing or req.rebase_from is not None):
            return self._scan(fix=False, prune=False)
        with self._db.transaction(immediate=True):
            rebased = self._rebase(req.rebase_from, req.rebase_to) if req.rebase_from and req.rebase_to else 0
            report = self._scan(fix=req.fix or rebased > 0, prune=req.prune_missing)
            report.rebased = rebased
        pruned_outputs = [m.id for m in report.missing if m.kind == "output"] if req.prune_missing else []
        if pruned_outputs:
            self._events.publish(OutputsRemovedEvent(ids=pruned_outputs))
        return report

    def _rebase(self, old: str, new: str) -> int:
        """Point the absolute records under ``old`` under ``new``."""
        old = old.rstrip("/\\")
        count = 0
        for field in FIELDS.values():
            for row in self._db.fetchall(f"SELECT id, {field.column} AS value FROM {field.table} WHERE {field.column} IS NOT NULL"):
                value = str(row["value"])
                if is_absolute_text(value) and value.startswith(old) and value[len(old) : len(old) + 1] in ("", "/", "\\"):
                    moved = Path(new).joinpath(*[p for p in value[len(old) :].replace("\\", "/").split("/") if p])
                    self._db.execute(f"UPDATE {field.table} SET {field.column} = ? WHERE id = ?", (self.store(moved), row["id"]))
                    count += 1
        return count

    def _scan(self, fix: bool, prune: bool, previous: list[str] | None = None) -> PathCheckReport:
        """Check every record; ``previous`` collects where the data directory was, read off old records found again."""
        report = PathCheckReport(data_dir=str(self.base), fixed=fix)
        for kind, field in FIELDS.items():
            size_sql = f", {field.size} AS size" if field.size else ", NULL AS size"
            for row in self._db.fetchall(f"SELECT id, {field.column} AS value{size_sql} FROM {field.table} WHERE {field.column} IS NOT NULL"):
                value = str(row["value"])
                size = row["size"]
                found: Path | None = None
                if not is_absolute_text(value):
                    if self.resolve(value).is_file():
                        report.relative += 1
                        continue
                    found = self.relocate(value, size)
                else:
                    stored = self.store(Path(value))
                    if not is_absolute_text(stored):  # inside the data directory: recorded the old way
                        if fix:
                            self._db.execute(f"UPDATE {field.table} SET {field.column} = ? WHERE id = ?", (stored, row["id"]))
                        if self.resolve(stored).is_file():
                            report.relative += 1
                            continue
                        found = self.relocate(stored, size)
                    else:
                        # A copy inside this data directory wins over the file it was copied from: a copied data
                        # directory must not keep reading, or deleting, the original's files.
                        found = self.relocate(value, size, roots=[self.base])
                        if found is None and Path(value).is_file():
                            report.external += 1
                            continue
                        if found is None:
                            found = self.relocate(value, size)
                        if found is not None and previous is not None and not previous:
                            _note_previous(value, found, self.base, previous)
                if found is not None:
                    report.relocated += 1
                    if fix:
                        self._db.execute(f"UPDATE {field.table} SET {field.column} = ? WHERE id = ?", (self.store(found), row["id"]))
                    continue
                report.missing.append(MissingFile(kind=kind, id=row["id"], name=PurePosixPath(value.replace("\\", "/")).name, recorded=value))
                if prune and kind != "source":
                    self._db.execute(f"DELETE FROM {field.table} WHERE id = ?", (row["id"],))
                    report.pruned += 1
        return report


def _note_previous(value: str, found: Path, base: Path, previous: list[str]) -> None:
    """Where the data directory was, when ``value`` (an old absolute record) and ``found`` (inside ``base``) are the same
    file at the same place relative to it."""
    try:
        rel = found.relative_to(base).as_posix()
    except ValueError:
        return
    text = value.replace("\\", "/")
    if text.endswith("/" + rel):
        previous.append(value[: len(value) - len(rel) - 1])


def _matches(path: Path, size: int | None) -> bool:
    """Whether ``path`` is a file of the recorded size."""
    try:
        st = path.stat()
    except OSError:
        return False
    return path.is_file() and (size is None or st.st_size == size)
