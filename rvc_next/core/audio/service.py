"""Audio inputs, outputs, peaks and server browsing."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from rvc_next.core.audio.models import AudioAnalysis, AudioFile, AudioInfo, AudioRef, BrowseEntry, BrowseListing, Output, OutputPage, Peaks, SpectrogramData
from rvc_next.core.clock import new_id, now_iso
from rvc_next.core.db import Database
from rvc_next.core.errors import InvalidPathError, NotFoundError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import OutputsAddedEvent, OutputsRemovedEvent
from rvc_next.core.files import is_within, reveal, trash, zip_stream
from rvc_next.core.params import VoiceParamsModel
from rvc_next.core.safety import unique_path, validate_name
from rvc_next.core.settings import SettingsService

logger = logging.getLogger(__name__)

UPLOAD_TTL_SECONDS = 24 * 3600
MIN_POINTS, MAX_POINTS = 64, 4096
MAX_ANALYSIS_SECONDS = 30 * 60


def _audio_ext(name: str) -> bool:
    from rvc_next.engine.audio.io import is_audio_path

    return is_audio_path(name)


class AudioFileService:
    def __init__(self, settings: SettingsService, db: Database, events: EventBus) -> None:
        self._settings = settings
        self._db = db
        self._events = events

    @property
    def uploads_dir(self) -> Path:
        return self._settings.data_dir / "uploads"

    @property
    def peaks_dir(self) -> Path:
        return self._settings.data_dir / "cache" / "peaks"

    # -- probing ------------------------------------------------------------------

    def probe(self, path: Path) -> AudioInfo:
        from rvc_next.engine.audio.io import probe
        from rvc_next.engine.errors import AudioError

        try:
            info = probe(path)
        except AudioError as e:
            raise ValidationError(str(e), {"reason": "audio"}) from e
        return AudioInfo(duration=info.duration, sample_rate=info.sample_rate, channels=info.channels, codec=info.codec)

    # -- inputs -------------------------------------------------------------------

    def upload(self, name: str, chunks: Iterable[bytes]) -> AudioFile:
        """Stream a raw upload body to ``uploads/<id>/<name>``, then probe it."""
        name = validate_name(os.path.basename(name.replace("\\", "/")) or "audio")
        file_id = new_id("a")
        folder = self.uploads_dir / file_id
        folder.mkdir(parents=True, exist_ok=True)
        part = folder / (name + ".part")
        size = 0
        try:
            with open(part, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
                    size += len(chunk)
            target = folder / name
            part.replace(target)
            info = self.probe(target)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return self._insert(file_id, target, "upload", info, size)

    def resolve_path(self, raw: str) -> AudioFile:
        """Register a server file; it must lie inside ``paths.browse_roots``."""
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise InvalidPathError("A server path must be absolute")
        self._check_root(path)
        if not path.is_file():
            raise NotFoundError(f"No such file: {path}")
        st = path.stat()
        row = self._db.fetchone("SELECT * FROM audio_files WHERE path = ? AND mtime_ns = ? AND origin = 'path'", (str(path.resolve()), st.st_mtime_ns))
        if row is not None:
            return self._row(row)
        info = self.probe(path)
        return self._insert(new_id("a"), path.resolve(), "path", info, st.st_size)

    def _insert(self, file_id: str, path: Path, origin: Literal["upload", "path"], info: AudioInfo, size: int) -> AudioFile:
        st = path.stat()
        created = now_iso()
        self._db.execute(
            "INSERT INTO audio_files(id, path, name, origin, duration, sample_rate, channels, size, mtime_ns, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (file_id, str(path), path.name, origin, info.duration, info.sample_rate, info.channels, size, st.st_mtime_ns, created),
        )
        return AudioFile(
            id=file_id, name=path.name, path=str(path), origin=origin, duration=info.duration, sample_rate=info.sample_rate, channels=info.channels, size=size, created_at=created
        )

    def get_file(self, file_id: str) -> AudioFile:
        row = self._db.fetchone("SELECT * FROM audio_files WHERE id = ?", (file_id,))
        if row is None:
            raise NotFoundError(f"Unknown audio file: {file_id}")
        return self._row(row)

    def file_path(self, file_id: str) -> Path:
        path = Path(self.get_file(file_id).path)
        if not path.is_file():
            raise NotFoundError(f"The file is gone: {path.name}")
        return path

    @staticmethod
    def _row(r: Any) -> AudioFile:
        return AudioFile(
            id=r["id"],
            name=r["name"],
            path=r["path"],
            origin=r["origin"],
            duration=r["duration"],
            sample_rate=r["sample_rate"],
            channels=r["channels"],
            size=r["size"],
            created_at=r["created_at"],
        )

    def resolve_ref(self, ref: AudioRef) -> tuple[Path, str]:
        """The file behind an input reference, and a display name."""
        if ref.kind == "upload":
            if not ref.id:
                raise ValidationError("An upload reference needs an id")
            path = self.file_path(ref.id)
            return path, path.name
        if ref.kind == "output-source":
            if not ref.id:
                raise ValidationError("An output-source reference needs an output id")
            src = self.get_output(ref.id).source_path
            if not src or not Path(src).is_file():
                raise NotFoundError("The input this output was made from is gone")
            return Path(src), Path(src).name
        if ref.kind == "output":
            if not ref.id:
                raise ValidationError("An output reference needs an id")
            out = self.get_output(ref.id)
            path = Path(out.path)
            if not path.is_file():
                raise NotFoundError(f"The output file is gone: {path.name}")
            return path, path.name
        if not ref.path:
            raise ValidationError("A path reference needs a path")
        path = Path(ref.path).expanduser()
        if not path.is_file():
            raise NotFoundError(f"No such file: {path}")
        return path, path.name

    def expand_inputs(self, refs: list[AudioRef], recursive: bool = False) -> list[tuple[Path, str]]:
        """Resolve references; a path reference to a folder expands to its audio files."""
        out: list[tuple[Path, str]] = []
        for ref in refs:
            if ref.kind == "path" and ref.path and Path(ref.path).expanduser().is_dir():
                folder = Path(ref.path).expanduser()
                it = folder.rglob("*") if recursive else folder.iterdir()
                out += [(p, p.relative_to(folder).as_posix()) for p in sorted(it) if p.is_file() and _audio_ext(p.name)]
            else:
                out.append(self.resolve_ref(ref))
        if not out:
            raise ValidationError("No audio files to process")
        return out

    def check_refs(self, refs: list[AudioRef]) -> None:
        """Refuse server paths outside ``paths.browse_roots`` (the API calls this; the command line is trusted)."""
        for ref in refs:
            if ref.kind == "path" and ref.path:
                self._check_root(Path(ref.path).expanduser())

    # -- browsing -----------------------------------------------------------------

    def roots(self) -> list[Path]:
        return self._settings.browse_roots()

    def _check_root(self, path: Path) -> None:
        if not is_within(path, self.roots()):
            raise InvalidPathError(f"{path} is outside the folders the server may browse", {"roots": [str(r) for r in self.roots()]})

    def browse(self, path: str | None = None) -> BrowseListing:
        roots = self.roots()
        if not path:
            entries = [BrowseEntry(name=str(r), path=str(r), is_dir=True) for r in roots if r.is_dir()]
            return BrowseListing(root="", path="", parent=None, entries=entries, roots=[str(r) for r in roots])
        target = Path(path).expanduser()
        self._check_root(target)
        if not target.is_dir():
            raise NotFoundError(f"No such folder: {target}")
        root = next((r for r in roots if target.resolve() == r or r in target.resolve().parents), roots[0])
        entries: list[BrowseEntry] = []
        try:
            children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError as e:
            raise InvalidPathError(f"Cannot read {target}: permission denied") from e
        for child in children:
            if child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
                st = child.stat()
            except OSError:
                continue
            is_audio = not is_dir and _audio_ext(child.name)
            if not is_dir and not is_audio:
                continue
            entries.append(BrowseEntry(name=child.name, path=str(child), is_dir=is_dir, is_audio=is_audio, size=None if is_dir else st.st_size, mtime=st.st_mtime))
        parent = None if target.resolve() == root else str(target.resolve().parent)
        return BrowseListing(root=str(root), path=str(target.resolve()), parent=parent, entries=entries, roots=[str(r) for r in roots])

    # -- peaks ----------------------------------------------------------------------

    def peaks(self, path: Path, points: int = 1024) -> Peaks:
        from rvc_next.engine.audio.io import decode
        from rvc_next.engine.audio.peaks import compute_peaks
        from rvc_next.engine.errors import AudioError

        points = max(MIN_POINTS, min(MAX_POINTS, int(points)))
        st = path.stat()
        key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{points}".encode()).hexdigest()
        cache = self.peaks_dir / key[:2] / f"{key}.json"
        if cache.is_file():
            try:
                return Peaks.model_validate_json(cache.read_text(encoding="utf-8"))
            except ValueError:
                pass
        try:
            audio, sr = decode(path, None, mono=True)
        except AudioError as e:
            raise ValidationError(str(e), {"reason": "audio"}) from e
        data = compute_peaks(audio, points)
        result = Peaks(points=points, duration=audio.shape[0] / sr if sr else 0.0, sample_rate=sr, data=[round(float(v), 4) for v in data])
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(), encoding="utf-8")
        tmp.replace(cache)
        return result

    # -- analysis ----------------------------------------------------------------------

    def analyse(self, path: Path, columns: int = 800) -> AudioAnalysis:
        """A spectrogram and the pitch curve of ``path`` (cached like the peaks)."""
        import base64

        from rvc_next.engine.audio.analysis import median_pitch, pitch_curve, spectrogram
        from rvc_next.engine.audio.io import decode
        from rvc_next.engine.errors import AudioError

        columns = max(64, min(4000, int(columns)))
        st = path.stat()
        key = hashlib.sha1(f"analysis|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{columns}".encode()).hexdigest()
        cache = self.peaks_dir / key[:2] / f"{key}.json"
        if cache.is_file():
            try:
                return AudioAnalysis.model_validate_json(cache.read_text(encoding="utf-8"))
            except ValueError:
                pass
        info = self.probe(path)
        if info.duration and info.duration > MAX_ANALYSIS_SECONDS:
            raise ValidationError(f"{path.name} is longer than {MAX_ANALYSIS_SECONDS // 60} minutes; analysis covers shorter audio", {"reason": "too_long"})
        try:
            audio, sr = decode(path, None, mono=True)
            audio_16k, _ = decode(path, 16000, mono=True)
        except AudioError as e:
            raise ValidationError(str(e), {"reason": "audio"}) from e
        spec = spectrogram(audio, sr, columns)
        f0 = pitch_curve(audio_16k)
        result = AudioAnalysis(
            duration=audio.shape[0] / sr if sr else 0.0,
            sample_rate=sr,
            channels=info.channels,
            codec=info.codec,
            spectrogram=SpectrogramData(rows=spec.rows, cols=spec.cols, fmax=spec.fmax, data=base64.b64encode(spec.data.tobytes()).decode("ascii")),
            pitch=[round(float(v), 2) for v in f0],
            pitch_median=median_pitch(f0),
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(), encoding="utf-8")
        tmp.replace(cache)
        return result

    # -- outputs ----------------------------------------------------------------------

    def job_output_dir(self, job_id: str) -> Path:
        day = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
        return self._settings.outputs_dir / day / job_id

    @staticmethod
    def output_path(folder: Path, stem: str, suffix: str, fmt: str) -> Path:
        """``<stem>.<suffix>.<fmt>`` in ``folder``, never overwriting."""
        safe = "".join("_" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in stem).strip(" .") or "audio"
        name = f"{safe}.{suffix}.{fmt}" if suffix else f"{safe}.{fmt}"
        return unique_path(folder / name)

    def add_output(
        self,
        job_id: str | None,
        path: Path,
        kind: str,
        label: str,
        source_path: Path | None = None,
        model_id: str | None = None,
        voice: VoiceParamsModel | None = None,
        publish: bool = True,
    ) -> Output:
        try:
            info: AudioInfo | None = self.probe(path)
        except ValidationError:
            info = None
        out_id = new_id("o")
        created = now_iso()
        size = path.stat().st_size if path.exists() else None
        self._db.execute(
            "INSERT INTO outputs(id, job_id, path, kind, label, source_path, model_id, voice, duration, sample_rate, channels, size, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                out_id,
                job_id,
                str(path),
                kind,
                label,
                str(source_path) if source_path else None,
                model_id,
                voice.model_dump_json() if voice else None,
                info.duration if info else None,
                info.sample_rate if info else None,
                info.channels if info else None,
                size,
                created,
            ),
        )
        output = self.get_output(out_id)
        if publish:
            self._events.publish(OutputsAddedEvent(job_id=job_id, outputs=[output]))
        return output

    def get_output(self, output_id: str) -> Output:
        row = self._db.fetchone("SELECT * FROM outputs WHERE id = ?", (output_id,))
        if row is None:
            raise NotFoundError(f"Unknown output: {output_id}")
        return self._output_row(row)

    def list_outputs(self, job_id: str | None = None, voice_id: str | None = None, kind: str | None = None, cursor: str | None = None, limit: int = 100) -> OutputPage:
        where, params = [], []
        if job_id:
            where.append("job_id = ?")
            params.append(job_id)
        if voice_id:
            where.append("model_id = ?")
            params.append(voice_id)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if cursor:
            created, _, oid = cursor.partition("|")
            where.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params += [created, created, oid]
        sql = "SELECT * FROM outputs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC, id DESC LIMIT ?"
        rows = self._db.fetchall(sql, (*params, limit + 1))
        items = [self._output_row(r) for r in rows[:limit]]
        next_cursor = f"{rows[limit - 1]['created_at']}|{rows[limit - 1]['id']}" if len(rows) > limit else None
        return OutputPage(items=items, next_cursor=next_cursor)

    def delete_output(self, output_id: str, delete_file: bool = True) -> None:
        out = self.get_output(output_id)
        if delete_file:
            path = Path(out.path)
            trash(path, self._settings.data_dir / "trash")
            try:
                path.parent.rmdir()
            except OSError:
                pass
        self._db.execute("DELETE FROM outputs WHERE id = ?", (output_id,))
        self._events.publish(OutputsRemovedEvent(ids=[output_id]))

    def output_file(self, output_id: str) -> Path:
        path = Path(self.get_output(output_id).path)
        if not path.is_file():
            raise NotFoundError(f"The output file is gone: {path.name}")
        return path

    def zip_outputs(self, ids: list[str]) -> Iterator[bytes]:
        """A zip of the chosen outputs, produced in pieces."""
        return zip_stream((p.name, p) for p in [self.output_file(i) for i in ids])

    def reveal(self, path: Path) -> None:
        reveal(path)

    def _output_row(self, r: Any) -> Output:
        path = Path(r["path"])
        source = r["source_path"]
        return Output(
            id=r["id"],
            job_id=r["job_id"],
            path=r["path"],
            name=path.name,
            kind=r["kind"],
            label=r["label"],
            source_path=source,
            source_name=Path(source).name if source else None,
            model_id=r["model_id"],
            voice=VoiceParamsModel.model_validate_json(r["voice"]) if r["voice"] else None,
            duration=r["duration"],
            sample_rate=r["sample_rate"],
            channels=r["channels"],
            size=r["size"],
            exists=path.is_file(),
            created_at=r["created_at"],
        )

    # -- housekeeping -----------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove uploads older than a day that no output references, and outputs past their retention."""
        now = time.time()
        referenced = {r["source_path"] for r in self._db.fetchall("SELECT DISTINCT source_path FROM outputs WHERE source_path IS NOT NULL")}
        if self.uploads_dir.is_dir():
            for folder in self.uploads_dir.iterdir():
                try:
                    if now - folder.stat().st_mtime < UPLOAD_TTL_SECONDS:
                        continue
                    files = list(folder.iterdir()) if folder.is_dir() else [folder]
                    if any(str(f) in referenced for f in files):
                        continue
                    shutil.rmtree(folder, ignore_errors=True) if folder.is_dir() else folder.unlink(missing_ok=True)
                    self._db.execute("DELETE FROM audio_files WHERE path LIKE ?", (str(folder) + "%",))
                except OSError:
                    continue
        days = self._settings.settings.convert.keep_outputs_days
        if days > 0:
            cutoff = datetime.fromtimestamp(now - days * 86400, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            for row in self._db.fetchall("SELECT id FROM outputs WHERE created_at < ?", (cutoff,)):
                try:
                    self.delete_output(row["id"])
                except Exception:
                    logger.exception("Could not remove an old output")
