"""The voice library.

A library voice is a folder ``models/<slug>/`` with ``model.pth``, its index (``model.index`` or
``spkid<N>.index``) and ``voice.json``. The ``.pth`` is never rewritten, so it stays usable in the
original. The database is a cache: ``rescan()`` rebuilds it from the folders and the legacy roots.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rvc_next.core.clock import new_id, now_iso
from rvc_next.core.db import Database
from rvc_next.core.engine_errors import map_engine_error
from rvc_next.core.errors import ConflictError, InvalidPathError, ModelFormatError, NotFoundError, RvcNextError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import ModelsChangedEvent
from rvc_next.core.files import slugify, trash
from rvc_next.core.jobs.service import JobContext, JobService, JobSpec
from rvc_next.core.models.importers import index_key, legacy_index_roots, legacy_weights, suggest_index
from rvc_next.core.models.models import (
    CheckpointInfo,
    ExtractRequest,
    ImportResult,
    IndexSuggestion,
    LegacyScanResult,
    MergeRequest,
    Speaker,
    VoiceModel,
    VoiceUpdate,
)
from rvc_next.core.safety import validate_name
from rvc_next.core.settings import SettingsService

if TYPE_CHECKING:
    from rvc_next.core.presets.service import PresetService

logger = logging.getLogger(__name__)

VOICE_FILE = "voice.json"
MODEL_FILE = "model.pth"
LEGACY_FILE = "legacy-voices.json"
# Folders of models/ that hold other model types.
RESERVED = {"voices", "indexes", "base", "separation", "attached-indexes"}


def _index_filename(key: str) -> str:
    return "model.index" if key == "default" else f"spkid{int(key[3:])}.index"


class VoiceModelService:
    def __init__(self, settings: SettingsService, db: Database, events: EventBus, jobs: JobService) -> None:
        self._settings = settings
        self._db = db
        self._events = events
        self._jobs = jobs
        self._lock = threading.RLock()
        self.presets: PresetService | None = None
        self.imports: Any = None
        """The import sessions; set by ``build_services``."""
        jobs.register("merge", self._merge_spec)
        jobs.register("extract", self._extract_spec)

    @property
    def root(self) -> Path:
        return self._settings.models_dir / "voices"

    def _migrate_layout(self) -> None:
        """Move voice folders from ``models/<slug>/`` (before 2026-10-04) to ``models/voices/<slug>/``."""
        top = self._settings.models_dir
        if not top.is_dir():
            return
        for folder in sorted(top.iterdir()):
            if folder.name in RESERVED or not folder.is_dir() or not (folder / MODEL_FILE).is_file():
                continue
            self.root.mkdir(parents=True, exist_ok=True)
            target = self.root / folder.name
            if target.exists():
                logger.warning("Not moving %s: %s exists", folder, target)
                continue
            folder.rename(target)
            logger.info("Moved voice %s into %s", folder.name, self.root)

    # -- reading --------------------------------------------------------------------

    def list_voices(self, include_hidden: bool = False) -> list[VoiceModel]:
        rows = self._db.fetchall("SELECT * FROM voice_models WHERE location != 'temporary'" + ("" if include_hidden else " AND hidden = 0") + " ORDER BY name COLLATE NOCASE")
        return [self._row(r) for r in rows]

    def get(self, voice_id: str) -> VoiceModel:
        row = self._db.fetchone("SELECT * FROM voice_models WHERE id = ?", (voice_id,))
        if row is None:
            raise NotFoundError(f"Unknown voice: {voice_id}")
        return self._row(row)

    def resolve(self, ref: str, allow_path: bool = True) -> VoiceModel:
        """A voice by id, by name (case-insensitive) or, for the command line, by ``.pth`` path."""
        row = self._db.fetchone("SELECT * FROM voice_models WHERE id = ?", (ref,))
        if row is None:
            row = self._db.fetchone("SELECT * FROM voice_models WHERE name = ? COLLATE NOCASE AND location != 'temporary' ORDER BY hidden LIMIT 1", (ref,))
        if row is not None:
            return self._row(row)
        path = Path(ref).expanduser()
        if allow_path and path.suffix.lower() == ".pth" and path.is_file():
            return self.temporary(path)
        raise NotFoundError(f"No voice named {ref!r}")

    def temporary(self, path: Path, index: Path | None = None) -> VoiceModel:
        """Register a ``.pth`` outside the library for one command or a training try-out."""
        path = path.resolve()
        voice_id = "tmp-" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
        indexes = {index_key(index): str(index.resolve())} if index else {}
        self._upsert_from_pth(voice_id, path, location="temporary", name=path.stem, indexes=indexes)
        return self.get(voice_id)

    def _row(self, r: Any) -> VoiceModel:
        indexes = json.loads(r["indexes"])
        return VoiceModel(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            tags=json.loads(r["tags"] or "[]"),
            location=r["location"],
            legacy=r["location"].startswith("legacy:"),
            model_path=r["model_path"],
            sample_rate=r["sample_rate"],
            version=r["version"],
            pitch_guidance=bool(r["pitch_guidance"]),
            speakers=[Speaker.model_validate(s) for s in json.loads(r["speakers"])],
            speaker_slots=r["speaker_slots"],
            indexes=indexes,
            has_index=any(Path(p).is_file() for p in indexes.values()),
            size=r["size"],
            info=r["info"],
            hidden=bool(r["hidden"]),
            catalog_id=r["catalog_id"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    def index_for(self, voice: VoiceModel, speaker_id: int) -> Path | None:
        """The index to use for a speaker: its own ``spk<N>`` index, else the default one."""
        for key in (f"spk{speaker_id}", "default"):
            p = voice.indexes.get(key)
            if p and Path(p).is_file():
                return Path(p)
        if voice.speakers and voice.indexes:
            return None
        p = next((Path(v) for v in voice.indexes.values() if Path(v).is_file()), None)
        return p

    # -- scanning ---------------------------------------------------------------------

    def rescan(self) -> None:
        """Rebuild the cache from the library folders and the legacy roots."""
        with self._lock:
            self._migrate_layout()
            seen: set[str] = set()
            if self.root.is_dir():
                for folder in sorted(self.root.iterdir()):
                    if folder.is_dir() and (folder / MODEL_FILE).is_file():
                        try:
                            self._load_library_folder(folder)
                            seen.add(folder.name)
                        except Exception as e:
                            logger.warning("Skipping %s: %s", folder, e)
            for root in self._settings.settings.paths.legacy_roots:
                try:
                    seen.update(v.id for v in self._scan_legacy_root(root.id, Path(root.path).expanduser(), suggest=False))
                except Exception as e:
                    logger.warning("Skipping legacy root %s: %s", root.path, e)
            rows = self._db.fetchall("SELECT id, location, model_path FROM voice_models")
            removed = [r["id"] for r in rows if r["id"] not in seen and (r["location"] != "temporary" or not Path(r["model_path"]).is_file())]
            for rid in removed:
                self._db.execute("DELETE FROM voice_models WHERE id = ?", (rid,))
        if removed:
            self._events.publish(ModelsChangedEvent(removed=removed))

    def _read_voice_json(self, folder: Path) -> dict[str, Any]:
        try:
            return json.loads((folder / VOICE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_voice_json(self, folder: Path, data: dict[str, Any]) -> None:
        tmp = folder / (VOICE_FILE + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(folder / VOICE_FILE)

    def _load_library_folder(self, folder: Path) -> VoiceModel:
        meta = self._read_voice_json(folder)
        indexes: dict[str, str] = {}
        for p in folder.glob("*.index"):
            key = "default" if p.name == "model.index" else index_key(p)
            indexes[key] = str(p)
        return self._upsert_from_pth(
            folder.name,
            folder / MODEL_FILE,
            location="library",
            name=meta.get("name") or folder.name,
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            speaker_names=meta.get("speakers"),
            indexes=indexes,
            created_at=meta.get("imported_at"),
            catalog_id=meta.get("catalog_id"),
        )

    def _upsert_from_pth(
        self,
        voice_id: str,
        pth: Path,
        *,
        location: str,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        speaker_names: list[dict] | None = None,
        indexes: dict[str, str] | None = None,
        hidden: bool = False,
        created_at: str | None = None,
        catalog_id: str | None = None,
    ) -> VoiceModel:
        st = pth.stat()
        row = self._db.fetchone("SELECT * FROM voice_models WHERE id = ?", (voice_id,))
        if row is not None and row["mtime_ns"] == st.st_mtime_ns and row["size"] == st.st_size and row["model_path"] == str(pth):
            fields = {
                "name": name,
                "description": description,
                "tags": json.dumps(tags or []),
                "indexes": json.dumps(indexes or {}),
                "hidden": int(hidden),
                "catalog_id": catalog_id,
            }
            if speaker_names is not None:
                fields["speakers"] = json.dumps(speaker_names)
            changed = {k: v for k, v in fields.items() if row[k] != v}
            if changed:
                self._db.execute(f"UPDATE voice_models SET {', '.join(f'{k} = ?' for k in changed)}, updated_at = ? WHERE id = ?", (*changed.values(), now_iso(), voice_id))
            return self.get(voice_id)
        summary = self._summarize(pth)
        if summary.kind != "small":
            raise ModelFormatError(f"{pth.name} is a training checkpoint, not a voice; extract it first", {"checkpoint": str(pth)})
        speakers = speaker_names if speaker_names is not None else [s.model_dump() for s in summary.speakers]
        now = now_iso()
        self._db.execute(
            """INSERT INTO voice_models(id, name, description, tags, location, model_path, sample_rate, version, pitch_guidance, speakers, speaker_slots, indexes, size, mtime_ns, info, hidden, catalog_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, description=excluded.description, tags=excluded.tags, location=excluded.location, model_path=excluded.model_path,
               sample_rate=excluded.sample_rate, version=excluded.version, pitch_guidance=excluded.pitch_guidance, speakers=excluded.speakers, speaker_slots=excluded.speaker_slots,
               indexes=excluded.indexes, size=excluded.size, mtime_ns=excluded.mtime_ns, info=excluded.info, hidden=excluded.hidden, catalog_id=excluded.catalog_id,
               updated_at=excluded.updated_at""",
            (
                voice_id,
                name,
                description,
                json.dumps(tags or []),
                location,
                str(pth),
                summary.sample_rate or 0,
                summary.version or "v1",
                int(bool(summary.pitch_guidance)),
                json.dumps(speakers),
                summary.speaker_slots,
                json.dumps(indexes or {}),
                st.st_size,
                st.st_mtime_ns,
                summary.info,
                int(hidden),
                catalog_id,
                created_at or now,
                now,
            ),
        )
        return self.get(voice_id)

    def _summarize(self, pth: Path) -> CheckpointInfo:
        from rvc_next.engine.models.checkpoint import summarize

        try:
            s = summarize(pth)
        except Exception as e:
            mapped = map_engine_error(e)
            raise (mapped or ModelFormatError(str(e))) from e
        return CheckpointInfo(
            kind=s.kind,  # ty: ignore[invalid-argument-type]
            sample_rate=s.sample_rate,
            version=s.version,  # ty: ignore[invalid-argument-type]
            pitch_guidance=s.pitch_guidance,
            speakers=[Speaker(id=i["id"], name=i["name"]) for i in s.speaker_info],
            speaker_slots=s.speaker_slots,
            info=s.info,
            iteration=s.iteration,
        )

    def inspect(self, path: str) -> CheckpointInfo:
        p = Path(path).expanduser()
        if not p.is_file():
            raise NotFoundError(f"No such file: {p}")
        return self._summarize(p)

    # -- legacy roots ------------------------------------------------------------------

    def _legacy_state(self) -> dict[str, Any]:
        try:
            return json.loads((self._settings.data_dir / LEGACY_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_legacy_state(self, data: dict[str, Any]) -> None:
        path = self._settings.data_dir / LEGACY_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _scan_legacy_root(self, root_id: str, root: Path, suggest: bool) -> list[VoiceModel]:
        state = self._legacy_state()
        voices: list[VoiceModel] = []
        for pth in legacy_weights(root):
            voice_id = f"legacy-{root_id}-{slugify(pth.stem)}"
            entry = state.setdefault(voice_id, {})
            # A scan suggests indexes again only where they are still suggestions: an index the user
            # attached or detached (``suggested`` false) is their choice and is kept.
            if "indexes" not in entry or (suggest and entry.get("suggested", True)):
                speakers = self._summarize(pth).speakers
                entry["indexes"] = {}
                if speakers:
                    for spk in speakers:
                        p = suggest_index(pth, legacy_index_roots(root), spk.id)
                        if p is not None:
                            entry["indexes"][f"spk{spk.id}"] = str(p)
                else:
                    p = suggest_index(pth, legacy_index_roots(root), None)
                    if p is not None:
                        entry["indexes"]["default"] = str(p)
                entry["suggested"] = True
            try:
                voices.append(
                    self._upsert_from_pth(
                        voice_id,
                        pth,
                        location=f"legacy:{root_id}",
                        name=entry.get("name") or pth.stem,
                        description=entry.get("description", ""),
                        tags=entry.get("tags", []),
                        speaker_names=entry.get("speakers"),
                        indexes=entry.get("indexes", {}),
                        hidden=bool(entry.get("hidden")),
                    )
                )
            except ModelFormatError:
                continue
        self._save_legacy_state(state)
        return voices

    def scan_legacy_root(self, root_id: str) -> LegacyScanResult:
        root = next((r for r in self._settings.settings.paths.legacy_roots if r.id == root_id), None)
        if root is None:
            raise NotFoundError(f"Unknown legacy root: {root_id}")
        path = Path(root.path).expanduser()
        if not path.is_dir():
            raise NotFoundError(f"Legacy root not found: {path}")
        with self._lock:
            voices = self._scan_legacy_root(root_id, path, suggest=True)
        self._events.publish(ModelsChangedEvent(updated=[v.id for v in voices]))
        suggestions = [IndexSuggestion(voice_id=v.id, key=k, path=p) for v in voices for k, p in (v.indexes.items() or [("default", None)])]
        return LegacyScanResult(root_id=root_id, voices=voices, suggestions=suggestions)

    def add_legacy_root(self, path: str, name: str = "") -> str:
        p = Path(path).expanduser().resolve()
        if not (p / "assets").is_dir():
            raise ValidationError(f"{p} does not look like an RVC install (no assets folder)")
        root_id = slugify(p.name, "rvc")
        existing = {r.id for r in self._settings.settings.paths.legacy_roots}
        base, n = root_id, 1
        while root_id in existing:
            n += 1
            root_id = f"{base}-{n}"

        def add(data: dict[str, Any]) -> None:
            data["paths"]["legacy_roots"].append({"id": root_id, "path": str(p), "name": name or p.name})

        self._settings.mutate(add)
        self.scan_legacy_root(root_id)
        return root_id

    def remove_legacy_root(self, root_id: str) -> None:
        """Forget a legacy root; its files are not touched."""
        if not any(r.id == root_id for r in self._settings.settings.paths.legacy_roots):
            raise NotFoundError(f"Unknown legacy root: {root_id}")

        def drop(data: dict[str, Any]) -> None:
            data["paths"]["legacy_roots"] = [r for r in data["paths"]["legacy_roots"] if r["id"] != root_id]

        self._settings.mutate(drop)
        rows = self._db.fetchall("SELECT id FROM voice_models WHERE location = ?", (f"legacy:{root_id}",))
        self._db.execute("DELETE FROM voice_models WHERE location = ?", (f"legacy:{root_id}",))
        state = self._legacy_state()
        for r in rows:
            state.pop(r["id"], None)
        self._save_legacy_state(state)
        self._events.publish(ModelsChangedEvent(removed=[r["id"] for r in rows]))

    # -- import ------------------------------------------------------------------------

    def import_paths(self, paths: Iterable[Path], name: str | None = None, index: Path | None = None) -> ImportResult:
        """Import in one step through an import session; ambiguous input raises ``ConflictError`` with the plan."""
        imports = self._imports()
        paths = list(paths)
        return imports.import_now(lambda sid: imports.add_paths(sid, paths), name=name, pair_index=index)

    def import_upload(self, filename: str, chunks: Iterable[bytes], name: str | None = None) -> ImportResult:
        """A raw upload (``.pth``, ``.index``, ``.zip``, …), imported in one step like ``import_paths``."""
        imports = self._imports()
        return imports.import_now(lambda sid: imports.add_upload(sid, filename, chunks), name=name)

    def _imports(self) -> Any:
        if self.imports is None:
            raise RvcNextError("Imports are not available")
        return self.imports

    def _unique_slug(self, name: str) -> str:
        base = slugify(name)
        slug, n = base, 1
        while (self.root / slug).exists() or self._db.fetchone("SELECT 1 FROM voice_models WHERE id = ?", (slug,)):
            n += 1
            slug = f"{base}-{n}"
        return slug

    def _add_to_library(
        self, pth: Path, indexes: dict[str, Path], name: str | None, summary: CheckpointInfo, description: str = "", source: dict[str, Any] | None = None
    ) -> VoiceModel:
        display = name or pth.stem
        with self._lock:
            slug = self._unique_slug(display)
            folder = self.root / slug
            tmp = self.root / f".{slug}.part"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            try:
                shutil.copy2(pth, tmp / MODEL_FILE)
                for key, idx in indexes.items():
                    shutil.copy2(idx, tmp / _index_filename(key))
                self._write_voice_json(
                    tmp,
                    {
                        "name": display,
                        "description": description,
                        "tags": [],
                        "speakers": [s.model_dump() for s in summary.speakers] or None,
                        "source_files": [pth.name, *(i.name for i in indexes.values())],
                        "index_sources": {key: idx.name for key, idx in indexes.items()},
                        "imported_at": now_iso(),
                        **(source or {}),
                    },
                )
                tmp.rename(folder)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            voice = self._load_library_folder(folder)
        if self.presets is not None:
            self.presets.ensure_default(voice.id)
        return voice

    def add_model_file(
        self, pth: Path, name: str, indexes: dict[str, Path] | None = None, description: str = "", source: dict[str, Any] | None = None, publish: bool = True
    ) -> VoiceModel:
        """Add a small model to the library (import, merge, extract, training export, downloads)."""
        summary = self._summarize(pth)
        if summary.kind != "small":
            raise ModelFormatError(f"{pth.name} is a training checkpoint, not a voice; extract it first")
        voice = self._add_to_library(pth, indexes or {}, name=name, summary=summary, description=description, source=source)
        if publish:
            self._events.publish(ModelsChangedEvent(added=[voice.id]))
        return voice

    def import_legacy_root(self, root_id: str) -> ImportResult:
        """Copy every voice of a legacy root into the library (Import all)."""
        voices = [v for v in self.list_voices(include_hidden=True) if v.location == f"legacy:{root_id}"]
        result = ImportResult()
        for v in voices:
            summary = self._summarize(Path(v.model_path))
            idx = {k: Path(p) for k, p in v.indexes.items() if Path(p).is_file()}
            result.voices.append(self._add_to_library(Path(v.model_path), idx, name=v.name, summary=summary, description=v.description))
        self._events.publish(ModelsChangedEvent(added=[v.id for v in result.voices]))
        return result

    # -- editing -------------------------------------------------------------------------

    def update(self, voice_id: str, patch: VoiceUpdate) -> VoiceModel:
        voice = self.get(voice_id)
        if voice.location == "temporary":
            raise ConflictError("A temporary voice cannot be edited; import it first")
        speakers = None
        if patch.speakers is not None:
            for s in patch.speakers:
                if not 0 <= s.id < max(voice.speaker_slots, 1):
                    raise ValidationError(f"Speaker id {s.id} is outside 0–{voice.speaker_slots - 1}")
            speakers = [s.model_dump() for s in sorted(patch.speakers, key=lambda s: s.id) if s.name.strip()]
        if voice.legacy:
            state = self._legacy_state()
            entry = state.setdefault(voice_id, {})
            for key in ("name", "description", "tags", "hidden"):
                value = getattr(patch, key)
                if value is not None:
                    entry[key] = value
            if speakers is not None:
                entry["speakers"] = speakers
            self._save_legacy_state(state)
            pth = Path(voice.model_path)
            self._upsert_from_pth(
                voice_id,
                pth,
                location=voice.location,
                name=entry.get("name") or pth.stem,
                description=entry.get("description", ""),
                tags=entry.get("tags", []),
                speaker_names=entry.get("speakers"),
                indexes=entry.get("indexes", {}),
                hidden=bool(entry.get("hidden")),
            )
        else:
            folder = Path(voice.model_path).parent
            meta = self._read_voice_json(folder)
            for key in ("name", "description", "tags"):
                value = getattr(patch, key)
                if value is not None:
                    meta[key] = value
            if speakers is not None:
                meta["speakers"] = speakers
            if patch.hidden is not None:
                raise ValidationError("Library voices are deleted, not hidden")
            self._write_voice_json(folder, meta)
            self._load_library_folder(folder)
        self._events.publish(ModelsChangedEvent(updated=[voice_id]))
        return self.get(voice_id)

    def delete(self, voice_id: str) -> None:
        voice = self.get(voice_id)
        if voice.legacy:
            self.update(voice_id, VoiceUpdate(hidden=True))
            return
        if voice.location == "library":
            trash(Path(voice.model_path).parent, self._settings.data_dir / "trash")
        self._db.execute("DELETE FROM voice_models WHERE id = ?", (voice_id,))
        self._events.publish(ModelsChangedEvent(removed=[voice_id]))

    def set_index(self, voice_id: str, key: str, source: Path | None, original_name: str | None = None, persist: bool | None = None) -> VoiceModel:
        """Attach or detach an index; any file name works, but the index must fit the voice.

        A library voice gets a copy in its folder. A legacy or temporary voice stores a path: to the
        file itself for a server path, or, with ``persist`` (uploads, the inbox, import sessions), to a
        copy kept in ``models/attached-indexes/<voice>/``.
        """
        from rvc_next.core.models.inbox import check_index_fits

        voice = self.get(voice_id)
        if key != "default" and not (key.startswith("spk") and key[3:].isdigit()):
            raise ValidationError("The index key is 'default' or 'spk<N>'")
        if source is not None and not source.is_file():
            raise NotFoundError(f"No such file: {source}")
        if source is not None:
            check_index_fits(source, voice)
            if persist is None:
                persist = original_name is not None
            if persist and voice.location != "library":
                keep = self._settings.models_dir / "attached-indexes" / slugify(voice_id) / _index_filename(key)
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, keep.with_name(keep.name + ".part"))
                keep.with_name(keep.name + ".part").replace(keep)
                source = keep
        if voice.location == "library" and source is not None:
            folder = Path(voice.model_path).parent
            meta = self._read_voice_json(folder)
            meta.setdefault("index_sources", {})[key] = original_name or source.name
            self._write_voice_json(folder, meta)
        if voice.legacy:
            state = self._legacy_state()
            entry = state.setdefault(voice_id, {})
            idx = dict(entry.get("indexes", {}))
            if source is None:
                idx.pop(key, None)
            else:
                idx[key] = str(source.resolve())
            entry["indexes"] = idx
            entry["suggested"] = False
            self._save_legacy_state(state)
            self._db.execute("UPDATE voice_models SET indexes = ?, updated_at = ? WHERE id = ?", (json.dumps(idx), now_iso(), voice_id))
        elif voice.location == "library":
            folder = Path(voice.model_path).parent
            target = folder / _index_filename(key)
            if source is None:
                trash(target, self._settings.data_dir / "trash")
            else:
                tmp = target.with_name(target.name + ".part")
                shutil.copy2(source, tmp)
                tmp.replace(target)
            self._load_library_folder(folder)
        else:
            idx = dict(voice.indexes)
            if source is None:
                idx.pop(key, None)
            else:
                idx[key] = str(source.resolve())
            self._db.execute("UPDATE voice_models SET indexes = ? WHERE id = ?", (json.dumps(idx), voice_id))
        self._events.publish(ModelsChangedEvent(updated=[voice_id]))
        return self.get(voice_id)

    def set_index_upload(self, voice_id: str, key: str, chunks: Iterable[bytes], filename: str | None = None) -> VoiceModel:
        staging = self._settings.data_dir / "uploads" / new_id("index")
        staging.mkdir(parents=True, exist_ok=True)
        try:
            target = staging / "upload.index"
            with open(target, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
            return self.set_index(voice_id, key, target, original_name=filename or "upload.index", persist=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def model_folder(self, voice_id: str) -> Path:
        voice = self.get(voice_id)
        return Path(voice.model_path).parent

    # -- merge and extract jobs -----------------------------------------------------------

    def merge(self, request: MergeRequest, foreground: bool = False) -> Any:
        validate_name(request.name)
        return (self._jobs.run_now if foreground else self._jobs.submit)("merge", request.model_dump())

    def extract(self, request: ExtractRequest, foreground: bool = False) -> Any:
        validate_name(request.name)
        return (self._jobs.run_now if foreground else self._jobs.submit)("extract", request.model_dump())

    def _merge_spec(self, raw: dict[str, Any]) -> JobSpec:
        req = MergeRequest.model_validate(raw)

        def run(ctx: JobContext) -> dict[str, Any]:
            from rvc_next.engine.models.merge import merge_models

            a, b = self.get(req.a), self.get(req.b)
            out_dir = self._settings.data_dir / "uploads" / new_id("merge")
            out = out_dir / f"{slugify(req.name)}.pth"
            try:
                ctx.log(f"Merging {a.name} ({req.alpha:g}) with {b.name} ({1 - req.alpha:g})")
                merge_models(a.model_path, b.model_path, out, alpha=req.alpha, info=req.info)
                ctx.progress(0.8)
                voice = self.add_model_file(out, req.name, description=req.info, source={"merged_from": [a.id, b.id], "alpha": req.alpha})
            finally:
                shutil.rmtree(out_dir, ignore_errors=True)
            return {"voice_id": voice.id}

        return JobSpec(title=f"Merge into {req.name}", run=run, resources=frozenset({"cpu"}))

    def _extract_spec(self, raw: dict[str, Any]) -> JobSpec:
        req = ExtractRequest.model_validate(raw)

        def run(ctx: JobContext) -> dict[str, Any]:
            from rvc_next.engine.models.extract import extract_small_model

            src = self._checkpoint_path(req.checkpoint)
            summary = self._summarize(src)
            if summary.kind != "checkpoint":
                raise ModelFormatError(f"{src.name} is already a small model")
            sr, version, f0, speaker_info = self._checkpoint_context(src)
            sample_rate = req.sample_rate or sr
            if sample_rate is None:
                raise ValidationError("The sample rate could not be read from the experiment; choose it")
            out_dir = self._settings.data_dir / "uploads" / new_id("extract")
            out = out_dir / f"{slugify(req.name)}.pth"
            try:
                extract_small_model(
                    src,
                    out,
                    sample_rate=sample_rate,
                    pitch_guidance=req.pitch_guidance if req.pitch_guidance is not None else (f0 if f0 is not None else bool(summary.pitch_guidance)),
                    version=req.version or version or summary.version or "v2",
                    info=req.info,
                    speaker_info=speaker_info,
                )
                voice = self.add_model_file(out, req.name, description=req.info, source={"extracted_from": str(src)})
            finally:
                shutil.rmtree(out_dir, ignore_errors=True)
            return {"voice_id": voice.id}

        return JobSpec(title=f"Extract {req.name}", run=run, resources=frozenset({"cpu"}))

    def _checkpoint_path(self, ref: str) -> Path:
        p = Path(ref).expanduser()
        if p.is_absolute():
            if not p.is_file():
                raise NotFoundError(f"No such file: {p}")
            return p
        exp, _, file = ref.replace("\\", "/").partition("/")
        if not file:
            raise InvalidPathError("Give a checkpoint path or <experiment>/<file>")
        p = self._settings.experiments_dir / validate_name(exp) / validate_name(file)
        if not p.is_file():
            raise NotFoundError(f"No such checkpoint: {ref}")
        return p

    @staticmethod
    def _checkpoint_context(src: Path) -> tuple[str | None, str | None, bool | None, list[dict] | None]:
        """Sample rate, version, pitch guidance and speakers from the checkpoint's experiment folder."""
        sr = version = None
        f0 = None
        speakers = None
        exp_json = src.parent / "experiment.json"
        cfg = src.parent / "config.json"
        try:
            if exp_json.is_file():
                data = json.loads(exp_json.read_text(encoding="utf-8"))
                sr, version, f0 = data.get("sample_rate"), data.get("version"), data.get("pitch_guidance")
            if cfg.is_file():
                data = json.loads(cfg.read_text(encoding="utf-8"))
                speakers = data.get("speaker_info")
                rate = data.get("data", {}).get("sampling_rate")
                if sr is None and rate:
                    sr = {32000: "32k", 40000: "40k", 48000: "48k"}.get(int(rate))
        except (OSError, ValueError):
            pass
        return sr, version, f0, speakers
