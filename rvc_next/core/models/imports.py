"""Import sessions: stage, inspect, propose, review, commit.

Files are staged in ``uploads/imports/<session>/``, identified by their content (not their
extension), and paired by ``pairing.py``. The plan is recomputed on demand, so it always reflects
the library as it is; the user's decisions are applied on commit. One-step imports (the command
line, ``PUT /models/import``) go through the same path and stop with the plan when it is ambiguous.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rvc_next.core.clock import new_id, now_iso
from rvc_next.core.engine_errors import map_engine_error
from rvc_next.core.errors import ConflictError, NotFoundError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import ModelsChangedEvent
from rvc_next.core.models.inbox import IndexInbox, library_candidates
from rvc_next.core.models.models import (
    AttachedIndex,
    GeneratorPlan,
    ImportDecisions,
    ImportPlan,
    ImportResult,
    IndexPlan,
    PairingCandidate,
    SeparationPlan,
    Speaker,
    StagedFile,
    VoicePlan,
)
from rvc_next.core.models.pairing import G_TO_D, IndexCandidate, Named, VoiceCandidate, pair_indexes, pair_partners
from rvc_next.core.safety import validate_name
from rvc_next.core.settings import SettingsService

if TYPE_CHECKING:
    from rvc_next.core.models.base import BaseModelService
    from rvc_next.core.models.service import VoiceModelService
    from rvc_next.core.separation.library import SeparationLibrary

logger = logging.getLogger(__name__)

SESSION_FILE = "session.json"
SESSION_TTL = 24 * 3600
MODEL_EXT = {".pth", ".pt"}
INDEX_EXT = {".index"}
CKPT_EXT = {".ckpt", ".bin", ".safetensors", ".chpt"}
YAML_EXT = {".yaml", ".yml"}
SUPPORTED = MODEL_EXT | INDEX_EXT | CKPT_EXT | YAML_EXT | {".zip"}
MAX_ZIP_MEMBER = 4 * 1024**3


def _stem(name: str) -> str:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


class ImportService:
    def __init__(self, settings: SettingsService, events: EventBus, models: VoiceModelService, inbox: IndexInbox) -> None:
        self._settings = settings
        self._events = events
        self._models = models
        self._inbox = inbox
        self.base: BaseModelService | None = None
        self.separation: SeparationLibrary | None = None

    @property
    def root(self) -> Path:
        return self._settings.data_dir / "uploads" / "imports"

    # -- sessions -----------------------------------------------------------------------

    def create(self) -> str:
        self.cleanup()
        session_id = new_id("s")
        folder = self.root / session_id
        (folder / "files").mkdir(parents=True)
        self._save(session_id, {"created_at": now_iso(), "files": []})
        return session_id

    def _folder(self, session_id: str) -> Path:
        folder = self.root / validate_name(session_id)
        if not (folder / SESSION_FILE).is_file():
            raise NotFoundError(f"No import session {session_id!r}")
        return folder

    def _load(self, session_id: str) -> dict[str, Any]:
        return json.loads((self._folder(session_id) / SESSION_FILE).read_text(encoding="utf-8"))

    def _save(self, session_id: str, data: dict[str, Any]) -> None:
        path = self.root / session_id / SESSION_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def delete(self, session_id: str) -> None:
        shutil.rmtree(self._folder(session_id), ignore_errors=True)

    def cleanup(self) -> None:
        if not self.root.is_dir():
            return
        now = time.time()
        for folder in self.root.iterdir():
            try:
                if now - folder.stat().st_mtime > SESSION_TTL:
                    shutil.rmtree(folder, ignore_errors=True)
            except OSError:
                continue

    def files(self, session_id: str) -> list[StagedFile]:
        return [StagedFile.model_validate({k: v for k, v in f.items() if k != "path"}) for f in self._load(session_id)["files"]]

    def _path(self, session_id: str, file_id: str) -> Path:
        for f in self._load(session_id)["files"]:
            if f["id"] == file_id:
                return Path(f["path"])
        raise NotFoundError(f"No file {file_id!r} in this import")

    # -- staging ---------------------------------------------------------------------------

    def add_upload(self, session_id: str, name: str, chunks: Iterable[bytes]) -> list[StagedFile]:
        """A raw upload; a ``.zip`` is unpacked and each usable member staged."""
        name = Path(name.replace("\\", "/")).name
        validate_name(name)
        folder = self._folder(session_id)
        tmp = folder / f".{new_id()}.part"
        with open(tmp, "wb") as f:
            for chunk in chunks:
                f.write(chunk)
        try:
            return self._stage(session_id, tmp, name, group=None, move=True)
        finally:
            tmp.unlink(missing_ok=True)

    def add_paths(self, session_id: str, paths: Iterable[Path]) -> list[StagedFile]:
        """Server files or folders. A folder adds its usable files; an original RVC install, its voices and indexes."""
        out: list[StagedFile] = []
        for p in paths:
            p = Path(p).expanduser()
            if not p.exists():
                raise NotFoundError(f"No such file: {p}")
            if p.is_dir():
                from rvc_next.core.models.importers import legacy_index_roots, legacy_weights

                if (p / "assets" / "weights").is_dir():
                    members = legacy_weights(p) + [x for r in legacy_index_roots(p) if r.is_dir() for x in r.rglob("added_*.index")]
                else:
                    members = sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in SUPPORTED)
                for m in members:
                    rel = m.relative_to(p).as_posix()
                    group = f"{p.name}/{Path(rel).parent.as_posix()}".rstrip("/.")
                    out += self._stage(session_id, m, rel, group=group, move=False)
            else:
                out += self._stage(session_id, p, p.name, group=p.parent.name or None, move=False)
        return out

    def _stage(self, session_id: str, source: Path, name: str, group: str | None, move: bool) -> list[StagedFile]:
        suffix = Path(name).suffix.lower()
        if suffix == ".zip":
            return self._stage_zip(session_id, source, name)
        file_id = new_id("f")
        target_dir = self.root / session_id / "files" / file_id
        target_dir.mkdir(parents=True)
        target = target_dir / Path(name).name
        if move:
            shutil.move(str(source), target)
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        staged = self._inspect(file_id, target, name, group)
        data = self._load(session_id)
        data["files"].append({**staged.model_dump(mode="json"), "path": str(target)})
        self._save(session_id, data)
        return [staged]

    def _stage_zip(self, session_id: str, archive: Path, name: str) -> list[StagedFile]:
        out: list[StagedFile] = []
        work = self.root / session_id / f".unzip-{new_id()}"
        try:
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    member = info.filename.replace("\\", "/")
                    base = member.rsplit("/", 1)[-1]
                    if info.is_dir() or not base or base.startswith(".") or "__MACOSX" in member or Path(base).suffix.lower() not in SUPPORTED - {".zip"}:
                        continue
                    if info.file_size > MAX_ZIP_MEMBER:
                        raise ValidationError(f"{base} in {name} is too large")
                    dest = work / new_id() / base
                    dest.parent.mkdir(parents=True)
                    with zf.open(info) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    folder = member.rsplit("/", 1)[0] if "/" in member else ""
                    out += self._stage(session_id, dest, member, group=f"{name}/{folder}".rstrip("/"), move=True)
        except zipfile.BadZipFile as e:
            raise ValidationError(f"{name} is not a zip archive") from e
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if not out:
            raise ValidationError(f"{name} holds no model, index or separation files")
        return out

    def _inspect(self, file_id: str, path: Path, name: str, group: str | None) -> StagedFile:
        base: dict[str, Any] = {"id": file_id, "name": name, "group": group, "size": path.stat().st_size}
        suffix = path.suffix.lower()
        try:
            if suffix in INDEX_EXT:
                from rvc_next.engine.index.inspect import inspect_index

                info = inspect_index(path)
                return StagedFile(**base, kind="index", dim=info.dim, vectors=info.ntotal, version=info.version)  # ty: ignore[invalid-argument-type]
            if suffix in YAML_EXT:
                from rvc_next.engine.separate.presets import inspect_config

                cfg = inspect_config(path)
                return StagedFile(**base, kind="separation_config", model_type=cfg.model_type, instruments=list(cfg.instruments), target_instrument=cfg.target_instrument)
            if suffix in CKPT_EXT:
                return StagedFile(**base, kind="separation_checkpoint")
            if suffix in MODEL_EXT:
                from rvc_next.engine.models.checkpoint import inspect_checkpoint

                c = inspect_checkpoint(path)
                if c.kind == "small" and c.summary is not None:
                    speakers = [Speaker(id=s["id"], name=s["name"]) for s in c.summary.speaker_info]
                    return StagedFile(**base, kind="voice", version=c.version, sample_rate=c.sample_rate, pitch_guidance=c.pitch_guidance, speakers=speakers)  # ty: ignore[invalid-argument-type]
                if c.kind == "G":
                    return StagedFile(**base, kind="generator", version=c.version, sample_rate=c.sample_rate, pitch_guidance=c.pitch_guidance)  # ty: ignore[invalid-argument-type]
                if c.kind == "D":
                    return StagedFile(**base, kind="discriminator", version=c.version)  # ty: ignore[invalid-argument-type]
                # Some separation models ship their weights as .pth; a YAML in the import can claim it.
                return StagedFile(**base, kind="separation_checkpoint", note=c.note)
        except Exception as e:
            mapped = map_engine_error(e)
            return StagedFile(**base, kind="unsupported", note=mapped.message if mapped else str(e))
        return StagedFile(**base, kind="unsupported", note="Not a model, index or separation file")

    # -- the plan ----------------------------------------------------------------------------

    def plan(self, session_id: str) -> ImportPlan:
        data = self._load(session_id)
        files = self.files(session_id)
        by_id = {f.id: f for f in files}
        voices = [f for f in files if f.kind == "voice"]
        library = self._models.list_voices()
        names = {f"file:{f.id}": _stem(f.name) for f in voices} | {f"voice:{v.id}": v.name for v in library}
        vcands = [VoiceCandidate(f"file:{f.id}", f.name, f.version or "v1", tuple(s.id for s in f.speakers), f.group) for f in voices] + library_candidates(library)
        icands = [IndexCandidate(f.id, f.name, f.dim or 0, f.vectors or 0, f.group) for f in files if f.kind == "index"]
        indexes = []
        for p in pair_indexes(icands, vcands):
            indexes.append(
                IndexPlan(
                    file_id=p.index,
                    target=p.voice,
                    key=p.key,
                    status=p.status,
                    confident=p.confident,
                    reasons=p.reasons,
                    candidates=[PairingCandidate(target=c.voice, name=names.get(c.voice, c.voice), score=c.score) for c in p.candidates],
                )
            )
        gens = [f for f in files if f.kind == "generator"]
        discs = [f for f in files if f.kind == "discriminator"]
        generators = []
        for p in pair_partners([Named(f.id, f.name, f.version, f.group) for f in gens], [Named(f.id, f.name, f.version, f.group) for f in discs], swap=G_TO_D, same_version=True):
            generators.append(
                GeneratorPlan(file_id=p.left, name=_stem(by_id[p.left].name), discriminator=f"file:{p.right}" if p.right else None, candidates=[f"file:{c}" for c in p.candidates])
            )
        configs = [f for f in files if f.kind == "separation_config"]
        ckpts = [f for f in files if f.kind == "separation_checkpoint"]
        separations = []
        for p in pair_partners([Named(f.id, f.name, None, f.group) for f in configs], [Named(f.id, f.name, None, f.group) for f in ckpts]):
            cfg = by_id[p.left]
            primary = cfg.target_instrument if cfg.target_instrument in cfg.instruments else cfg.instruments[0]
            secondary = next(i for i in cfg.instruments if i != primary)
            name = _stem(by_id[p.right].name) if p.right else _stem(cfg.name)
            separations.append(
                SeparationPlan(
                    config_id=p.left,
                    checkpoint_id=p.right,
                    name=name,
                    primary=primary,
                    secondary=secondary,
                    primary_label=primary,
                    secondary_label="instrumental" if secondary == "other" and primary == "vocals" else secondary,
                    candidates=list(p.candidates),
                )
            )
        plan = ImportPlan(
            session_id=session_id,
            files=files,
            voices=[VoicePlan(file_id=f.id, name=_stem(f.name)) for f in voices],
            indexes=indexes,
            generators=generators,
            separations=separations,
            created_at=data["created_at"],
        )
        plan.confident = self._confident(plan, discs)
        return plan

    @staticmethod
    def _confident(plan: ImportPlan, discs: list[StagedFile]) -> bool:
        actionable = plan.voices or plan.generators or plan.separations or any(i.status not in ("empty",) for i in plan.indexes)
        if not actionable:
            return False
        if any(i.status not in ("empty",) and not i.confident for i in plan.indexes):
            return False
        if any(s.checkpoint_id is None for s in plan.separations):
            return False
        paired = {g.discriminator for g in plan.generators if g.discriminator}
        return not any(f"file:{d.id}" not in paired for d in discs)

    @staticmethod
    def apply(plan: ImportPlan, decisions: ImportDecisions | None) -> ImportPlan:
        """The plan with the user's decisions in place of the proposals."""
        if decisions is None:
            return plan
        out = plan.model_copy(deep=True)
        if decisions.voices is not None:
            out.voices = decisions.voices
        if decisions.indexes is not None:
            given = {i.file_id: i for i in decisions.indexes}
            out.indexes = [given.get(i.file_id, i).model_copy(update={"status": "manual" if i.file_id in given else i.status}) for i in plan.indexes]
        if decisions.generators is not None:
            out.generators = decisions.generators
        if decisions.separations is not None:
            out.separations = decisions.separations
        return out

    # -- commit --------------------------------------------------------------------------------

    def commit(self, session_id: str, decisions: ImportDecisions | None = None) -> ImportResult:
        plan = self.apply(self.plan(session_id), decisions)
        by_id = {f.id: f for f in plan.files}
        path = {f.id: self._path(session_id, f.id) for f in plan.files}
        result = ImportResult()
        self._validate(plan, by_id)

        new_voices: dict[str, str] = {}
        for vp in plan.voices:
            if vp.action != "import":
                result.skipped.append(by_id[vp.file_id].name)
                continue
            mine = {i.key: path[i.file_id] for i in plan.indexes if i.target == f"file:{vp.file_id}"}
            name = vp.name.strip() or _stem(by_id[vp.file_id].name)
            voice = self._models.add_model_file(path[vp.file_id], name, indexes=mine, publish=False)
            new_voices[vp.file_id] = voice.id
            result.voices.append(voice)
            for i in plan.indexes:
                if i.target == f"file:{vp.file_id}":
                    result.attached.append(AttachedIndex(voice_id=voice.id, key=i.key, name=by_id[i.file_id].name))
        for i in plan.indexes:
            f = by_id[i.file_id]
            if i.target and i.target.startswith("voice:"):
                voice_id = i.target.split(":", 1)[1]
                self._models.set_index(voice_id, i.key, path[i.file_id], original_name=f.name)
                result.attached.append(AttachedIndex(voice_id=voice_id, key=i.key, name=f.name))
            elif i.target and i.target.startswith("file:"):
                if i.target[5:] not in new_voices:
                    result.inbox.append(self._inbox.add(path[i.file_id], f.name, f.dim or 0, f.vectors or 0))
            elif i.status == "empty" or (f.vectors or 0) == 0:
                result.skipped.append(f"{f.name}: holds no features (a trained_ index)")
            else:
                result.inbox.append(self._inbox.add(path[i.file_id], f.name, f.dim or 0, f.vectors or 0))
        for g in plan.generators:
            f = by_id[g.file_id]
            if g.action == "skip":
                result.skipped.append(f.name)
            elif g.action == "extract":
                result.voices.append(self._extract(path[g.file_id], f, g.name))
            else:
                if self.base is None:
                    raise ValidationError("Base models cannot be imported yet")
                d = path[g.discriminator[5:]] if g.discriminator else None
                result.base_models.append(self.base.add(path[g.file_id], d, g.name, f, by_id[g.discriminator[5:]].name if g.discriminator else None).id)
        for s in plan.separations:
            if s.action == "skip" or s.checkpoint_id is None:
                result.skipped.append(by_id[s.config_id].name + ("" if s.checkpoint_id else ": no checkpoint for this YAML"))
                continue
            if self.separation is None:
                raise ValidationError("Separation models cannot be imported yet")
            result.separation_models.append(self.separation.add(path[s.checkpoint_id], path[s.config_id], s, by_id[s.config_id], by_id[s.checkpoint_id]).id)
        used = {v.file_id for v in plan.voices} | {i.file_id for i in plan.indexes} | {g.file_id for g in plan.generators}
        used |= (
            {g.discriminator[5:] for g in plan.generators if g.discriminator}
            | {s.config_id for s in plan.separations}
            | {s.checkpoint_id for s in plan.separations if s.checkpoint_id}
        )
        for f in plan.files:
            if f.id not in used:
                result.skipped.append(f"{f.name}: {f.note or 'not used'}" if f.kind == "unsupported" or f.note else f"{f.name}: not used")
        self.delete(session_id)
        added = [v.id for v in result.voices]
        if added or result.attached or result.inbox or result.base_models or result.separation_models:
            self._events.publish(ModelsChangedEvent(added=added, updated=sorted({a.voice_id for a in result.attached} - set(added))))
        return result

    def _validate(self, plan: ImportPlan, by_id: dict[str, StagedFile]) -> None:
        voices = {f"file:{v.file_id}": by_id[v.file_id] for v in plan.voices if v.action == "import"}
        slots: set[tuple[str, str]] = set()
        for i in plan.indexes:
            if not i.target:
                continue
            f = by_id[i.file_id]
            if (f.vectors or 0) == 0:
                raise ValidationError(f"{f.name} holds no features and cannot be attached")
            if i.target.startswith("file:"):
                v = voices.get(i.target)
                if v is None:
                    continue
                if f.version != v.version:
                    raise ValidationError(f"{f.name} is a {f.version or str(f.dim) + '-dimensional'} index; {v.name} is a {v.version} voice", {"reason": "version_mismatch"})
            else:
                voice = self._models.get(i.target.split(":", 1)[1])
                if f.version != voice.version:
                    raise ValidationError(
                        f"{f.name} is a {f.version or str(f.dim) + '-dimensional'} index; {voice.name} is a {voice.version} voice", {"reason": "version_mismatch"}
                    )
            if i.key != "default" and not (i.key.startswith("spk") and i.key[3:].isdigit()):
                raise ValidationError("An index key is 'default' or 'spk<N>'")
            if (i.target, i.key) in slots:
                raise ValidationError(f"Two indexes are set for the same voice and speaker ({i.key})")
            slots.add((i.target, i.key))

    def _extract(self, g_path: Path, f: StagedFile, name: str) -> Any:
        from rvc_next.engine.models.extract import extract_small_model

        if not f.sample_rate or not f.version:
            raise ValidationError(f"The sample rate of {f.name} could not be read; extract it in Models › Tools")
        out = self.root / f".extract-{new_id()}" / "model.pth"
        try:
            extract_small_model(g_path, out, sample_rate=f.sample_rate, pitch_guidance=bool(f.pitch_guidance), version=f.version, info=f"Extracted from {Path(f.name).name}")
            return self._models.add_model_file(out, name or _stem(f.name), publish=False)
        finally:
            shutil.rmtree(out.parent, ignore_errors=True)

    # -- one step -------------------------------------------------------------------------------

    def import_now(self, add: Any, name: str | None = None, pair_index: Path | None = None, decisions: ImportDecisions | None = None) -> ImportResult:
        """Stage with ``add(session_id)``, then commit a confident plan; otherwise raise ``ConflictError`` with the plan."""
        session_id = self.create()
        try:
            add(session_id)
            if pair_index is not None:
                self.add_paths(session_id, [pair_index])
            plan = self.plan(session_id)
            if name and len(plan.voices) == 1:
                decisions = decisions or ImportDecisions()
                decisions.voices = [plan.voices[0].model_copy(update={"name": name})]
            if pair_index is not None and plan.voices:
                index_file = next(f for f in reversed(plan.files) if f.kind == "index")
                decisions = decisions or ImportDecisions()
                if len(plan.voices) != 1:
                    raise ValidationError("--index pairs with a single voice; use --pair for several")
                decisions.indexes = [IndexPlan(file_id=index_file.id, target=f"file:{plan.voices[0].file_id}", key="default")]
            if not (plan.voices or plan.indexes or plan.generators or plan.separations):
                raise ValidationError("Nothing to import: " + "; ".join(f"{f.name}: {f.note or f.kind}" for f in plan.files))
            if not plan.confident and decisions is None:
                raise ConflictError("The import needs a decision: some files could not be paired for sure", {"plan": plan.model_dump(mode="json")})
            return self.commit(session_id, decisions)
        finally:
            shutil.rmtree(self.root / session_id, ignore_errors=True)
