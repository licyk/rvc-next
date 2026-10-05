"""Indexes waiting for a voice: ``models/indexes/<id>/model.index`` with ``index.json``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from rvc_next.core.clock import new_id, now_iso
from rvc_next.core.errors import NotFoundError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import ModelsChangedEvent
from rvc_next.core.files import trash
from rvc_next.core.models.models import AssignIndexRequest, InboxIndex, PairingCandidate, VoiceModel
from rvc_next.core.models.pairing import IndexCandidate, VoiceCandidate, pair_indexes
from rvc_next.core.settings import SettingsService

if TYPE_CHECKING:
    from rvc_next.core.models.service import VoiceModelService

SIDECAR = "index.json"
FILE = "model.index"


def library_candidates(voices: list[VoiceModel]) -> list[VoiceCandidate]:
    return [VoiceCandidate(f"voice:{v.id}", v.name, v.version, tuple(s.id for s in v.speakers)) for v in voices if v.location != "temporary"]


class IndexInbox:
    def __init__(self, settings: SettingsService, events: EventBus, models: VoiceModelService) -> None:
        self._settings = settings
        self._events = events
        self._models = models

    @property
    def root(self) -> Path:
        return self._settings.models_dir / "indexes"

    def add(self, source: Path, name: str, dim: int, vectors: int) -> str:
        """Move ``source`` into the inbox; return its id."""
        index_id = new_id("i")
        folder = self.root / index_id
        folder.mkdir(parents=True)
        shutil.move(str(source), folder / FILE)
        (folder / SIDECAR).write_text(json.dumps({"name": name, "dim": dim, "vectors": vectors, "imported_at": now_iso()}, ensure_ascii=False, indent=2), encoding="utf-8")
        self._events.publish(ModelsChangedEvent(added=[f"index:{index_id}"]))
        return index_id

    def list_indexes(self) -> list[InboxIndex]:
        if not self.root.is_dir():
            return []
        items = [self._read(f) for f in sorted(self.root.iterdir()) if (f / SIDECAR).is_file() and (f / FILE).is_file()]
        voices = self._models.list_voices()
        names = {f"voice:{v.id}": v.name for v in voices}
        proposals = {p.index: p for p in pair_indexes([IndexCandidate(i.id, i.name, i.dim, i.vectors) for i in items], library_candidates(voices))}
        for item in items:
            p = proposals[item.id]
            item.candidates = [PairingCandidate(target=c.voice, name=names.get(c.voice, c.voice), score=c.score) for c in p.candidates]
            item.proposed = p.voice
        return items

    def _read(self, folder: Path) -> InboxIndex:
        from rvc_next.core.models.pairing import DIM_VERSION

        meta = json.loads((folder / SIDECAR).read_text(encoding="utf-8"))
        dim = int(meta.get("dim", 0))
        return InboxIndex(
            id=folder.name,
            name=meta.get("name", FILE),
            dim=dim,
            vectors=int(meta.get("vectors", 0)),
            version=DIM_VERSION.get(dim),  # ty: ignore[invalid-argument-type]
            size=(folder / FILE).stat().st_size,
            imported_at=meta.get("imported_at", ""),
        )

    def get(self, index_id: str) -> InboxIndex:
        folder = self.root / index_id
        if "/" in index_id or "\\" in index_id or not (folder / SIDECAR).is_file():
            raise NotFoundError(f"No unassigned index {index_id!r}")
        return self._read(folder)

    def assign(self, index_id: str, request: AssignIndexRequest) -> VoiceModel:
        item = self.get(index_id)
        folder = self.root / index_id
        voice = self._models.set_index(request.voice_id, request.key, folder / FILE, original_name=item.name)
        shutil.rmtree(folder, ignore_errors=True)
        self._events.publish(ModelsChangedEvent(removed=[f"index:{index_id}"]))
        return voice

    def delete(self, index_id: str) -> None:
        self.get(index_id)
        trash(self.root / index_id, self._settings.data_dir / "trash")
        self._events.publish(ModelsChangedEvent(removed=[f"index:{index_id}"]))


def check_index_fits(path: Path, voice: VoiceModel) -> None:
    """Refuse an index that cannot work with ``voice``: wrong width, or no vectors."""
    from rvc_next.engine.index.inspect import inspect_index

    try:
        info = inspect_index(path)
    except ValueError as e:
        raise ValidationError(str(e)) from e
    if info.empty:
        raise ValidationError("This index holds no features (a trained_ index); use the added_ one", {"reason": "empty_index"})
    if info.version != voice.version:
        found = info.version or f"{info.dim}-dimensional"
        raise ValidationError(
            f"This is a {found} index; {voice.name} is a {voice.version} voice", {"reason": "version_mismatch", "index_dim": info.dim, "voice_version": voice.version}
        )
