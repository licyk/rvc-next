"""Presets. Each voice has one default preset, created on import from the global defaults."""

from __future__ import annotations

import json
from typing import Any

from rvc_next.core.clock import new_id, now_iso
from rvc_next.core.db import Database
from rvc_next.core.errors import NotFoundError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import PresetsChangedEvent
from rvc_next.core.params import StreamParamsModel, VoiceParamsModel
from rvc_next.core.presets.models import Preset, PresetCreate, PresetUpdate
from rvc_next.core.settings import SettingsService

DEFAULT_NAME = "Default"


class PresetService:
    def __init__(self, settings: SettingsService, db: Database, events: EventBus) -> None:
        self._settings = settings
        self._db = db
        self._events = events

    def list_presets(self, voice_id: str | None = None) -> list[Preset]:
        if voice_id:
            rows = self._db.fetchall("SELECT * FROM presets WHERE model_id = ? OR model_id IS NULL ORDER BY model_id IS NULL, is_default DESC, name COLLATE NOCASE", (voice_id,))
        else:
            rows = self._db.fetchall("SELECT * FROM presets ORDER BY model_id IS NULL, model_id, is_default DESC, name COLLATE NOCASE")
        return [self._row(r) for r in rows]

    def get(self, preset_id: str) -> Preset:
        row = self._db.fetchone("SELECT * FROM presets WHERE id = ?", (preset_id,))
        if row is None:
            raise NotFoundError(f"Unknown preset: {preset_id}")
        return self._row(row)

    def resolve(self, ref: str, voice_id: str | None = None) -> Preset:
        """A preset by id, or by name (the voice's own first, then a global one)."""
        row = self._db.fetchone("SELECT * FROM presets WHERE id = ?", (ref,))
        if row is None:
            row = self._db.fetchone(
                "SELECT * FROM presets WHERE name = ? COLLATE NOCASE AND (model_id = ? OR model_id IS NULL) ORDER BY model_id IS NULL LIMIT 1",
                (ref, voice_id),
            )
        if row is None:
            raise NotFoundError(f"No preset named {ref!r}")
        return self._row(row)

    def default_for(self, voice_id: str) -> Preset:
        row = self._db.fetchone("SELECT * FROM presets WHERE model_id = ? AND is_default = 1 LIMIT 1", (voice_id,))
        if row is None:
            return self.ensure_default(voice_id)
        return self._row(row)

    def ensure_default(self, voice_id: str) -> Preset:
        row = self._db.fetchone("SELECT * FROM presets WHERE model_id = ? AND is_default = 1 LIMIT 1", (voice_id,))
        if row is not None:
            return self._row(row)
        exists = self._db.fetchone("SELECT 1 FROM voice_models WHERE id = ?", (voice_id,))
        if exists is None:
            raise NotFoundError(f"Unknown voice: {voice_id}")
        return self.create(PresetCreate(name=DEFAULT_NAME, voice_id=voice_id, params=self._settings.settings.convert.default_params, is_default=True))

    def create(self, data: PresetCreate) -> Preset:
        name = data.name.strip()
        if not name:
            raise ValidationError("A preset needs a name")
        if data.is_default and not data.voice_id:
            raise ValidationError("Only a voice's own preset can be its default")
        preset_id = new_id("p")
        with self._db.transaction() as conn:
            if data.is_default:
                conn.execute("UPDATE presets SET is_default = 0 WHERE model_id = ?", (data.voice_id,))
            conn.execute(
                "INSERT INTO presets(id, name, model_id, is_default, voice, stream, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (preset_id, name, data.voice_id, int(data.is_default), data.params.model_dump_json(), data.stream.model_dump_json() if data.stream else None, now_iso()),
            )
        self._events.publish(PresetsChangedEvent(voice_id=data.voice_id))
        return self.get(preset_id)

    def update(self, preset_id: str, patch: PresetUpdate) -> Preset:
        current = self.get(preset_id)
        fields: dict[str, Any] = {}
        if patch.name is not None:
            if not patch.name.strip():
                raise ValidationError("A preset needs a name")
            fields["name"] = patch.name.strip()
        if patch.params is not None:
            fields["voice"] = patch.params.model_dump_json()
        if patch.stream is not None:
            fields["stream"] = patch.stream.model_dump_json()
        with self._db.transaction() as conn:
            if patch.is_default:
                if not current.voice_id:
                    raise ValidationError("Only a voice's own preset can be its default")
                conn.execute("UPDATE presets SET is_default = 0 WHERE model_id = ?", (current.voice_id,))
                fields["is_default"] = 1
            elif patch.is_default is False:
                fields["is_default"] = 0
            if fields:
                conn.execute(f"UPDATE presets SET {', '.join(f'{k} = ?' for k in fields)}, updated_at = ? WHERE id = ?", (*fields.values(), now_iso(), preset_id))
        self._events.publish(PresetsChangedEvent(voice_id=current.voice_id))
        return self.get(preset_id)

    def delete(self, preset_id: str) -> None:
        current = self.get(preset_id)
        self._db.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
        if current.is_default and current.voice_id:
            self.ensure_default(current.voice_id)
        self._events.publish(PresetsChangedEvent(voice_id=current.voice_id))

    @staticmethod
    def _row(r: Any) -> Preset:
        return Preset(
            id=r["id"],
            name=r["name"],
            voice_id=r["model_id"],
            is_default=bool(r["is_default"]),
            params=VoiceParamsModel.model_validate(json.loads(r["voice"])),
            stream=StreamParamsModel.model_validate(json.loads(r["stream"])) if r["stream"] else None,
            updated_at=r["updated_at"],
        )
