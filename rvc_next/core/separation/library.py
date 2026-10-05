"""Imported separation models: ``models/separation/<slug>/`` with the checkpoint, its YAML and
``separation.json``. Each becomes a one-step preset beside the built-in ones."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field

from rvc_next.core.clock import now_iso
from rvc_next.core.errors import NotFoundError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import ModelsChangedEvent
from rvc_next.core.files import slugify, trash
from rvc_next.core.process import worker_command
from rvc_next.core.record import Record
from rvc_next.core.settings import SettingsService

if TYPE_CHECKING:
    from rvc_next.core.models.models import SeparationPlan, StagedFile

SIDECAR = "separation.json"
PREFIX = "user-"


class SeparationModel(Record):
    id: str
    """``user-<slug>``; also the id of its preset."""
    name: str
    model_type: str
    instruments: list[str]
    primary: str
    secondary: str
    primary_label: str
    secondary_label: str
    size: int
    source_files: list[str] = Field(default_factory=list)
    imported_at: str | None = None


class SeparationModelUpdate(Record):
    name: str | None = None
    primary_label: str | None = None
    secondary_label: str | None = None


def _check_labels(primary_label: str, secondary_label: str) -> None:
    for label in (primary_label, secondary_label):
        if not label or not label.replace("_", "").replace("-", "").isalnum():
            raise ValidationError(f"A stem label uses letters, digits, - and _: {label!r}")
    if primary_label == secondary_label:
        raise ValidationError("The two stems need different labels")


class SeparationLibrary:
    def __init__(self, settings: SettingsService, events: EventBus) -> None:
        self._settings = settings
        self._events = events

    @property
    def root(self) -> Path:
        return self._settings.models_dir / "separation"

    def _folder(self, model_id: str) -> Path:
        if not model_id.startswith(PREFIX) or "/" in model_id or "\\" in model_id:
            raise NotFoundError(f"No imported separation model {model_id!r}")
        folder = self.root / model_id[len(PREFIX) :]
        if not (folder / SIDECAR).is_file():
            raise NotFoundError(f"No imported separation model {model_id!r}")
        return folder

    def _read(self, folder: Path) -> SeparationModel:
        meta = json.loads((folder / SIDECAR).read_text(encoding="utf-8"))
        ckpt = folder / meta["checkpoint"]
        return SeparationModel(
            id=PREFIX + folder.name,
            name=meta["name"],
            model_type=meta["model_type"],
            instruments=meta["instruments"],
            primary=meta["primary"],
            secondary=meta["secondary"],
            primary_label=meta["primary_label"],
            secondary_label=meta["secondary_label"],
            size=ckpt.stat().st_size if ckpt.is_file() else 0,
            source_files=meta.get("source_files", []),
            imported_at=meta.get("imported_at"),
        )

    def list_models(self) -> list[SeparationModel]:
        if not self.root.is_dir():
            return []
        return [self._read(f) for f in sorted(self.root.iterdir()) if (f / SIDECAR).is_file()]

    def get(self, model_id: str) -> SeparationModel:
        return self._read(self._folder(model_id))

    def spec(self, model_id: str) -> dict[str, Any]:
        """The worker's model spec for an imported model."""
        from rvc_next.engine.separate.presets import custom_model, inspect_config

        folder = self._folder(model_id)
        meta = json.loads((folder / SIDECAR).read_text(encoding="utf-8"))
        cfg = inspect_config(folder / "config.yaml")
        return custom_model(
            model_id,
            meta["name"],
            meta["model_type"],
            folder / meta["checkpoint"],
            folder / "config.yaml",
            meta["primary"],
            meta["secondary"],
            meta["primary_label"],
            meta["secondary_label"],
            chunk_size=cfg.chunk_size,
            batch_size=cfg.batch_size,
        )

    def presets(self) -> list[dict[str, Any]]:
        """One single-step preset per imported model, shaped like the built-in presets."""
        return [
            {
                "id": m.id,
                "title": m.name,
                "description": f"Imported {m.model_type} model: {m.primary_label} and {m.secondary_label}.",
                "kind": "model",
                "steps": [m.id],
                "stems": [m.primary_label, m.secondary_label],
                "primary_stem": m.primary_label,
                "assets": [],
                "source": "imported",
            }
            for m in self.list_models()
        ]

    def add(self, ckpt: Path, config: Path, plan: SeparationPlan, config_file: StagedFile, ckpt_file: StagedFile) -> SeparationModel:
        if plan.primary not in config_file.instruments or plan.secondary not in config_file.instruments or plan.primary == plan.secondary:
            raise ValidationError(f"Choose two different stems of {', '.join(config_file.instruments)}")
        _check_labels(plan.primary_label, plan.secondary_label)
        base = slugify(plan.name or Path(ckpt_file.name).stem, "separation")
        slug, n = base, 1
        while (self.root / slug).exists():
            n += 1
            slug = f"{base}-{n}"
        tmp = self.root / f".{slug}.part"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            checkpoint = "model" + ckpt.suffix.lower()
            shutil.copy2(ckpt, tmp / checkpoint)
            shutil.copy2(config, tmp / "config.yaml")
            meta = {
                "name": plan.name or Path(ckpt_file.name).stem,
                "model_type": config_file.model_type,
                "instruments": config_file.instruments,
                "primary": plan.primary,
                "secondary": plan.secondary,
                "primary_label": plan.primary_label,
                "secondary_label": plan.secondary_label,
                "checkpoint": checkpoint,
                "source_files": [ckpt_file.name, config_file.name],
                "imported_at": now_iso(),
            }
            (tmp / SIDECAR).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            self._check_loads(tmp, meta)
            tmp.rename(self.root / slug)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._events.publish(ModelsChangedEvent(added=[PREFIX + slug]))
        return self.get(PREFIX + slug)

    def _check_loads(self, folder: Path, meta: dict[str, Any]) -> None:
        """Load the model once in the separation worker; a wrong checkpoint for the YAML fails here."""
        import subprocess
        import tempfile

        from rvc_next.engine.separate.presets import custom_model
        from rvc_next.protocol.jsonl import parse_line
        from rvc_next.protocol.messages import ErrorEvent, ResultEvent

        spec = custom_model("check", meta["name"], meta["model_type"], folder / meta["checkpoint"], folder / "config.yaml", meta["primary"], meta["secondary"], "a", "b")
        fd, name = tempfile.mkstemp(prefix="rvc-next-sepcheck-", suffix=".json")
        with open(fd, "w", encoding="utf-8") as f:
            json.dump({"action": "check", "model": spec}, f)
        try:
            out = subprocess.run(worker_command("rvc_next.workers.separate", name), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        finally:
            Path(name).unlink(missing_ok=True)
        ok = False
        message = ""
        for line in out.stdout.splitlines():
            event = parse_line(line)
            if isinstance(event, ResultEvent) and event.data.get("ok"):
                ok = True
            elif isinstance(event, ErrorEvent):
                message = event.message
        if not ok:
            detail = message or (out.stderr.strip().splitlines() or ["the model did not load"])[-1]
            raise ValidationError(f"The checkpoint does not load with this YAML: {detail}", {"reason": "separation_load"})

    def update(self, model_id: str, patch: SeparationModelUpdate) -> SeparationModel:
        folder = self._folder(model_id)
        meta = json.loads((folder / SIDECAR).read_text(encoding="utf-8"))
        if patch.name is not None and patch.name.strip():
            meta["name"] = patch.name.strip()
        primary_label = patch.primary_label or meta["primary_label"]
        secondary_label = patch.secondary_label or meta["secondary_label"]
        _check_labels(primary_label, secondary_label)
        meta["primary_label"], meta["secondary_label"] = primary_label, secondary_label
        (folder / SIDECAR).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._events.publish(ModelsChangedEvent(updated=[model_id]))
        return self.get(model_id)

    def delete(self, model_id: str) -> None:
        folder = self._folder(model_id)
        trash(folder, self._settings.data_dir / "trash")
        self._events.publish(ModelsChangedEvent(removed=[model_id]))
