"""Training base models: the official ones from the catalog, and imported G/D pairs.

An imported base model is ``models/base/<slug>/G.pth`` (and ``D.pth`` when given) with
``base.json``. Its sample rate, version and pitch guidance come from the G's weights, so Train can
offer only the base models that fit an experiment.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from rvc_next.core.clock import now_iso
from rvc_next.core.errors import ConflictError, NotFoundError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import ModelsChangedEvent
from rvc_next.core.files import slugify, trash
from rvc_next.core.record import Record
from rvc_next.core.settings import SettingsService

if TYPE_CHECKING:
    from rvc_next.core.assets.service import AssetService
    from rvc_next.core.models.models import StagedFile

SIDECAR = "base.json"
RATES = ("32k", "40k", "48k")


class BaseModel(Record):
    id: str
    """``official-<v>-<rate>[-f0]`` or the slug of an imported one."""
    name: str
    source: Literal["official", "imported"]
    sample_rate: Literal["32k", "40k", "48k"]
    version: Literal["v1", "v2"]
    pitch_guidance: bool
    has_discriminator: bool
    installed: bool
    """False for an official base model whose asset is not downloaded."""
    asset_id: str | None = None
    size: int = 0
    source_files: list[str] = Field(default_factory=list)
    imported_at: str | None = None


class BaseModelUpdate(Record):
    name: str


def official_id(version: str, sample_rate: str, pitch_guidance: bool) -> str:
    return f"official-{version}-{sample_rate}{'-f0' if pitch_guidance else ''}"


class BaseModelService:
    def __init__(self, settings: SettingsService, events: EventBus, assets: AssetService) -> None:
        self._settings = settings
        self._events = events
        self._assets = assets

    @property
    def root(self) -> Path:
        return self._settings.models_dir / "base"

    # -- listing ----------------------------------------------------------------------

    def _official(self) -> list[BaseModel]:
        out = []
        for version in ("v1", "v2"):
            for rate in RATES:
                for f0 in (True, False):
                    g, d = self._official_paths(version, rate, f0)
                    installed = g.is_file()
                    out.append(
                        BaseModel(
                            id=official_id(version, rate, f0),
                            name=f"Official {version} {rate}{', pitch' if f0 else ''}",
                            source="official",
                            sample_rate=rate,
                            version=version,
                            pitch_guidance=f0,
                            has_discriminator=d.is_file() or not installed,
                            installed=installed,
                            asset_id=f"pretrained-{version}-{rate}",
                            size=(g.stat().st_size if g.is_file() else 0) + (d.stat().st_size if d.is_file() else 0),
                        )
                    )
        return out

    def _official_paths(self, version: str, rate: str, f0: bool) -> tuple[Path, Path]:
        folder = self._assets.root / ("pretrained_v2" if version == "v2" else "pretrained")
        prefix = "f0" if f0 else ""
        return folder / f"{prefix}G{rate}.pth", folder / f"{prefix}D{rate}.pth"

    def _imported(self) -> list[BaseModel]:
        if not self.root.is_dir():
            return []
        out = []
        for folder in sorted(self.root.iterdir()):
            meta_path = folder / SIDECAR
            if not meta_path.is_file() or not (folder / "G.pth").is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            d = folder / "D.pth"
            out.append(
                BaseModel(
                    id=folder.name,
                    name=meta.get("name", folder.name),
                    source="imported",
                    sample_rate=meta["sample_rate"],
                    version=meta["version"],
                    pitch_guidance=bool(meta["pitch_guidance"]),
                    has_discriminator=d.is_file(),
                    installed=True,
                    size=(folder / "G.pth").stat().st_size + (d.stat().st_size if d.is_file() else 0),
                    source_files=meta.get("source_files", []),
                    imported_at=meta.get("imported_at"),
                )
            )
        return out

    def list_base(self, sample_rate: str | None = None, version: str | None = None, pitch_guidance: bool | None = None) -> list[BaseModel]:
        """Official and imported base models, filtered to those that fit when filters are given."""
        items = self._official() + self._imported()
        return [
            b
            for b in items
            if (sample_rate is None or b.sample_rate == sample_rate)
            and (version is None or b.version == version)
            and (pitch_guidance is None or b.pitch_guidance == pitch_guidance)
        ]

    def get(self, base_id: str) -> BaseModel:
        for b in self.list_base():
            if b.id == base_id:
                return b
        raise NotFoundError(f"No base model {base_id!r}")

    def paths(self, base_id: str) -> tuple[Path, Path | None]:
        """The G and D files of a base model (D None when it has none); requires the official asset."""
        b = self.get(base_id)
        if b.source == "official":
            self._assets.require([b.asset_id or ""], "Training with this base model")
            g, d = self._official_paths(b.version, b.sample_rate, b.pitch_guidance)
            return g, d if d.is_file() else None
        folder = self.root / base_id
        d = folder / "D.pth"
        return folder / "G.pth", d if d.is_file() else None

    def check_fits(self, base_id: str, sample_rate: str, version: str, pitch_guidance: bool) -> BaseModel:
        b = self.get(base_id)
        if (b.sample_rate, b.version, b.pitch_guidance) != (sample_rate, version, pitch_guidance):
            want = f"{version} {sample_rate}{' with pitch' if pitch_guidance else ' without pitch'}"
            have = f"{b.version} {b.sample_rate}{' with pitch' if b.pitch_guidance else ' without pitch'}"
            raise ValidationError(f"{b.name} is a {have} base model; this experiment is {want}", {"reason": "base_model_mismatch"})
        return b

    # -- import and editing -------------------------------------------------------------------

    def add(self, g_path: Path, d_path: Path | None, name: str, g_file: StagedFile, d_name: str | None = None) -> BaseModel:
        if not g_file.sample_rate or not g_file.version or g_file.pitch_guidance is None:
            raise ValidationError(f"The sample rate or version of {g_file.name} could not be read from its weights")
        base = slugify(name or Path(g_file.name).stem, "base")
        slug, n = base, 1
        while (self.root / slug).exists() or slug.startswith("official-"):
            n += 1
            slug = f"{base}-{n}"
        tmp = self.root / f".{slug}.part"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            shutil.copy2(g_path, tmp / "G.pth")
            if d_path is not None:
                shutil.copy2(d_path, tmp / "D.pth")
            meta = {
                "name": name or Path(g_file.name).stem,
                "sample_rate": g_file.sample_rate,
                "version": g_file.version,
                "pitch_guidance": g_file.pitch_guidance,
                "source_files": [g_file.name, *([d_name] if d_name else [])],
                "imported_at": now_iso(),
            }
            (tmp / SIDECAR).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.rename(self.root / slug)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._events.publish(ModelsChangedEvent(added=[f"base:{slug}"]))
        return self.get(slug)

    def update(self, base_id: str, patch: BaseModelUpdate) -> BaseModel:
        b = self.get(base_id)
        if b.source != "imported":
            raise ConflictError("Official base models cannot be renamed")
        path = self.root / base_id / SIDECAR
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["name"] = patch.name.strip() or meta.get("name", base_id)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._events.publish(ModelsChangedEvent(updated=[f"base:{base_id}"]))
        return self.get(base_id)

    def delete(self, base_id: str) -> None:
        """An imported base model goes to the trash; an official one has its own G and D files
        removed (its asset then shows as partly downloaded, and can be downloaded again)."""
        b = self.get(base_id)
        if b.source == "official":
            g, d = self._official_paths(b.version, b.sample_rate, b.pitch_guidance)
            self._assets.delete_files([p.relative_to(self._assets.root).as_posix() for p in (g, d)])
            self._events.publish(ModelsChangedEvent(removed=[f"base:{base_id}"]))
            return
        trash(self.root / base_id, self._settings.data_dir / "trash")
        self._events.publish(ModelsChangedEvent(removed=[f"base:{base_id}"]))
