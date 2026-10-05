"""Build every service once. The API and the command line both start here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from rvc_next.core.assets.service import AssetService
from rvc_next.core.audio.service import AudioFileService
from rvc_next.core.compute.service import ComputeService
from rvc_next.core.conversion.service import ConversionService
from rvc_next.core.db import Database
from rvc_next.core.events import EventBus, LocalEventBus
from rvc_next.core.events.models import SettingsChangedEvent
from rvc_next.core.jobs.service import JobService
from rvc_next.core.models.base import BaseModelService
from rvc_next.core.models.imports import ImportService
from rvc_next.core.models.inbox import IndexInbox
from rvc_next.core.models.service import VoiceModelService
from rvc_next.core.net.http import HttpClientProvider
from rvc_next.core.presets.service import PresetService
from rvc_next.core.separation.library import SeparationLibrary
from rvc_next.core.separation.service import SeparationService
from rvc_next.core.settings import SettingsService

if TYPE_CHECKING:
    from rvc_next.core.live.service import LiveService
    from rvc_next.core.training.service import TrainingService

DB_FILE_NAME = "rvc-next.db"


@dataclass
class Services:
    settings: SettingsService
    events: EventBus
    db: Database
    http: HttpClientProvider
    compute: ComputeService
    jobs: JobService
    assets: AssetService
    audio: AudioFileService
    models: VoiceModelService
    inbox: IndexInbox
    imports: ImportService
    base_models: BaseModelService
    separation_models: SeparationLibrary
    presets: PresetService
    separation: SeparationService
    conversion: ConversionService
    training: TrainingService = field(default=None)  # ty: ignore[invalid-assignment]
    live: LiveService = field(default=None)  # ty: ignore[invalid-assignment]

    def close(self) -> None:
        if self.live is not None:
            self.live.close()
        self.jobs.close()
        self.compute.close()
        self.http.close()
        self.db.close()


def build_services(
    data_dir: Path | None = None,
    settings_path: Path | None = None,
    environ: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    settings_overrides: dict[str, Any] | None = None,
    start_background: bool = False,
    config_dir: Path | None = None,
) -> Services:
    """Create all services.

    ``start_background`` is false for the command line, which runs a job in the foreground, and
    true for the server, whose job scheduler, idle unloading and live supervisor then run.
    ``environ`` replaces ``os.environ`` and ``transport`` the network in tests.
    ``config_dir`` and ``settings_path`` put the settings file outside the data directory, and
    ``settings_overrides`` pin values; both come from a host application embedding rvc-next.
    """
    from rvc_next.core.live.service import LiveService
    from rvc_next.core.training.service import TrainingService

    settings = SettingsService(data_dir=data_dir, settings_path=settings_path, environ=environ, overrides=settings_overrides, config_dir=config_dir)
    events = LocalEventBus()
    db = Database(settings.data_dir / DB_FILE_NAME)
    http = HttpClientProvider(transport)
    compute = ComputeService(settings, events)
    jobs = JobService(settings, db, events, compute)
    assets = AssetService(settings, events, http, jobs)
    audio = AudioFileService(settings, db, events)
    models = VoiceModelService(settings, db, events, jobs)
    presets = PresetService(settings, db, events)
    models.presets = presets
    assets.models = models
    inbox = IndexInbox(settings, events, models)
    imports = ImportService(settings, events, models, inbox)
    models.imports = imports
    separation = SeparationService(settings, jobs, audio, assets, compute)
    conversion = ConversionService(settings, jobs, audio, models, assets, compute, separation)
    training = TrainingService(settings, db, events, jobs, models, assets, separation, compute)
    base_models = BaseModelService(settings, events, assets)
    separation_models = SeparationLibrary(settings, events)
    imports.base = base_models
    imports.separation = separation_models
    training.base = base_models
    separation.library = separation_models
    live = LiveService(settings, events, models, assets, compute, jobs)
    services = Services(
        settings=settings,
        events=events,
        db=db,
        http=http,
        compute=compute,
        jobs=jobs,
        assets=assets,
        audio=audio,
        models=models,
        inbox=inbox,
        imports=imports,
        base_models=base_models,
        separation_models=separation_models,
        presets=presets,
        separation=separation,
        conversion=conversion,
        training=training,
        live=live,
    )
    models.rescan()

    def on_settings_keys(keys: list[str]) -> None:
        # Every change (the API, the command line, or settings.toml edited while the server runs)
        # tells the clients, and new folders bring the library up to date.
        events.publish(SettingsChangedEvent(keys=keys))
        if any(key.startswith("paths.") for key in keys):
            models.rescan()

    settings.on_keys_change(on_settings_keys)
    if start_background:
        audio.cleanup()
        compute.start()
        jobs.start()
        live.start_background()
    return services
