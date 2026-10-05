"""Event models. Each has an event name and is published on the EventBus.

Every event carries a snapshot, never a live object: the socket bridge serialises it later, on
the event loop, while workers keep writing.
"""

from typing import ClassVar

from pydantic import Field

from rvc_next.core.assets.models import AssetStatus
from rvc_next.core.audio.models import Output
from rvc_next.core.compute.models import ComputeUsage
from rvc_next.core.jobs.models import Job
from rvc_next.core.live.models import DeviceList, LiveState, LiveStats
from rvc_next.core.record import Record
from rvc_next.core.training.models import StageState


class EventBase(Record):
    __event_name__: ClassVar[str] = ""

    @classmethod
    def get_events(cls) -> list[type["EventBase"]]:
        """Every concrete event class, for the OpenAPI schema."""
        out: list[type[EventBase]] = []
        stack = list(cls.__subclasses__())
        while stack:
            sub = stack.pop()
            stack.extend(sub.__subclasses__())
            if sub.__event_name__:
                out.append(sub)
        return sorted(out, key=lambda c: c.__event_name__)


class JobUpdatedEvent(EventBase):
    __event_name__ = "job_updated"
    job: Job


class JobLogEvent(EventBase):
    __event_name__ = "job_log"
    job_id: str
    offset: int
    lines: list[str]
    """At most 50; the client fetches the rest over REST."""


class JobsClearedEvent(EventBase):
    __event_name__ = "jobs_cleared"
    ids: list[str]


class OutputsAddedEvent(EventBase):
    __event_name__ = "outputs_added"
    job_id: str | None
    outputs: list[Output]


class OutputsRemovedEvent(EventBase):
    __event_name__ = "outputs_removed"
    ids: list[str]


class ModelsChangedEvent(EventBase):
    __event_name__ = "models_changed"
    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)


class PresetsChangedEvent(EventBase):
    __event_name__ = "presets_changed"
    voice_id: str | None = None


class AssetsChangedEvent(EventBase):
    __event_name__ = "assets_changed"
    assets: list[AssetStatus]


class ExperimentChangedEvent(EventBase):
    __event_name__ = "experiment_changed"
    name: str
    stages: dict[str, StageState]
    running_job: str | None = None
    removed: bool = False


class TrainMetricsEvent(EventBase):
    __event_name__ = "train_metrics"
    name: str
    epoch: int
    step: int
    losses: dict[str, float]
    lr: float | None = None


class LiveStateEvent(EventBase):
    __event_name__ = "live_state"
    state: LiveState


class LiveStatsEvent(EventBase):
    __event_name__ = "live_stats"
    stats: LiveStats


class DevicesChangedEvent(EventBase):
    __event_name__ = "devices_changed"
    devices: DeviceList


class ComputeChangedEvent(EventBase):
    __event_name__ = "compute_changed"
    usage: ComputeUsage


class SettingsChangedEvent(EventBase):
    __event_name__ = "settings_changed"
    keys: list[str]
