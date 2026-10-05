"""Training records."""

from typing import Literal

from pydantic import Field

from rvc_next.core.record import Record

StageId = Literal["clean", "slice", "f0", "features", "fit", "index"]
STAGE_ORDER: tuple[str, ...] = ("clean", "slice", "f0", "features", "fit", "index")
StageStatus = Literal["pending", "running", "done", "stale", "failed", "skipped"]
SampleRate = Literal["32k", "40k", "48k"]


class StageState(Record):
    status: StageStatus = "pending"
    fingerprint: str | None = None
    finished_at: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    job_id: str | None = None
    error: str | None = None


class SpeakerEntry(Record):
    name: str
    id: int = Field(ge=0, le=109)
    folder: str
    repeat: int = Field(default=1, ge=1, le=100)


class Dataset(Record):
    mode: Literal["single", "multi"] = "single"
    folder: str | None = None
    speakers: list[SpeakerEntry] = Field(default_factory=list)
    clean_preset: str | None = None
    """A separation preset run over the dataset before slicing."""


class FitSettings(Record):
    epochs: int = Field(default=20, ge=1, le=10000)
    save_every: int = Field(default=5, ge=1)
    batch_size: int | None = Field(default=None, ge=1, le=256)
    base_model: str | None = None
    """A base model id (``official-…`` or an imported one); None: the official one for the rate, version and pitch."""
    pretrained_g: str | None = None
    """Explicit G path (older experiments); used only when ``base_model`` is None. Empty string: none."""
    pretrained_d: str | None = None
    gpus: str = "auto"
    cache_in_gpu: bool = False
    save_small_every: bool = False
    save_latest_only: bool = False


class Experiment(Record):
    name: str
    sample_rate: SampleRate = "40k"
    version: Literal["v1", "v2"] = "v2"
    pitch_guidance: bool = True
    f0_method: Literal["pm", "rmvpe"] = "rmvpe"
    dataset: Dataset = Field(default_factory=Dataset)
    fit: FitSettings = Field(default_factory=FitSettings)
    stages: dict[str, StageState] = Field(default_factory=dict)
    voice_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    path: str = ""
    legacy: bool = False
    running_job: str | None = None


class ExperimentCreate(Record):
    name: str
    dataset: Dataset = Field(default_factory=Dataset)
    sample_rate: SampleRate | None = None
    version: Literal["v1", "v2"] | None = None
    pitch_guidance: bool | None = None
    f0_method: Literal["pm", "rmvpe"] | None = None
    fit: FitSettings | None = None


class ExperimentUpdate(Record):
    sample_rate: SampleRate | None = None
    version: Literal["v1", "v2"] | None = None
    pitch_guidance: bool | None = None
    f0_method: Literal["pm", "rmvpe"] | None = None
    dataset: Dataset | None = None
    fit: FitSettings | None = None


class DatasetIssue(Record):
    path: str
    problem: Literal["too_short", "too_long", "clipped", "silent", "unreadable"]
    detail: str = ""


class DatasetReport(Record):
    clips: int
    total_seconds: float
    sample_rates: dict[str, int]
    issues: list[DatasetIssue]
    speakers: dict[str, int] = Field(default_factory=dict)
    """Clips per speaker name, in multi-speaker mode."""


class RunRequest(Record):
    stages: list[StageId] | None = None
    """None runs every stage that is not done (Run all)."""
    force: bool = False
    """Run the chosen stages even when their inputs are unchanged."""


class TrainMetric(Record):
    epoch: int
    step: int
    losses: dict[str, float]
    lr: float | None = None
    time: str | None = None


class Checkpoint(Record):
    name: str
    path: str
    kind: Literal["G", "D", "small"]
    epoch: int | None = None
    step: int | None = None
    size: int
    modified_at: str


class ExportRequest(Record):
    checkpoint: str | None = None
    """A small model or G checkpoint name in the experiment; None: the latest."""
    voice_name: str | None = None


class ImportExperimentRequest(Record):
    path: str
    name: str | None = None


class FolderSpeakersRequest(Record):
    folder: str
    """A folder of ``Name_ID_Repeat`` subfolders."""


class ExperimentSummary(Record):
    name: str
    sample_rate: SampleRate
    version: Literal["v1", "v2"]
    pitch_guidance: bool
    mode: Literal["single", "multi"]
    stages: dict[str, StageState]
    running_job: str | None = None
    last_activity: str | None = None
    voice_id: str | None = None
    legacy: bool = False
