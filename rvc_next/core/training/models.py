"""Training records."""

from typing import Literal

from pydantic import Field

from rvc_next.core.record import Record

StageId = Literal["clean", "slice", "f0", "features", "fit", "index"]
STAGE_ORDER: tuple[str, ...] = ("clean", "slice", "f0", "features", "fit", "index")
StageStatus = Literal["pending", "running", "done", "stale", "failed", "skipped"]
SampleRate = Literal["32k", "40k", "48k"]
TrainingF0Method = Literal["pm", "rmvpe", "fcpe", "crepe", "crepe-tiny", "swift"]
TrainingPrecision = Literal["auto", "fp32", "bf16"]


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


class SliceSettings(Record):
    """How the dataset is cut into training slices; the defaults are the original's."""

    cut: Literal["auto", "fixed", "none"] = "auto"
    """auto: at silences, then pieces; fixed: pieces straight through; none: files as they are (already cut)."""
    chunk_seconds: float | None = Field(default=None, ge=0.5, le=10)
    """Piece length; None: 3.7 s with a GPU, 3.0 s without, as the original."""
    overlap: float = Field(default=0.3, ge=0, le=0.4)
    highpass: bool = True
    """The original's 48 Hz high-pass, which removes rumble."""
    normalize: Literal["slice", "file", "none"] = "slice"
    """Loudness normalisation per slice (the original), per file, or none."""
    denoise: float = Field(default=0.0, ge=0, le=1)
    """Noise reduction over each file before cutting; 0 turns it off."""


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
    precision: TrainingPrecision = "auto"
    """auto: fp16 when every chosen GPU qualifies (the original); bf16 on GPUs that have it; fp32."""
    tf32: bool = False
    """TF32 matrix maths on NVIDIA GPUs (Ampere and newer): faster fp32, slightly less precise."""
    checkpointing: bool = False
    """Gradient checkpointing: recompute activations in the backward pass; less memory, slower."""
    fresh_speakers: bool = False
    """Start the speakers from new vectors instead of the base model's (whose speakers 0 and 1 are near-identical)."""
    previews: bool = True
    """At every save, render a training clip with the current generator (Results › Previews)."""


class Experiment(Record):
    name: str
    sample_rate: SampleRate = "40k"
    version: Literal["v1", "v2"] = "v2"
    pitch_guidance: bool = True
    f0_method: TrainingF0Method = "rmvpe"
    dataset: Dataset = Field(default_factory=Dataset)
    slicing: SliceSettings = Field(default_factory=SliceSettings)
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
    f0_method: TrainingF0Method | None = None
    fit: FitSettings | None = None


class ExperimentUpdate(Record):
    sample_rate: SampleRate | None = None
    version: Literal["v1", "v2"] | None = None
    pitch_guidance: bool | None = None
    f0_method: TrainingF0Method | None = None
    dataset: Dataset | None = None
    slicing: SliceSettings | None = None
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
