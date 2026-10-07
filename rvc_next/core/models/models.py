"""Voice library records."""

from typing import Literal

from pydantic import Field

from rvc_next.core.record import Record


class Speaker(Record):
    id: int
    name: str


class Provenance(Record):
    """Training details a voice file carries (written by Applio and by rvc-next's exports)."""

    author: str | None = None
    epoch: int | None = None
    step: int | None = None
    created: str | None = None
    """When the file was made, as written (ISO 8601)."""
    dataset_length: str | None = None
    """Length of the training audio, as written (``HH:MM:SS``)."""

    @classmethod
    def from_file(cls, values: dict[str, str]) -> "Provenance":
        def number(key: str) -> int | None:
            try:
                return int(float(values[key]))
            except (KeyError, ValueError):
                return None

        return cls(author=values.get("author"), epoch=number("epoch"), step=number("step"), created=values.get("creation_date"), dataset_length=values.get("dataset_length"))


class VoiceModel(Record):
    """A voice: a small model, its index or indexes, its speakers and its default parameters."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    location: str
    """``library``, ``legacy:<root id>`` or ``temporary``."""
    legacy: bool = False
    model_path: str
    sample_rate: int
    version: Literal["v1", "v2"]
    pitch_guidance: bool
    speakers: list[Speaker] = Field(default_factory=list)
    """Named speakers; empty for a single-speaker voice."""
    speaker_slots: int = 1
    """Rows of the speaker embedding."""
    indexes: dict[str, str] = Field(default_factory=dict)
    """``default`` or ``spk<N>`` → index path."""
    has_index: bool = False
    size: int = 0
    info: str = ""
    provenance: Provenance = Field(default_factory=Provenance)
    hidden: bool = False
    catalog_id: str | None = None
    """The download catalog entry this voice came from."""
    created_at: str
    updated_at: str


class VoiceUpdate(Record):
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    speakers: list[Speaker] | None = None
    hidden: bool | None = None


class ImportPathRequest(Record):
    paths: list[str]
    name: str | None = None
    index: str | None = None


class ImportResult(Record):
    voices: list[VoiceModel] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)
    """G checkpoints found instead of small models: offered to Extract."""
    skipped: list[str] = Field(default_factory=list)
    attached: list["AttachedIndex"] = Field(default_factory=list)
    """Indexes added to voices, new or in the library."""
    inbox: list[str] = Field(default_factory=list)
    """Indexes kept unassigned, by inbox id."""
    base_models: list[str] = Field(default_factory=list)
    separation_models: list[str] = Field(default_factory=list)


class IndexSuggestion(Record):
    voice_id: str
    key: str
    path: str | None


class LegacyScanResult(Record):
    root_id: str
    voices: list[VoiceModel]
    suggestions: list[IndexSuggestion]


class IndexPairing(Record):
    key: str = "default"
    path: str | None
    """None detaches."""


class MergeRequest(Record):
    a: str
    b: str
    alpha: float = Field(default=0.5, ge=0, le=1)
    """Weight of ``a``."""
    name: str
    info: str = ""


class ExtractRequest(Record):
    checkpoint: str
    """A G checkpoint path, or ``<experiment>/<file>``."""
    name: str
    sample_rate: Literal["32k", "40k", "48k"] | None = None
    version: Literal["v1", "v2"] | None = None
    pitch_guidance: bool | None = None
    info: str = ""


class BuildIndexRequest(Record):
    experiment: str


class CheckpointInfo(Record):
    """What a ``.pth`` holds, without loading it into a model."""

    kind: Literal["small", "checkpoint"]
    sample_rate: int | None = None
    version: Literal["v1", "v2"] | None = None
    pitch_guidance: bool | None = None
    speakers: list[Speaker] = Field(default_factory=list)
    speaker_slots: int = 1
    info: str = ""
    iteration: int | None = None
    provenance: Provenance = Field(default_factory=Provenance)


# -- import sessions -------------------------------------------------------------

StagedKind = Literal["voice", "index", "generator", "discriminator", "separation_checkpoint", "separation_config", "unsupported"]


class StagedFile(Record):
    """A file in an import session, identified by its content."""

    id: str
    name: str
    """The name it arrived with, with its folder inside an archive."""
    group: str | None = None
    """The archive or folder it came in; files of one group likely belong together."""
    kind: StagedKind
    size: int
    note: str = ""
    version: Literal["v1", "v2"] | None = None
    sample_rate: str | None = None
    pitch_guidance: bool | None = None
    speakers: list[Speaker] = Field(default_factory=list)
    dim: int | None = None
    vectors: int | None = None
    model_type: str | None = None
    instruments: list[str] = Field(default_factory=list)
    target_instrument: str | None = None


class PairingCandidate(Record):
    target: str
    """``file:<id>`` (in this import) or ``voice:<id>`` (in the library)."""
    name: str
    score: float


class VoicePlan(Record):
    file_id: str
    name: str
    action: Literal["import", "skip"] = "import"


class IndexPlan(Record):
    file_id: str
    target: str | None = None
    """``file:<id>`` of a voice in this import, ``voice:<id>`` of a library voice, or None: keep it unassigned."""
    key: str = "default"
    status: Literal["unique", "evidence", "choose", "incompatible", "empty", "duplicate", "manual"] = "manual"
    confident: bool = False
    reasons: list[str] = Field(default_factory=list)
    candidates: list[PairingCandidate] = Field(default_factory=list)


class GeneratorPlan(Record):
    file_id: str
    action: Literal["base", "extract", "skip"] = "base"
    name: str
    discriminator: str | None = None
    """``file:<id>`` of the paired D, if any."""
    candidates: list[str] = Field(default_factory=list)


class SeparationPlan(Record):
    config_id: str
    checkpoint_id: str | None = None
    name: str
    primary: str
    secondary: str
    primary_label: str
    secondary_label: str
    candidates: list[str] = Field(default_factory=list)
    action: Literal["import", "skip"] = "import"


class ImportPlan(Record):
    session_id: str
    files: list[StagedFile]
    voices: list[VoicePlan] = Field(default_factory=list)
    indexes: list[IndexPlan] = Field(default_factory=list)
    generators: list[GeneratorPlan] = Field(default_factory=list)
    separations: list[SeparationPlan] = Field(default_factory=list)
    confident: bool = False
    """True when nothing needs a decision: the plan can be committed as proposed."""
    created_at: str


class ImportDecisions(Record):
    """The user's changes to a plan; a list left out keeps the proposals."""

    voices: list[VoicePlan] | None = None
    indexes: list[IndexPlan] | None = None
    generators: list[GeneratorPlan] | None = None
    separations: list[SeparationPlan] | None = None


class InboxIndex(Record):
    """An index waiting for a voice."""

    id: str
    name: str
    """The name it was imported with."""
    dim: int
    vectors: int
    version: Literal["v1", "v2"] | None = None
    size: int
    imported_at: str
    candidates: list[PairingCandidate] = Field(default_factory=list)
    """Compatible library voices, best first."""
    proposed: str | None = None


class AssignIndexRequest(Record):
    voice_id: str
    key: str = "default"


class AttachedIndex(Record):
    voice_id: str
    key: str
    name: str


ImportResult.model_rebuild()
