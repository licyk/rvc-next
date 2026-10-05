"""Asset records."""

from typing import Literal

from pydantic import Field

from rvc_next.core.record import Record

AssetGroup = Literal["inference", "training", "separation"]
AssetState = Literal["installed", "missing", "partial", "corrupt", "downloading"]


RepositoryId = Literal["rvc-model", "official"]


class AssetFile(Record):
    path: str
    """Relative to the assets folder, in the original's layout."""
    size: int
    sha256: str | None = None
    sources: dict[str, str]
    """Repository id → path of this file in that repository."""


class AssetSpec(Record):
    id: str
    title: str
    description: str
    group: AssetGroup
    files: list[AssetFile]


class AssetStatus(Record):
    id: str
    title: str
    description: str
    group: AssetGroup
    size: int
    installed_bytes: int
    state: AssetState
    verified: bool | None = None
    """True after a checksum check passed, False when one failed, None if not checked."""
    job_id: str | None = None
    missing_files: list[str] = Field(default_factory=list)


class Repository(Record):
    id: RepositoryId
    repo: str
    """The Hugging Face repository, ``owner/name``."""
    revision: str
    title: str
    selected: bool = False


class CatalogVoiceFile(Record):
    role: Literal["model", "index"]
    size: int
    sha256: str
    sources: dict[str, str]


class CatalogVoice(Record):
    """A ready-made voice that can be downloaded into the library."""

    id: str
    name: str
    description: str
    sample_rate: int
    version: Literal["v1", "v2"]
    pitch_guidance: bool
    files: list[CatalogVoiceFile]
    size: int = 0
    has_index: bool = False
    installed_voice_id: str | None = None
    """The library voice downloaded from this entry, if any."""
    job_id: str | None = None
    repositories: list[str] = Field(default_factory=list)
    """Repositories that host every file of this voice."""


class VoiceDownloadRequest(Record):
    ids: list[str] = Field(default_factory=list)
    all: bool = False
    """Every catalog voice not yet in the library."""


class AssetDownloadRequest(Record):
    ids: list[str] = Field(default_factory=list)
    group: Literal["inference", "training", "separation", "all"] | None = None
