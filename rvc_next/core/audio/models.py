"""Audio inputs and outputs."""

from typing import Literal

from pydantic import Field

from rvc_next.core.params import VoiceParamsModel
from rvc_next.core.record import Record

OutputKind = Literal["converted", "stem", "remix", "preview", "recording", "sample"]


class AudioRef(Record):
    """An input: an uploaded file, a server path or an earlier output."""

    kind: Literal["upload", "path", "output", "output-source"]
    id: str | None = None
    """The upload's or the output's id; ``output-source`` is the input an output was made from."""
    path: str | None = None
    """A server path, for ``kind = "path"``."""


class AudioFile(Record):
    id: str
    name: str
    path: str
    origin: Literal["upload", "path"]
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    size: int
    created_at: str


class AudioInfo(Record):
    """What probing an audio file found."""

    duration: float
    sample_rate: int
    channels: int
    codec: str | None = None


class Output(Record):
    id: str
    job_id: str | None = None
    path: str
    name: str
    kind: OutputKind
    label: str
    """vocals, instrumental, converted, …"""
    source_path: str | None = None
    source_name: str | None = None
    model_id: str | None = None
    voice: VoiceParamsModel | None = None
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    size: int | None = None
    exists: bool = True
    created_at: str


class OutputPage(Record):
    items: list[Output]
    next_cursor: str | None = None


class Peaks(Record):
    """Waveform peaks: ``points`` (min, max) pairs over the whole file, mixed to mono."""

    points: int
    duration: float
    sample_rate: int
    data: list[float]
    """Interleaved min and max, each in -1..1."""


class SpectrogramData(Record):
    rows: int
    cols: int
    fmax: float
    """Frequency of the top row, in Hz (half the sample rate); rows are mel-spaced from 0."""
    data: str
    """``rows × cols`` bytes, row 0 the lowest band, base64: 0 is 80 dB below the loudest, 255 the loudest."""


class AudioAnalysis(Record):
    """A spectrogram and the pitch curve of one file, for judging a result."""

    duration: float
    sample_rate: int
    channels: int
    codec: str | None = None
    spectrogram: SpectrogramData
    pitch_hop: float = 0.01
    pitch: list[float]
    """Hz every ``pitch_hop`` seconds; 0 where there is no pitch."""
    pitch_median: float | None = None


class BrowseEntry(Record):
    name: str
    path: str
    is_dir: bool
    is_audio: bool = False
    size: int | None = None
    mtime: float | None = None


class BrowseListing(Record):
    root: str
    path: str
    parent: str | None
    entries: list[BrowseEntry]
    roots: list[str] = Field(default_factory=list)


class ResolvePathRequest(Record):
    path: str


class ZipRequest(Record):
    ids: list[str]
