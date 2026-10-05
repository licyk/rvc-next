"""Settings models."""

from typing import Literal

from pydantic import Field

from rvc_next.core.live.models import LiveDevices
from rvc_next.core.params import StreamParamsModel, VoiceParamsModel
from rvc_next.core.record import Record

DEFAULT_PORT = 7868


class ServerSettings(Record):
    host: str = "127.0.0.1"
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    strict_port: bool = False
    open_browser: bool = True
    access_token: str | None = None
    allowed_origins: list[str] = Field(default_factory=list)


class LegacyRoot(Record):
    """An original RVC install whose ``assets/weights`` voices are read in place."""

    id: str
    path: str
    name: str = ""


class PathSettings(Record):
    """Empty folders mean "inside the data directory"."""

    models_dir: str = ""
    experiments_dir: str = ""
    outputs_dir: str = ""
    assets_dir: str = ""
    legacy_roots: list[LegacyRoot] = Field(default_factory=list)
    browse_roots: list[str] = Field(default_factory=list)
    """Server folders the UI may browse. Empty: home, the legacy roots, the data directory and every drive or volume."""


class ComputeSettings(Record):
    device: str = "auto"
    """auto, cpu, cuda:N (NVIDIA, or AMD on ROCm), xpu:N (Intel), dml, or mps (experimental)."""
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    cuda_graph_offline: bool = False
    cuda_graph_live: bool = True
    unload_after_minutes: int = Field(default=10, ge=0)
    gpu_jobs: int = Field(default=1, ge=1, le=16)


class DownloadSettings(Record):
    repository: Literal["rvc-model", "official"] = "rvc-model"
    """Where model files come from: licyk/rvc-model (every model by category, and the demo voices)
    or the official lj1995/VoiceConversionWebUI. A file the chosen one lacks comes from the other."""
    source: Literal["huggingface", "hf-mirror", "custom"] = "huggingface"
    """The Hugging Face endpoint the repository is reached through."""
    endpoint: str = ""
    verify_checksums: bool = True


OutputFormat = Literal["wav", "flac", "mp3", "m4a"]


class ConvertSettings(Record):
    output_format: OutputFormat = "wav"
    output_dir: str = ""
    keep_outputs_days: int = Field(default=0, ge=0)
    default_params: VoiceParamsModel = Field(default_factory=VoiceParamsModel)
    """Copied into a voice's default preset when the voice is imported."""


class SeparationSettings(Record):
    default_preset: str = "vocals"
    output_format: OutputFormat = "flac"


class TrainingSettings(Record):
    sample_rate: Literal["32k", "40k", "48k"] = "40k"
    version: Literal["v1", "v2"] = "v2"
    pitch_guidance: bool = True
    f0_method: Literal["auto", "pm", "rmvpe"] = "auto"
    """auto: rmvpe on a GPU, pm otherwise."""
    epochs: int = Field(default=20, ge=1, le=10000)
    save_every: int = Field(default=5, ge=1)
    batch_size: int | None = Field(default=None, ge=1, le=256)
    """None: the smallest GPU's memory in GiB divided by 2, as the original."""
    cache_in_gpu: bool = False
    save_small_every: bool = False
    cpu_workers: int | None = Field(default=None, ge=1)
    """None: the CPU count divided by 1.5."""
    gpus: str = "auto"
    """auto, cpu, or GPU indexes joined by '-', as the original ("0-1")."""


class LiveSettings(Record):
    devices: LiveDevices = Field(default_factory=LiveDevices)
    stream: StreamParamsModel = Field(default_factory=lambda: StreamParamsModel())
    last_voice: str | None = None
    last_params: VoiceParamsModel = Field(default_factory=lambda: VoiceParamsModel(index_rate=0.0, rms_mix_rate=0.0))
    """The realtime GUI's fresh-install values: no index and the converted loudness."""
    auto_reconnect: bool = True
    show_meters: bool = True
    enable_asio: bool = False
    show_all_devices: bool = False
    """List every device in both the input and the output menus, whatever its direction, for routing
    audio into an input device (a virtual cable's recording side, say). Off: each menu lists only the
    devices with channels in its direction."""


class Settings(Record):
    """Everything saved in ``settings.toml``."""

    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    downloads: DownloadSettings = Field(default_factory=DownloadSettings)
    convert: ConvertSettings = Field(default_factory=ConvertSettings)
    separation: SeparationSettings = Field(default_factory=SeparationSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    live: LiveSettings = Field(default_factory=LiveSettings)


# Public views: the access token never leaves the server.


class ServerSettingsView(Record):
    host: str
    port: int
    strict_port: bool
    open_browser: bool
    access_token_configured: bool
    allowed_origins: list[str]


class ResolvedPaths(Record):
    """The folders in use, with the data-directory defaults filled in."""

    models_dir: str
    experiments_dir: str
    outputs_dir: str
    assets_dir: str
    browse_roots: list[str]


class SettingsView(Record):
    """Settings as returned to clients, with the token replaced by a flag."""

    data_dir: str
    settings_file: str
    server: ServerSettingsView
    paths: PathSettings
    resolved_paths: ResolvedPaths
    compute: ComputeSettings
    downloads: DownloadSettings
    convert: ConvertSettings
    separation: SeparationSettings
    training: TrainingSettings
    live: LiveSettings
    env_overrides: list[str]
    pinned: list[str]
