"""Live session and audio device records."""

from typing import Literal

from pydantic import Field

from rvc_next.core.params import StreamParamsModel, VoiceParamsModel
from rvc_next.core.record import Record

Direction = Literal["input", "output"]
LiveStateName = Literal["stopped", "starting", "loading", "prewarming", "running", "stopping", "error", "reconnecting"]
ResolutionStatus = Literal["exact", "matched", "default_fallback", "missing"]


class HostApi(Record):
    id: str
    """wasapi | mme | directsound | wdm-ks | asio | coreaudio | alsa | jack | pulse | oss | other"""
    name: str
    rank: int
    """Lower is preferred on this platform."""
    low_latency: bool


class AudioDevice(Record):
    id: str
    """Stable across enumerations: a hash of host API, full name, direction and ordinal."""
    physical_key: str
    name: str
    raw_name: str
    host_api: str
    direction: Direction
    channels: int
    default_sample_rate: int
    supported_rates: list[int]
    latency_ms: list[float]
    """PortAudio's default low and high latency."""
    is_default: bool
    is_virtual: bool
    portaudio_index: int
    """Valid for this enumeration only; never stored."""


class PhysicalDevice(Record):
    key: str
    name: str
    direction: Direction
    is_default: bool
    is_virtual: bool
    variants: list[AudioDevice]
    recommended_id: str


class DeviceList(Record):
    host: str
    enumerated_at: str
    host_apis: list[HostApi]
    inputs: list[PhysicalDevice]
    outputs: list[PhysicalDevice]
    errors: list[str] = Field(default_factory=list)


class DeviceSelection(Record):
    device_id: str | None = None
    """None follows the system default."""
    physical_key: str | None = None
    name: str | None = None
    host_api: str | None = None
    channels: list[int] | None = None
    """Input: 1-based channels mixed to mono. Output: channels written. None: the first (input) or first two (output)."""
    sample_rate: int | None = None
    exclusive: bool = False


class LiveDevices(Record):
    input: DeviceSelection = Field(default_factory=DeviceSelection)
    output: DeviceSelection = Field(default_factory=DeviceSelection)
    monitor: DeviceSelection | None = None
    monitor_source: Literal["converted", "input", "both"] = "converted"
    monitor_gain_db: float = Field(default=0.0, ge=-60, le=12)
    output_gain_db: float = Field(default=0.0, ge=-60, le=12)


class ResolvedDevice(Record):
    role: Literal["input", "output", "monitor"]
    status: ResolutionStatus
    device: AudioDevice | None = None
    message: str | None = None


class LiveConfig(Record):
    voice_id: str
    params: VoiceParamsModel = Field(default_factory=VoiceParamsModel)
    stream: StreamParamsModel = Field(default_factory=StreamParamsModel)
    devices: LiveDevices = Field(default_factory=LiveDevices)
    allow_output_fallback: bool = False
    """Start even though the saved output device is gone and the system default would be used."""


class LiveStats(Record):
    input_peak_db: float = -120.0
    input_rms_db: float = -120.0
    output_peak_db: float = -120.0
    output_rms_db: float = -120.0
    infer_ms_p50: float = 0.0
    infer_ms_p95: float = 0.0
    block_ms: float = 0.0
    est_latency_ms: float = 0.0
    underruns: int = 0
    overruns: int = 0
    drift_ppm: float | None = None
    vram_mb: float | None = None


class LiveState(Record):
    state: LiveStateName = "stopped"
    voice_id: str | None = None
    config: LiveConfig | None = None
    resolved: list[ResolvedDevice] = Field(default_factory=list)
    topology: Literal["duplex", "split"] | None = None
    sample_rate: int | None = None
    meter: bool = False
    passthrough: bool = False
    error: dict | None = None
    started_at: str | None = None


class DeviceProblem(Record):
    role: Literal["input", "output", "monitor"]
    reason: Literal["missing", "busy", "format", "permission", "fallback"]
    message: str
    action: str | None = None
    """A suggested fix the picker shows as a button: "choose", "use_48k", "disable_exclusive", "grant_permission"."""


class DeviceCheck(Record):
    ok: bool
    resolved: list[ResolvedDevice]
    problems: list[DeviceProblem] = Field(default_factory=list)
    topology: Literal["duplex", "split"] | None = None
    sample_rate: int | None = None
    est_latency_ms: float | None = None


class TestToneRequest(Record):
    role: Literal["output", "monitor"] = "output"
    device: DeviceSelection | None = None
    """None: the saved selection for ``role``."""


class Toggle(Record):
    on: bool


class LiveVoiceRequest(Record):
    voice_id: str
