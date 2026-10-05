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


class LatencyMeasurement(Record):
    """A loopback measurement: test bursts played on the output and found again in the input."""

    ok: bool
    """Whether the bursts came back and agreed; the latencies are set only then."""
    reason: Literal["no_signal", "inconsistent"] | None = None
    latency_ms: float | None = None
    """End to end, from the microphone to the output: the measured round trip plus the engine's delay."""
    round_trip_ms: float | None = None
    """Measured: from the converter's output, out of the output device, back in through the input device,
    to the converter, with the session's own buffers; ``latency_ms`` minus the engine's part."""
    engine_ms: float
    """The conversion's own delay for these stream settings (crossfade + 10 ms, and the input denoise
    buffer), computed; SOLA may take up to 10 ms off it per block."""
    estimated_ms: float
    """The estimate for the same settings, from the latency PortAudio reported for the open streams."""
    jitter_ms: float | None = None
    """How far apart the found bursts' round trips were."""
    pings: int
    detected: int
    snr_db: float | None = None
    """The weakest found burst's correlation peak over the noise floor."""
    input_peak_db: float
    clipped: bool
    underruns: int = 0
    topology: Literal["duplex", "split"]
    sample_rate: int
    reported_ms: list[float]
    """PortAudio's input and output latency for the streams the measurement opened."""
    devices: LiveDevices
    stream: StreamParamsModel
    measured_at: str


class LatencyTestRequest(Record):
    devices: LiveDevices | None = None
    """None: the saved devices."""
    stream: StreamParamsModel | None = None
    """None: the saved stream settings. The block sets the buffers measured; crossfade and input
    denoise set the engine's part."""
    level_db: float = Field(default=-12.0, ge=-40, le=-3)
    """Peak level of the test bursts in dBFS."""


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
    latency_test: LatencyMeasurement | None = None
    """The last latency measurement, while it still applies: cleared when the input, the output or
    the block length changes; recomputed when the crossfade or input denoise does."""


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
