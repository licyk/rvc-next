"""The live worker's control protocol.

The core and the live worker exchange one JSON object per message over a
``multiprocessing.connection`` connection (``send_bytes``/``recv_bytes``). Every object carries a
``type``. Commands go to the worker; events come back.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

# -- commands (core → worker) ---------------------------------------------------------------


@dataclass
class Start:
    """Load the voice, open the devices and run.

    ``config`` keys: ``voice_path``, ``index_path`` (or None), ``params`` (VoiceParams fields),
    ``stream`` (StreamParams fields) and ``devices`` (see ``SetDevices``).
    """

    config: dict[str, Any]
    type: str = "start"


@dataclass
class Stop:
    type: str = "stop"


@dataclass
class Shutdown:
    type: str = "shutdown"


@dataclass
class UpdateVoice:
    """Hot: pitch, formant, index rate, protect, loudness match, speaker, F0 method."""

    params: dict[str, Any]
    type: str = "update_voice"


@dataclass
class UpdateStream:
    """Threshold and denoise are hot; block, crossfade and context rebuffer without reloading."""

    stream: dict[str, Any]
    type: str = "update_stream"


@dataclass
class SetVoice:
    voice_path: str
    index_path: str | None = None
    speaker_id: int | None = None
    type: str = "set_voice"


@dataclass
class SetDevices:
    """``devices`` keys: ``input``, ``output``, ``monitor`` (each an endpoint or None).

    An endpoint is ``{"device": <AudioDevice dict or None for the system default>,
    "channels": [1-based] or None, "sample_rate": int or None, "exclusive": bool}``. Top-level
    keys ``monitor_source`` (converted, input or both), ``monitor_gain_db`` and
    ``output_gain_db`` complete it.
    """

    devices: dict[str, Any]
    type: str = "set_devices"


@dataclass
class Meter:
    """Open the input alone and report levels, without a voice. ``input`` is an endpoint."""

    on: bool
    input: dict[str, Any] | None = None
    type: str = "meter"


@dataclass
class TestTone:
    """Play a short chime on ``device`` (an endpoint; None for the system default output)."""

    device: dict[str, Any] | None = None
    type: str = "test_tone"


@dataclass
class Passthrough:
    """Send the input straight to the monitor (or to the output when there is no monitor)."""

    on: bool
    type: str = "passthrough"


@dataclass
class ReleaseMemory:
    """Unload the voice, HuBERT and the F0 models and empty the GPU cache; ignored while a session runs."""

    type: str = "release_memory"


# -- events (worker → core) -------------------------------------------------------------------


@dataclass
class State:
    """``state``: stopped, starting, loading, prewarming, running, stopping, error or reconnecting."""

    state: str
    error: dict[str, Any] | None = None
    """``{code, message, detail}``; codes as the core's errors (device_unavailable, asset_missing, …)."""
    topology: str | None = None
    """duplex or split, while running."""
    sample_rate: int | None = None
    meter: bool = False
    passthrough: bool = False
    cached: list[str] = field(default_factory=list)
    """The models the worker keeps loaded (its runtime's ``cached()``)."""
    voice_path: str | None = None
    """Set only in answer to ``SetVoice`` during a session: the voice now converting, which is the
    previous one when the new one could not be loaded (``error.detail.reason == "voice_load"``)."""
    stage: str | None = None
    """While ``loading``, the step under way: runtime (importing torch and transformers, opening the
    compute device), voice, index, hubert or pitch. None otherwise."""
    type: str = "state"


@dataclass
class Stats:
    """The fields of the core's ``LiveStats``."""

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
    type: str = "stats"


@dataclass
class DeviceLost:
    direction: str
    """input, output or monitor."""
    device_id: str | None
    reason: str
    """missing, busy, format or permission."""
    message: str = ""
    type: str = "device_lost"


@dataclass
class Log:
    level: str
    message: str
    type: str = "log"


Message = Start | Stop | Shutdown | UpdateVoice | UpdateStream | SetVoice | SetDevices | Meter | TestTone | Passthrough | State | Stats | DeviceLost | Log

MESSAGE_TYPES: dict[str, Any] = {
    cls.type: cls for cls in (Start, Stop, Shutdown, UpdateVoice, UpdateStream, SetVoice, SetDevices, Meter, TestTone, Passthrough, ReleaseMemory, State, Stats, DeviceLost, Log)
}

STATS_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Stats) if f.name != "type")


def encode(message: Message) -> bytes:
    return json.dumps(asdict(message), ensure_ascii=False, default=str).encode("utf-8")


def decode(data: bytes | str) -> Message | None:
    """Rebuild a message; None for an unknown type or malformed fields (unknown keys are ignored)."""
    try:
        obj = json.loads(data)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    cls = MESSAGE_TYPES.get(str(obj.get("type")))
    if cls is None:
        return None
    names = {f.name for f in fields(cls)}
    try:
        return cls(**{k: v for k, v in obj.items() if k in names})
    except TypeError:
        return None


@dataclass
class WorkerRequest:
    """The live worker's request file."""

    address: list[Any]
    authkey: str
    assets_dir: str
    device: str = "auto"
    precision: str = "auto"
    cuda_graph: bool = True
    enable_asio: bool = False
    backend: str = "sounddevice"
    """``fake`` only in tests."""
    fake: dict[str, Any] = field(default_factory=dict)
    """Configuration for the fake backend (tests)."""
