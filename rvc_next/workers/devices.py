"""Device enumeration in a short subprocess.

``python -m rvc_next.workers.devices <request.json>``. Enumerating in its own process never disturbs
a running stream, and needs none of the private re-initialisation calls the original used.

Request: ``{"action": "enumerate" | "check" | "test_tone" | "measure_latency", "enable_asio": bool,
"platform"?: str, "backend"?: "sounddevice" | "fake", "fake"?: {...}}``. ``check`` also takes
``input``, ``output`` and ``monitor`` (each a ``DeviceSelection`` dict plus ``portaudio_index``, or
null) and ``sample_rate``; ``test_tone`` takes ``device`` (the same shape, or null for the default
output); ``measure_latency`` takes ``input``, ``output``, ``block_ms`` and ``level_db`` and runs a
loopback measurement (``engine/audio_io/loopback.py``). One ``ResultEvent`` carries the answer.
"""

from __future__ import annotations

import sys
from typing import Any

from rvc_next.engine.audio_io.devices import device_list
from rvc_next.engine.audio_io.fake import backend_from_request
from rvc_next.engine.audio_io.streams import DeviceOpenError, Endpoint, choose_topology, engine_rate
from rvc_next.engine.audio_io.tone import chime
from rvc_next.protocol.jsonl import Emitter
from rvc_next.protocol.messages import ResultEvent
from rvc_next.workers.base import run_worker


def endpoint_from_selection(selection: dict[str, Any] | None, raw_devices: list[dict[str, Any]], direction: str) -> Endpoint | None:
    """An ``Endpoint`` for a selection that carries ``portaudio_index``; None without one."""
    if not selection:
        return None
    index = selection.get("portaudio_index")
    device: dict[str, Any] | None = None
    if index is not None:
        raw = next((d for d in raw_devices if int(d.get("index", -1)) == int(index)), None)
        key = "max_input_channels" if direction == "input" else "max_output_channels"
        device = {
            "id": selection.get("device_id"),
            "portaudio_index": int(index),
            "host_api": selection.get("host_api"),
            "channels": int(raw.get(key, 2)) if raw else 2,
            "channel_counts": list(raw.get(f"{direction}_channel_counts") or []) if raw else [],
            "default_sample_rate": int(round(float(raw.get("default_samplerate", 48000)))) if raw else 48000,
            "supported_rates": list(raw.get("supported_rates", [])) if raw else [],
        }
    return Endpoint(device=device, channels=selection.get("channels"), sample_rate=selection.get("sample_rate"), exclusive=bool(selection.get("exclusive", False)))


def check(backend: Any, request: dict[str, Any]) -> dict[str, Any]:
    _, raw_devices, _ = backend.query()
    inp = endpoint_from_selection(request.get("input"), raw_devices, "input")
    out = endpoint_from_selection(request.get("output"), raw_devices, "output")
    mon = endpoint_from_selection(request.get("monitor"), raw_devices, "output")
    rate = int(request.get("sample_rate") or engine_rate(out, inp))
    topology = choose_topology(inp or Endpoint(), out, rate) if out is not None else "input"
    for role, ep, direction in (("input", inp, "input"), ("output", out, "output"), ("monitor", mon, "output")):
        if ep is None:
            continue
        ep_rate = rate
        if role == "input" and topology == "split" and ep.device is not None and not ep.supports(rate):
            ep_rate = int(ep.device.get("default_sample_rate") or rate)
        try:
            backend.check(ep, direction, ep.open_channels(direction), int(ep.sample_rate or ep_rate) if role == "input" else ep_rate)
        except DeviceOpenError as e:
            return {"ok": False, "reason": e.reason, "message": e.message, "role": role, "sample_rate": rate, "topology": topology}
    return {"ok": True, "reason": None, "message": None, "role": None, "sample_rate": rate, "topology": topology}


def test_tone(backend: Any, request: dict[str, Any]) -> dict[str, Any]:
    _, raw_devices, _ = backend.query()
    ep = endpoint_from_selection(request.get("device"), raw_devices, "output")
    rate = int((ep.sample_rate if ep else None) or engine_rate(ep))
    try:
        backend.play(ep, chime(rate), rate)
    except DeviceOpenError as e:
        return {"ok": False, "reason": e.reason, "message": e.message, "role": "output"}
    return {"ok": True, "reason": None, "message": None, "role": None}


def measure_latency(backend: Any, request: dict[str, Any]) -> dict[str, Any]:
    from rvc_next.engine.audio_io.loopback import measure
    from rvc_next.engine.audio_io.streams import SessionConfig
    from rvc_next.engine.stream.latency import block_frames

    _, raw_devices, _ = backend.query()
    inp = endpoint_from_selection(request.get("input"), raw_devices, "input") or Endpoint()
    out = endpoint_from_selection(request.get("output"), raw_devices, "output") or Endpoint()
    block = block_frames(float(request.get("block_ms") or 250), engine_rate(out, inp))
    try:
        run = measure(backend, SessionConfig(input=inp, output=out), block, level_db=float(request.get("level_db", -12.0)), timeout_scale=1.0 / getattr(backend, "time_scale", 1.0))
    except DeviceOpenError as e:
        return {"ok": False, "reason": e.reason, "message": e.message, "role": e.role}
    r = run.result
    return {
        "ok": True,
        "found": r.ok,
        "failure": r.reason,
        "round_trip_ms": r.round_trip_ms,
        "jitter_ms": r.jitter_ms,
        "pings": r.pings,
        "detected": len(r.delays),
        "snr_db": r.snr_db,
        "input_peak_db": r.input_peak_db,
        "clipped": r.clipped,
        "topology": run.topology,
        "sample_rate": run.sample_rate,
        "reported_ms": list(run.reported_ms),
        "underruns": run.underruns,
    }


def main(request: dict[str, Any], emitter: Emitter) -> None:
    action = request.get("action", "enumerate")
    enable_asio = bool(request.get("enable_asio", False))
    backend = backend_from_request(request.get("backend", "sounddevice"), enable_asio, request.get("fake"))
    if action == "enumerate":
        hostapis, devices, default_api = backend.query()
        platform = request.get("platform") or sys.platform
        errors = []
        if enable_asio and platform.startswith("win") and not any("asio" in str(h.get("name", "")).lower() for h in hostapis):
            errors.append("ASIO was requested, but no ASIO driver is installed")
        data = device_list(hostapis, devices, platform, default_api, errors)
    elif action == "check":
        data = check(backend, request)
    elif action == "test_tone":
        data = test_tone(backend, request)
    elif action == "measure_latency":
        data = measure_latency(backend, request)
    else:
        raise ValueError(f"Unknown action {action!r}")
    emitter.emit(ResultEvent(data=data))


if __name__ == "__main__":
    raise SystemExit(run_worker(main))
