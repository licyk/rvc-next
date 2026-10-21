"""Device enumeration in a short subprocess.

``python -m rvc_next.workers.devices <request.json>``. Enumerating in its own process never disturbs
a running stream, and needs none of the private re-initialisation calls the original used.

Request: ``{"action": "enumerate" | "check" | "test_tone" | "measure_latency", "enable_asio": bool,
"loopback"?: bool, "platform"?: str, "backend"?: "sounddevice" | "fake", "fake"?: {...}}``.
``loopback`` lists output devices as loopback inputs too (``engine/audio_io/capture.py``); a fake
machine's ``fake.platform`` stands in for ``platform``. ``check``
also takes ``input``, ``output`` and ``monitor`` (each a ``DeviceSelection`` dict plus
``portaudio_index`` and, for a loopback input, ``loopback_source``, or null), ``sample_rate`` and ``probe``
(the roles to open for a moment, ``streams.probe_stream``, once every setting passed); ``test_tone`` takes ``device`` (the same shape, or null for the default
output); ``measure_latency`` takes ``input``, ``output``, ``block_ms`` and ``level_db`` and runs a
loopback measurement (``engine/audio_io/loopback.py``). One ``ResultEvent`` carries the answer.

A selection with ``candidates`` (an automatic driver: ``[{device_id, host_api, portaudio_index,
loopback_source}]``, the first being the selection's own) is tried through each in turn
(``fallback.open_with_fallback``); answers then add ``opened`` (role → the device id that passed, when
it ran or another driver refused first) and ``attempts`` (role → the drivers that refused).
"""

from __future__ import annotations

import sys
from typing import Any

from rvc_next.engine.audio_io.devices import device_list
from rvc_next.engine.audio_io.fake import backend_from_request
from rvc_next.engine.audio_io.fallback import describe_attempts, every_driver_refused, open_with_fallback
from rvc_next.engine.audio_io.streams import DeviceOpenError, Endpoint, SessionConfig, choose_topology, engine_rate, probe_stream
from rvc_next.engine.audio_io.tone import chime
from rvc_next.protocol.jsonl import Emitter
from rvc_next.protocol.messages import ResultEvent
from rvc_next.workers.base import run_worker


def _device(entry: dict[str, Any], raw_devices: list[dict[str, Any]], direction: str) -> dict[str, Any] | None:
    """The device dict an ``Endpoint`` opens, for an entry carrying ``portaudio_index``; None without one."""
    index = entry.get("portaudio_index")
    if index is None:
        return None
    source = entry.get("loopback_source")
    # A loopback source is found by its provider id: its index is only a position in the list.
    if source is not None:
        raw = next((d for d in raw_devices if d.get("loopback_source") == source), None)
    else:
        raw = next((d for d in raw_devices if int(d.get("index", -1)) == int(index)), None)
    key = "max_input_channels" if direction == "input" else "max_output_channels"
    return {
        "id": entry.get("device_id"),
        "portaudio_index": int(index),
        "host_api": entry.get("host_api"),
        "channels": int(raw.get(key, 2)) if raw else 2,
        "channel_counts": list(raw.get(f"{direction}_channel_counts") or []) if raw else [],
        "loopback_source": source,
        "default_sample_rate": int(round(float(raw.get("default_samplerate", 48000)))) if raw else 48000,
        "supported_rates": list(raw.get("supported_rates", [])) if raw else [],
    }


def endpoint_from_selection(selection: dict[str, Any] | None, raw_devices: list[dict[str, Any]], direction: str) -> Endpoint | None:
    """An ``Endpoint`` for a selection that carries ``portaudio_index``, with its ``candidates``; None without a selection."""
    if not selection:
        return None
    candidates = [d for d in (_device(c, raw_devices, direction) for c in selection.get("candidates") or []) if d is not None]
    return Endpoint(
        device=_device(selection, raw_devices, direction),
        channels=selection.get("channels"),
        sample_rate=selection.get("sample_rate"),
        exclusive=bool(selection.get("exclusive", False)),
        candidates=candidates,
    )


def _alone(role: str, ep: Endpoint) -> SessionConfig:
    """A configuration holding ``ep`` alone in ``role``, so one role's drivers are tried at a time."""
    return SessionConfig(input=ep if role == "input" else Endpoint(), output=ep if role == "output" else None, monitor=ep if role == "monitor" else None)


def _failure(e: DeviceOpenError) -> str:
    """The message for a refusal; one naming each driver when every one was tried."""
    tried = every_driver_refused(e)
    return f"No driver opens it ({describe_attempts(tried)})" if tried else e.message


def check(backend: Any, request: dict[str, Any]) -> dict[str, Any]:
    _, raw_devices, _ = backend.query()
    inp = endpoint_from_selection(request.get("input"), raw_devices, "input")
    out = endpoint_from_selection(request.get("output"), raw_devices, "output")
    mon = endpoint_from_selection(request.get("monitor"), raw_devices, "output")
    rate = int(request.get("sample_rate") or engine_rate(out, inp))
    probe = set(request.get("probe") or [])
    chosen: dict[str, Endpoint | None] = {"input": inp, "output": out, "monitor": mon}
    opened: dict[str, str] = {}
    attempts: dict[str, list[dict[str, Any]]] = {}

    def role_rate(role: str, ep: Endpoint) -> int:
        if role != "input":
            return rate
        topology = choose_topology(ep, chosen["output"], rate) if chosen["output"] is not None else "input"
        ep_rate = int(ep.device.get("default_sample_rate") or rate) if topology == "split" and ep.device is not None and not ep.supports(rate) else rate
        return int(ep.sample_rate or ep_rate)

    def topology() -> str:
        return choose_topology(chosen["input"] or Endpoint(), chosen["output"], rate) if chosen["output"] is not None else "input"

    try:
        for role, direction in (("input", "input"), ("output", "output"), ("monitor", "output")):
            first = chosen[role]
            if first is None:
                continue

            def run(c: SessionConfig, role: str = role, direction: str = direction) -> None:
                ep = c.input if role == "input" else c.output if role == "output" else c.monitor
                assert ep is not None
                ep_rate = role_rate(role, ep)
                try:
                    backend.check(ep, direction, ep.open_channels(direction), ep_rate)
                    # The settings pass; some drivers still give up once a stream runs. One device at a time.
                    if role in probe:
                        probe_stream(backend, ep, direction, ep_rate)
                except DeviceOpenError as e:
                    e.role = role
                    raise

            _, used, tried = open_with_fallback(_alone(role, first), run)
            ep = used.input if role == "input" else used.output if role == "output" else used.monitor
            chosen[role] = ep
            if tried.get(role):
                attempts[role] = tried[role]
            # A driver counts as opening the device when the device ran, or the ones before it refused:
            # settings a driver accepts may still fail as the stream starts.
            if ep is not None and ep.candidates and ep.device_id is not None and (role in probe or tried.get(role)):
                opened[role] = ep.device_id
    except DeviceOpenError as e:
        attempts.update(e.attempts)
        return {"ok": False, "reason": e.reason, "message": _failure(e), "role": e.role, "sample_rate": rate, "topology": topology(), "opened": opened, "attempts": attempts}
    return {"ok": True, "reason": None, "message": None, "role": None, "sample_rate": rate, "topology": topology(), "opened": opened, "attempts": attempts}


def test_tone(backend: Any, request: dict[str, Any]) -> dict[str, Any]:
    _, raw_devices, _ = backend.query()
    ep = endpoint_from_selection(request.get("device"), raw_devices, "output")

    def play(c: SessionConfig) -> None:
        rate = int((c.output.sample_rate if c.output else None) or engine_rate(c.output))
        backend.play(c.output, chime(rate), rate)

    try:
        _, used, tried = open_with_fallback(SessionConfig(input=Endpoint(), output=ep), play)
    except DeviceOpenError as e:
        e.role = e.role or "output"
        return {"ok": False, "reason": e.reason, "message": _failure(e), "role": "output", "attempts": e.attempts}
    opened = {"output": used.output.device_id} if used.output is not None and used.output.candidates and used.output.device_id else {}
    return {"ok": True, "reason": None, "message": None, "role": None, "opened": opened, "attempts": tried}


def measure_latency(backend: Any, request: dict[str, Any]) -> dict[str, Any]:
    from rvc_next.engine.audio_io.loopback import measure
    from rvc_next.engine.stream.latency import block_frames

    _, raw_devices, _ = backend.query()
    inp = endpoint_from_selection(request.get("input"), raw_devices, "input") or Endpoint()
    out = endpoint_from_selection(request.get("output"), raw_devices, "output") or Endpoint()
    block = block_frames(float(request.get("block_ms") or 250), engine_rate(out, inp))

    def run_measure(c: SessionConfig) -> Any:
        return measure(backend, c, block, level_db=float(request.get("level_db", -12.0)), timeout_scale=1.0 / getattr(backend, "time_scale", 1.0))

    try:
        run, _, _ = open_with_fallback(SessionConfig(input=inp, output=out), run_measure)
    except DeviceOpenError as e:
        return {"ok": False, "reason": e.reason, "message": _failure(e), "role": e.role}
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
    backend = backend_from_request(request.get("backend", "sounddevice"), enable_asio, request.get("fake"), bool(request.get("loopback", False)))
    if action == "enumerate":
        hostapis, devices, default_api = backend.query()
        platform = request.get("platform") or (request.get("fake") or {}).get("platform") or sys.platform
        errors = [backend.loopback_error] if getattr(backend, "loopback_error", None) else []
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
