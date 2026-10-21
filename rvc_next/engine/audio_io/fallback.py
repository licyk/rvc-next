"""Automatic driver choice: open a device through the next driver when one refuses.

On Windows one microphone is listed under WASAPI, WDM-KS, DirectSound and MME, and a driver can
refuse a device the others open (a WDM-KS pin another program holds, an ``Unanticipated host
error`` when the stream starts). An endpoint with ``candidates`` (an automatic selection,
``devices.resolve_drivers``) is tried under each of them in turn: ``open_with_fallback`` calls
``start`` with a session configuration, and when it raises ``DeviceOpenError`` for a role that has
another driver left, swaps that one in and calls again. The drivers that refused are recorded per
role; when every one has, the last error is raised with all of them in ``attempts``.

A duplex stream's error may name neither device (``role`` None): the input moves on first, then
the output (a pair on two drivers opens as two streams). A refused permission is never retried,
since an operating system's privacy setting blocks every driver alike.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

from rvc_next.engine.audio_io.devices import host_api_label
from rvc_next.engine.audio_io.streams import DeviceOpenError, Endpoint, SessionConfig

T = TypeVar("T")

MAX_ATTEMPTS = 8
"""Opens tried in all, over every role: a refusal can take a while (WDM-KS), and Windows lists four drivers."""

RETRY_REASONS = frozenset({"busy", "missing", "format", "channels", "stopped"})
"""Refusals another driver may not share."""

ROLES = ("input", "output", "monitor")


def next_candidate(ep: Endpoint | None) -> Endpoint | None:
    """``ep`` through its next driver, or None when it has no other left."""
    if ep is None or not ep.candidates:
        return None
    ids = [d.get("id") for d in ep.candidates]
    at = ids.index(ep.device_id) if ep.device_id in ids else -1
    if at + 1 >= len(ep.candidates):
        return None
    return replace(ep, device=ep.candidates[at + 1])


def attempt_record(ep: Endpoint | None, error: DeviceOpenError) -> dict[str, Any]:
    """What one refused driver said, as the protocol carries it."""
    return {"device_id": ep.device_id if ep else error.device_id, "host_api": ep.host_api if ep else None, "reason": error.reason, "error": error.message}


def _endpoint(cfg: SessionConfig, role: str) -> Endpoint | None:
    return {"input": cfg.input, "output": cfg.output, "monitor": cfg.monitor}.get(role)


def open_with_fallback(
    cfg: SessionConfig, start: Callable[[SessionConfig], T], on_fallback: Callable[[str, Endpoint, DeviceOpenError, Endpoint], None] | None = None
) -> tuple[T, SessionConfig, dict[str, list[dict[str, Any]]]]:
    """Call ``start(cfg)``, moving a refusing role on to its next driver until it succeeds.

    Returns what ``start`` returned, the configuration it succeeded with and, by role, the drivers
    that refused before. ``on_fallback(role, refused, error, next)`` is told of each move. Raises the last
    ``DeviceOpenError``, its ``attempts`` holding every refusal by role, when a role has no driver
    left, the reason is one no other driver changes, or ``MAX_ATTEMPTS`` opens failed.
    """
    attempts: dict[str, list[dict[str, Any]]] = {}
    current = cfg
    for tried in range(1, MAX_ATTEMPTS + 1):
        try:
            return start(current), current, attempts
        except DeviceOpenError as e:
            role = e.role
            if role is None:
                # A duplex stream whose devices could not be told apart: move the input first.
                role = next((r for r in ("input", "output") if next_candidate(_endpoint(current, r)) is not None), None)
            ep = _endpoint(current, role) if role else None
            if role is not None and ep is not None and ep.candidates:
                attempts.setdefault(role, []).append(attempt_record(ep, e))
            following = next_candidate(ep) if e.reason in RETRY_REASONS and tried < MAX_ATTEMPTS else None
            if role is None or ep is None or following is None:
                e.attempts = attempts
                raise
            if on_fallback is not None:
                on_fallback(role, ep, e, following)
            current = replace(current, **{role: following})
    raise AssertionError("unreachable")  # the last pass returns or raises


def describe_attempts(attempts: list[dict[str, Any]], separator: str = "; ") -> str:
    """``WASAPI: <error>; DirectSound: <error>``, for a message naming every driver that refused."""
    return separator.join(f"{host_api_label(a.get('host_api'))}: {a.get('error') or a.get('reason')}" for a in attempts)


def every_driver_refused(error: DeviceOpenError) -> list[dict[str, Any]]:
    """The drivers tried for the failing role when more than one was, else empty."""
    tried = error.attempts.get(error.role or "", [])
    return tried if len(tried) > 1 else []
