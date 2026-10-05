"""Device enumeration, grouping and selection.

PortAudio lists one device per host API, so the same microphone can appear four times on Windows,
with MME cutting names to 31 characters. ``group_devices`` turns the raw lists into physical
devices with full names, one variant per host API, best first; ``resolve_selection`` finds a saved
selection in a fresh list without silently swapping devices. Both are pure functions over the
dictionaries ``sounddevice.query_hostapis()`` and ``query_devices()`` return, so they are tested
against captured enumerations. ``sounddevice`` itself is imported only inside ``enumerate_raw``.
"""

from __future__ import annotations

import hashlib
import re
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

COMMON_RATES = (44100, 48000, 88200, 96000)

HOST_API_IDS: tuple[tuple[str, str], ...] = (
    ("wasapi", "wasapi"),
    ("wdm-ks", "wdm-ks"),
    ("wdmks", "wdm-ks"),
    ("directsound", "directsound"),
    ("mme", "mme"),
    ("asio", "asio"),
    ("core audio", "coreaudio"),
    ("jack", "jack"),
    ("alsa", "alsa"),
    ("pulseaudio", "pulse"),
    ("pulse", "pulse"),
    ("oss", "oss"),
)

HOST_API_RANKS: dict[str, dict[str, int]] = {
    "win32": {"asio": 0, "wasapi": 1, "wdm-ks": 2, "directsound": 3, "mme": 4},
    "darwin": {"coreaudio": 0},
    "linux": {"jack": 0, "alsa": 1, "pulse": 2, "oss": 3},
}
"""Lower is preferred. Anything unlisted ranks 9."""

LOW_LATENCY = frozenset({"asio", "wasapi", "wdm-ks", "coreaudio", "jack", "alsa"})

# Default-device aliases: "follow the system default" covers them, so they are not listed.
_ALIASES = re.compile(r"^(microsoft sound mapper - (input|output)|primary sound (capture )?driver)$", re.IGNORECASE)

_VIRTUAL = re.compile(
    r"vb-audio|\bcable (input|output)\b|voicemeeter|blackhole|\bloopback\b|soundflower|null (sink|output|input)|\bvirtual\b|vb-cable",
    re.IGNORECASE,
)

MME_NAME_LIMIT = 31


def platform_key(platform: str | None = None) -> str:
    p = platform or sys.platform
    if p.startswith("win"):
        return "win32"
    if p == "darwin":
        return "darwin"
    return "linux"


def host_api_id(name: str) -> str:
    low = name.lower()
    for needle, key in HOST_API_IDS:
        if needle in low:
            return key
    return "other"


def host_api_rank(api_id: str, platform: str | None = None) -> int:
    return HOST_API_RANKS.get(platform_key(platform), {}).get(api_id, 9)


def is_virtual(name: str) -> bool:
    """Virtual cables and loopback drivers: VB-Audio, VoiceMeeter, BlackHole, Loopback, Soundflower, null sinks."""
    return bool(_VIRTUAL.search(name or ""))


def normalize_name(name: str, host_api_name: str = "") -> str:
    """Case-fold, collapse whitespace and strip a trailing ``(<host api>)`` some drivers add."""
    text = " ".join((name or "").split()).casefold()
    if host_api_name:
        suffix = f"({host_api_name.casefold()})"
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return text


def device_id(api_id: str, raw_name: str, direction: str, ordinal: int) -> str:
    return hashlib.sha1(f"{api_id}|{raw_name}|{direction}|{ordinal}".encode()).hexdigest()[:12]


def _latency(raw: dict[str, Any], direction: str) -> tuple[float, float]:
    prefix = "input" if direction == "input" else "output"
    low = float(raw.get(f"default_low_{prefix}_latency", 0.0) or 0.0) * 1000
    high = float(raw.get(f"default_high_{prefix}_latency", 0.0) or 0.0) * 1000
    return (round(low, 2), round(high, 2))


def group_devices(raw_hostapis: list[dict[str, Any]], raw_devices: list[dict[str, Any]], platform: str | None = None, default_hostapi: int = 0) -> dict[str, Any]:
    """Group raw PortAudio devices into physical devices.

    Returns ``{"host_apis", "inputs", "outputs", "errors"}`` shaped like the core's ``DeviceList``
    without ``host`` and ``enumerated_at``. A raw device may carry ``supported_rates`` (from
    ``enumerate_raw``); without it the default rate is assumed supported. ``default_hostapi`` is
    PortAudio's default host API, whose default devices are the system defaults.
    """
    plat = platform_key(platform)
    host_apis = []
    api_ids: list[str] = []
    for api in raw_hostapis:
        aid = host_api_id(str(api.get("name", "")))
        api_ids.append(aid)
        if aid != "other" and aid in [h["id"] for h in host_apis]:
            continue
        host_apis.append({"id": aid, "name": str(api.get("name", "")), "rank": host_api_rank(aid, plat), "low_latency": aid in LOW_LATENCY})
    out: dict[str, Any] = {"host_apis": sorted(host_apis, key=lambda h: (h["rank"], h["name"])), "inputs": [], "outputs": [], "errors": []}
    for direction in ("input", "output"):
        out[f"{direction}s"] = _group_direction(raw_hostapis, raw_devices, api_ids, direction, plat, default_hostapi)
    return out


def _variants(raw_hostapis: list[dict[str, Any]], raw_devices: list[dict[str, Any]], api_ids: list[str], direction: str, plat: str, default_hostapi: int) -> list[dict[str, Any]]:
    channels_key = "max_input_channels" if direction == "input" else "max_output_channels"
    default_key = "default_input_device" if direction == "input" else "default_output_device"
    ordinals: dict[tuple[str, str], int] = {}
    variants = []
    for idx, raw in enumerate(raw_devices):
        channels = int(raw.get(channels_key, 0) or 0)
        if channels <= 0:
            continue
        hi = int(raw.get("hostapi", 0))
        api = raw_hostapis[hi] if 0 <= hi < len(raw_hostapis) else {"name": "other"}
        aid = api_ids[hi] if 0 <= hi < len(api_ids) else "other"
        raw_name = str(raw.get("name", ""))
        if _ALIASES.match(raw_name.strip()):
            continue
        ordinal = ordinals.get((aid, raw_name), 0)
        ordinals[(aid, raw_name)] = ordinal + 1
        pa_index = int(raw.get("index", idx))
        default_rate = int(round(float(raw.get("default_samplerate", 48000) or 48000)))
        rates = sorted({int(r) for r in raw.get("supported_rates", [default_rate])})
        variants.append(
            {
                "id": device_id(aid, raw_name, direction, ordinal),
                "physical_key": "",
                "name": raw_name,
                "raw_name": raw_name,
                "host_api": aid,
                "direction": direction,
                "channels": channels,
                "default_sample_rate": default_rate,
                "supported_rates": rates,
                "latency_ms": _latency(raw, direction),
                "is_default": int(api.get(default_key, -1)) == pa_index,
                "is_virtual": is_virtual(raw_name),
                "portaudio_index": pa_index,
                "_norm": normalize_name(raw_name, str(api.get("name", ""))),
                "_rank": host_api_rank(aid, plat),
                "_system_default": hi == default_hostapi and int(api.get(default_key, -1)) == pa_index,
            }
        )
    return variants


@dataclass
class _Group:
    norm: str
    apis: set[str] = field(default_factory=set)
    variants: list[dict[str, Any]] = field(default_factory=list)


def _group_direction(
    raw_hostapis: list[dict[str, Any]], raw_devices: list[dict[str, Any]], api_ids: list[str], direction: str, plat: str, default_hostapi: int
) -> list[dict[str, Any]]:
    variants = _variants(raw_hostapis, raw_devices, api_ids, direction, plat, default_hostapi)
    # Long names first, so MME's truncated names find the full name they are a prefix of.
    variants.sort(key=lambda v: (-len(v["_norm"]), v["_rank"], v["portaudio_index"]))
    groups: list[_Group] = []
    for v in variants:
        exact = [g for g in groups if g.norm == v["_norm"] and v["host_api"] not in g.apis]
        candidates = exact
        if not candidates and v["host_api"] == "mme" and len(v["raw_name"]) >= MME_NAME_LIMIT:
            candidates = [g for g in groups if g.norm.startswith(v["_norm"]) and v["host_api"] not in g.apis]
        if len(candidates) == 1:
            group = candidates[0]
        else:
            # Nothing matches, or the prefix is ambiguous: keep it apart (the safe failure).
            group = _Group(v["_norm"])
            groups.append(group)
        group.apis.add(v["host_api"])
        group.variants.append(v)
    seen_keys: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for g in groups:
        variants_sorted = sorted(g.variants, key=lambda v: (v["_rank"], v["host_api"], v["portaudio_index"]))
        longest = max(g.variants, key=lambda v: (len(v["raw_name"]), -v["_rank"]))
        key = g.norm
        n = seen_keys.get(key, 0)
        seen_keys[key] = n + 1
        if n:
            key = f"{key}#{n + 1}"
        system_default = any(v["_system_default"] for v in g.variants)
        clean = []
        for v in variants_sorted:
            item = {k: val for k, val in v.items() if not k.startswith("_")}
            item["physical_key"] = key
            item["name"] = longest["raw_name"]
            clean.append(item)
        out.append(
            {
                "key": key,
                "name": longest["raw_name"],
                "direction": direction,
                "is_default": system_default,
                "is_virtual": any(v["is_virtual"] for v in clean),
                "variants": clean,
                "recommended_id": clean[0]["id"],
            }
        )
    out.sort(key=lambda p: (not p["is_default"], str(p["name"]).casefold()))
    return out


def enumerate_raw(enable_asio: bool = False, rates: tuple[int, ...] = COMMON_RATES) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Query PortAudio once: host APIs, devices with their supported common rates, and the default host API."""
    import os

    if enable_asio:
        os.environ["SD_ENABLE_ASIO"] = "1"
    import sounddevice as sd

    hostapis = [dict(h) for h in sd.query_hostapis()]
    devices = [dict(d) for d in sd.query_devices()]
    for d in devices:
        idx = int(d.get("index", devices.index(d)))
        supported: set[int] = set()
        default_rate = int(round(float(d.get("default_samplerate", 0) or 0)))
        for rate in sorted({*rates, default_rate} - {0}):
            ok = False
            try:
                if int(d.get("max_input_channels", 0)) > 0:
                    sd.check_input_settings(device=idx, channels=1, samplerate=rate, dtype="float32")
                    ok = True
                if int(d.get("max_output_channels", 0)) > 0:
                    sd.check_output_settings(device=idx, channels=1, samplerate=rate, dtype="float32")
                    ok = True
            except Exception:
                ok = False
            if ok:
                supported.add(rate)
        d["supported_rates"] = sorted(supported or {default_rate or 48000})
    try:
        default_api = int(sd.default.hostapi)
    except Exception:
        default_api = 0
    return hostapis, devices, default_api


def device_list(
    raw_hostapis: list[dict[str, Any]], raw_devices: list[dict[str, Any]], platform: str | None = None, default_hostapi: int = 0, errors: list[str] | None = None
) -> dict[str, Any]:
    """The full ``DeviceList`` dict, with this host's name and the time."""
    grouped = group_devices(raw_hostapis, raw_devices, platform, default_hostapi)
    grouped["errors"] = list(errors or [])
    return {
        "host": socket.gethostname(),
        "enumerated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        **grouped,
    }


def _physicals(device_list: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    return list(device_list.get("inputs" if direction == "input" else "outputs", []))


def _recommended(physical: dict[str, Any]) -> dict[str, Any]:
    for v in physical["variants"]:
        if v["id"] == physical["recommended_id"]:
            return v
    return physical["variants"][0]


def system_default(device_list: dict[str, Any], direction: str) -> dict[str, Any] | None:
    """The system default device's recommended variant, else the first device, else None.

    Only devices of ``direction`` count: a list may also hold the other direction's devices
    (``live.show_all_devices``), and they are never a default.
    """
    physicals = [p for p in _physicals(device_list, direction) if p.get("direction", direction) == direction]
    for p in physicals:
        if p["is_default"]:
            return _recommended(p)
    return _recommended(physicals[0]) if physicals else None


def resolve_selection(selection: dict[str, Any] | None, device_list: dict[str, Any], direction: str) -> tuple[str, dict[str, Any] | None, str | None]:
    """Find a saved selection in a fresh device list.

    Tried in order: the same ``device_id``; the same physical key on the same host API; the same
    physical key on the recommended host API; the same name on any host API; the system default.
    Returns ``(status, device, message)`` with status ``exact``, ``matched``, ``default_fallback``
    or ``missing``. A selection with no device follows the system default and is ``exact``.
    """
    selection = selection or {}
    physicals = _physicals(device_list, direction)
    wanted_id = selection.get("device_id")
    key = selection.get("physical_key")
    name = selection.get("name")
    api = selection.get("host_api")
    if not wanted_id and not key and not name:
        default = system_default(device_list, direction)
        if default is None:
            return "missing", None, f"No {direction} device is available"
        return "exact", default, None
    if wanted_id:
        for p in physicals:
            for v in p["variants"]:
                if v["id"] == wanted_id:
                    return "exact", v, None
    if key:
        for p in physicals:
            if p["key"] != key:
                continue
            if api:
                for v in p["variants"]:
                    if v["host_api"] == api:
                        return "matched", v, None
            return "matched", _recommended(p), (f"{p['name']} is now used through {_recommended(p)['host_api']}" if api else None)
    if name:
        wanted = normalize_name(name)
        best: dict[str, Any] | None = None
        for p in physicals:
            names = {normalize_name(p["name"]), *(normalize_name(v["raw_name"]) for v in p["variants"])}
            if wanted in names or any(n.startswith(wanted) or wanted.startswith(n) for n in names if len(n) >= MME_NAME_LIMIT - 1):
                best = p
                break
        if best is not None:
            if api:
                for v in best["variants"]:
                    if v["host_api"] == api:
                        return "matched", v, None
            return "matched", _recommended(best), None
    default = system_default(device_list, direction)
    label = name or key or wanted_id
    if default is None:
        return "missing", None, f"{label} was not found, and there is no {direction} device"
    return "default_fallback", default, f"{label} was not found — using {default['name']}"
