import json
from pathlib import Path
from typing import Any

import pytest

from rvc_next.engine.audio_io.devices import group_devices, is_virtual, normalize_name, resolve_selection

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "devices"


def load(name: str) -> dict[str, Any]:
    data = json.loads((FIXTURES / f"{name}.json").read_text())
    return group_devices(data["hostapis"], data["devices"], data["platform"], data["default_hostapi"])


def by_name(device_list: dict[str, Any], direction: str) -> dict[str, dict[str, Any]]:
    return {p["name"]: p for p in device_list[f"{direction}s"]}


def variant(device_list: dict[str, Any], direction: str, name: str, api: str) -> dict[str, Any]:
    return next(v for v in by_name(device_list, direction)[name]["variants"] if v["host_api"] == api)


def test_windows_groups_one_physical_device_per_endpoint() -> None:
    dl = load("windows")
    inputs = by_name(dl, "input")
    # MME's 31-character names join the full WASAPI/DirectSound names.
    cable = inputs["CABLE Output (VB-Audio Virtual Cable)"]
    assert [v["host_api"] for v in cable["variants"]] == ["wasapi", "directsound", "mme"]
    assert cable["variants"][-1]["raw_name"] == "CABLE Output (VB-Audio Virtual "
    assert all(v["name"] == "CABLE Output (VB-Audio Virtual Cable)" for v in cable["variants"])
    assert cable["is_virtual"] and cable["recommended_id"] == cable["variants"][0]["id"]
    assert inputs["VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)"]["is_virtual"]
    # Default first, then alphabetical; the sound-mapper aliases are not listed.
    names = list(inputs)
    assert names[0] == "Microphone (USB Audio Device)" and inputs[names[0]]["is_default"]
    assert names[1:] == sorted(names[1:], key=str.casefold)
    assert not any("Sound Mapper" in n or "Primary Sound" in n for n in names)
    # WDM-KS reports different names; it stays separate rather than guessing.
    assert [v["host_api"] for v in inputs["Microphone (Realtek HD Audio Mic input)"]["variants"]] == ["wdm-ks"]
    assert [h["id"] for h in dl["host_apis"]] == ["wasapi", "wdm-ks", "directsound", "mme"]
    outputs = by_name(dl, "output")
    assert len(outputs) == 5 and outputs["Speakers (Realtek(R) Audio)"]["is_default"]
    assert variant(dl, "output", "Speakers (Realtek(R) Audio)", "wasapi")["supported_rates"] == [48000]


def test_ids_are_stable_and_distinct() -> None:
    a, b = load("windows"), load("windows")
    ids_a = [v["id"] for p in a["inputs"] + a["outputs"] for v in p["variants"]]
    ids_b = [v["id"] for p in b["inputs"] + b["outputs"] for v in p["variants"]]
    assert ids_a == ids_b and len(set(ids_a)) == len(ids_a)


def test_macos_virtual_devices_in_both_directions() -> None:
    dl = load("macos")
    assert by_name(dl, "input")["BlackHole 2ch"]["is_virtual"] and by_name(dl, "output")["BlackHole 2ch"]["is_virtual"]
    assert by_name(dl, "input")["Loopback Audio"]["is_virtual"]
    assert not by_name(dl, "output")["External Headphones"]["is_virtual"]
    assert by_name(dl, "input")["MacBook Pro Microphone"]["is_default"]
    assert dl["host_apis"][0] == {"id": "coreaudio", "name": "Core Audio", "rank": 0, "low_latency": True}


def test_linux_pipewire_keeps_every_route() -> None:
    dl = load("linux_pipewire")
    inputs = by_name(dl, "input")
    assert set(inputs) == {"default", "HDA Intel PCH: ALC892 Analog (hw:0,0)", "pipewire", "pulse", "sysdefault", "USB Audio Device: - (hw:1,0)"}
    assert inputs["default"]["is_default"] and not inputs["pipewire"]["is_virtual"]
    assert "HDA Intel PCH: HDMI 0 (hw:0,3)" in by_name(dl, "output")


@pytest.mark.parametrize(
    ("name", "virtual"),
    [("CABLE Input (VB-Audio Virtual Cable)", True), ("VoiceMeeter Aux Input", True), ("BlackHole 16ch", True), ("Null Output", True), ("Speakers (Realtek(R) Audio)", False)],
)
def test_virtual_badge(name: str, virtual: bool) -> None:
    assert is_virtual(name) is virtual


def test_normalize_strips_host_api_suffix() -> None:
    assert normalize_name("  Speakers   (Realtek)  (Windows WASAPI)", "Windows WASAPI") == "speakers (realtek)"


# -- resolution ------------------------------------------------------------------------------


def selection_for(v: dict[str, Any], **override: Any) -> dict[str, Any]:
    return {"device_id": v["id"], "physical_key": v["physical_key"], "name": v["name"], "host_api": v["host_api"], **override}


def test_resolve_exact_matched_fallback_missing() -> None:
    dl = load("windows")
    usb = variant(dl, "input", "Microphone (USB Audio Device)", "directsound")
    assert resolve_selection(selection_for(usb), dl, "input") == ("exact", usb, None)
    # The id changed (a different ordinal, say) but the physical device is on the same host API.
    status, dev, _ = resolve_selection(selection_for(usb, device_id="gone"), dl, "input")
    assert status == "matched" and dev is not None and dev["id"] == usb["id"]
    # Saved on a host API that no longer lists it: the recommended variant.
    status, dev, _ = resolve_selection(selection_for(usb, device_id="gone", host_api="asio"), dl, "input")
    assert status == "matched" and dev is not None and dev["host_api"] == "wasapi"
    # Only an MME name, cut at 31 characters, is known.
    status, dev, _ = resolve_selection({"name": "CABLE Output (VB-Audio Virtual "}, dl, "input")
    assert status == "matched" and dev is not None and dev["host_api"] == "wasapi" and dev["is_virtual"]
    # Unplugged: the system default, said out loud.
    status, dev, message = resolve_selection({"device_id": "x", "physical_key": "usb microphone (gone)", "name": "USB Microphone"}, dl, "input")
    assert status == "default_fallback" and dev is not None and dev["name"] == "Microphone (USB Audio Device)"
    assert message == "USB Microphone was not found — using Microphone (USB Audio Device)"
    # No selection follows the system default.
    assert resolve_selection(None, dl, "output")[0] == "exact"
    empty = {"inputs": [], "outputs": []}
    assert resolve_selection({"name": "x"}, empty, "input")[0] == "missing"
    assert resolve_selection(None, empty, "input")[0] == "missing"
