import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rvc_next.protocol.jsonl import parse_line
from rvc_next.protocol.messages import ResultEvent

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "devices"
ROOT = Path(__file__).resolve().parents[2]


def run(tmp_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    path = tmp_path / "req.json"
    path.write_text(json.dumps(request))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-m", "rvc_next.workers.devices", str(path)], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    results = [e for e in map(parse_line, out.stdout.splitlines()) if isinstance(e, ResultEvent)]
    assert len(results) == 1
    return results[0].data


def windows() -> dict[str, Any]:
    data = json.loads((FIXTURES / "windows.json").read_text())
    return {"hostapis": data["hostapis"], "devices": data["devices"]}


def test_enumerate(tmp_path: Path) -> None:
    data = run(tmp_path, {"action": "enumerate", "backend": "fake", "platform": "win32", "enable_asio": True, "fake": windows()})
    assert set(data) == {"host", "enumerated_at", "host_apis", "inputs", "outputs", "errors"}
    assert data["errors"] == ["ASIO was requested, but no ASIO driver is installed"]
    names = [p["name"] for p in data["inputs"]]
    assert names[0] == "Microphone (USB Audio Device)"
    variant = data["inputs"][0]["variants"][0]
    assert set(variant) == {
        "id",
        "physical_key",
        "name",
        "raw_name",
        "host_api",
        "direction",
        "channels",
        "channel_counts",
        "default_sample_rate",
        "supported_rates",
        "latency_ms",
        "is_default",
        "is_virtual",
        "portaudio_index",
        "loopback_of",
        "loopback_source",
    }


def test_check_reports_a_refused_rate(tmp_path: Path) -> None:
    fake = windows()
    wasapi_out = next(d for d in fake["devices"] if d["name"] == "Speakers (Realtek(R) Audio)" and d["hostapi"] == 2)
    wasapi_in = next(d for d in fake["devices"] if d["name"] == "Microphone (USB Audio Device)" and d["hostapi"] == 2)
    ok = run(
        tmp_path,
        {
            "action": "check",
            "backend": "fake",
            "fake": fake,
            "input": {"portaudio_index": wasapi_in["index"], "host_api": "wasapi"},
            "output": {"portaudio_index": wasapi_out["index"], "host_api": "wasapi"},
            "monitor": None,
        },
    )
    assert ok["ok"] and ok["sample_rate"] == 48000 and ok["topology"] == "duplex"
    bad = run(
        tmp_path, {"action": "check", "backend": "fake", "fake": fake, "output": {"portaudio_index": wasapi_out["index"], "sample_rate": 44100}, "input": None, "monitor": None}
    )
    assert bad == {
        "ok": False,
        "reason": "format",
        "message": "Speakers (Realtek(R) Audio) does not support 44100 Hz",
        "role": "output",
        "sample_rate": 44100,
        "topology": "split",
        "opened": {},
        "attempts": {},
    }


def test_enumerate_and_check_a_loopback_input(tmp_path: Path) -> None:
    fake = windows()
    assert not any(p["is_loopback"] for p in run(tmp_path, {"action": "enumerate", "backend": "fake", "platform": "win32", "fake": fake})["inputs"])
    data = run(tmp_path, {"action": "enumerate", "backend": "fake", "platform": "win32", "loopback": True, "fake": fake})
    speakers = next(p for p in data["inputs"] if p["is_loopback"] and p["name"] == "Speakers (Realtek(R) Audio)")
    variant = speakers["variants"][0]
    assert speakers["key"] == "loopback:speakers (realtek(r) audio)" and len(speakers["variants"]) == 1
    assert variant["host_api"] == "loopback" and variant["loopback_of"] == "Speakers (Realtek(R) Audio)" and variant["portaudio_index"] < 0
    selection = {"portaudio_index": variant["portaudio_index"], "host_api": "loopback", "device_id": variant["id"], "loopback_source": variant["loopback_source"]}
    ok = run(tmp_path, {"action": "check", "backend": "fake", "fake": fake, "loopback": True, "input": selection, "output": None, "monitor": None})
    assert ok["ok"] and ok["topology"] == "input"


def test_check_opens_a_stereo_only_microphone(tmp_path: Path) -> None:
    fake = windows()
    wdmks_in = next(d for d in fake["devices"] if d["name"] == "Microphone (Realtek HD Audio Mic input)")
    wdmks_in["input_channel_counts"] = [2]
    request = {"action": "check", "backend": "fake", "fake": fake, "input": {"portaudio_index": wdmks_in["index"], "host_api": "wdm-ks"}, "output": None, "monitor": None}
    assert run(tmp_path, request)["ok"]
    variant = next(p for p in run(tmp_path, {"action": "enumerate", "backend": "fake", "platform": "win32", "fake": fake})["inputs"] if p["name"] == wdmks_in["name"])
    assert variant["variants"][0]["channel_counts"] == [2]


def test_test_tone(tmp_path: Path) -> None:
    assert run(tmp_path, {"action": "test_tone", "backend": "fake", "device": None})["ok"]


def test_real_portaudio_enumerates(tmp_path: Path) -> None:
    """This machine has PortAudio and no devices: enumeration still succeeds."""
    data = run(tmp_path, {"action": "enumerate"})
    assert isinstance(data["inputs"], list) and data["host"]


def auto_selection(listing: dict[str, Any], direction: str, name: str) -> dict[str, Any]:
    """An automatic selection as the core sends it: the first driver's index and every driver as a candidate."""
    physical = next(p for p in listing[f"{direction}s"] if p["name"] == name)
    variants = physical["variants"]
    first = variants[0]
    return {
        "physical_key": physical["key"],
        "name": physical["name"],
        "driver": "auto",
        "device_id": first["id"],
        "host_api": first["host_api"],
        "portaudio_index": first["portaudio_index"],
        "candidates": [{"device_id": v["id"], "host_api": v["host_api"], "portaudio_index": v["portaudio_index"], "loopback_source": None} for v in variants],
    }


def test_check_and_test_sound_try_the_next_driver(tmp_path: Path) -> None:
    from rvc_next.engine.audio_io.devices import group_devices

    fake = windows()
    listing = group_devices(fake["hostapis"], fake["devices"], "win32", 2)
    mic = auto_selection(listing, "input", "Microphone (Realtek(R) Audio)")
    speakers = auto_selection(listing, "output", "Speakers (Realtek(R) Audio)")
    # WASAPI's microphone stream will not start: only running it shows (the settings pass).
    fake["refuse_start"] = {"26": "Unanticipated host error [PaErrorCode -9999]"}
    request = {"action": "check", "backend": "fake", "fake": fake, "input": mic, "output": speakers, "monitor": None, "probe": ["input", "output"]}
    result = run(tmp_path, request)
    assert result["ok"] and result["topology"] == "split"
    assert result["opened"] == {"input": mic["candidates"][1]["device_id"], "output": speakers["device_id"]}
    assert [a["host_api"] for a in result["attempts"]["input"]] == ["wasapi"] and "output" not in result["attempts"]
    # Without running the input the check cannot tell, and claims no driver for it.
    assert "input" not in run(tmp_path, {**request, "probe": ["output"]})["opened"]
    # Every driver refusing.
    fake["refuse_start"] = {"26": "host error", "13": "host error", "3": "host error"}
    failed = run(tmp_path, {**request, "fake": fake})
    assert not failed["ok"] and failed["role"] == "input" and failed["reason"] == "busy"
    assert failed["message"].startswith("No driver opens it (WASAPI: Error starting stream: host error; DirectSound: ")
    # The test sound plays through the next driver of a refusing output.
    fake["refuse_start"] = {"20": "host error"}
    tone = run(tmp_path, {"action": "test_tone", "backend": "fake", "fake": fake, "device": speakers})
    assert tone["ok"] and tone["opened"] == {"output": speakers["candidates"][1]["device_id"]}
