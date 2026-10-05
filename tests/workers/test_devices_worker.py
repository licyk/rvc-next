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
        "default_sample_rate",
        "supported_rates",
        "latency_ms",
        "is_default",
        "is_virtual",
        "portaudio_index",
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
    assert bad == {"ok": False, "reason": "format", "message": "Speakers (Realtek(R) Audio) does not support 44100 Hz", "role": "output", "sample_rate": 44100, "topology": "split"}


def test_test_tone(tmp_path: Path) -> None:
    assert run(tmp_path, {"action": "test_tone", "backend": "fake", "device": None})["ok"]


def test_real_portaudio_enumerates(tmp_path: Path) -> None:
    """This machine has PortAudio and no devices: enumeration still succeeds."""
    data = run(tmp_path, {"action": "enumerate"})
    assert isinstance(data["inputs"], list) and data["host"]
