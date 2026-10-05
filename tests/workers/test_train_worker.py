"""The training worker as a subprocess: request file in, JSON-line events out."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from rvc_next.protocol.jsonl import parse_line
from rvc_next.protocol.messages import ErrorEvent, LogEvent, ProgressEvent, ResultEvent, StepEvent
from tests.tiny import tone


def run_worker(tmp_path: Path, request: dict, name: str) -> tuple[int, list]:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(request))
    proc = subprocess.run([sys.executable, "-m", "rvc_next.workers.train", str(path)], capture_output=True, text=True, timeout=600)
    return proc.returncode, [parse_line(line) for line in proc.stdout.splitlines()]


def test_slice_f0_features_index_through_the_worker(tmp_path: Path, tiny_assets_dir: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    sf.write(data / "a.wav", tone(5.0, 44100), 44100)
    exp = tmp_path / "exp"
    code, events = run_worker(tmp_path, {"stage": "slice", "exp_dir": str(exp), "sample_rate": "40k", "folder": str(data), "n_workers": 1}, "slice")
    assert code == 0
    assert isinstance(events[0], StepEvent) and events[0].state == "running"
    result = events[-1]
    assert isinstance(result, ResultEvent) and result.data["stage"] == "slice" and result.data["files"] == 1
    assert any(isinstance(e, ProgressEvent) and e.value == 1.0 for e in events)

    code, events = run_worker(tmp_path, {"stage": "f0", "exp_dir": str(exp), "method": "pm", "device": "cpu"}, "f0")
    assert code == 0 and events[-1].data["done"] >= 1
    code, events = run_worker(tmp_path, {"stage": "features", "exp_dir": str(exp), "version": "v1", "assets_dir": str(tiny_assets_dir), "device": "cpu"}, "features")
    assert code == 0 and events[-1].data["done"] >= 1
    code, events = run_worker(tmp_path, {"stage": "index", "exp_dir": str(exp), "version": "v1", "seed": 0}, "index")
    assert code == 0 and events[-1].data["built"] == 1


def test_missing_asset_is_a_domain_error(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    (exp / "1_16k_wavs").mkdir(parents=True)
    sf.write(exp / "1_16k_wavs" / "0_0.wav", tone(1.0), 16000)
    code, events = run_worker(tmp_path, {"stage": "features", "exp_dir": str(exp), "assets_dir": str(tmp_path / "none"), "device": "cpu"}, "features")
    assert code == 1
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert errors and errors[-1].code == "asset_missing" and errors[-1].detail["assets"] == ["hubert"]


def test_unknown_stage(tmp_path: Path) -> None:
    code, events = run_worker(tmp_path, {"stage": "nope"}, "bad")
    assert code == 1 and any(isinstance(e, ErrorEvent) for e in events)
    assert all(isinstance(e, (LogEvent, ErrorEvent, StepEvent)) for e in events)
