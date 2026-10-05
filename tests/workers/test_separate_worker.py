"""The separation worker as a subprocess, with the fake separator."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rvc_next.engine.separate.presets import resolve
from rvc_next.protocol.jsonl import parse_line
from rvc_next.protocol.messages import EXIT_RETRY, ErrorEvent, OutputEvent, ProgressEvent, ResultEvent, StepEvent


def _clip(path: Path, seconds: float = 1.0) -> Path:
    t = np.arange(int(44100 * seconds)) / 44100
    sf.write(str(path), np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)], axis=1).astype(np.float32) * 0.3, 44100)
    return path


def _request(tmp_path: Path, preset_id: str, inputs: list[Path], fake: dict[str, Any] | None = None, **extra: Any) -> Path:
    preset, models, _ = resolve(preset_id, tmp_path / "assets")
    request = {
        "inputs": [{"path": str(p), "name": p.name} for p in inputs],
        "preset": preset,
        "models": models,
        "output_dir": str(tmp_path / "out"),
        "output_format": "flac",
        "device": "cpu",
        "precision": "auto",
        "assets_dir": str(tmp_path / "assets"),
        "test_fake": fake if fake is not None else {},
        **extra,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    return path


def _run(request: Path, timeout: float = 120) -> tuple[int, list]:
    proc = subprocess.run([sys.executable, "-m", "rvc_next.workers.separate", str(request)], capture_output=True, text=True, timeout=timeout)
    return proc.returncode, [parse_line(line) for line in proc.stdout.splitlines() if line.strip()]


def test_chain_event_sequence(tmp_path: Path) -> None:
    a = _clip(tmp_path / "a.wav")
    b = _clip(tmp_path / "b.wav", 0.5)
    code, events = _run(_request(tmp_path, "vocals-clean", [a, b]))
    assert code == 0, events
    steps = [(e.id, e.state) for e in events if isinstance(e, StepEvent)]
    assert steps == [
        ("input-0", "running"),
        ("input-0/0", "running"),
        ("input-0/0", "done"),
        ("input-0/1", "running"),
        ("input-0/1", "done"),
        ("input-0", "done"),
        ("input-1", "running"),
        ("input-1/0", "running"),
        ("input-1/0", "done"),
        ("input-1/1", "running"),
        ("input-1/1", "done"),
        ("input-1", "done"),
    ]
    outputs = [(Path(e.path).name, e.label, e.kind, Path(e.source or "").name) for e in events if isinstance(e, OutputEvent)]
    assert outputs == [
        ("a.instrumental.flac", "instrumental", "stem", "a.wav"),
        ("a.vocals.flac", "vocals", "stem", "a.wav"),
        ("a.reverb.flac", "reverb", "stem", "a.wav"),
        ("b.instrumental.flac", "instrumental", "stem", "b.wav"),
        ("b.vocals.flac", "vocals", "stem", "b.wav"),
        ("b.reverb.flac", "reverb", "stem", "b.wav"),
    ]
    progress = [e.value for e in events if isinstance(e, ProgressEvent)]
    assert progress == sorted(progress) and progress[-1] == 1.0
    result = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result) == 1 and result[0].data["processed"] == 2 and len(result[0].data["outputs"]) == 6 and result[0].data["failed"] == []
    # The chain's vocals passed through two halvings: a quarter of the input.
    vocals, sr = sf.read(str(tmp_path / "out" / "a.vocals.flac"))
    source, _ = sf.read(str(a))
    assert sr == 44100 and vocals.shape == source.shape
    assert np.allclose(vocals, source * 0.25, atol=1e-3)


def test_outputs_get_a_suffix_instead_of_overwriting(tmp_path: Path) -> None:
    a = _clip(tmp_path / "a.wav")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a.vocals.flac").write_bytes(b"keep")
    code, events = _run(_request(tmp_path, "vocals", [a]))
    assert code == 0
    names = sorted(Path(e.path).name for e in events if isinstance(e, OutputEvent))
    assert names == ["a.instrumental.flac", "a.vocals-2.flac"]
    assert (tmp_path / "out" / "a.vocals.flac").read_bytes() == b"keep"


def test_unreadable_input_fails_alone(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    good = _clip(tmp_path / "good.wav")
    code, events = _run(_request(tmp_path, "vocals", [bad, good]))
    assert code == 0
    assert ("input-0", "failed") in [(e.id, e.state) for e in events if isinstance(e, StepEvent)]
    result = next(e for e in events if isinstance(e, ResultEvent)).data
    assert [Path(f["path"]).name for f in result["failed"]] == ["bad.wav"] and result["processed"] == 2


def test_all_inputs_failing_is_an_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    code, events = _run(_request(tmp_path, "vocals", [bad]))
    assert code == 1
    assert any(isinstance(e, ErrorEvent) for e in events)


def test_missing_weights_report_asset_ids(tmp_path: Path) -> None:
    a = _clip(tmp_path / "a.wav")
    request = _request(tmp_path, "lead-clean", [a])
    data = json.loads(request.read_text())
    del data["test_fake"]
    request.write_text(json.dumps(data))
    code, events = _run(request)
    assert code == 1
    error = next(e for e in events if isinstance(e, ErrorEvent))
    assert error.code == "asset_missing"
    assert error.detail["assets"] == ["separation-dereverb", "separation-karaoke", "separation-vocals"]


def test_directml_fp16_failure_asks_for_fp32_retry(tmp_path: Path) -> None:
    a = _clip(tmp_path / "a.wav")
    fake = {"pretend_dml": True, "fail_fp16": True}
    code, events = _run(_request(tmp_path, "vocals", [a], fake))
    assert code == EXIT_RETRY
    assert not any(isinstance(e, ResultEvent) for e in events)
    code, events = _run(_request(tmp_path, "vocals", [a], fake, precision="fp32"))
    assert code == 0
    assert next(e for e in events if isinstance(e, ResultEvent)).data["precision"] == "fp32"


def test_cancel_by_killing_the_process(tmp_path: Path) -> None:
    a = _clip(tmp_path / "a.wav")
    request = _request(tmp_path, "vocals", [a], {"delay": 30, "chunks": 30})
    proc = subprocess.Popen([sys.executable, "-m", "rvc_next.workers.separate", str(request)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    deadline = time.monotonic() + 60
    started = False
    while time.monotonic() < deadline:
        event = parse_line(proc.stdout.readline())
        if isinstance(event, ProgressEvent) and (event.value or 0) > 0:
            started = True
            break
    assert started
    proc.kill()
    proc.wait(timeout=10)
    assert proc.returncode != 0
    assert not list((tmp_path / "out").glob("*.flac"))
