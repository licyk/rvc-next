"""Separation worker: ``python -m rvc_next.workers.separate <request.json>``.

The request::

    {
      "inputs": [{"path": "/abs/song.mp3", "name": "song.mp3"}],      # name: display name, also the output stem
      "preset": {...},                       # one entry of core/separation/presets.json
      "models": {"vocals": {...}},           # engine.separate.presets.resolve_model() for every preset step
      "output_dir": "/abs/out",
      "output_format": "flac",               # wav | flac | mp3 | m4a | ogg
      "device": "auto",                      # compute.device: auto | cpu | cuda:N | xpu:N | dml | mps
      "precision": "auto",                   # auto | fp16 | fp32; the runner re-runs with fp32 after exit 75
      "allow_fp32_retry": true,              # DirectML only
      "assets_dir": "/abs/assets"            # informational; the model paths are already resolved
    }

Events: progress (seconds of audio done over the whole job), a step per input (``input-<i>``) and,
for chains, per chain step (``input-<i>/<n>``), an output per stem written, then the result
``{outputs: [{path, label, source}], failed: [{path, error}], processed, device, precision}``.
Exit codes: 0 done, 1 failed (an error event says why), 75 retry in fp32, 130 cancelled.
"""

from __future__ import annotations

import sys
from typing import Any

from rvc_next.engine.separate.runner import RetryInFp32, run_separation
from rvc_next.protocol.jsonl import Emitter
from rvc_next.protocol.messages import EXIT_RETRY, LogEvent, OutputEvent, ProgressEvent, ResultEvent, StepEvent
from rvc_next.workers.base import run_worker


class _EventReporter:
    def __init__(self, emitter: Emitter) -> None:
        self.emitter = emitter

    def progress(self, value: float, step: str | None, message: str | None) -> None:
        self.emitter.emit(ProgressEvent(value=round(value, 4), step=step, message=message))

    def step(self, id: str, state: str, progress: float | None = None, title: str | None = None) -> None:
        self.emitter.emit(StepEvent(id=id, state=state, progress=progress, title=title))

    def output(self, path: str, label: str, source: str) -> None:
        self.emitter.emit(OutputEvent(path=path, label=label, kind="stem", source=source))

    def log(self, level: str, message: str) -> None:
        self.emitter.emit(LogEvent(level=level, message=message))


def check(request: dict[str, Any], emitter: Emitter) -> None:
    """Load an imported model once on the CPU, so a wrong checkpoint/YAML pair fails at import."""
    import torch

    from rvc_next.engine.separate.runner import PymssSeparator, ensure_plain_checkpoint

    model = request["model"]
    ensure_plain_checkpoint(model["ckpt"], model["model_type"])
    PymssSeparator(model, torch.device("cpu"), "cpu", False).close()
    emitter.emit(ResultEvent(data={"ok": True}))


def main(request: dict[str, Any], emitter: Emitter) -> None:
    if request.get("action") == "check":
        check(request, emitter)
        return
    try:
        result = run_separation(request, _EventReporter(emitter))
    except RetryInFp32 as e:
        emitter.emit(LogEvent(level="warning", message=f"Retrying in fp32: {e}"))
        sys.exit(EXIT_RETRY)
    emitter.emit(ResultEvent(data=result.to_dict()))


if __name__ == "__main__":
    raise SystemExit(run_worker(main))
