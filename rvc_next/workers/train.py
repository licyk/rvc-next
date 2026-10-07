"""Training worker: one stage per process.

    python -m rvc_next.workers.train <request.json>

The request is ``{"stage": "slice" | "f0" | "features" | "fit" | "index", ...}`` plus the stage's
request fields (see ``engine/train`` and ``engine/index/build.py``), and optionally ``assets_dir``,
``device`` (auto, cpu, cuda:N, xpu:N, dml) and ``precision`` (auto, fp32, fp16). Events go to stdout as
JSON lines; the stage's result dict is the final ``result`` event. Cancel is a process-tree kill.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from rvc_next.protocol.jsonl import Emitter
from rvc_next.protocol.messages import LogEvent, MetricEvent, OutputEvent, ProgressEvent, ResultEvent, StepEvent
from rvc_next.workers.base import run_worker

STAGES = ("slice", "f0", "features", "fit", "index")
SAMPLE_RATES = {"32k": 32000, "40k": 40000, "48k": 48000}


def build_request(cls: Any, data: dict[str, Any], **overrides: Any) -> Any:
    """A frozen request dataclass from JSON: unknown keys dropped, lists turned into tuples."""
    names = {f.name for f in dataclasses.fields(cls)}
    values: dict[str, Any] = {}
    for key, value in {**data, **overrides}.items():
        if key in names:
            values[key] = tuple(value) if isinstance(value, list) and key != "config_override" else value
    return cls(**values)


def resolve_device(data: dict[str, Any]) -> tuple[str, bool]:
    """The device string and half precision for F0 and features, by the runtime's GPU rule."""
    from rvc_next.engine.runtime import choose_device, gpu_backend

    requested = str(data.get("device") or "auto")
    gpus = list(data.get("gpus") or [])
    if requested == "auto" and len(gpus) == 1:
        requested = f"{gpu_backend() or 'cuda'}:{gpus[0]}"
    choice = choose_device(requested, str(data.get("precision") or "auto"))
    return choice.id, choice.fp16


def main(request: dict[str, Any], emitter: Emitter) -> None:
    stage = request.get("stage")
    if stage not in STAGES:
        raise ValueError(f"Unknown training stage: {stage!r}")
    last = [0.0]

    def progress(value: float | None, step: str | None = None) -> None:
        now = time.monotonic()
        # At most four updates a second; the first and the last always go out.
        if value is None or value >= 1.0 or now - last[0] >= 0.25:
            last[0] = now
            emitter.emit(ProgressEvent(value=value, step=stage, message=step))

    def log(message: str) -> None:
        emitter.emit(LogEvent(level="info", message=message))

    emitter.emit(StepEvent(id=stage, state="running"))
    result = _run(stage, request, progress, log, emitter)
    emitter.emit(StepEvent(id=stage, state="done", progress=1.0))
    emitter.emit(ResultEvent(data={"stage": stage, **result}))


def _run(stage: str, request: dict[str, Any], progress: Any, log: Any, emitter: Emitter) -> dict[str, Any]:
    assets = str(request.get("assets_dir") or "")
    if stage == "slice":
        from rvc_next.engine.train import preprocess

        rate = request.get("sample_rate", 40000)
        speakers = tuple(preprocess.SpeakerSpec(**s) for s in request.get("speakers") or [])
        req = build_request(preprocess.SliceRequest, {k: v for k, v in request.items() if k != "speakers"}, sample_rate=SAMPLE_RATES.get(rate, rate), speakers=speakers)
        return preprocess.run(req, progress, log)
    if stage == "f0":
        from rvc_next.engine.train import f0_extract

        device, half = resolve_device(request)
        return f0_extract.run(build_request(f0_extract.F0Request, request, assets_dir=assets, device=device, is_half=half), progress, log)
    if stage == "features":
        from rvc_next.engine.train import feature_extract

        device, half = resolve_device(request)
        return feature_extract.run(build_request(feature_extract.FeatureRequest, request, assets_dir=assets, device=device, is_half=half), progress, log)
    if stage == "fit":
        from rvc_next.engine.train import loop

        def metric(item: dict[str, Any]) -> None:
            emitter.emit(MetricEvent(epoch=int(item["epoch"]), step=int(item["step"]), values=dict(item["losses"]), lr=item.get("lr")))

        def output(path: str, kind: str, label: str = "", source: str | None = None) -> None:
            if kind == "preview":
                emitter.emit(OutputEvent(path=path, label=label, kind="preview", source=source))
            else:
                emitter.emit(OutputEvent(path=path, label=kind, kind="small" if kind == "small" else "checkpoint"))

        return loop.run(build_request(loop.TrainRequest, request), progress, log, None, metric, output)
    from rvc_next.engine.index import build

    return build.run(build_request(build.IndexRequest, request), progress, log)


if __name__ == "__main__":
    raise SystemExit(run_worker(main))
