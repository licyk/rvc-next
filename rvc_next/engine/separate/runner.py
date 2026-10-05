"""Run a separation preset over audio files (the original ``tools/pymss_webui.py``, without its UI).

Each input is decoded once at 44.1 kHz, passed through the preset's steps (a chain feeds each
step's primary stem to the next), and the kept stems are written as ``<input>.<label>.<format>``.
Models are loaded once per job. On DirectML the first attempt runs in fp16; an fp16 failure before
any file succeeded raises ``RetryInFp32`` so the caller can start a fresh fp32 process, as the
original does with exit code 75.
"""

from __future__ import annotations

import gc
import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rvc_next.engine.cancel import CancelToken
from rvc_next.engine.errors import MissingAssetError
from rvc_next.engine.separate.presets import MODEL_SAMPLE_RATE, plan_chain

logger = logging.getLogger("rvc_next.separate")

DML_CHUNK_SIZE = 88200
DML_OVERLAP_SIZE = 22050
DEFAULT_CHUNK_SIZE = 352800
OUTPUT_FORMATS = ("wav", "flac", "mp3", "m4a")


class RetryInFp32(Exception):
    """The fp16 DirectML path failed; run the job again in a new process with ``precision = "fp32"``."""


class Reporter(Protocol):
    """Where the runner reports; the worker turns these calls into protocol events."""

    def progress(self, value: float, step: str | None, message: str | None) -> None: ...

    def step(self, id: str, state: str, progress: float | None = None, title: str | None = None) -> None: ...

    def output(self, path: str, label: str, source: str) -> None: ...

    def log(self, level: str, message: str) -> None: ...


ProgressHook = Callable[[float], None]


class Separator(Protocol):
    def separate(self, mix: np.ndarray, progress: ProgressHook) -> dict[str, np.ndarray]:
        """``mix`` is ``[C, T]`` at 44.1 kHz; returns stem name → ``[T, C]``."""
        ...

    def close(self) -> None: ...


class PymssSeparator:
    """``pymss.MSSeparator`` with the options the original passes, adapted to pymss 2.1.7.

    pymss 2.1.7 knows cpu, cuda and mps. For DirectML the model is loaded on the CPU and then moved
    (and cast to half for the fp16 attempt); demixing follows ``separator.device``.
    """

    def __init__(self, model: dict[str, Any], device: Any, kind: str, half: bool) -> None:
        from pymss import MSSeparator

        self._hook: ProgressHook | None = None
        dml = kind == "dml"
        use_amp = bool(half and kind == "cuda")
        if kind == "cuda":
            index = getattr(device, "index", None)
            pymss_device, device_ids = "cuda", [index or 0]
        elif kind == "mps":
            pymss_device, device_ids = "mps", [0]
        else:
            pymss_device, device_ids = "cpu", [0]
        self.separator = MSSeparator(
            model_type=model["model_type"],
            model_path=model["ckpt"],
            config_path=model["config"],
            device=pymss_device,
            device_ids=device_ids,
            output_format="wav",
            use_tta=False,
            store_dirs={},
            logger=logging.getLogger("rvc_next.separate.pymss"),
            debug=False,
            progress_callback=self._on_progress,
            inference_params={
                "batch_size": 1 if dml else int(model["batch_size"]),
                "chunk_size": DML_CHUNK_SIZE if dml else int(model.get("chunk_size") or DEFAULT_CHUNK_SIZE),
                "overlap_size": DML_OVERLAP_SIZE if dml else int(model["overlap_size"]),
                "standardize": False,
                "normalize": False,
                "use_amp": use_amp,
            },
        )
        config: Any = self.separator.config
        config.training.use_amp = use_amp
        if dml:
            net: Any = self.separator.model
            net = net.to(device)
            if half:
                net = net.half()
            self.separator.model = net
            self.separator.device = device

    def _on_progress(self, done: float, total: float, message: str | None = None) -> None:
        if self._hook is not None and total:
            self._hook(min(1.0, max(0.0, float(done) / float(total))))

    def separate(self, mix: np.ndarray, progress: ProgressHook) -> dict[str, np.ndarray]:
        self._hook = progress
        try:
            return self.separator.separate(mix, pbar=False)
        finally:
            self._hook = None

    def close(self) -> None:
        try:
            self.separator.close()
        except Exception:
            logger.exception("Failed to close the separator")


class FakeSeparator:
    """For tests: halves the input into the two stems, with optional delay and an fp16 failure."""

    def __init__(self, model: dict[str, Any], half: bool, options: dict[str, Any]) -> None:
        self.model = model
        self.half = half
        self.delay = float(options.get("delay", 0.0))
        self.chunks = max(1, int(options.get("chunks", 4)))
        self.fail_fp16 = bool(options.get("fail_fp16", False))

    def separate(self, mix: np.ndarray, progress: ProgressHook) -> dict[str, np.ndarray]:
        for k in range(self.chunks):
            if self.delay:
                time.sleep(self.delay / self.chunks)
            if self.fail_fp16 and self.half:
                raise RuntimeError("fp16 is not supported by this operator")
            progress((k + 1) / self.chunks)
        stem = np.ascontiguousarray((mix * 0.5).T, dtype=np.float32)
        return {self.model["primary"]: stem, self.model["secondary"]: stem.copy()}

    def close(self) -> None:
        pass


def write_stem(path: Path, audio: np.ndarray, sample_rate: int, fmt: str) -> None:
    """Write ``[T, C]`` float audio: WAV as 32-bit float, FLAC as 24-bit, mp3/m4a through the engine encoder."""
    if fmt in ("wav", "flac"):
        import soundfile as sf

        tmp = path.with_name(f".{path.name}.part")
        try:
            sf.write(str(tmp), audio, sample_rate, format=fmt.upper(), subtype="FLOAT" if fmt == "wav" else "PCM_24")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        return
    from rvc_next.engine.audio.io import encode

    encode(path, np.ascontiguousarray(audio.T), sample_rate, fmt)


def _retryable(error: BaseException) -> bool:
    message = str(error).lower()
    oom = any(m in message for m in ("out of memory", "e_outofmemory", "not enough memory", "failed to allocate", "allocation failed"))
    return isinstance(error, (RuntimeError, NotImplementedError, FloatingPointError, TypeError, ValueError)) and not oom


@dataclass
class JobResult:
    outputs: list[dict[str, str]] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    processed: int = 0
    device: str = "cpu"
    precision: str = "fp32"

    def to_dict(self) -> dict[str, Any]:
        return {"outputs": self.outputs, "failed": self.failed, "processed": self.processed, "device": self.device, "precision": self.precision}


def run_separation(request: dict[str, Any], reporter: Reporter, cancel: CancelToken | None = None) -> JobResult:
    """Separate every input of ``request`` (the worker request; see ``workers/separate.py``)."""
    from rvc_next.engine.audio.io import decode, probe
    from rvc_next.engine.separate.presets import output_name

    preset: dict[str, Any] = request["preset"]
    models: dict[str, dict[str, Any]] = request["models"]
    inputs: list[dict[str, Any]] = request["inputs"]
    fmt = str(request.get("output_format") or "flac").lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {fmt}")
    out_dir = Path(request["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fake: dict[str, Any] | None = request.get("test_fake")
    plan = plan_chain(preset, models)
    chain = preset.get("kind") == "chain"

    if fake is None:
        missing = sorted({m["asset_id"] for m in models.values() if not Path(m["ckpt"]).is_file()})
        if missing:
            raise MissingAssetError(f"Separation weights are not installed: {', '.join(missing)}", missing)

    precision = str(request.get("precision") or "auto")
    if fake is not None:
        device, kind, fp16 = "cpu", "dml" if fake.get("pretend_dml") else "cpu", False
    else:
        from rvc_next.engine.runtime import choose_device

        choice = choose_device(str(request.get("device") or "auto"), precision)
        device, kind, fp16 = choice.device, choice.kind, choice.fp16
    dml = kind == "dml"
    # DirectML tries fp16 first unless fp32 was asked for (the original's "auto" model dtype).
    half = fp16 or (dml and precision != "fp32")
    allow_retry = dml and half and bool(request.get("allow_fp32_retry", True))
    result = JobResult(device="dml" if dml else str(device), precision="fp16" if half else "fp32")
    reporter.log("info", f"Separating {len(inputs)} file(s) with {preset['id']} on {result.device}, {result.precision}")

    durations: list[float] = []
    for item in inputs:
        try:
            durations.append(max(probe(item["path"]).duration, 0.0))
        except Exception:
            durations.append(0.0)
    total = max(sum(durations) * len(plan), 1e-9)
    done_seconds = 0.0
    last_emit = [0.0]

    def emit_progress(seconds: float, step: str, message: str | None, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - last_emit[0] >= 0.1:
            last_emit[0] = now
            reporter.progress(min(1.0, seconds / total), step, message)

    separators: dict[str, Separator] = {}

    def separator_for(model_id: str) -> Separator:
        if model_id not in separators:
            model = models[model_id]
            reporter.log("info", f"Loading {model.get('title', model_id)}")
            separators[model_id] = FakeSeparator(model, half, fake) if fake is not None else PymssSeparator(model, device, kind, half)
        return separators[model_id]

    names: set[str] = set()
    successes = 0
    try:
        for i, item in enumerate(inputs):
            if cancel:
                cancel.check()
            path = Path(item["path"])
            name = str(item.get("name") or path.name)
            stem = Path(name).stem
            step_id = f"input-{i}"
            duration = durations[i]
            input_start = done_seconds
            reporter.step(step_id, "running", 0.0, name)
            emit_progress(done_seconds, step_id, name, force=True)
            written: list[dict[str, str]] = []
            try:
                mix, _ = decode(path, MODEL_SAMPLE_RATE, mono=False)
                current = mix
                for ps in plan:
                    if cancel:
                        cancel.check()
                    sub_id = f"{step_id}/{ps.index}"
                    if chain:
                        reporter.step(sub_id, "running", 0.0, models[ps.model_id].get("title", ps.model_id))
                    base = done_seconds

                    def hook(fraction: float, base: float = base, span: float = duration, at: str = sub_id if chain else step_id, label: str = name) -> None:
                        emit_progress(base + fraction * span, at, label)

                    results = separator_for(ps.model_id).separate(current, hook)
                    spec = models[ps.model_id]
                    for needed in {spec["primary"], spec["secondary"]}:
                        if needed not in results:
                            raise RuntimeError(f"The model returned no '{needed}' stem")
                        if not np.isfinite(np.asarray(results[needed])).all():
                            raise FloatingPointError(f"The model returned NaN or Inf in '{needed}'")
                    for model_stem, label in ps.keep:
                        filename = output_name(stem, label, fmt, names, out_dir)
                        names.add(filename)
                        target = out_dir / filename
                        write_stem(target, np.asarray(results[model_stem], dtype=np.float32), MODEL_SAMPLE_RATE, fmt)
                        written.append({"path": str(target), "label": label, "source": str(path)})
                        reporter.output(str(target), label, str(path))
                    if ps.feeds_next is not None:
                        current = np.ascontiguousarray(np.asarray(results[ps.feeds_next], dtype=np.float32).T)
                    del results
                    done_seconds = base + duration
                    if chain:
                        reporter.step(sub_id, "done", 1.0)
                successes += 1
                result.outputs.extend(written)
                reporter.step(step_id, "done", 1.0)
            except (MissingAssetError, KeyboardInterrupt):
                raise
            except Exception as error:
                if type(error).__name__ == "Cancelled":
                    raise
                if allow_retry and successes == 0 and _retryable(error):
                    reporter.log("warning", f"The DirectML fp16 path failed ({error}); retrying in fp32")
                    raise RetryInFp32(str(error)) from error
                reporter.log("error", f"{name}: {error}\n{traceback.format_exc()}")
                reporter.step(step_id, "failed")
                result.failed.append({"path": str(path), "error": str(error) or type(error).__name__})
                done_seconds = input_start + duration * len(plan)
            result.processed += 1
            emit_progress(done_seconds, step_id, name, force=True)
    finally:
        for sep in separators.values():
            sep.close()
        separators.clear()
        gc.collect()
        _empty_cache()
    if successes == 0 and result.failed:
        raise RuntimeError(f"Separation failed: {result.failed[0]['error']}")
    reporter.progress(1.0, None, None)
    return result


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
