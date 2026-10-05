"""Separation: presets and chains over pymss, one worker subprocess per job."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rvc_next.core.assets.service import AssetService
from rvc_next.core.audio.service import AudioFileService
from rvc_next.core.compute.service import ComputeService
from rvc_next.core.errors import NotFoundError, ValidationError
from rvc_next.core.jobs.models import Job, JobStep
from rvc_next.core.jobs.service import JobContext, JobService, JobSpec, WorkerExit
from rvc_next.core.separation.models import SeparateRequest, SeparationPreset
from rvc_next.core.settings import SettingsService

logger = logging.getLogger(__name__)

WORKER = "rvc_next.workers.separate"


class SeparationService:
    def __init__(self, settings: SettingsService, jobs: JobService, audio: AudioFileService, assets: AssetService, compute: ComputeService) -> None:
        self._settings = settings
        self._jobs = jobs
        self._audio = audio
        self._assets = assets
        self._compute = compute
        jobs.register("separate", self._spec)

    library: Any = None
    """Imported separation models; set by ``build_services``."""

    def presets(self) -> list[SeparationPreset]:
        """The built-in presets and chains, then one preset per imported model."""
        from rvc_next.engine.separate.presets import load_presets

        imported = self.library.presets() if self.library is not None else []
        return [SeparationPreset.model_validate(p) for p in [*load_presets(), *imported]]

    def preset(self, preset_id: str) -> SeparationPreset:
        for p in self.presets():
            if p.id == preset_id:
                return p
        raise NotFoundError(f"Unknown separation preset: {preset_id}")

    def require(self, preset_id: str) -> None:
        preset = self.preset(preset_id)
        self._assets.require(preset.assets, f"The {preset.title} preset")

    def separate(self, request: SeparateRequest, foreground: bool = False) -> Job:
        self.require(request.preset)
        self._audio.expand_inputs(request.inputs)
        return (self._jobs.run_now if foreground else self._jobs.submit)("separate", request.model_dump())

    def _spec(self, raw: dict[str, Any]) -> JobSpec:
        req = SeparateRequest.model_validate(raw)
        preset = self.preset(req.preset)
        files = self._audio.expand_inputs(req.inputs)
        steps = [JobStep(id=f"sep:{i}", title=name) for i, (_, name) in enumerate(files)]
        fmt = req.output_format or self._settings.settings.separation.output_format

        def run(ctx: JobContext) -> dict[str, Any]:
            out_dir = Path(req.output_dir).expanduser() if req.output_dir else self._audio.job_output_dir(ctx.job_id)
            results = self.run_files(ctx, files, preset.id, out_dir, fmt, register=True, step_prefix="sep")
            return {"outputs": {str(src): {k: str(v) for k, v in stems.items()} for src, stems in results.items()}}

        title = f"Separate {files[0][1]}" + (f" and {len(files) - 1} more" if len(files) > 1 else "") + f" · {preset.title}"
        return JobSpec(title=title, run=run, resources=frozenset({self.resource()}), steps=steps)

    def resource(self) -> str:
        return "cpu" if self._settings.settings.compute.device == "cpu" else "gpu"

    def run_files(
        self,
        ctx: JobContext,
        files: list[tuple[Path, str]],
        preset_id: str,
        out_dir: Path,
        fmt: str,
        register: bool,
        step_prefix: str | None,
        progress_span: tuple[float, float] = (0.0, 1.0),
    ) -> dict[Path, dict[str, Path]]:
        """Separate ``files`` with a preset in one worker; return each source's stems by label."""
        from rvc_next.engine.separate.presets import resolve

        chosen = self.preset(preset_id)
        if chosen.source == "imported":
            preset: dict[str, Any] = chosen.model_dump()
            models, missing = {preset_id: self.library.spec(preset_id)}, []
        else:
            preset, models, missing = resolve(preset_id, str(self._settings.assets_dir))
        if missing:
            self._assets.require(list(missing), "Separation")
        out_dir.mkdir(parents=True, exist_ok=True)
        c = self._settings.settings.compute
        request = {
            "inputs": [{"path": str(p), "name": name} for p, name in files],
            "preset": preset,
            "models": models,
            "output_dir": str(out_dir),
            "output_format": fmt,
            "device": c.device,
            "precision": c.precision,
            "assets_dir": str(self._settings.assets_dir),
        }
        stems: dict[Path, dict[str, Path]] = {p: {} for p, _ in files}
        by_source = {str(p): p for p, _ in files}
        lo, hi = progress_span

        def on_event(event: Any) -> None:
            from rvc_next.protocol.messages import OutputEvent, ProgressEvent, StepEvent

            if isinstance(event, ProgressEvent):
                ctx.progress(None if event.value is None else lo + (hi - lo) * event.value)
            elif isinstance(event, StepEvent):
                # Inputs are "input-<i>"; chain steps inside an input are "input-<i>/<n>" and stay internal.
                if step_prefix and event.id.startswith("input-") and "/" not in event.id and event.id[6:].isdigit():
                    ctx.step(f"{step_prefix}:{event.id[6:]}", event.state, event.progress)
            elif isinstance(event, OutputEvent):
                src = by_source.get(event.source or "")
                if src is None:
                    return
                path = Path(event.path)
                stems[src][event.label] = path
                if register:
                    self._audio.add_output(ctx.job_id, path, "stem", event.label, source_path=src)

        try:
            result = ctx.run_worker(WORKER, request, on_event=on_event)
        except WorkerExit as e:
            from rvc_next.protocol.messages import EXIT_RETRY

            if e.exit_code != EXIT_RETRY:
                raise
            ctx.log("Retrying in fp32")
            request["precision"] = "fp32"
            result = ctx.run_worker(WORKER, request, on_event=on_event)
        for failure in result.get("failed", []):
            ctx.log(f"Could not separate {Path(failure.get('path', '')).name}: {failure.get('error')}")
        if step_prefix:
            for i, (src, _) in enumerate(files):
                ctx.step(f"{step_prefix}:{i}", "done" if stems[src] else "failed")
        if not any(stems.values()):
            raise ValidationError("Separation produced no output")
        return stems
