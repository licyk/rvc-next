"""Conversion: one job per request, a step per input and per pipeline stage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from rvc_next.core.assets.service import AssetService
from rvc_next.core.audio.service import AudioFileService
from rvc_next.core.compute.service import ComputeService
from rvc_next.core.conversion.models import ConvertRequest
from rvc_next.core.errors import ValidationError
from rvc_next.core.files import slugify
from rvc_next.core.jobs.models import Job, JobStep
from rvc_next.core.jobs.service import JobContext, JobService, JobSpec
from rvc_next.core.models.models import VoiceModel
from rvc_next.core.models.service import VoiceModelService
from rvc_next.core.params import f0_assets
from rvc_next.core.separation.service import SeparationService
from rvc_next.core.settings import SettingsService

logger = logging.getLogger(__name__)

PREVIEW_PRIORITY = 10


class ConversionService:
    def __init__(
        self,
        settings: SettingsService,
        jobs: JobService,
        audio: AudioFileService,
        models: VoiceModelService,
        assets: AssetService,
        compute: ComputeService,
        separation: SeparationService,
    ) -> None:
        self._settings = settings
        self._jobs = jobs
        self._audio = audio
        self._models = models
        self._assets = assets
        self._compute = compute
        self._separation = separation
        jobs.register("convert", self._spec)

    def required_assets(self, voice: VoiceModel, f0_method: str) -> list[str]:
        from rvc_next.engine.features.embedders import asset_id

        ids = [asset_id(voice.embedder)]
        if voice.pitch_guidance:
            ids += f0_assets(f0_method, self._settings.settings.compute.device == "dml")
        return ids

    def validate(self, request: ConvertRequest) -> VoiceModel:
        voice = self._models.get(request.voice_id)
        if request.params.speaker_id >= max(voice.speaker_slots, 1):
            raise ValidationError(f"Speaker {request.params.speaker_id} is outside this voice's 0–{voice.speaker_slots - 1}")
        self._assets.require(self.required_assets(voice, request.params.f0_method), f"Converting with {voice.name}")
        if request.separate:
            self._separation.require(request.separate.preset)
        if request.remix and not request.separate:
            raise ValidationError("Adding the accompaniment back needs a separation step")
        return voice

    def convert(self, request: ConvertRequest, foreground: bool = False) -> Job:
        self.validate(request)
        self._audio.expand_inputs(request.inputs, request.recursive)
        return (self._jobs.run_now if foreground else self._jobs.submit)("convert", request.model_dump())

    def _spec(self, raw: dict[str, Any]) -> JobSpec:
        req = ConvertRequest.model_validate(raw)
        voice = self._models.get(req.voice_id)
        files = self._audio.expand_inputs(req.inputs, req.recursive)
        steps: list[JobStep] = []
        if req.separate:
            steps.append(JobStep(id="separate", title="Separate"))
        for i, (_, name) in enumerate(files):
            steps.append(JobStep(id=f"conv:{i}", title=name))
        preview = req.preview_seconds is not None
        what = files[0][1] + (f" and {len(files) - 1} more" if len(files) > 1 else "")
        title = f"{'Preview' if preview else 'Convert'} {what} · {voice.name}"

        def run(ctx: JobContext) -> dict[str, Any]:
            return self._run(ctx, req, voice, files)

        resource = "cpu" if self._settings.settings.compute.device == "cpu" else "gpu"
        return JobSpec(title=title, run=run, resources=frozenset({resource}), steps=steps, priority=PREVIEW_PRIORITY if preview else 0)

    def _run(self, ctx: JobContext, req: ConvertRequest, voice: VoiceModel, files: list[tuple[Path, str]]) -> dict[str, Any]:
        from rvc_next.engine.audio.io import decode, encode
        from rvc_next.engine.convert.offline import OfflineConverter

        self._assets.require(self.required_assets(voice, req.params.f0_method), f"Converting with {voice.name}")
        fmt = req.output_format or self._settings.settings.convert.output_format
        out_dir = Path(req.output_dir).expanduser() if req.output_dir else self._audio.job_output_dir(ctx.job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        preview = req.preview_seconds
        n = len(files)
        sep_share = 0.4 if req.separate else 0.0

        # Separate first, in one worker for every input, then convert each vocal stem.
        stems: dict[Path, dict[str, Path]] = {}
        if req.separate:
            ctx.step("separate", "running")
            sep_dir = out_dir / "stems"
            stems = self._separation.run_files(ctx, files, req.separate.preset, sep_dir, "wav", register=False, step_prefix=None, progress_span=(0.0, sep_share))
            ctx.step("separate", "done")
        primary = self._separation.preset(req.separate.preset).primary_stem if req.separate else None

        runtime = self._compute.runtime()
        loaded = runtime.voice(voice.model_path)
        params = req.params.to_engine(voice.provenance.pitch_median)
        index_path = self._models.index_for(voice, req.params.speaker_id)
        index = runtime.index(index_path) if index_path and req.params.index_rate > 0 else None
        if index is None and req.params.index_rate > 0:
            ctx.log("No index for this voice; index strength has no effect")
        converter = OfflineConverter(runtime)
        written: list[str] = []
        voice_slug = slugify(voice.name)
        for i, (src, name) in enumerate(files):
            ctx.check_cancel()
            step = f"conv:{i}"
            ctx.step(step, "running")
            source = stems.get(src, {}).get(primary or "", src) if req.separate else src
            ctx.log(f"Converting {name}")
            audio, _ = decode(source, 16000, mono=True, max_seconds=preview)
            base = sep_share + (1 - sep_share) * i / n

            def on_progress(p: float, base: float = base, step: str = step) -> None:
                ctx.progress(base + (1 - sep_share) * p / n, step)

            out, sr = converter.convert(
                audio, loaded, params, index=index, resample_to=req.resample_to, progress=on_progress, cancel=ctx.cancel_token(), denoise=req.output_denoise
            )
            if params.auto_pitch != "off" and voice.pitch_guidance:
                ctx.log(f"Automatic key: {converter.last_auto_key:+d} semitones (towards {params.auto_pitch_target:.0f} Hz)")
            stem = Path(name).stem
            kind = "preview" if preview else "converted"
            if req.remix and req.separate:
                accompaniment = self._accompaniment(stems.get(src, {}), primary)
                if accompaniment is not None:
                    out = self._remix(out, sr, accompaniment, req.remix.gain_db, req.remix.vocal_gain_db, preview)
                    kind = "preview" if preview else "remix"
            target = self._target(out_dir, stem, voice_slug + (".preview" if preview else ""), fmt, req)
            encode(target, out, sr, fmt)
            written.append(str(target))
            self._audio.add_output(ctx.job_id, target, kind, "remix" if kind == "remix" else "converted", source_path=src, model_id=voice.id, voice=req.params)
            ctx.step(step, "done")
        if req.separate:
            # The stems were a means; Separate keeps them when they are wanted.
            import shutil

            shutil.rmtree(out_dir / "stems", ignore_errors=True)
        return {"outputs": written}

    def _target(self, folder: Path, stem: str, suffix: str, fmt: str, req: ConvertRequest) -> Path:
        if req.output_dir and req.overwrite:
            return folder / f"{stem}.{suffix}.{fmt}"
        return self._audio.output_path(folder, stem, suffix, fmt)

    @staticmethod
    def _accompaniment(stems: dict[str, Path], primary: str | None) -> Path | None:
        for label in ("instrumental", "accompaniment", "other", "no_vocals"):
            if label in stems:
                return stems[label]
        rest = [p for k, p in stems.items() if k != primary]
        return rest[0] if rest else None

    @staticmethod
    def _remix(vocal_int16: np.ndarray, sr: int, accompaniment: Path, gain_db: float, vocal_gain_db: float, preview: float | None) -> np.ndarray:
        """Mix the converted vocal over the accompaniment at the vocal's rate; return stereo ``[2, T]`` float."""
        from rvc_next.engine.audio.dsp import db_to_gain
        from rvc_next.engine.audio.io import decode

        vocal = vocal_int16.astype(np.float32) / 32768.0 * db_to_gain(vocal_gain_db)
        acc, _ = decode(accompaniment, sr, mono=False, max_seconds=preview)
        if acc.shape[0] == 1:
            acc = np.repeat(acc, 2, axis=0)
        acc = acc[:2] * db_to_gain(gain_db)
        length = max(acc.shape[1], vocal.shape[0])
        mix = np.zeros((2, length), dtype=np.float32)
        mix[:, : acc.shape[1]] += acc
        mix[:, : vocal.shape[0]] += vocal[None, :]
        peak = float(np.abs(mix).max()) if mix.size else 0.0
        if peak > 0.99:
            mix *= 0.99 / peak
        return mix
