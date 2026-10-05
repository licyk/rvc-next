"""Training.

An experiment is a folder with ``experiment.json`` and the original's stage folders, so a folder
copied in from elsewhere just appears. Each stage records a fingerprint of its inputs; **Run all**
runs the stages that never finished or whose inputs changed. Every stage runs in the training
worker, one process per stage, so cancel is a process-tree kill and CUDA starts clean.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

from rvc_next.core.assets.service import AssetService
from rvc_next.core.clock import now_iso
from rvc_next.core.compute.service import ComputeService
from rvc_next.core.db import Database
from rvc_next.core.errors import BusyError, ConflictError, NotFoundError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import ExperimentChangedEvent, TrainMetricsEvent
from rvc_next.core.files import trash
from rvc_next.core.jobs.models import Job, JobStep
from rvc_next.core.jobs.service import JobContext, JobService, JobSpec
from rvc_next.core.models.models import VoiceModel
from rvc_next.core.models.service import VoiceModelService
from rvc_next.core.safety import validate_name
from rvc_next.core.separation.service import SeparationService
from rvc_next.core.settings import SettingsService
from rvc_next.core.training.models import (
    STAGE_ORDER,
    Checkpoint,
    Dataset,
    DatasetReport,
    Experiment,
    ExperimentCreate,
    ExperimentSummary,
    ExperimentUpdate,
    ExportRequest,
    FitSettings,
    ImportExperimentRequest,
    RunRequest,
    SpeakerEntry,
    StageState,
    TrainMetric,
)

logger = logging.getLogger(__name__)

EXPERIMENT_FILE = "experiment.json"
WORKER = "rvc_next.workers.train"
CLEAN_DIR = "dataset_clean"
STAGE_TITLES = {"clean": "Clean up the dataset", "slice": "Slice", "f0": "Extract pitch", "features": "Extract features", "fit": "Train", "index": "Build the index"}
# How much of a full run each stage takes, for one progress bar.
STAGE_WEIGHT = {"clean": 2.0, "slice": 0.5, "f0": 1.0, "features": 1.0, "fit": 10.0, "index": 0.3}
LATER = {s: STAGE_ORDER[i + 1 :] for i, s in enumerate(STAGE_ORDER)}


class TrainingService:
    def __init__(
        self,
        settings: SettingsService,
        db: Database,
        events: EventBus,
        jobs: JobService,
        models: VoiceModelService,
        assets: AssetService,
        separation: SeparationService,
        compute: ComputeService,
    ) -> None:
        self._settings = settings
        self._db = db
        self._events = events
        self._jobs = jobs
        self._models = models
        self._assets = assets
        self._separation = separation
        self._compute = compute
        self._lock = threading.RLock()
        self._running: dict[str, str] = {}
        self.base: Any = None
        """The base model service; set by ``build_services``."""
        jobs.register("train", self._run_spec)
        jobs.register("index", self._index_spec)
        jobs.register("export", self._export_spec)

    @property
    def root(self) -> Path:
        return self._settings.experiments_dir

    # -- storage -------------------------------------------------------------------

    def _dir(self, name: str) -> Path:
        return self.root / validate_name(name)

    def _load(self, name: str) -> Experiment:
        folder = self._dir(name)
        path = folder / EXPERIMENT_FILE
        if path.is_file():
            try:
                exp = Experiment.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError as e:
                raise ValidationError(f"{path} is damaged: {e}") from e
        elif folder.is_dir():
            # A folder copied in from an original install: describe it from what it holds.
            exp = self._from_legacy(name, folder)
            self._save(exp)
        else:
            raise NotFoundError(f"Unknown experiment: {name}")
        exp.name = name
        exp.path = str(folder)
        exp.running_job = self._running.get(name)
        return exp

    def _save(self, exp: Experiment) -> None:
        folder = self._dir(exp.name)
        folder.mkdir(parents=True, exist_ok=True)
        exp.updated_at = now_iso()
        data = exp.model_dump(exclude={"path", "running_job"})
        tmp = folder / (EXPERIMENT_FILE + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(folder / EXPERIMENT_FILE)

    def _emit(self, exp: Experiment, removed: bool = False) -> None:
        self._events.publish(
            ExperimentChangedEvent(name=exp.name, stages={k: v.model_copy() for k, v in exp.stages.items()}, running_job=self._running.get(exp.name), removed=removed)
        )

    def _from_legacy(self, name: str, folder: Path) -> Experiment:
        from rvc_next.engine.train.experiment import read_legacy

        info = read_legacy(folder)
        speakers = [SpeakerEntry.model_validate(r) for r in info.get("speaker_rows", [])]
        exp = Experiment(
            name=name,
            sample_rate=info.get("sample_rate") or "40k",
            version=info.get("version") or "v2",
            pitch_guidance=bool(info.get("pitch_guidance", True)),
            f0_method="rmvpe",
            dataset=Dataset(mode="multi" if info.get("multi_speaker") else "single", speakers=speakers),
            created_at=now_iso(),
            legacy=True,
        )
        for stage, exists in info.get("stages", {}).items():
            if exists:
                exp.stages[stage] = StageState(status="done", finished_at=now_iso(), fingerprint=self._fingerprint(exp, stage))
        return exp

    # -- listing and editing -----------------------------------------------------------

    def list_experiments(self) -> list[ExperimentSummary]:
        out: list[ExperimentSummary] = []
        if not self.root.is_dir():
            return out
        for folder in sorted(self.root.iterdir(), key=lambda p: p.name.lower()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            if not (folder / EXPERIMENT_FILE).is_file() and not (folder / "0_gt_wavs").is_dir():
                continue
            try:
                exp = self._load(folder.name)
            except Exception as e:
                logger.warning("Skipping experiment %s: %s", folder.name, e)
                continue
            out.append(
                ExperimentSummary(
                    name=exp.name,
                    sample_rate=exp.sample_rate,
                    version=exp.version,
                    pitch_guidance=exp.pitch_guidance,
                    mode=exp.dataset.mode,
                    stages=exp.stages,
                    running_job=exp.running_job,
                    last_activity=exp.updated_at or exp.created_at,
                    voice_id=exp.voice_id,
                    legacy=exp.legacy,
                )
            )
        return out

    def get(self, name: str) -> Experiment:
        return self._load(name)

    def _default_f0(self) -> str:
        method = self._settings.settings.training.f0_method
        if method != "auto":
            return method
        return "pm" if self._settings.settings.compute.device == "cpu" or not self._has_cuda() else "rmvpe"

    @staticmethod
    def _has_cuda() -> bool:
        from rvc_next.engine.runtime import cuda_profiles

        try:
            return any(p.eligible for p in cuda_profiles())
        except Exception:
            return False

    def create(self, data: ExperimentCreate) -> Experiment:
        name = validate_name(data.name.strip())
        if self._dir(name).exists():
            raise ConflictError(f"An experiment named {name} exists")
        t = self._settings.settings.training
        fit = data.fit or FitSettings(
            epochs=t.epochs, save_every=t.save_every, batch_size=t.batch_size, gpus=t.gpus, cache_in_gpu=t.cache_in_gpu, save_small_every=t.save_small_every
        )
        exp = Experiment(
            name=name,
            sample_rate=data.sample_rate or t.sample_rate,
            version=data.version or t.version,
            pitch_guidance=t.pitch_guidance if data.pitch_guidance is None else data.pitch_guidance,
            f0_method=data.f0_method or self._default_f0(),  # ty: ignore[invalid-argument-type]
            dataset=data.dataset,
            fit=fit,
            created_at=now_iso(),
        )
        self._check_dataset(exp.dataset)
        self._save(exp)
        exp = self._load(name)
        self._emit(exp)
        return exp

    def _check_dataset(self, ds: Dataset) -> None:
        folders = [ds.folder] if ds.mode == "single" else [s.folder for s in ds.speakers]
        for f in folders:
            if f and not Path(f).expanduser().is_dir():
                raise NotFoundError(f"No such folder: {f}")
        if ds.mode == "multi":
            ids = [s.id for s in ds.speakers]
            if len(ids) != len(set(ids)):
                raise ValidationError("Speaker ids must be unique")
            names = [s.name for s in ds.speakers]
            if len(names) != len(set(names)):
                raise ValidationError("Speaker names must be unique")

    def update(self, name: str, patch: ExperimentUpdate) -> Experiment:
        with self._lock:
            if name in self._running:
                raise BusyError(f"{name} is running; stop it first")
            exp = self._load(name)
            stale_from: str | None = None
            if patch.dataset is not None and patch.dataset != exp.dataset:
                self._check_dataset(patch.dataset)
                stale_from = "clean" if patch.dataset.clean_preset != exp.dataset.clean_preset else "slice"
                exp.dataset = patch.dataset
            if patch.sample_rate and patch.sample_rate != exp.sample_rate:
                exp.sample_rate = patch.sample_rate
                stale_from = _earliest(stale_from, "slice")
            if patch.version and patch.version != exp.version:
                exp.version = patch.version
                stale_from = _earliest(stale_from, "features")
            if patch.pitch_guidance is not None and patch.pitch_guidance != exp.pitch_guidance:
                exp.pitch_guidance = patch.pitch_guidance
                stale_from = _earliest(stale_from, "f0")
            if patch.f0_method and patch.f0_method != exp.f0_method:
                exp.f0_method = patch.f0_method
                stale_from = _earliest(stale_from, "f0")
            if patch.fit is not None:
                exp.fit = patch.fit
            if stale_from:
                for stage in (stale_from, *LATER[stale_from]):
                    st = exp.stages.get(stage)
                    if st is not None and st.status == "done":
                        st.status = "stale"
            self._save(exp)
        exp = self._load(name)
        self._emit(exp)
        return exp

    def delete(self, name: str) -> None:
        if name in self._running:
            raise BusyError(f"{name} is running; stop it first")
        exp = self._load(name)
        trash(self._dir(name), self._settings.data_dir / "trash")
        self._emit(exp, removed=True)

    def set_speakers(self, name: str, entries: list[SpeakerEntry]) -> Experiment:
        exp = self._load(name)
        return self.update(name, ExperimentUpdate(dataset=exp.dataset.model_copy(update={"mode": "multi", "speakers": entries, "folder": None})))

    def speakers_from_folders(self, name: str, folder: str) -> Experiment:
        from rvc_next.engine.train.multispeaker import speakers_from_folders

        root = Path(folder).expanduser()
        if not root.is_dir():
            raise NotFoundError(f"No such folder: {root}")
        rows = speakers_from_folders(root)
        if not rows:
            raise ValidationError("No Name_ID_Repeat subfolders found")
        return self.set_speakers(name, [SpeakerEntry.model_validate(r) for r in rows])

    def scan_dataset(self, name: str) -> DatasetReport:
        from rvc_next.engine.train.scan import scan_dataset

        exp = self._load(name)
        ds = exp.dataset
        folders: list[tuple[str | None, str]] = [(None, ds.folder)] if ds.mode == "single" and ds.folder else [(s.name, s.folder) for s in ds.speakers]
        if not folders:
            raise ValidationError("The experiment has no dataset folder yet")
        return DatasetReport.model_validate(scan_dataset(folders))

    # -- reading results ------------------------------------------------------------------

    def metrics(self, name: str, since: int = 0) -> list[TrainMetric]:
        from rvc_next.engine.train.experiment import read_metrics

        self._load(name)
        out = []
        for m in read_metrics(self._dir(name), since):
            try:
                out.append(
                    TrainMetric(
                        epoch=int(m.get("epoch", 0)),
                        step=int(m.get("step", 0)),
                        losses={k: float(v) for k, v in (m.get("losses") or {}).items()},
                        lr=m.get("lr"),
                        time=m.get("time"),
                    )
                )
            except (TypeError, ValueError):
                continue
        return out

    def checkpoints(self, name: str) -> list[Checkpoint]:
        from datetime import datetime, timezone

        from rvc_next.engine.train.experiment import list_checkpoints

        self._load(name)
        out = []
        for c in list_checkpoints(self._dir(name)):
            mtime = c.get("mtime")
            modified = datetime.fromtimestamp(float(mtime), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if mtime else ""
            out.append(Checkpoint(name=c["name"], path=c["path"], kind=c["kind"], epoch=c.get("epoch"), step=c.get("step"), size=int(c.get("size", 0)), modified_at=modified))
        return out

    # -- running stages -----------------------------------------------------------------------

    def plan(self, exp: Experiment, request: RunRequest) -> list[str]:
        """The stages a run will execute, in order.

        Without chosen stages ("Run all"): every stage that is not done, or whose inputs changed since
        it ran (its stored fingerprint no longer matches), and from the first such stage on every later
        one, since they read its output. ``fit`` also runs alone when ``epochs`` grew, to continue.
        """
        applicable = [s for s in STAGE_ORDER if self._applies(exp, s)]
        if request.stages:
            chosen: list[str] = [s for s in applicable if s in request.stages]
            if not chosen:
                raise ValidationError("None of the chosen stages applies to this experiment")
            return chosen
        out: list[str] = []
        cascade = False
        for s in applicable:
            st = exp.stages.get(s)
            if request.force or cascade or st is None or st.status != "done" or self._inputs_changed(exp, s, st):
                out.append(s)
                cascade = True
            elif s == "fit" and st.counts.get("epochs", 0) < exp.fit.epochs:
                out.append(s)
        return out

    def _inputs_changed(self, exp: Experiment, stage: str, state: StageState) -> bool:
        """Whether what ``stage`` read has changed since it ran; unknown (no fingerprint, a legacy run) counts as unchanged."""
        return state.fingerprint is not None and state.fingerprint != self._fingerprint(exp, stage)

    @staticmethod
    def _applies(exp: Experiment, stage: str) -> bool:
        if stage == "clean":
            return bool(exp.dataset.clean_preset)
        if stage == "f0":
            return exp.pitch_guidance
        return True

    def run(self, name: str, request: RunRequest, foreground: bool = False, fit_override: dict[str, Any] | None = None) -> Job:
        """Run stages as one job. ``fit_override`` adds fit-worker fields (tests use a tiny model)."""
        with self._lock:
            if name in self._running:
                raise BusyError(f"{name} is already running")
            exp = self._load(name)
            stages = self.plan(exp, request)
            if not stages:
                raise ConflictError("Every stage is done and its inputs are unchanged; choose stages to run them again")
            self._preflight(exp, stages)
            payload: dict[str, Any] = {"name": name, "stages": stages, "force": request.force}
            if fit_override:
                payload["config_override"] = fit_override
            return (self._jobs.run_now if foreground else self._jobs.submit)("train", payload)

    def _preflight(self, exp: Experiment, stages: list[str]) -> None:
        ds = exp.dataset
        if "slice" in stages and not (ds.folder if ds.mode == "single" else ds.speakers):
            raise ValidationError("Choose the dataset folder (or the speaker table) first")
        needed: list[str] = []
        if "features" in stages:
            needed.append("hubert")
        if "f0" in stages and exp.f0_method == "rmvpe":
            needed.append("rmvpe")
        if "fit" in stages:
            if exp.fit.base_model and self.base is not None:
                base = self.base.check_fits(exp.fit.base_model, exp.sample_rate, exp.version, exp.pitch_guidance)
                if base.source == "official":
                    needed.append(base.asset_id or "")
            elif exp.fit.pretrained_g is None:
                needed.append(f"pretrained-{exp.version}-{exp.sample_rate}")
        if "clean" in stages and ds.clean_preset:
            needed += self._separation.preset(ds.clean_preset).assets
        self._assets.require(needed, f"Training {exp.name}")

    def stop(self, name: str) -> Experiment:
        job_id = self._running.get(name)
        if job_id is None:
            raise ConflictError(f"{name} is not running")
        self._jobs.cancel(job_id)
        return self._load(name)

    def _run_spec(self, raw: dict[str, Any]) -> JobSpec:
        name = raw["name"]
        stages = list(raw.get("stages") or [])
        force = bool(raw.get("force"))
        exp = self._load(name)
        steps = [JobStep(id=s, title=STAGE_TITLES[s]) for s in stages]
        resource = "cpu" if self._settings.settings.compute.device == "cpu" or exp.fit.gpus == "cpu" else "gpu"
        title = f"Train {name}" + ("" if len(stages) > 1 else f" · {STAGE_TITLES[stages[0]]}")

        override = raw.get("config_override") or {}

        def run(ctx: JobContext) -> dict[str, Any]:
            return self._run_stages(ctx, name, stages, force, override)

        return JobSpec(title=title, run=run, resources=frozenset({resource}), steps=steps)

    def _run_stages(self, ctx: JobContext, name: str, stages: list[str], force: bool, override: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if name in self._running:
                raise BusyError(f"{name} is already running")
            self._running[name] = ctx.job_id
        results: dict[str, Any] = {}
        current: str | None = None
        try:
            total = sum(STAGE_WEIGHT[s] for s in stages)
            done_weight = 0.0
            for stage in stages:
                ctx.check_cancel()
                current = stage
                exp = self._load(name)
                exp.stages[stage] = StageState(status="running", job_id=ctx.job_id, counts=exp.stages.get(stage, StageState()).counts)
                self._save(exp)
                self._emit(exp)
                ctx.step(stage, "running")
                base, span = done_weight / total, STAGE_WEIGHT[stage] / total
                result = self._run_stage(ctx, exp, stage, force, base, span, override)
                results[stage] = result
                exp = self._load(name)
                counts = {k: int(v) for k, v in result.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                if stage == "fit":
                    counts["epochs"] = int(result.get("epochs", exp.fit.epochs))
                exp.stages[stage] = StageState(status="done", fingerprint=self._fingerprint(exp, stage), finished_at=now_iso(), counts=counts)
                for later in LATER[stage]:
                    st = exp.stages.get(later)
                    if st is not None and st.status == "done" and later not in stages:
                        st.status = "stale"
                self._save(exp)
                self._emit(exp)
                ctx.step(stage, "done")
                done_weight += STAGE_WEIGHT[stage]
                current = None
            return {"stages": results}
        except BaseException as e:
            if current is not None:
                try:
                    exp = self._load(name)
                    cancelled = ctx.cancelled
                    exp.stages[current] = StageState(
                        status="pending" if cancelled else "failed", error=None if cancelled else str(e) or type(e).__name__, counts=exp.stages.get(current, StageState()).counts
                    )
                    self._save(exp)
                    self._emit(exp)
                except Exception:
                    logger.exception("Could not record the failed stage")
            raise
        finally:
            with self._lock:
                self._running.pop(name, None)
            try:
                self._emit(self._load(name))
            except Exception:
                pass

    def _gpus(self, spec: str) -> list[int]:
        """The experiment's ``fit.gpus`` (new experiments take ``training.gpus`` from the settings)."""
        if spec == "cpu" or self._settings.settings.compute.device == "cpu":
            return []
        if spec == "auto":
            from rvc_next.engine.runtime import cuda_profiles

            try:
                return [int(p.id.split(":")[1]) for p in cuda_profiles() if p.eligible]
            except Exception:
                return []
        try:
            return [int(x) for x in spec.replace(",", "-").split("-") if x.strip() != ""]
        except ValueError as e:
            raise ValidationError(f"GPUs must be auto, cpu or ids like 0-1, not {spec!r}") from e

    def _cpu_workers(self) -> int:
        from rvc_next.engine.train.experiment import default_cpu_workers

        return self._settings.settings.training.cpu_workers or default_cpu_workers()

    def _base_request(self, exp: Experiment, stage: str) -> dict[str, Any]:
        c = self._settings.settings.compute
        return {"stage": stage, "exp_dir": str(self._dir(exp.name)), "assets_dir": str(self._settings.assets_dir), "device": c.device, "precision": c.precision}

    def _run_stage(self, ctx: JobContext, exp: Experiment, stage: str, force: bool, base: float, span: float, override: dict[str, Any] | None = None) -> dict[str, Any]:
        from rvc_next.protocol.messages import LogEvent, MetricEvent, OutputEvent, ProgressEvent

        def on_event(event: Any) -> None:
            if isinstance(event, ProgressEvent):
                if event.value is not None:
                    ctx.progress(base + span * float(event.value), stage)
            elif isinstance(event, MetricEvent):
                self._events.publish(TrainMetricsEvent(name=exp.name, epoch=event.epoch, step=event.step, losses=event.values, lr=event.lr))
            elif isinstance(event, OutputEvent):
                ctx.log(f"Saved {Path(event.path).name}")
            elif isinstance(event, LogEvent):
                pass

        if stage == "clean":
            return self._clean(ctx, exp, base, span)
        req = self._base_request(exp, stage)
        gpus = self._gpus(exp.fit.gpus)
        prev = exp.stages.get(stage)
        clean = force or (prev is not None and prev.status == "stale")
        if stage == "slice":
            ds = exp.dataset
            src = self._dir(exp.name) / CLEAN_DIR if ds.clean_preset else None
            req |= {"sample_rate": exp.sample_rate, "n_workers": self._cpu_workers(), "per": 3.7 if gpus else 3.0, "clean": clean}
            if ds.mode == "multi":
                req["speakers"] = [
                    {"name": s.name, "id": s.id, "folder": str(src / str(s.id)) if src else str(Path(s.folder).expanduser()), "repeat": s.repeat} for s in ds.speakers
                ]
            else:
                req["folder"] = str(src / "0") if src else str(Path(ds.folder or "").expanduser())
        elif stage == "f0":
            req |= {"method": exp.f0_method, "n_workers": self._cpu_workers(), "gpus": gpus, "clean": clean}
        elif stage == "features":
            req |= {"version": exp.version, "gpus": gpus, "clean": clean}
        elif stage == "fit":
            from rvc_next.engine.train.experiment import default_batch_size, pretrained_paths

            fit = exp.fit
            if fit.base_model and self.base is not None:
                self.base.check_fits(fit.base_model, exp.sample_rate, exp.version, exp.pitch_guidance)
                g_path, d_path = self.base.paths(fit.base_model)
                g, d = str(g_path), str(d_path) if d_path else ""
            elif fit.pretrained_g is None:
                g, d = pretrained_paths(self._settings.assets_dir, exp.version, exp.pitch_guidance, exp.sample_rate)
                if not g or not d:
                    self._assets.require([f"pretrained-{exp.version}-{exp.sample_rate}"], f"Training {exp.name}")
            else:
                g, d = fit.pretrained_g, fit.pretrained_d or ""
            req |= {
                "sample_rate": exp.sample_rate,
                "version": exp.version,
                "pitch_guidance": exp.pitch_guidance,
                "epochs": fit.epochs,
                "save_every": fit.save_every,
                "batch_size": fit.batch_size or default_batch_size(),
                "pretrained_g": g,
                "pretrained_d": d,
                "gpus": gpus,
                "cache_in_gpu": fit.cache_in_gpu,
                "save_small_every": fit.save_small_every,
                "save_latest_only": fit.save_latest_only,
                "name": exp.name,
                "multi_speaker": exp.dataset.mode == "multi",
                "num_workers": min(4, self._cpu_workers()),
            }
            if override:
                req |= override
        elif stage == "index":
            req |= {"version": exp.version, "name": exp.name, "n_cpu": self._cpu_workers(), "force": force or clean}
        return ctx.run_worker(WORKER, req, on_event=on_event)

    def _clean(self, ctx: JobContext, exp: Experiment, base: float, span: float) -> dict[str, Any]:
        """Run the clean-up separation preset over the dataset into ``dataset_clean/<speaker id>/``."""
        ds = exp.dataset
        preset = self._separation.preset(ds.clean_preset or "")
        out_root = self._dir(exp.name) / CLEAN_DIR
        if out_root.exists():
            shutil.rmtree(out_root)
        groups = [("0", ds.folder)] if ds.mode == "single" else [(str(s.id), s.folder) for s in ds.speakers]
        written = 0
        for n, (key, folder) in enumerate(groups):
            files = self._separation_inputs(folder)
            if not files:
                continue
            target = out_root / key
            tmp = out_root / f".{key}"
            lo = base + span * n / len(groups)
            hi = base + span * (n + 1) / len(groups)
            stems = self._separation.run_files(ctx, files, preset.id, tmp, "wav", register=False, step_prefix=None, progress_span=(lo, hi))
            target.mkdir(parents=True, exist_ok=True)
            for src, by_label in stems.items():
                path = by_label.get(preset.primary_stem)
                if path is not None:
                    shutil.move(str(path), str(target / f"{Path(src).stem}.wav"))
                    written += 1
            shutil.rmtree(tmp, ignore_errors=True)
        return {"files": written}

    @staticmethod
    def _separation_inputs(folder: str | None) -> list[tuple[Path, str]]:
        from rvc_next.engine.audio.io import is_audio_path

        if not folder:
            return []
        root = Path(folder).expanduser()
        return [(p, p.name) for p in sorted(root.iterdir()) if p.is_file() and is_audio_path(p)]

    def _fingerprint(self, exp: Experiment, stage: str) -> str:
        from rvc_next.engine.train.experiment import feature_dir
        from rvc_next.engine.train.fingerprint import of_paths

        folder = self._dir(exp.name)
        ds = exp.dataset
        sources = [ds.folder] if ds.mode == "single" else [s.folder for s in ds.speakers]
        sources = [str(Path(s).expanduser()) for s in sources if s]
        speakers = [s.model_dump() for s in ds.speakers]
        if stage == "clean":
            return of_paths(sources, {"preset": ds.clean_preset})
        if stage == "slice":
            inputs = [folder / CLEAN_DIR] if ds.clean_preset else sources
            return of_paths(inputs, {"sr": exp.sample_rate, "speakers": speakers})
        if stage == "f0":
            return of_paths([folder / "1_16k_wavs"], {"method": exp.f0_method})
        if stage == "features":
            return of_paths([folder / "1_16k_wavs"], {"version": exp.version})
        if stage == "fit":
            dirs = [feature_dir(folder, exp.version), folder / "0_gt_wavs"] + ([folder / "2b-f0nsf"] if exp.pitch_guidance else [])
            return of_paths(dirs, {"sr": exp.sample_rate, "version": exp.version, "f0": exp.pitch_guidance, "speakers": speakers})
        return of_paths([feature_dir(folder, exp.version)], {"version": exp.version})

    # -- results into the library ---------------------------------------------------------------

    def _pick_model(self, exp: Experiment, ref: str | None) -> Path:
        cps = self.checkpoints(exp.name)
        if ref:
            for c in cps:
                if c.name == ref or c.path == ref:
                    return Path(c.path)
            raise NotFoundError(f"No checkpoint named {ref} in {exp.name}")
        smalls = [c for c in cps if c.kind == "small"]
        gs = [c for c in cps if c.kind == "G"]
        pick = (smalls or gs or [None])[-1]
        if pick is None:
            raise NotFoundError(f"{exp.name} has no trained model yet")
        return Path(pick.path)

    def _indexes(self, exp: Experiment) -> dict[str, Path]:
        from rvc_next.core.models.importers import index_key

        folder = self._dir(exp.name)
        return {index_key(p): p for p in sorted(folder.glob("added_*.index"))}

    def export(self, name: str, request: ExportRequest, foreground: bool = False) -> Job:
        exp = self._load(name)
        self._pick_model(exp, request.checkpoint)
        return (self._jobs.run_now if foreground else self._jobs.submit)("export", {"name": name, **request.model_dump()})

    def _export_spec(self, raw: dict[str, Any]) -> JobSpec:
        name = raw["name"]
        req = ExportRequest.model_validate({k: v for k, v in raw.items() if k != "name"})

        def run(ctx: JobContext) -> dict[str, Any]:
            from rvc_next.engine.models.checkpoint import summarize
            from rvc_next.engine.models.extract import extract_small_model

            exp = self._load(name)
            src = self._pick_model(exp, req.checkpoint)
            voice_name = req.voice_name or name
            tmp_dir = self._dir(name) / ".export"
            tmp_dir.mkdir(exist_ok=True)
            try:
                if summarize(src).kind == "checkpoint":
                    ctx.log(f"Extracting a small model from {src.name}")
                    small = tmp_dir / f"{name}.pth"
                    speakers = [{"id": s.id, "name": s.name} for s in exp.dataset.speakers] if exp.dataset.mode == "multi" else None
                    extract_small_model(
                        src, small, sample_rate=exp.sample_rate, pitch_guidance=exp.pitch_guidance, version=exp.version, info=f"{name} {src.stem}", speaker_info=speakers
                    )
                    src = small
                voice = self._models.add_model_file(src, voice_name, indexes=self._indexes(exp), source={"experiment": name, "checkpoint": src.name})
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            exp = self._load(name)
            exp.voice_id = voice.id
            self._save(exp)
            self._emit(exp)
            return {"voice_id": voice.id}

        return JobSpec(title=f"Export {name} to the library", run=run, resources=frozenset({"io"}))

    def try_checkpoint(self, name: str, request: ExportRequest) -> VoiceModel:
        """A temporary voice for a saved small model or a G checkpoint, with the experiment's index.

        A G checkpoint is extracted to a small model once, under ``.try/`` in the experiment (again
        only when the checkpoint is newer), as export does; D checkpoints cannot be tried.
        """
        from rvc_next.engine.models.extract import extract_small_model

        exp = self._load(name)
        src = self._pick_model(exp, request.checkpoint)
        kind = next((c.kind for c in self.checkpoints(name) if Path(c.path) == src), None)
        if kind == "G":
            small = self._dir(name) / ".try" / f"{src.stem}.pth"
            if not small.is_file() or small.stat().st_mtime_ns < src.stat().st_mtime_ns:
                small.parent.mkdir(exist_ok=True)
                speakers = [{"id": s.id, "name": s.name} for s in exp.dataset.speakers] if exp.dataset.mode == "multi" else None
                extract_small_model(
                    src, small, sample_rate=exp.sample_rate, pitch_guidance=exp.pitch_guidance, version=exp.version, info=f"{name} {src.stem}", speaker_info=speakers
                )
            src = small
        elif kind != "small":
            raise ConflictError(f"{src.name} is a discriminator; try a G checkpoint or a small model")
        idx = self._indexes(exp)
        return self._models.temporary(src, idx.get("default") or next(iter(idx.values()), None))

    def build_index_for_voice(self, voice_id: str, experiment: str, foreground: bool = False) -> Job:
        self._models.get(voice_id)
        self._load(experiment)
        return (self._jobs.run_now if foreground else self._jobs.submit)("index", {"voice_id": voice_id, "experiment": experiment})

    def _index_spec(self, raw: dict[str, Any]) -> JobSpec:
        voice_id, name = raw["voice_id"], raw["experiment"]

        def run(ctx: JobContext) -> dict[str, Any]:
            exp = self._load(name)
            if not exp.stages.get("features") or exp.stages["features"].status not in ("done", "stale"):
                from rvc_next.engine.train.experiment import stage_outputs_exist

                if not stage_outputs_exist(self._dir(name), exp.version).get("features"):
                    raise ConflictError(f"{name} has no extracted features; run its features stage first")
            result = self._run_stage(ctx, exp, "index", False, 0.0, 0.9)
            for key, path in self._indexes(exp).items():
                self._models.set_index(voice_id, key, path)
            exp = self._load(name)
            exp.stages["index"] = StageState(status="done", fingerprint=self._fingerprint(exp, "index"), finished_at=now_iso())
            self._save(exp)
            self._emit(exp)
            return {"voice_id": voice_id, **{k: v for k, v in result.items() if isinstance(v, (int, str))}}

        return JobSpec(title=f"Build the index from {name}", run=run, resources=frozenset({"cpu"}))

    # -- legacy import ---------------------------------------------------------------------------

    def import_legacy(self, request: ImportExperimentRequest) -> Experiment:
        src = Path(request.path).expanduser().resolve()
        if not src.is_dir():
            raise NotFoundError(f"No such folder: {src}")
        if not (src / "0_gt_wavs").is_dir() and not (src / "config.json").is_file():
            raise ValidationError(f"{src} does not look like an RVC experiment (no 0_gt_wavs or config.json)")
        name = validate_name(request.name or src.name)
        target = self._dir(name)
        if target.exists():
            raise ConflictError(f"An experiment named {name} exists")
        tmp = self.root / f".{name}.part"
        if tmp.exists():
            shutil.rmtree(tmp)
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("eval", "*.tmp"))
        tmp.rename(target)
        exp = self._from_legacy(name, target)
        self._save(exp)
        exp = self._load(name)
        self._emit(exp)
        return exp


def _earliest(current: str | None, stage: str) -> str:
    if current is None:
        return stage
    return current if STAGE_ORDER.index(current) <= STAGE_ORDER.index(stage) else stage
