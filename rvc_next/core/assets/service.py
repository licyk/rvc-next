"""Model assets: catalog, status, downloads and verification.

``assets/`` uses the original's layout, so ``paths.assets_dir`` can point at an existing RVC
``assets/`` folder and nothing is downloaded twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from importlib import resources
from pathlib import Path
from typing import Any

import httpx

from rvc_next.core.assets.models import AssetFile, AssetSpec, AssetStatus, CatalogVoice, CatalogVoiceFile, Repository
from rvc_next.core.errors import AssetMissingError, ConflictError, NotFoundError, RvcNextError, ValidationError
from rvc_next.core.events import EventBus
from rvc_next.core.events.models import AssetsChangedEvent
from rvc_next.core.jobs.models import Job, JobStep
from rvc_next.core.jobs.service import JobContext, JobService, JobSpec
from rvc_next.core.net.http import HttpClientProvider
from rvc_next.core.settings import SettingsService

logger = logging.getLogger(__name__)

ENDPOINTS = {"huggingface": "https://huggingface.co", "hf-mirror": "https://hf-mirror.com"}
CHUNK = 1024 * 1024
RETRIES = 4
VERIFIED_FILE = "assets-verified.json"


def load_catalog() -> tuple[list[Repository], list[AssetSpec], list[CatalogVoice]]:
    data = json.loads(resources.files("rvc_next.core.assets").joinpath("catalog.json").read_text(encoding="utf-8"))
    repos = [Repository(id=key, **value) for key, value in data["repositories"].items()]
    voices = []
    for v in data.get("voices", []):
        voice = CatalogVoice.model_validate(v)
        voice.size = sum(f.size for f in voice.files)
        voice.has_index = any(f.role == "index" for f in voice.files)
        voice.repositories = [r.id for r in repos if all(r.id in f.sources for f in voice.files)]
        voices.append(voice)
    return repos, [AssetSpec.model_validate(a) for a in data["assets"]], voices


class AssetService:
    def __init__(self, settings: SettingsService, events: EventBus, http: HttpClientProvider, jobs: JobService) -> None:
        self._settings = settings
        self._events = events
        self._http = http
        self._jobs = jobs
        self._repos, specs, voices = load_catalog()
        self._specs = {s.id: s for s in specs}
        self._voices = {v.id: v for v in voices}
        self.models: Any = None
        """The voice library; set by ``build_services`` (voice downloads import into it)."""
        self._lock = threading.Lock()
        self._downloading: dict[str, str] = {}
        self._verified_path = settings.data_dir / VERIFIED_FILE
        jobs.register("download", self._job_spec)

    @property
    def root(self) -> Path:
        return self._settings.assets_dir

    def specs(self) -> list[AssetSpec]:
        return list(self._specs.values())

    def spec(self, asset_id: str) -> AssetSpec:
        spec = self._specs.get(asset_id)
        if spec is None:
            raise NotFoundError(f"Unknown asset: {asset_id}")
        return spec

    def add_specs(self, specs: list[AssetSpec]) -> None:
        """Extend the catalog (separation weights are declared by the separation presets)."""
        for s in specs:
            self._specs.setdefault(s.id, s)

    # -- status ----------------------------------------------------------------

    def _verified(self) -> dict[str, Any]:
        try:
            return json.loads(self._verified_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_verified(self, data: dict[str, Any]) -> None:
        self._verified_path.parent.mkdir(parents=True, exist_ok=True)
        self._verified_path.write_text(json.dumps(data), encoding="utf-8")

    def _file_key(self, f: AssetFile) -> str | None:
        path = self.root / f.path
        try:
            st = path.stat()
        except OSError:
            return None
        return f"{f.path}|{st.st_size}|{st.st_mtime_ns}"

    def status(self, asset_id: str) -> AssetStatus:
        spec = self.spec(asset_id)
        verified = self._verified()
        installed = 0
        missing: list[str] = []
        checks: list[bool | None] = []
        for f in spec.files:
            path = self.root / f.path
            if path.is_file() and path.stat().st_size == f.size:
                installed += f.size
                key = self._file_key(f)
                checks.append(verified.get(key) if key else None)
            else:
                missing.append(f.path)
                if path.is_file():
                    installed += min(path.stat().st_size, f.size)
        size = sum(f.size for f in spec.files)
        if asset_id in self._downloading:
            state = "downloading"
        elif not missing:
            state = "corrupt" if any(c is False for c in checks) else "installed"
        elif len(missing) == len(spec.files) and installed == 0:
            state = "missing"
        else:
            state = "partial"
        verified_flag = None if not checks or any(c is None for c in checks) or missing else all(checks)
        return AssetStatus(
            id=spec.id,
            title=spec.title,
            description=spec.description,
            group=spec.group,
            size=size,
            installed_bytes=installed,
            state=state,
            verified=verified_flag,
            job_id=self._downloading.get(asset_id),
            missing_files=missing,
            base_model=spec.base_model,
        )

    def list_assets(self) -> list[AssetStatus]:
        return [self.status(i) for i in self._specs]

    def is_installed(self, asset_id: str) -> bool:
        return self.status(asset_id).state in ("installed", "corrupt")

    def require(self, asset_ids: list[str], what: str = "This") -> None:
        """Raise ``AssetMissingError`` naming every asset that is not installed."""
        missing = [i for i in asset_ids if not self.is_installed(i)]
        if missing:
            titles = ", ".join(self.spec(i).title for i in missing)
            raise AssetMissingError(f"{what} needs {titles}", missing, {"size": sum(self.status(i).size - self.status(i).installed_bytes for i in missing)})

    def delete(self, asset_id: str) -> AssetStatus:
        """Remove an asset's files (and partial downloads); it can be downloaded again."""
        spec = self.spec(asset_id)
        with self._lock:
            if asset_id in self._downloading:
                raise ConflictError(f"{spec.title} is downloading")
        verified = self._verified()
        for f in spec.files:
            path = self.root / f.path
            key = self._file_key(f)
            if key:
                verified.pop(key, None)
            for p in (path, path.with_name(path.name + ".part")):
                if p.is_file():
                    p.unlink()
        self._save_verified(verified)
        self._emit()
        return self.status(asset_id)

    def delete_files(self, paths: list[str]) -> None:
        """Remove some files of the assets (an official base model's own G and D)."""
        known = {f.path: f for s in self._specs.values() for f in s.files}
        verified = self._verified()
        for rel in paths:
            f = known.get(rel)
            if f is None:
                raise NotFoundError(f"{rel} is not an asset file")
            key = self._file_key(f)
            if key:
                verified.pop(key, None)
            path = self.root / rel
            if path.is_file():
                path.unlink()
        self._save_verified(verified)
        self._emit()

    def ids_for_group(self, group: str) -> list[str]:
        """The assets of a catalog group (``all``: every one but the community base models and the other
        content-feature models, large and needed only for some voices, fetched one by one). The ONNX RMVPE is only for DirectML and is left out;
        ``inference`` is HuBERT and every pitch model, the default download."""
        return [i for i, s in self._specs.items() if s.id != "rmvpe-onnx" and (s.group == group or (group == "all" and s.group not in ("community", "embedder")))]

    # -- downloads -------------------------------------------------------------

    def download(self, ids: list[str] | None = None, group: str | None = None, verify_only: bool = False, foreground: bool = False) -> Job:
        chosen = list(dict.fromkeys((ids or []) + (self.ids_for_group(group) if group else [])))
        if not chosen:
            raise ValidationError("Choose assets or a group to download")
        for i in chosen:
            self.spec(i)
        request = {"ids": chosen, "verify_only": verify_only}
        return self._jobs.run_now("download", request) if foreground else self._jobs.submit("download", request)

    # -- repositories ------------------------------------------------------------

    def repositories(self) -> list[Repository]:
        """The repositories to choose from in Settings (the community models' own are not)."""
        chosen = self._settings.settings.downloads.repository
        return [r.model_copy(update={"selected": r.id == chosen}) for r in self._repos if not r.upstream]

    def _repo(self, repo_id: str) -> Repository:
        return next(r for r in self._repos if r.id == repo_id)

    def url(self, f: AssetFile | CatalogVoiceFile) -> str:
        """The download URL of ``f``: the chosen repository if it hosts the file, else another that does."""
        d = self._settings.settings.downloads
        repo_id = d.repository if d.repository in f.sources else next(iter(f.sources))
        repo = self._repo(repo_id)
        base = d.endpoint.rstrip("/") if d.source == "custom" and d.endpoint else ENDPOINTS.get(d.source, ENDPOINTS["huggingface"])
        return f"{base}/{repo.repo}/resolve/{repo.revision}/{f.sources[repo_id]}"

    def _job_spec(self, request: dict[str, Any]) -> JobSpec:
        if request.get("voices"):
            return self._voice_job_spec(request)
        ids = list(request.get("ids", []))
        verify_only = bool(request.get("verify_only"))
        specs = [self.spec(i) for i in ids]
        title = ("Verify " if verify_only else "Download ") + ", ".join(s.title for s in specs)
        steps = [JobStep(id=s.id, title=s.title) for s in specs]

        def run(ctx: JobContext) -> dict[str, Any]:
            return self._run(ctx, specs, verify_only)

        return JobSpec(title=title, run=run, resources=frozenset({"io"}), steps=steps)

    def _run(self, ctx: JobContext, specs: list[AssetSpec], verify_only: bool) -> dict[str, Any]:
        total = sum(f.size for s in specs for f in s.files)
        done = 0
        results: dict[str, str] = {}
        verified = self._verified()
        check = verify_only or self._settings.settings.downloads.verify_checksums
        try:
            for s in specs:
                with self._lock:
                    self._downloading[s.id] = ctx.job_id
                self._emit()
                ctx.step(s.id, "running")
                ok = True
                for f in s.files:
                    ctx.check_cancel()
                    path = self.root / f.path
                    base = done

                    def on_bytes(n: int, base: int = base, sid: str = s.id) -> None:
                        ctx.progress((base + n) / total if total else None, sid)

                    def on_fetch(n: int, base: int = base) -> None:
                        on_bytes(n)
                        ctx.transfer(base + n, total)

                    if not verify_only and not (path.is_file() and path.stat().st_size == f.size):
                        ctx.log(f"Downloading {self.url(f)} ({f.size / 2**20:.1f} MiB)")
                        self._fetch(ctx, f, path, on_fetch)
                    if not path.is_file():
                        ok = False
                    elif check and f.sha256:
                        good = _sha256(path, on_bytes, ctx) == f.sha256
                        key = self._file_key(f)
                        if key:
                            verified[key] = good
                        if not good:
                            ok = False
                            ctx.log(f"Checksum mismatch: {f.path}")
                    done += f.size
                self._save_verified(verified)
                results[s.id] = "ok" if ok else "failed"
                ctx.step(s.id, "done" if ok else "failed")
                with self._lock:
                    self._downloading.pop(s.id, None)
                self._emit()
        finally:
            with self._lock:
                for s in specs:
                    self._downloading.pop(s.id, None)
            self._emit()
        failed = [k for k, v in results.items() if v != "ok"]
        if failed:
            raise RvcNextError(f"{'Verification' if verify_only else 'Download'} failed for {', '.join(failed)}", {"assets": failed})
        return {"assets": results}

    def _fetch(self, ctx: JobContext, f: AssetFile | CatalogVoiceFile, path: Path, on_bytes: Any) -> None:
        part = path.with_name(path.name + ".part")
        path.parent.mkdir(parents=True, exist_ok=True)
        url = self.url(f)
        name = url.rsplit("/", 1)[-1]
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            ctx.check_cancel()
            have = part.stat().st_size if part.exists() else 0
            if have > f.size:
                part.unlink()
                have = 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                with self._http.client().stream("GET", url, headers=headers) as r:
                    if r.status_code == 416 and have == f.size:
                        break
                    if r.status_code not in (200, 206):
                        raise RvcNextError(f"HTTP {r.status_code} for {url}")
                    if r.status_code == 200 and have:
                        have = 0
                    with open(part, "ab" if have else "wb") as out:
                        for chunk in r.iter_bytes(CHUNK):
                            ctx.check_cancel()
                            out.write(chunk)
                            have += len(chunk)
                            on_bytes(have)
                if have >= f.size:
                    break
            except (httpx.HTTPError, OSError) as e:
                last_error = e
                ctx.log(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(min(2**attempt, 10))
        else:
            raise RvcNextError(f"Could not download {name}: {last_error}")
        if part.stat().st_size != f.size:
            raise RvcNextError(f"Size mismatch for {name}: {part.stat().st_size} != {f.size}")
        part.replace(path)

    # -- the voice catalog -----------------------------------------------------------

    def voice_catalog(self) -> list[CatalogVoice]:
        """Ready-made voices, with the library voice each became once downloaded."""
        installed: dict[str, str] = {}
        if self.models is not None:
            for v in self.models.list_voices(include_hidden=True):
                if v.catalog_id and v.location == "library":
                    installed.setdefault(v.catalog_id, v.id)
        out = []
        for v in self._voices.values():
            out.append(v.model_copy(update={"installed_voice_id": installed.get(v.id), "job_id": self._downloading.get(f"voice:{v.id}")}))
        return out

    def catalog_voice(self, voice_id: str) -> CatalogVoice:
        voice = self._voices.get(voice_id)
        if voice is None:
            raise NotFoundError(f"No downloadable voice {voice_id!r}")
        return voice

    def download_voices(self, ids: list[str], all_missing: bool = False, foreground: bool = False) -> Job:
        """Download catalog voices and add them to the library, with their indexes."""
        if all_missing:
            ids = [v.id for v in self.voice_catalog() if v.installed_voice_id is None]
            if not ids:
                raise ValidationError("Every downloadable voice is already in the library")
        chosen = list(dict.fromkeys(ids))
        if not chosen:
            raise ValidationError("Choose voices to download")
        for i in chosen:
            self.catalog_voice(i)
        request = {"voices": chosen}
        return self._jobs.run_now("download", request) if foreground else self._jobs.submit("download", request)

    def _voice_job_spec(self, request: dict[str, Any]) -> JobSpec:
        voices = [self.catalog_voice(i) for i in request["voices"]]
        title = "Download " + ", ".join(v.name for v in voices)
        steps = [JobStep(id=f"voice:{v.id}", title=v.name) for v in voices]

        def run(ctx: JobContext) -> dict[str, Any]:
            return self._run_voices(ctx, voices)

        return JobSpec(title=title, run=run, resources=frozenset({"io"}), steps=steps)

    def _run_voices(self, ctx: JobContext, voices: list[CatalogVoice]) -> dict[str, Any]:
        import shutil

        from rvc_next.core.clock import new_id

        if self.models is None:
            raise RvcNextError("The voice library is not available")
        total = sum(v.size for v in voices) or 1
        done = 0
        added: dict[str, str] = {}
        try:
            for v in voices:
                step = f"voice:{v.id}"
                with self._lock:
                    self._downloading[step] = ctx.job_id
                ctx.step(step, "running")
                staging = self._settings.data_dir / "uploads" / new_id("voice")
                try:
                    paths: dict[str, Path] = {}
                    for f in v.files:
                        ctx.check_cancel()
                        target = staging / f"{v.name}.{'pth' if f.role == 'model' else 'index'}"
                        base = done

                        def on_bytes(n: int, base: int = base, step: str = step) -> None:
                            ctx.progress((base + n) / total, step)
                            ctx.transfer(base + n, total)

                        ctx.log(f"Downloading {self.url(f)} ({f.size / 2**20:.1f} MiB)")
                        self._fetch(ctx, f, target, on_bytes)
                        if _sha256(target, lambda n: None, ctx) != f.sha256:
                            raise RvcNextError(f"Checksum mismatch for {v.name} ({f.role})")
                        paths[f.role] = target
                        done += f.size
                    indexes = {"default": paths["index"]} if "index" in paths else {}
                    voice = self.models.add_model_file(paths["model"], v.name, indexes=indexes, description=v.description, source={"catalog_id": v.id})
                    added[v.id] = voice.id
                    ctx.log(f"Added {v.name} to the library as {voice.id}")
                    ctx.step(step, "done")
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
                    with self._lock:
                        self._downloading.pop(step, None)
        finally:
            with self._lock:
                for v in voices:
                    self._downloading.pop(f"voice:{v.id}", None)
        return {"voices": added}

    def _emit(self) -> None:
        self._events.publish(AssetsChangedEvent(assets=self.list_assets()))


def _sha256(path: Path, on_bytes: Any, ctx: JobContext | None = None) -> str:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK * 4)
            if not block:
                break
            h.update(block)
            n += len(block)
            if ctx is not None and n % (CHUNK * 64) < CHUNK * 4:
                ctx.check_cancel()
                on_bytes(n)
    return h.hexdigest()
