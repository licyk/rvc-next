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
from rvc_next.core.settings.models import DownloadSettings

logger = logging.getLogger(__name__)

ENDPOINTS = {"huggingface": "https://huggingface.co", "hf-mirror": "https://hf-mirror.com"}
MODELSCOPE = "https://www.modelscope.cn"
SOURCES = ("huggingface", "hf-mirror", "modelscope")
"""The public sources, tried in this order after the chosen one when it lacks a file or serves it damaged."""
CHUNK = 1024 * 1024
RETRIES = 4
VERIFIED_FILE = "assets-verified.json"


def hf_endpoint(d: DownloadSettings) -> str:
    """The Hugging Face endpoint of the download settings (huggingface.co when the source is ModelScope)."""
    return d.endpoint.rstrip("/") if d.source == "custom" and d.endpoint else ENDPOINTS.get(d.source, ENDPOINTS["huggingface"])


def _sizes(f: AssetFile | CatalogVoiceFile) -> set[int]:
    return {f.size, *(a.size for a in getattr(f, "alternates", []))}


def _digests(f: AssetFile | CatalogVoiceFile) -> set[str]:
    return {d for d in (f.sha256, *(a.sha256 for a in getattr(f, "alternates", []))) if d}


class _WrongContent(Exception):
    """The source serves something else under this file's name (another size or checksum)."""


class _Unavailable(Exception):
    """The source does not have the file, or could not be reached."""


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
            if path.is_file() and path.stat().st_size in _sizes(f):
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
        return self.urls(f)[0]

    def urls(self, f: AssetFile | CatalogVoiceFile) -> list[str]:
        """Where ``f`` can be downloaded, in the order tried: the chosen source first, then the other
        public ones; at each, the chosen repository if it hosts the file, then the others that do. Only
        rvc-model has a ModelScope copy: with ModelScope chosen, a file elsewhere only (the community
        models' own repositories, Applio's) comes from Hugging Face."""
        d = self._settings.settings.downloads
        repo_ids = sorted(f.sources, key=lambda r: r != d.repository)
        out: list[str] = []
        for source in (d.source, *(s for s in SOURCES if s != d.source)):
            for repo_id in repo_ids:
                repo = self._repo(repo_id)
                if source == "modelscope":
                    if not repo.modelscope:
                        continue
                    url = f"{MODELSCOPE}/models/{repo.modelscope}/resolve/{repo.modelscope_revision}/{f.sources[repo_id]}"
                else:
                    base = hf_endpoint(d) if source == d.source else ENDPOINTS[source]
                    url = f"{base}/{repo.repo}/resolve/{repo.revision}/{f.sources[repo_id]}"
                if url not in out:
                    out.append(url)
        return out

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

                    checked = False
                    if not verify_only and not (path.is_file() and path.stat().st_size in _sizes(f)):
                        checked = self._fetch(ctx, f, path, on_fetch, on_bytes if check else None)
                    if not path.is_file():
                        ok = False
                    elif check and f.sha256:
                        good = checked or _sha256(path, on_bytes, ctx) in _digests(f)
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

    def _fetch(self, ctx: JobContext, f: AssetFile | CatalogVoiceFile, path: Path, on_bytes: Any, on_hash: Any = None) -> bool:
        """Download ``f`` to ``path`` from the first source that has it whole and, with ``on_hash``,
        matching its checksum; return whether the checksum was checked. A source that lacks the file,
        serves another size or content, or stays unreachable after its retries gives way to the next."""
        part = path.with_name(path.name + ".part")
        path.parent.mkdir(parents=True, exist_ok=True)
        name = path.name
        errors: list[str] = []
        urls = self.urls(f)
        for i, url in enumerate(urls):
            ctx.log(f"Downloading {url} ({f.size / 2**20:.1f} MiB)")
            try:
                self._fetch_from(ctx, url, f, part, on_bytes)
                if on_hash is not None and f.sha256 and _sha256(part, on_hash, ctx) not in _digests(f):
                    raise _WrongContent("its checksum does not match")
            except _WrongContent as e:
                part.unlink(missing_ok=True)
                errors.append(f"{url}: {e}")
            except _Unavailable as e:
                errors.append(f"{url}: {e}")
            else:
                part.replace(path)
                return on_hash is not None and bool(f.sha256)
            if i + 1 < len(urls):
                ctx.log(f"{errors[-1]}; trying the next source")
        raise RvcNextError(f"Could not download {name}", {"file": name, "errors": errors[-6:]})

    def _fetch_from(self, ctx: JobContext, url: str, f: AssetFile | CatalogVoiceFile, part: Path, on_bytes: Any) -> None:
        """Download ``url`` into ``part``, resuming what is there, until it holds one of ``f``'s sizes."""
        sizes = _sizes(f)
        last: str = ""
        for attempt in range(RETRIES):
            ctx.check_cancel()
            have = part.stat().st_size if part.exists() else 0
            if have > max(sizes):
                part.unlink()
                have = 0
            if have in sizes:
                return
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                with self._http.client().stream("GET", url, headers=headers) as r:
                    if r.status_code in (401, 403, 404, 410):
                        raise _Unavailable(f"HTTP {r.status_code}")
                    if r.status_code == 416:
                        raise _WrongContent("the file there is shorter")
                    if r.status_code not in (200, 206):
                        raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                    if r.status_code == 200:
                        have = 0
                    total = _stated_size(r, have)
                    if total is not None and total not in sizes:
                        raise _WrongContent(f"the file there is {total} bytes, not {f.size}")
                    got = 0
                    with open(part, "ab" if have else "wb") as out:
                        for chunk in r.iter_bytes(CHUNK):
                            ctx.check_cancel()
                            out.write(chunk)
                            have += len(chunk)
                            got += len(chunk)
                            on_bytes(have)
                if have in sizes:
                    return
                if have > max(sizes) or not got:
                    raise _WrongContent(f"the file there is not {f.size} bytes")
                last = "the transfer ended early"
            except (httpx.HTTPError, OSError) as e:
                last = str(e) or type(e).__name__
                ctx.log(f"Attempt {attempt + 1} failed: {last}")
                time.sleep(min(2**attempt, 10))
        raise _Unavailable(last)

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

                        self._fetch(ctx, f, target, on_bytes, lambda n: None)
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


def _stated_size(r: httpx.Response, offset: int) -> int | None:
    """The whole file's size as the response states it (``Content-Range``, or ``Content-Length`` past ``offset``)."""
    if r.status_code == 206:
        total = r.headers.get("content-range", "").rpartition("/")[2]
        return int(total) if total.isdigit() else None
    length = r.headers.get("content-length", "")
    return offset + int(length) if length.isdigit() and "content-encoding" not in r.headers else None


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
