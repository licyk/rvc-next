"""The voice library."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from rvc_next.api.deps import LocalDep, ServicesDep
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.api.files import content_disposition
from rvc_next.api.uploads import RAW_BODY, read_chunks, spool
from rvc_next.core.assets.models import CatalogVoice, VoiceDownloadRequest
from rvc_next.core.jobs.models import Job
from rvc_next.core.models.base import BaseModel, BaseModelUpdate
from rvc_next.core.models.models import (
    AssignIndexRequest,
    BuildIndexRequest,
    CheckpointInfo,
    ExtractRequest,
    ImportDecisions,
    ImportPathRequest,
    ImportPlan,
    ImportResult,
    InboxIndex,
    IndexPairing,
    LegacyScanResult,
    MergeRequest,
    StagedFile,
    StageUrlsRequest,
    VoiceModel,
    VoiceUpdate,
)
from rvc_next.core.record import Record
from rvc_next.core.separation.library import SeparationModel, SeparationModelUpdate

router = APIRouter(prefix="/v1/models", tags=["models"], responses=ERROR_RESPONSES)


class ImportSession(Record):
    session_id: str


class StagePathsRequest(Record):
    paths: list[str]


class InspectRequest(Record):
    path: str


class LegacyRootCreate(Record):
    path: str
    name: str = ""


class LegacyRootCreated(Record):
    id: str
    scan: LegacyScanResult


@router.get("", operation_id="list_models")
def list_models(services: ServicesDep, include_hidden: bool = False) -> list[VoiceModel]:
    return services.models.list_voices(include_hidden=include_hidden)


@router.get("/catalog", operation_id="list_catalog_voices")
def list_catalog(services: ServicesDep) -> list[CatalogVoice]:
    """Ready-made voices (the official RVC demo voices) that can be downloaded into the library."""
    return services.assets.voice_catalog()


@router.post("/catalog/download", operation_id="download_catalog_voices")
def download_catalog(services: ServicesDep, body: VoiceDownloadRequest) -> Job:
    return services.assets.download_voices(body.ids, all_missing=body.all)


# -- import sessions: stage, inspect, review, commit --------------------------------------------


@router.post("/imports", operation_id="create_import")
def create_import(services: ServicesDep) -> ImportSession:
    """Start an import; add files, read the proposed plan, then commit it with any changes."""
    return ImportSession(session_id=services.imports.create())


@router.put("/imports/{session_id}/files", operation_id="add_import_file", openapi_extra=RAW_BODY)
async def add_import_file(request: Request, services: ServicesDep, session_id: str, name: Annotated[str, Query(max_length=255)]) -> list[StagedFile]:
    """Add a file as the raw request body; a .zip adds every usable member."""
    tmp = await spool(request, services.settings.data_dir / "uploads" / ".incoming")
    try:
        return await run_in_threadpool(services.imports.add_upload, session_id, name, read_chunks(tmp))
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/imports/{session_id}/paths", operation_id="add_import_paths")
def add_import_paths(services: ServicesDep, session_id: str, body: StagePathsRequest) -> list[StagedFile]:
    return services.imports.add_paths(session_id, [Path(p) for p in body.paths])


@router.post("/imports/{session_id}/urls", operation_id="add_import_urls")
def add_import_urls(services: ServicesDep, session_id: str, body: StageUrlsRequest) -> Job:
    """Download links into the import as a job: Hugging Face files, repositories and folders, Google
    Drive files, or any http(s) file. Links must point at public addresses (a reverse proxy can make
    a remote browser look local, so the web never reaches the server's own network; the command
    line can). When the job is done, the plan includes the files."""
    return services.imports.add_urls(session_id, body.urls)


@router.get("/imports/{session_id}", operation_id="get_import_plan")
def get_import_plan(services: ServicesDep, session_id: str) -> ImportPlan:
    """What each file is, and the proposed pairing, with its reasons."""
    return services.imports.plan(session_id)


@router.post("/imports/{session_id}/commit", operation_id="commit_import")
def commit_import(services: ServicesDep, session_id: str, body: ImportDecisions) -> ImportResult:
    """Apply the plan; lists given in the body replace the proposals."""
    return services.imports.commit(session_id, body)


@router.delete("/imports/{session_id}", operation_id="discard_import", status_code=status.HTTP_204_NO_CONTENT)
def discard_import(services: ServicesDep, session_id: str) -> None:
    services.imports.delete(session_id)


# -- unassigned indexes --------------------------------------------------------------------


@router.get("/indexes", operation_id="list_inbox_indexes")
def list_inbox(services: ServicesDep) -> list[InboxIndex]:
    """Indexes waiting for a voice, each with the compatible voices."""
    return services.inbox.list_indexes()


@router.post("/indexes/{index_id}/assign", operation_id="assign_inbox_index")
def assign_inbox(services: ServicesDep, index_id: str, body: AssignIndexRequest) -> VoiceModel:
    return services.inbox.assign(index_id, body)


@router.delete("/indexes/{index_id}", operation_id="delete_inbox_index", status_code=status.HTTP_204_NO_CONTENT)
def delete_inbox(services: ServicesDep, index_id: str) -> None:
    services.inbox.delete(index_id)


# -- base models and separation models -------------------------------------------------------


@router.get("/base", operation_id="list_base_models")
def list_base(services: ServicesDep, sample_rate: str | None = None, version: str | None = None, pitch_guidance: bool | None = None) -> list[BaseModel]:
    """Official and imported training base models; the filters keep those that fit an experiment."""
    return services.base_models.list_base(sample_rate, version, pitch_guidance)


@router.patch("/base/{base_id}", operation_id="update_base_model")
def update_base(services: ServicesDep, base_id: str, body: BaseModelUpdate) -> BaseModel:
    return services.base_models.update(base_id, body)


@router.delete("/base/{base_id}", operation_id="delete_base_model", status_code=status.HTTP_204_NO_CONTENT)
def delete_base(services: ServicesDep, base_id: str) -> None:
    services.base_models.delete(base_id)


@router.get("/separation", operation_id="list_separation_models")
def list_separation(services: ServicesDep) -> list[SeparationModel]:
    """Imported separation models; each is also a preset in Separate."""
    return services.separation_models.list_models()


@router.patch("/separation/{model_id}", operation_id="update_separation_model")
def update_separation(services: ServicesDep, model_id: str, body: SeparationModelUpdate) -> SeparationModel:
    return services.separation_models.update(model_id, body)


@router.delete("/separation/{model_id}", operation_id="delete_separation_model", status_code=status.HTTP_204_NO_CONTENT)
def delete_separation(services: ServicesDep, model_id: str) -> None:
    services.separation_models.delete(model_id)


@router.put("/import", operation_id="import_model_upload", openapi_extra=RAW_BODY)
async def import_upload(request: Request, services: ServicesDep, filename: Annotated[str, Query()], name: str | None = None) -> ImportResult:
    """One-step import of a raw body (``.pth``, ``.index``, ``.zip``, …); ``409`` with ``detail.plan`` when it needs a decision."""
    tmp = await spool(request, services.settings.data_dir / "uploads" / ".incoming")
    try:
        return await run_in_threadpool(services.models.import_upload, filename, read_chunks(tmp), name)
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/import-path", operation_id="import_model_paths")
def import_paths(services: ServicesDep, body: ImportPathRequest) -> ImportResult:
    return services.models.import_paths([Path(p) for p in body.paths], name=body.name, index=Path(body.index) if body.index else None)


@router.post("/inspect", operation_id="inspect_model")
def inspect(services: ServicesDep, body: InspectRequest) -> CheckpointInfo:
    return services.models.inspect(body.path)


@router.post("/merge", operation_id="merge_models")
def merge(services: ServicesDep, body: MergeRequest) -> Job:
    return services.models.merge(body)


@router.post("/extract", operation_id="extract_model")
def extract(services: ServicesDep, body: ExtractRequest) -> Job:
    return services.models.extract(body)


@router.post("/legacy-roots", operation_id="add_legacy_root")
def add_legacy_root(services: ServicesDep, body: LegacyRootCreate) -> LegacyRootCreated:
    root_id = services.models.add_legacy_root(body.path, body.name)
    return LegacyRootCreated(id=root_id, scan=services.models.scan_legacy_root(root_id))


@router.delete("/legacy-roots/{root_id}", operation_id="remove_legacy_root", status_code=status.HTTP_204_NO_CONTENT)
def remove_legacy_root(services: ServicesDep, root_id: str) -> None:
    """Forget an RVC install; its files are not touched."""
    services.models.remove_legacy_root(root_id)


@router.post("/legacy-roots/{root_id}/scan", operation_id="scan_legacy_root")
def scan_legacy_root(services: ServicesDep, root_id: str) -> LegacyScanResult:
    return services.models.scan_legacy_root(root_id)


@router.post("/legacy-roots/{root_id}/import", operation_id="import_legacy_root")
def import_legacy_root(services: ServicesDep, root_id: str) -> ImportResult:
    return services.models.import_legacy_root(root_id)


@router.get("/{voice_id}", operation_id="get_model")
def get_model(services: ServicesDep, voice_id: str) -> VoiceModel:
    return services.models.get(voice_id)


@router.get("/{voice_id}/archive", operation_id="export_model_archive", response_class=StreamingResponse, responses={200: {"content": {"application/zip": {}}}})
def export_archive(services: ServicesDep, voice_id: str) -> StreamingResponse:
    """The voice as a zip: its ``.pth`` (with the speaker names) and its indexes, ready to import elsewhere."""
    name, stream = services.models.export_archive(voice_id)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": content_disposition("attachment", name)})


@router.patch("/{voice_id}", operation_id="update_model")
def update_model(services: ServicesDep, voice_id: str, body: VoiceUpdate) -> VoiceModel:
    return services.models.update(voice_id, body)


@router.delete("/{voice_id}", operation_id="delete_model", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(services: ServicesDep, voice_id: str) -> None:
    services.models.delete(voice_id)


@router.put("/{voice_id}/index", operation_id="upload_model_index", openapi_extra=RAW_BODY)
async def upload_index(request: Request, services: ServicesDep, voice_id: str, key: str = "default", filename: str | None = None) -> VoiceModel:
    """Attach an index uploaded as the raw request body; any file name works, but it must fit the voice."""
    tmp = await spool(request, services.settings.data_dir / "uploads" / ".incoming")
    try:
        return await run_in_threadpool(services.models.set_index_upload, voice_id, key, read_chunks(tmp), filename)
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/{voice_id}/index", operation_id="set_model_index")
def set_index(services: ServicesDep, voice_id: str, body: IndexPairing) -> VoiceModel:
    """Pair a server-side index file, or detach one (``path: null``)."""
    return services.models.set_index(voice_id, body.key, Path(body.path) if body.path else None)


@router.post("/{voice_id}/index/build", operation_id="build_model_index")
def build_index(services: ServicesDep, voice_id: str, body: BuildIndexRequest) -> Job:
    return services.training.build_index_for_voice(voice_id, body.experiment)


@router.post("/{voice_id}/reveal", operation_id="reveal_model", status_code=status.HTTP_204_NO_CONTENT, dependencies=[LocalDep])
def reveal(services: ServicesDep, voice_id: str) -> None:
    services.audio.reveal(Path(services.models.get(voice_id).model_path))
