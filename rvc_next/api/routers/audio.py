"""Audio inputs: uploads, server files and browsing."""

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from rvc_next.api.deps import ServicesDep, require_trusted
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.api.files import audio_response
from rvc_next.api.uploads import RAW_BODY, read_chunks, spool
from rvc_next.core.audio.models import AudioAnalysis, AudioFile, BrowseListing, Peaks, ResolvePathRequest

router = APIRouter(prefix="/v1/audio", tags=["audio"], responses=ERROR_RESPONSES)


@router.put("/uploads", operation_id="upload_audio", openapi_extra=RAW_BODY)
async def upload(request: Request, services: ServicesDep, name: Annotated[str, Query(max_length=255)]) -> AudioFile:
    """Upload an audio file as the raw request body."""
    tmp = await spool(request, services.audio.uploads_dir / ".incoming")
    try:
        return await run_in_threadpool(services.audio.upload, name, read_chunks(tmp))
    finally:
        tmp.unlink(missing_ok=True)


@router.get("/files/{file_id}", operation_id="get_audio_file")
def get_file(services: ServicesDep, file_id: str) -> AudioFile:
    return services.audio.get_file(file_id)


@router.get("/files/{file_id}/content", operation_id="get_audio_content", response_class=FileResponse)
def get_content(services: ServicesDep, file_id: str, download: bool = False) -> FileResponse:
    return audio_response(services.audio.file_path(file_id), download)


@router.get("/files/{file_id}/analysis", operation_id="analyse_audio_file")
def analyse_file(services: ServicesDep, file_id: str, columns: int = 800) -> AudioAnalysis:
    """A spectrogram and the pitch curve of an uploaded file (cached)."""
    return services.audio.analyse(services.audio.file_path(file_id), columns)


@router.get("/files/{file_id}/peaks", operation_id="get_audio_peaks")
def get_peaks(services: ServicesDep, file_id: str, points: int = 1024) -> Peaks:
    return services.audio.peaks(services.audio.file_path(file_id), points)


@router.get("/browse", operation_id="browse_server")
def browse(request: Request, services: ServicesDep, path: str = "") -> BrowseListing:
    require_trusted(request, services)
    return services.audio.browse(path or None)


@router.post("/resolve-path", operation_id="resolve_audio_path")
def resolve_path(request: Request, services: ServicesDep, body: ResolvePathRequest) -> AudioFile:
    require_trusted(request, services)
    return services.audio.resolve_path(body.path)


@router.get("/path-peaks", operation_id="get_path_peaks")
def path_peaks(request: Request, services: ServicesDep, path: str, points: int = 1024) -> Peaks:
    require_trusted(request, services)
    return services.audio.peaks(services.audio.file_path(services.audio.resolve_path(path).id), points)
