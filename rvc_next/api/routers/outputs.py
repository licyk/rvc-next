"""Results of conversion and separation."""

from pathlib import Path

from fastapi import APIRouter, status
from fastapi.responses import FileResponse, StreamingResponse

from rvc_next.api.deps import LocalDep, ServicesDep
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.api.files import audio_response, content_disposition
from rvc_next.core.audio.models import Output, OutputPage, Peaks, ZipRequest

router = APIRouter(prefix="/v1/outputs", tags=["outputs"], responses=ERROR_RESPONSES)


@router.get("", operation_id="list_outputs")
def list_outputs(services: ServicesDep, job_id: str | None = None, voice_id: str | None = None, kind: str | None = None, cursor: str | None = None, limit: int = 100) -> OutputPage:
    return services.audio.list_outputs(job_id, voice_id, kind, cursor, max(1, min(limit, 500)))


@router.post("/zip", operation_id="zip_outputs", response_class=StreamingResponse, responses={200: {"content": {"application/zip": {}}}})
def zip_outputs(services: ServicesDep, body: ZipRequest) -> StreamingResponse:
    stream = services.audio.zip_outputs(body.ids)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": content_disposition("attachment", "rvc-next-outputs.zip")})


@router.get("/{output_id}", operation_id="get_output")
def get_output(services: ServicesDep, output_id: str) -> Output:
    return services.audio.get_output(output_id)


@router.get("/{output_id}/content", operation_id="get_output_content", response_class=FileResponse)
def get_content(services: ServicesDep, output_id: str, download: bool = False) -> FileResponse:
    return audio_response(services.audio.output_file(output_id), download)


@router.get("/{output_id}/source", operation_id="get_output_source", response_class=FileResponse)
def get_source(services: ServicesDep, output_id: str) -> FileResponse:
    """The input this output was made from, for A/B comparison."""
    from rvc_next.core.errors import NotFoundError

    out = services.audio.get_output(output_id)
    if not out.source_path or not Path(out.source_path).is_file():
        raise NotFoundError("The source of this output is gone")
    return audio_response(Path(out.source_path))


@router.get("/{output_id}/peaks", operation_id="get_output_peaks")
def get_peaks(services: ServicesDep, output_id: str, points: int = 1024) -> Peaks:
    return services.audio.peaks(services.audio.output_file(output_id), points)


@router.get("/{output_id}/source-peaks", operation_id="get_output_source_peaks")
def get_source_peaks(services: ServicesDep, output_id: str, points: int = 1024) -> Peaks:
    from rvc_next.core.errors import NotFoundError

    out = services.audio.get_output(output_id)
    if not out.source_path or not Path(out.source_path).is_file():
        raise NotFoundError("The source of this output is gone")
    return services.audio.peaks(Path(out.source_path), points)


@router.delete("/{output_id}", operation_id="delete_output", status_code=status.HTTP_204_NO_CONTENT)
def delete_output(services: ServicesDep, output_id: str) -> None:
    services.audio.delete_output(output_id)


@router.post("/{output_id}/reveal", operation_id="reveal_output", status_code=status.HTTP_204_NO_CONTENT, dependencies=[LocalDep])
def reveal(services: ServicesDep, output_id: str) -> None:
    services.audio.reveal(services.audio.output_file(output_id))
