"""Results of conversion and separation."""

from fastapi import APIRouter, status
from fastapi.responses import FileResponse, StreamingResponse

from rvc_next.api.deps import LocalDep, ServicesDep
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.api.files import audio_response, content_disposition
from rvc_next.core.audio.models import AudioAnalysis, Output, OutputPage, Peaks, ZipRequest
from rvc_next.core.storage import PathCheckReport, PathCheckRequest

router = APIRouter(prefix="/v1/outputs", tags=["outputs"], responses=ERROR_RESPONSES)


@router.get("", operation_id="list_outputs")
def list_outputs(services: ServicesDep, job_id: str | None = None, voice_id: str | None = None, kind: str | None = None, cursor: str | None = None, limit: int = 100) -> OutputPage:
    return services.audio.list_outputs(job_id, voice_id, kind, cursor, max(1, min(limit, 500)))


@router.post("/zip", operation_id="zip_outputs", response_class=StreamingResponse, responses={200: {"content": {"application/zip": {}}}})
def zip_outputs(services: ServicesDep, body: ZipRequest) -> StreamingResponse:
    stream = services.audio.zip_outputs(body.ids)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": content_disposition("attachment", "rvc-next-outputs.zip")})


@router.post("/check", operation_id="check_output_files")
def check_files(services: ServicesDep, body: PathCheckRequest) -> PathCheckReport:
    """Where every recorded output, source and audio file is; with ``fix``, records rewritten to where the files are now."""
    return services.storage.check(body)


@router.get("/{output_id}", operation_id="get_output")
def get_output(services: ServicesDep, output_id: str) -> Output:
    return services.audio.get_output(output_id)


@router.get("/{output_id}/content", operation_id="get_output_content", response_class=FileResponse)
def get_content(services: ServicesDep, output_id: str, download: bool = False) -> FileResponse:
    return audio_response(services.audio.output_file(output_id), download)


@router.get("/{output_id}/source", operation_id="get_output_source", response_class=FileResponse)
def get_source(services: ServicesDep, output_id: str) -> FileResponse:
    """The input this output was made from, for A/B comparison."""
    return audio_response(services.audio.source_file(output_id))


@router.get("/{output_id}/peaks", operation_id="get_output_peaks")
def get_peaks(services: ServicesDep, output_id: str, points: int = 1024) -> Peaks:
    return services.audio.peaks(services.audio.output_file(output_id), points)


@router.get("/{output_id}/analysis", operation_id="analyse_output")
def analyse_output(services: ServicesDep, output_id: str, columns: int = 800) -> AudioAnalysis:
    """A spectrogram and the pitch curve of an output (cached)."""
    return services.audio.analyse(services.audio.output_file(output_id), columns)


@router.get("/{output_id}/source-analysis", operation_id="analyse_output_source")
def analyse_output_source(services: ServicesDep, output_id: str, columns: int = 800) -> AudioAnalysis:
    """The same for the file the output was made from."""
    return services.audio.analyse(services.audio.source_file(output_id), columns)


@router.get("/{output_id}/source-peaks", operation_id="get_output_source_peaks")
def get_source_peaks(services: ServicesDep, output_id: str, points: int = 1024) -> Peaks:
    return services.audio.peaks(services.audio.source_file(output_id), points)


@router.delete("/{output_id}", operation_id="delete_output", status_code=status.HTTP_204_NO_CONTENT)
def delete_output(services: ServicesDep, output_id: str) -> None:
    services.audio.delete_output(output_id)


@router.post("/{output_id}/reveal", operation_id="reveal_output", status_code=status.HTTP_204_NO_CONTENT, dependencies=[LocalDep])
def reveal(services: ServicesDep, output_id: str) -> None:
    services.audio.reveal(services.audio.output_file(output_id))
