"""Training experiments."""

from typing import Annotated

from fastapi import APIRouter, Query, Request, status
from starlette.concurrency import run_in_threadpool

from rvc_next.api.deps import ServicesDep, require_trusted
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.api.uploads import RAW_BODY, read_chunks, spool
from rvc_next.core.audio.models import Output
from rvc_next.core.jobs.models import Job
from rvc_next.core.models.models import VoiceModel
from rvc_next.core.training.models import (
    Checkpoint,
    DatasetReport,
    Experiment,
    ExperimentCreate,
    ExperimentSummary,
    ExperimentUpdate,
    ExportRequest,
    FolderSpeakersRequest,
    ImportExperimentRequest,
    RunRequest,
    SpeakerEntry,
    TrainMetric,
)

router = APIRouter(prefix="/v1/train", tags=["train"], responses=ERROR_RESPONSES)


@router.get("/experiments", operation_id="list_experiments")
def list_experiments(services: ServicesDep) -> list[ExperimentSummary]:
    return services.training.list_experiments()


@router.post("/experiments", operation_id="create_experiment")
def create_experiment(request: Request, services: ServicesDep, body: ExperimentCreate) -> Experiment:
    if body.dataset.folder or body.dataset.speakers:
        require_trusted(request, services)
    return services.training.create(body)


@router.post("/import", operation_id="import_experiment")
def import_experiment(request: Request, services: ServicesDep, body: ImportExperimentRequest) -> Experiment:
    require_trusted(request, services)
    return services.training.import_legacy(body)


@router.get("/experiments/{name}", operation_id="get_experiment")
def get_experiment(services: ServicesDep, name: str) -> Experiment:
    return services.training.get(name)


@router.patch("/experiments/{name}", operation_id="update_experiment")
def update_experiment(request: Request, services: ServicesDep, name: str, body: ExperimentUpdate) -> Experiment:
    if body.dataset is not None:
        require_trusted(request, services)
    return services.training.update(name, body)


@router.delete("/experiments/{name}", operation_id="delete_experiment", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(services: ServicesDep, name: str) -> None:
    services.training.delete(name)


@router.post("/experiments/{name}/dataset/scan", operation_id="scan_dataset")
def scan_dataset(services: ServicesDep, name: str) -> DatasetReport:
    return services.training.scan_dataset(name)


@router.put("/experiments/{name}/speakers", operation_id="set_speakers")
def set_speakers(request: Request, services: ServicesDep, name: str, body: list[SpeakerEntry]) -> Experiment:
    require_trusted(request, services)
    return services.training.set_speakers(name, body)


@router.put("/experiments/{name}/dataset/files", operation_id="upload_dataset_file", openapi_extra=RAW_BODY)
async def upload_dataset_file(request: Request, services: ServicesDep, name: str, filename: Annotated[str, Query(max_length=255)]) -> Experiment:
    """Add an audio file, or a zip of them, to the experiment's own dataset folder (the raw request
    body). In multi-speaker mode a zip's top-level ``Name_ID_Repeat`` folders become the speakers."""
    tmp = await spool(request, services.training.experiment_dir(name) / ".incoming")
    try:
        return await run_in_threadpool(services.training.add_dataset_file, name, filename, read_chunks(tmp))
    finally:
        tmp.unlink(missing_ok=True)


@router.delete("/experiments/{name}/dataset/files", operation_id="clear_dataset_uploads")
def clear_dataset_uploads(services: ServicesDep, name: str) -> Experiment:
    return services.training.clear_dataset_uploads(name)


@router.post("/experiments/{name}/speakers/from-folders", operation_id="speakers_from_folders")
def speakers_from_folders(request: Request, services: ServicesDep, name: str, body: FolderSpeakersRequest) -> Experiment:
    """Fill the speaker table from ``Name_ID_Repeat`` subfolders."""
    require_trusted(request, services)
    return services.training.speakers_from_folders(name, body.folder)


@router.post("/experiments/{name}/run", operation_id="run_experiment")
def run_experiment(services: ServicesDep, name: str, body: RunRequest) -> Job:
    return services.training.run(name, body)


@router.post("/experiments/{name}/stop", operation_id="stop_experiment")
def stop_experiment(services: ServicesDep, name: str) -> Experiment:
    return services.training.stop(name)


@router.get("/experiments/{name}/metrics", operation_id="get_experiment_metrics")
def get_metrics(services: ServicesDep, name: str, since: int = 0) -> list[TrainMetric]:
    return services.training.metrics(name, since)


@router.get("/experiments/{name}/checkpoints", operation_id="list_checkpoints")
def list_checkpoints(services: ServicesDep, name: str) -> list[Checkpoint]:
    return services.training.checkpoints(name)


@router.get("/experiments/{name}/samples", operation_id="list_training_samples")
def list_samples(services: ServicesDep, name: str) -> list[Output]:
    """One training clip rendered at every save, newest first; each plays against the clip itself."""
    return services.training.samples(name)


@router.post("/experiments/{name}/export", operation_id="export_experiment")
def export(services: ServicesDep, name: str, body: ExportRequest) -> Job:
    return services.training.export(name, body)


@router.post("/experiments/{name}/try", operation_id="try_checkpoint")
def try_checkpoint(services: ServicesDep, name: str, body: ExportRequest) -> VoiceModel:
    """A temporary voice for a saved small model, to open in Convert."""
    return services.training.try_checkpoint(name, body)
