"""Separation."""

from fastapi import APIRouter

from rvc_next.api.deps import ServicesDep
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.core.errors import ValidationError
from rvc_next.core.jobs.models import Job
from rvc_next.core.separation.models import SeparateRequest, SeparationPreset

router = APIRouter(prefix="/v1/separate", tags=["separate"], responses=ERROR_RESPONSES)


@router.get("/presets", operation_id="list_separation_presets")
def presets(services: ServicesDep) -> list[SeparationPreset]:
    return services.separation.presets()


@router.post("", operation_id="separate")
def separate(services: ServicesDep, body: SeparateRequest) -> Job:
    if any(ref.kind == "path" for ref in body.inputs):
        services.audio.check_refs(body.inputs)
    if body.output_dir:
        raise ValidationError("output_dir is for the command line")
    return services.separation.separate(body)
