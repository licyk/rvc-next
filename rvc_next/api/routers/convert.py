"""Conversion."""

from fastapi import APIRouter, Request

from rvc_next.api.deps import ServicesDep, require_trusted
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.core.conversion.models import ConvertRequest
from rvc_next.core.errors import ValidationError
from rvc_next.core.jobs.models import Job

router = APIRouter(prefix="/v1/convert", tags=["convert"], responses=ERROR_RESPONSES)


@router.post("", operation_id="convert")
def convert(request: Request, services: ServicesDep, body: ConvertRequest) -> Job:
    if any(ref.kind == "path" for ref in body.inputs):
        require_trusted(request, services)
        services.audio.check_refs(body.inputs)
    if body.output_dir or body.overwrite:
        raise ValidationError("output_dir and overwrite are for the command line")
    return services.conversion.convert(body)
