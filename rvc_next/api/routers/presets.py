"""Presets."""

from fastapi import APIRouter, status

from rvc_next.api.deps import ServicesDep
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.core.presets.models import Preset, PresetCreate, PresetUpdate

router = APIRouter(prefix="/v1/presets", tags=["presets"], responses=ERROR_RESPONSES)


@router.get("", operation_id="list_presets")
def list_presets(services: ServicesDep, voice_id: str | None = None) -> list[Preset]:
    if voice_id:
        services.presets.ensure_default(voice_id)
    return services.presets.list_presets(voice_id)


@router.post("", operation_id="create_preset")
def create_preset(services: ServicesDep, body: PresetCreate) -> Preset:
    return services.presets.create(body)


@router.patch("/{preset_id}", operation_id="update_preset")
def update_preset(services: ServicesDep, preset_id: str, body: PresetUpdate) -> Preset:
    return services.presets.update(preset_id, body)


@router.delete("/{preset_id}", operation_id="delete_preset", status_code=status.HTTP_204_NO_CONTENT)
def delete_preset(services: ServicesDep, preset_id: str) -> None:
    services.presets.delete(preset_id)
