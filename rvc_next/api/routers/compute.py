"""Compute devices, memory and the model cache."""

from fastapi import APIRouter

from rvc_next.api.deps import ServicesDep
from rvc_next.core.compute.models import ComputeInfo, ComputeUsage

router = APIRouter(prefix="/v1/compute", tags=["compute"])


@router.get("/devices", operation_id="get_compute_devices")
def get_devices(services: ServicesDep) -> ComputeInfo:
    return services.compute.info()


@router.get("/usage", operation_id="get_compute_usage")
def get_usage(services: ServicesDep) -> ComputeUsage:
    return services.compute.usage()


@router.post("/release", operation_id="release_compute")
def release(services: ServicesDep) -> ComputeUsage:
    """Free GPU memory: unload every cached model."""
    return services.compute.release()
