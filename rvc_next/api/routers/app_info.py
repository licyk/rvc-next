"""App version, health and what the interface needs to know about this server."""

import importlib.util
import socket

from fastapi import APIRouter, Request

from rvc_next.api.deps import ServicesDep, is_local
from rvc_next.core.record import Record
from rvc_next.version import VERSION

router = APIRouter(prefix="/v1/app", tags=["app"])


class AppVersion(Record):
    version: str


class Health(Record):
    status: str
    auth_required: bool


class AppMeta(Record):
    local: bool
    """Whether this request comes from the server's own machine: only then can it reveal files or use server devices directly."""
    host: str
    """The server's host name; the audio device picker names it."""
    compute: str
    """The configured compute device (auto, cpu, cuda:N, dml, mps)."""
    live_available: bool
    """Whether audio device access (sounddevice) is installed on the server."""
    assets_ready: bool
    """Whether HuBERT and RMVPE are installed."""
    features: list[str]
    data_dir: str
    trusted: bool
    """Whether this client may browse server folders."""


@router.get("/version", operation_id="get_app_version")
def get_version() -> AppVersion:
    return AppVersion(version=VERSION)


@router.get("/health", operation_id="get_health")
def get_health(services: ServicesDep) -> Health:
    return Health(status="ok", auth_required=bool(services.settings.settings.server.access_token))


@router.get("/meta", operation_id="get_app_meta")
def get_meta(request: Request, services: ServicesDep) -> AppMeta:
    local = is_local(request)
    return AppMeta(
        local=local,
        host=socket.gethostname(),
        compute=services.settings.settings.compute.device,
        live_available=importlib.util.find_spec("sounddevice") is not None,
        assets_ready=services.assets.is_installed("hubert") and services.assets.is_installed("rmvpe"),
        features=["convert", "live", "separate", "train", "models"],
        data_dir=str(services.settings.data_dir),
        trusted=local or bool(services.settings.settings.server.access_token),
    )
