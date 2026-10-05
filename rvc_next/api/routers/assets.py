"""Model assets: status, downloads and verification."""

from fastapi import APIRouter

from rvc_next.api.deps import ServicesDep
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.core.assets.models import AssetDownloadRequest, AssetStatus, Repository
from rvc_next.core.jobs.models import Job

router = APIRouter(prefix="/v1/assets", tags=["assets"], responses=ERROR_RESPONSES)


@router.get("", operation_id="list_assets")
def list_assets(services: ServicesDep) -> list[AssetStatus]:
    return services.assets.list_assets()


@router.get("/repositories", operation_id="list_asset_repositories")
def repositories(services: ServicesDep) -> list[Repository]:
    """The Hugging Face repositories models can come from; ``downloads.repository`` chooses one."""
    return services.assets.repositories()


@router.post("/download", operation_id="download_assets")
def download(services: ServicesDep, body: AssetDownloadRequest) -> Job:
    return services.assets.download(body.ids, body.group)


@router.post("/verify", operation_id="verify_assets")
def verify(services: ServicesDep, body: AssetDownloadRequest) -> Job:
    return services.assets.download(body.ids, body.group, verify_only=True)


@router.delete("/{asset_id}", operation_id="delete_asset")
def delete(services: ServicesDep, asset_id: str) -> AssetStatus:
    """Remove a downloaded asset's files; it shows as not installed and can be downloaded again."""
    return services.assets.delete(asset_id)
