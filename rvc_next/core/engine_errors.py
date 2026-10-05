"""Map engine exceptions to domain errors. The engine never imports the core."""

from __future__ import annotations

from rvc_next.core.errors import AssetMissingError, ComputeError, ModelFormatError, RvcNextError, ValidationError


def map_engine_error(e: BaseException) -> RvcNextError | None:
    """The domain error for an engine exception, or None if it is not one the domain knows."""
    from rvc_next.engine import errors as ee

    if isinstance(e, RvcNextError):
        return e
    if isinstance(e, ee.MissingAssetError):
        return AssetMissingError(str(e), e.assets)
    if isinstance(e, ee.ModelFormatError):
        return ModelFormatError(str(e))
    if isinstance(e, ee.AudioError):
        return ValidationError(str(e), {"reason": "audio"})
    from rvc_next.engine.runtime import is_oom
    from rvc_next.protocol.messages import OUT_OF_MEMORY

    if is_oom(e):
        return ComputeError(f"Out of GPU memory: {e}", {"reason": OUT_OF_MEMORY})
    if "cuda error" in str(e).lower():
        return ComputeError(str(e))
    return None


def is_out_of_memory(error: RvcNextError) -> bool:
    """Whether a domain error is a device running out of memory (in this process or a worker)."""
    from rvc_next.protocol.messages import OUT_OF_MEMORY

    return isinstance(error, ComputeError) and error.detail.get("reason") == OUT_OF_MEMORY


def worker_error(code: str, message: str, detail: dict | None = None) -> RvcNextError:
    """Rebuild a domain error from a worker's ``error`` event."""
    from rvc_next.core import errors as de

    detail = detail or {}
    if code == "asset_missing":
        return de.AssetMissingError(message, list(detail.get("assets", [])), detail)
    if code == de.DeviceError.code:
        return de.DeviceError(message, str(detail.get("reason", "missing")), detail)
    for cls in (de.NotFoundError, de.ConflictError, de.BusyError, de.InvalidPathError, de.ValidationError, de.ModelFormatError, de.ComputeError):
        if cls.code == code:
            return cls(message, detail)
    return RvcNextError(message, detail)
