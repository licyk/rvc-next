"""Domain exceptions shared by the API and the command line."""

from typing import Any


class RvcNextError(Exception):
    """Base class for every error the wrappers translate."""

    code = "internal_error"
    http_status = 500
    exit_code = 1

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class NotFoundError(RvcNextError):
    """A model, job, file or experiment is unknown."""

    code = "not_found"
    http_status = 404
    exit_code = 2


class ConflictError(RvcNextError):
    """A name clash, or an import over an existing model."""

    code = "conflict"
    http_status = 409
    exit_code = 3


class BusyError(ConflictError):
    """Live is already running, or a stage of the same experiment is running."""

    code = "busy"


class InvalidPathError(RvcNextError):
    """A path escapes its root, or a server path is outside the browse roots."""

    code = "invalid_path"
    http_status = 400
    exit_code = 4


class ValidationError(RvcNextError):
    """A parameter is out of range."""

    code = "invalid_input"
    http_status = 400
    exit_code = 4


class ModelFormatError(ValidationError):
    """A ``.pth`` is not an RVC small model or a G checkpoint."""

    code = "invalid_model"


class AssetMissingError(RvcNextError):
    """HuBERT, RMVPE or a base model is not installed. ``detail["assets"]`` lists the ids."""

    code = "asset_missing"
    http_status = 409
    exit_code = 7

    def __init__(self, message: str, assets: list[str], detail: dict[str, Any] | None = None) -> None:
        super().__init__(message, {**(detail or {}), "assets": list(assets)})
        self.assets = list(assets)


class DeviceError(RvcNextError):
    """An audio device is missing, busy or refuses the format. ``detail["reason"]`` says which."""

    code = "device_unavailable"
    http_status = 409
    exit_code = 8

    def __init__(self, message: str, reason: str = "missing", detail: dict[str, Any] | None = None) -> None:
        super().__init__(message, {**(detail or {}), "reason": reason})
        self.reason = reason


class ComputeError(RvcNextError):
    """The chosen GPU is not usable, or it ran out of memory."""

    code = "compute_unavailable"
    http_status = 503
    exit_code = 9
