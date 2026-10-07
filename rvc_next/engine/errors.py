"""Engine exceptions. The core maps them to its domain errors; the engine never imports the core."""


class EngineError(Exception):
    """Base class for engine failures."""


class MissingAssetError(EngineError):
    """A required asset file is not installed. ``assets`` holds the catalog ids."""

    def __init__(self, message: str, assets: list[str]) -> None:
        super().__init__(message)
        self.assets = assets


class ModelFormatError(EngineError):
    """A ``.pth`` is neither an RVC small model nor a G checkpoint. ``detail`` says why, for clients."""

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


class AudioError(EngineError):
    """Audio could not be decoded or encoded."""


class Cancelled(EngineError):
    """The operation was cancelled through its token."""
