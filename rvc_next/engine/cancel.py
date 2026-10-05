"""Cooperative cancellation for in-process work."""

import threading

from rvc_next.engine.errors import Cancelled


class CancelToken:
    """Set from any thread; checked by the engine between chunks."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise ``Cancelled`` if cancellation was requested."""
        if self._event.is_set():
            raise Cancelled("Cancelled")
