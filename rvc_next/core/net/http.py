"""One shared HTTP client, replaceable in tests."""

from __future__ import annotations

import threading
from typing import Any

import httpx

from rvc_next.version import VERSION

USER_AGENT = f"rvc-next/{VERSION}"


class HttpClientProvider:
    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()

    def client(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                kwargs: dict[str, Any] = {"follow_redirects": True, "headers": {"User-Agent": USER_AGENT}, "timeout": httpx.Timeout(30.0, read=60.0)}
                if self._transport is not None:
                    kwargs["transport"] = self._transport
                self._client = httpx.Client(**kwargs)
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
