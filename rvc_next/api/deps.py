"""FastAPI dependencies. Services come from ``app.state`` rather than a global locator."""

from typing import Annotated

from fastapi import Depends, Request

from rvc_next.core.context import Services
from rvc_next.core.errors import RvcNextError
from rvc_next.core.net.ports import is_loopback


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


class NotLocalError(RvcNextError):
    code = "not_local"
    http_status = 403
    exit_code = 4


def is_local(request: Request) -> bool:
    client = request.client.host if request.client else ""
    return bool(client) and (is_loopback(client) or client == "testclient")


def require_local(request: Request) -> None:
    """Revealing files works only for a browser on the server's own machine."""
    if not is_local(request):
        raise NotLocalError("This works only in a browser on the server's own machine")


def require_trusted(request: Request, services: Services) -> None:
    """Server paths: a loopback client, or any client once an access token is set (the middleware checked it)."""
    if not is_local(request) and not services.settings.settings.server.access_token:
        raise NotLocalError("Server files can be browsed off this machine only when an access token is set")


LocalDep = Depends(require_local)
