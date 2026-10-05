"""rvc-next: retrieval-based voice conversion with one web UI and one command line.

To embed the web UI and API in another application:

    from rvc_next import RvcNextServer

    server = RvcNextServer(data_dir="./rvc-next-data", config_dir="./config", port=0)
    print(server.start())   # http://127.0.0.1:54123
"""

from typing import TYPE_CHECKING, Any

from rvc_next.version import VERSION

if TYPE_CHECKING:
    from rvc_next.embed import RvcNextServer, serve

__version__ = VERSION
__all__ = ["VERSION", "RvcNextServer", "__version__", "serve"]

_LAZY = {"RvcNextServer", "serve"}


def __getattr__(name: str) -> Any:
    """Import the server lazily, so ``rvc-next version`` does not pay for FastAPI."""
    if name in _LAZY:
        from rvc_next import embed

        return getattr(embed, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
