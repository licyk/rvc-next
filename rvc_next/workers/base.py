"""Shared start-up for subprocess workers."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rvc_next.engine.errors import Cancelled, MissingAssetError, ModelFormatError
from rvc_next.engine.runtime import is_oom
from rvc_next.protocol.jsonl import Emitter
from rvc_next.protocol.messages import EXIT_CANCELLED, OUT_OF_MEMORY, ErrorEvent, LogEvent

WorkerMain = Callable[[dict[str, Any], Emitter], None]


class _LogHandler(logging.Handler):
    def __init__(self, emitter: Emitter) -> None:
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.emit(LogEvent(level=record.levelname.lower(), message=self.format(record)))
        except Exception:
            pass


def run_worker(main: WorkerMain, argv: list[str] | None = None) -> int:
    """Read the request file named by the first argument, run ``main``, and map failures to events.

    The events go to the real stdout; anything else a library prints to stdout is moved to stderr,
    which the runner keeps as log text.
    """
    args = sys.argv[1:] if argv is None else argv
    real_stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    emitter = Emitter(real_stdout)
    handler = _LogHandler(emitter)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("rvc_next")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(EXIT_CANCELLED))
    try:
        if not args:
            raise SystemExit("usage: python -m rvc_next.workers.<name> <request.json>")
        request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        main(request, emitter)
        return 0
    except Cancelled:
        return EXIT_CANCELLED
    except MissingAssetError as e:
        emitter.emit(ErrorEvent(code="asset_missing", message=str(e), detail={"assets": e.assets}))
        return 1
    except ModelFormatError as e:
        emitter.emit(ErrorEvent(code="invalid_model", message=str(e)))
        return 1
    except KeyboardInterrupt:
        return EXIT_CANCELLED
    except Exception as e:
        emitter.emit(LogEvent(level="error", message=traceback.format_exc()))
        if is_oom(e):
            # The process exits next, which frees its memory; the server unloads its own models too.
            emitter.emit(ErrorEvent(code="compute_unavailable", message=f"Out of GPU memory: {e}", detail={"reason": OUT_OF_MEMORY}))
        else:
            emitter.emit(ErrorEvent(code="internal_error", message=str(e) or type(e).__name__))
        return 1
    finally:
        try:
            real_stdout.flush()
        except Exception:
            pass
