"""Run a stage's parts in child processes and gather their progress through one queue.

Children are spawned (not forked: CUDA and some BLAS builds break across fork) and are daemons, so
they end with the stage. A cancel token terminates them.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import traceback
from collections.abc import Callable
from typing import Any

from rvc_next.engine.cancel import CancelToken
from rvc_next.engine.errors import Cancelled, EngineError

Progress = Callable[[float | None, str | None], None]
Log = Callable[[str], None]


class PartReporter:
    """What a part uses to report: ``done(label)`` per finished item, ``log(message)``."""

    def __init__(self, q: Any | None, on_done: Callable[[str], None] | None = None, on_log: Log | None = None) -> None:
        self.q = q
        self.on_done = on_done
        self.on_log = on_log

    def done(self, label: str = "") -> None:
        if self.q is not None:
            self.q.put(("done", label))
        elif self.on_done:
            self.on_done(label)

    def log(self, message: str) -> None:
        if self.q is not None:
            self.q.put(("log", message))
        elif self.on_log:
            self.on_log(message)


def _child(target: Callable[..., dict[str, Any]], args: tuple, q: Any) -> None:
    try:
        result = target(*args, PartReporter(q))
        q.put(("result", result))
    except BaseException as e:  # reported to the parent, which raises
        q.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def run_parts(
    target: Callable[..., dict[str, Any]],
    parts: list[tuple],
    total_items: int,
    progress: Progress | None = None,
    log: Log | None = None,
    cancel: CancelToken | None = None,
    label: str | None = None,
    env: list[dict[str, str] | None] | None = None,
) -> list[dict[str, Any]]:
    """Call ``target(*part, reporter)`` for every part; in this process when there is one part.

    ``env`` may give per-part environment variables (``CUDA_VISIBLE_DEVICES``), applied in the child.
    Returns each part's result dict.
    """
    done = 0

    def tick(item: str) -> None:
        nonlocal done
        done += 1
        if progress and total_items:
            progress(min(1.0, done / total_items), item or label)

    if len(parts) <= 1:
        results = []
        for part in parts:
            if cancel:
                cancel.check()
            results.append(target(*part, _InlineReporter(tick, log, cancel)))
        return results

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for i, part in enumerate(parts):
        p = ctx.Process(target=_env_child, args=(target, part, q, (env or [None] * len(parts))[i]), daemon=True)
        p.start()
        procs.append(p)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        finished = 0
        while finished < len(procs):
            if cancel and cancel.cancelled:
                raise Cancelled("Cancelled")
            try:
                kind, payload = q.get(timeout=0.2)
            except queue_mod.Empty:
                if all(not p.is_alive() for p in procs):
                    # Drain whatever is left, then stop waiting for parts that died silently.
                    try:
                        while True:
                            kind, payload = q.get_nowait()
                            finished += _handle(kind, payload, tick, log, results, errors)
                    except queue_mod.Empty:
                        pass
                    break
                continue
            finished += _handle(kind, payload, tick, log, results, errors)
    finally:
        for p in procs:
            if p.is_alive() and (cancel is not None and cancel.cancelled or errors):
                p.terminate()
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
    if errors:
        raise EngineError(errors[0])
    if len(results) < len(parts):
        raise EngineError(f"{len(parts) - len(results)} of {len(parts)} parts ended without a result")
    return results


def _env_child(target: Callable[..., dict[str, Any]], part: tuple, q: Any, env: dict[str, str] | None) -> None:
    if env:
        import os

        os.environ.update(env)
    _child(target, part, q)


def _handle(kind: str, payload: Any, tick: Callable[[str], None], log: Log | None, results: list, errors: list) -> int:
    if kind == "done":
        tick(payload)
    elif kind == "log":
        if log:
            log(payload)
    elif kind == "result":
        results.append(payload)
        return 1
    elif kind == "error":
        errors.append(payload)
        return 1
    return 0


class _InlineReporter(PartReporter):
    def __init__(self, tick: Callable[[str], None], log: Log | None, cancel: CancelToken | None) -> None:
        super().__init__(None, tick, log)
        self.cancel = cancel

    def done(self, label: str = "") -> None:
        super().done(label)
        if self.cancel:
            self.cancel.check()


def merge_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        for k, v in r.items():
            if isinstance(v, int):
                out[k] = out.get(k, 0) + v
    return out
