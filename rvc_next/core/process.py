"""Starting and stopping worker processes."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


def worker_command(module: str, *args: str) -> list[str]:
    """``python -m <module> <args>`` with this interpreter (the original's ``--pycmd`` is gone)."""
    return [sys.executable, "-m", module, *args]


def start_process(cmd: Sequence[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.Popen[str]:
    """Start ``cmd`` in its own process group, with stdout piped as text and stderr merged into it."""
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    full_env = {**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=full_env,
        cwd=str(cwd) if cwd else None,
        **kwargs,
    )


def kill_process_tree(process: subprocess.Popen | None) -> bool:
    """Terminate a process and every child it created (from the original ``tools/process_utils.py``)."""
    if process is None or process.poll() is not None:
        return False
    pid = process.pid
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/t", "/f", "/pid", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass
    else:
        try:
            group = os.getpgid(pid)
        except OSError:
            group = None
        own = group is not None and group != os.getpgrp()
        try:
            os.killpg(group, signal.SIGTERM) if own and group is not None else os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(20):
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if process.poll() is None:
            try:
                os.killpg(group, signal.SIGKILL) if own and group is not None else os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass
    logger.debug("Process tree %s terminated", pid)
    return True


WINDOWS_STATUS = {
    0xC0000005: "an access violation",
    0xC000001D: "an illegal instruction",
    0xC0000094: "an integer division by zero",
    0xC00000FD: "a stack overflow",
    0xC0000374: "heap corruption",
    0xC0000409: "a stack buffer overrun",
    0xC000013A: "Ctrl+C",
    0x40010004: "being terminated",
}
"""NTSTATUS codes a crashed Windows process exits with."""


def describe_exit(code: int | None) -> str:
    """A process's exit code in words: ``exit code 1``, ``killed by SIGSEGV (signal 11)``, or a
    Windows crash status such as ``0xC0000005 (an access violation)``."""
    if code is None:
        return "no exit code"
    if code < 0 and os.name != "nt":
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = "a signal"
        return f"killed by {name} (signal {-code})"
    status = code & 0xFFFFFFFF
    if status in WINDOWS_STATUS:
        return f"exit code 0x{status:08X}, {WINDOWS_STATUS[status]}"
    return f"exit code {code}"
