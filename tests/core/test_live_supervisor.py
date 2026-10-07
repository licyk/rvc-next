"""The live worker's crash report: how it ended, why, and its last output."""

import os
import signal
import subprocess
import sys
from typing import Any, cast

import pytest

from rvc_next.core.live.supervisor import _exit_code, _Run, crash_error, last_error_line
from rvc_next.core.process import describe_exit
from rvc_next.protocol.messages import ErrorEvent


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_describe_exit_on_posix() -> None:
    assert describe_exit(1) == "exit code 1"
    assert describe_exit(-signal.SIGSEGV) == f"killed by SIGSEGV (signal {int(signal.SIGSEGV)})"
    assert describe_exit(None) == "no exit code"


def test_describe_exit_names_windows_crashes() -> None:
    assert describe_exit(0xC0000005) == "exit code 0xC0000005, an access violation"
    if os.name == "nt":
        assert describe_exit(-1073741819) == "exit code 0xC0000005, an access violation"


def test_last_error_line() -> None:
    traceback = ["Traceback (most recent call last):", '  File "x.py", line 1, in <module>', "    boom()", "RuntimeError: boom"]
    assert last_error_line(["loading", *traceback]) == "RuntimeError: boom"
    assert (
        last_error_line(["Fatal Python error: Segmentation fault", "", "Current thread 0x1 (most recent call first):", '  File "x.py", line 3'])
        == "Fatal Python error: Segmentation fault"
    )
    # An older traceback, handled and followed by more output, is not why the process ended.
    assert last_error_line([*traceback, *(f"Live: block {i}" for i in range(10))]) is None
    assert last_error_line(["just output"]) is None


def test_crash_error_prefers_the_workers_error_event() -> None:
    run = _Run(process=cast(Any, None))
    run.output.extend(["Traceback (most recent call last):", "  ...", "OSError: no PortAudio"])
    error = crash_error(run, 1, "The live worker did not start")
    assert error == {
        "code": "internal_error",
        "message": "The live worker did not start (exit code 1): OSError: no PortAudio",
        "detail": {"exit_code": 1, "log": "Traceback (most recent call last):\n  ...\nOSError: no PortAudio"},
    }
    run.fatal = ErrorEvent(code="compute_unavailable", message="Out of GPU memory: CUDA", detail={"reason": "out_of_memory"})
    error = crash_error(run, 1, "The live worker stopped unexpectedly")
    assert error["code"] == "compute_unavailable" and error["detail"]["reason"] == "out_of_memory"
    assert error["message"] == "The live worker stopped unexpectedly (exit code 1): Out of GPU memory: CUDA"
    if os.name != "nt":
        assert "run out of memory" in crash_error(_Run(process=cast(Any, None)), -9, "The live worker stopped unexpectedly")["message"]


def test_exit_code_reads_the_output_to_the_end() -> None:
    process = subprocess.Popen([sys.executable, "-c", "print('last words'); raise SystemExit(3)"], stdout=subprocess.PIPE, text=True)
    run = _Run(process=process)
    with process:
        assert _exit_code(run, 10.0) == 3
