"""The live worker process and its connection."""

from __future__ import annotations

import collections
import json
import logging
import secrets
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Any

from rvc_next.core.errors import RvcNextError
from rvc_next.core.process import describe_exit, kill_process_tree, start_process, worker_command
from rvc_next.protocol import live as P
from rvc_next.protocol.jsonl import parse_line
from rvc_next.protocol.messages import ErrorEvent, LogEvent

logger = logging.getLogger(__name__)

WORKER = "rvc_next.workers.live"
CONNECT_TIMEOUT = 60.0
OUTPUT_LINES = 60
"""The worker's last lines of output kept for a crash report."""
EXIT_WAIT = 5.0
"""How long a crash report waits for the process to end once its connection closed."""
CAUSE_LINES = 5
"""An exception counts as the cause only when it is among the last lines the worker printed."""


@dataclass
class _Run:
    """One worker process and what it printed."""

    process: subprocess.Popen[str]
    output: collections.deque[str] = field(default_factory=lambda: collections.deque(maxlen=OUTPUT_LINES))
    fatal: ErrorEvent | None = None
    """The error event ``run_worker`` emits when ``main`` raised."""
    drain: threading.Thread | None = None


def last_error_line(lines: list[str]) -> str | None:
    """The line that says why a process died: a fatal error from ``faulthandler``, or the last
    exception of the last traceback, when they end the output (an older, handled traceback is no cause)."""
    for i in range(len(lines) - 1, max(-1, len(lines) - 1 - OUTPUT_LINES), -1):
        line = lines[i]
        if line.startswith(("Fatal Python error:", "Windows fatal exception:")):
            return line.strip()
        if line.startswith("Traceback (most recent call last):"):
            # The frames are indented; the exception is the first line that is not.
            j = next((j for j in range(i + 1, len(lines)) if lines[j].strip() and not lines[j].startswith((" ", "\t"))), None)
            return lines[j].strip() if j is not None and j >= len(lines) - CAUSE_LINES else None
    return None


def crash_error(run: _Run | None, code: int | None, lead: str) -> dict[str, Any]:
    """The ``{code, message, detail}`` for a worker that ended unasked: how it ended (exit code or
    signal), why when its output says so, and that output's last lines in ``detail.log``."""
    lines = list(run.output) if run is not None else []
    fatal = run.fatal if run is not None else None
    how = describe_exit(code) if code is not None else "its connection closed"
    if code == -9:
        how += "; the system may have run out of memory"
    cause = fatal.message if fatal is not None else last_error_line(lines)
    detail: dict[str, Any] = {**(fatal.detail if fatal is not None else {}), "exit_code": code, "log": "\n".join(lines)}
    return {"code": fatal.code if fatal is not None else "internal_error", "message": f"{lead} ({how})" + (f": {cause}" if cause else ""), "detail": detail}


class LiveSupervisor:
    """Starts ``workers.live`` on demand, sends commands, and hands every event to ``on_event``.

    ``on_exit(expected, error)`` is called once when the worker's connection closes, expectedly or
    not; for an unexpected end, ``error`` is the crash report (``crash_error``).
    """

    def __init__(self, on_event: Callable[[Any], None], on_exit: Callable[[bool, dict[str, Any] | None], None], on_audio: Callable[[bytes], None] | None = None) -> None:
        self._on_event = on_event
        self._on_exit = on_exit
        self._on_audio = on_audio
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._run: _Run | None = None
        self._conn: Connection | None = None
        self._reader: threading.Thread | None = None
        self._stopping = False
        self._request_path: Path | None = None

    @property
    def running(self) -> bool:
        return self._conn is not None and self._process is not None and self._process.poll() is None

    def ensure(self, request: dict[str, Any], env: dict[str, str] | None = None) -> None:
        """Start the worker unless it runs; block until it connects."""
        with self._lock:
            if self.running:
                return
            self._cleanup()
            authkey = secrets.token_bytes(32)
            listener = Listener(("127.0.0.1", 0), authkey=authkey)
            try:
                address = listener.address
                assert isinstance(address, tuple)
                full = {**request, "address": [address[0], address[1]], "authkey": authkey.hex()}
                fd, name = tempfile.mkstemp(prefix="rvc-next-live-", suffix=".json")
                with open(fd, "w", encoding="utf-8") as f:
                    json.dump(full, f)
                self._request_path = Path(name)
                self._stopping = False
                self._process = start_process(worker_command(WORKER, name), env=env)
                run = self._run = _Run(self._process)
                run.drain = threading.Thread(target=self._drain_output, args=(run,), name="live-output", daemon=True)
                run.drain.start()
                conn = self._accept(listener, run)
            finally:
                listener.close()
            self._conn = conn
            self._reader = threading.Thread(target=self._read, args=(conn, run), name="live-reader", daemon=True)
            self._reader.start()

    def _accept(self, listener: Listener, run: _Run) -> Connection:
        result: dict[str, Any] = {}

        def accept() -> None:
            try:
                result["conn"] = listener.accept()
            except Exception as e:  # the listener closes when the worker never came
                result["error"] = e

        t = threading.Thread(target=accept, daemon=True)
        t.start()
        process = self._process
        waited = 0.0
        while t.is_alive() and waited < CONNECT_TIMEOUT:
            t.join(0.2)
            waited += 0.2
            if process is not None and process.poll() is not None:
                break
        if "conn" not in result:
            ended = process is not None and process.poll() is not None
            kill_process_tree(process)
            self._process = None
            if ended:
                error = crash_error(run, _exit_code(run, EXIT_WAIT), "The live worker did not start")
            else:
                error = crash_error(run, None, "The live worker did not connect")
                error["message"] = f"The live worker did not connect within {CONNECT_TIMEOUT:.0f} s"
            raise RvcNextError(error["message"], error["detail"])
        return result["conn"]

    def _drain_output(self, run: _Run) -> None:
        """The worker's stdout and stderr (library warnings, tracebacks) go to the server log, and the
        last lines stay for a crash report. ``run_worker``'s events are unpacked: a log event's text,
        and its error event, which is the cause when the worker ends."""
        stdout = run.process.stdout
        assert stdout is not None
        for line in stdout:
            event = parse_line(line)
            if isinstance(event, ErrorEvent):
                run.fatal = event
                text = f"{event.code}: {event.message}"
            elif isinstance(event, LogEvent):
                text = event.message.rstrip()
            else:
                continue
            if text:
                run.output.extend(text.splitlines())
                logger.info("live worker: %s", text)

    def send(self, message: Any) -> None:
        with self._lock:
            conn = self._conn
            if conn is None:
                raise RvcNextError("The live worker is not running")
            try:
                conn.send_bytes(P.encode(message))
            except OSError as e:
                raise RvcNextError(f"The live worker is gone: {e}") from e

    def send_audio(self, pcm: bytes) -> bool:
        """Microphone samples for a browser-audio session; False when no worker runs."""
        with self._lock:
            conn = self._conn
            if conn is None:
                return False
            try:
                conn.send_bytes(P.encode_audio(pcm))
            except OSError:
                return False
            return True

    def _read(self, conn: Connection, run: _Run) -> None:
        try:
            while True:
                data = conn.recv_bytes()
                pcm = P.audio_payload(data)
                if pcm is not None:
                    if self._on_audio is not None:
                        self._on_audio(pcm)
                    continue
                message = P.decode(data)
                if message is not None:
                    try:
                        self._on_event(message)
                    except Exception:
                        logger.exception("Live event handler failed")
        except (EOFError, OSError):
            pass
        finally:
            expected = self._stopping
            with self._lock:
                if self._conn is conn:
                    self._conn = None
            self._on_exit(expected, None if expected else crash_error(run, _exit_code(run, EXIT_WAIT), "The live worker stopped unexpectedly"))

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._stopping = True
            conn, process = self._conn, self._process
        if conn is not None:
            try:
                conn.send_bytes(P.encode(P.Shutdown()))
            except OSError:
                pass
        if process is not None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                kill_process_tree(process)
        with self._lock:
            self._cleanup()

    def _cleanup(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._process is not None and self._process.poll() is None:
            kill_process_tree(self._process)
        self._process = None
        self._run = None
        if self._request_path is not None:
            self._request_path.unlink(missing_ok=True)
            self._request_path = None


def _exit_code(run: _Run, timeout: float) -> int | None:
    """The process's exit code once it ends (within ``timeout``), with its output read to the end;
    None while it still runs."""
    try:
        code = run.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if run.drain is not None and run.drain is not threading.current_thread():
        run.drain.join(timeout=2.0)
    return code
