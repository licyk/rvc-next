"""The live worker process and its connection."""

from __future__ import annotations

import json
import logging
import secrets
import subprocess
import tempfile
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Any

from rvc_next.core.errors import RvcNextError
from rvc_next.core.process import kill_process_tree, start_process, worker_command
from rvc_next.protocol import live as P

logger = logging.getLogger(__name__)

WORKER = "rvc_next.workers.live"
CONNECT_TIMEOUT = 60.0


class LiveSupervisor:
    """Starts ``workers.live`` on demand, sends commands, and hands every event to ``on_event``.

    ``on_exit`` is called once when the worker's connection closes, expectedly or not.
    """

    def __init__(self, on_event: Callable[[Any], None], on_exit: Callable[[bool], None]) -> None:
        self._on_event = on_event
        self._on_exit = on_exit
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._conn: Connection | None = None
        self._reader: threading.Thread | None = None
        self._drain: threading.Thread | None = None
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
                self._drain = threading.Thread(target=self._drain_output, args=(self._process,), name="live-output", daemon=True)
                self._drain.start()
                conn = self._accept(listener)
            finally:
                listener.close()
            self._conn = conn
            self._reader = threading.Thread(target=self._read, args=(conn,), name="live-reader", daemon=True)
            self._reader.start()

    def _accept(self, listener: Listener) -> Connection:
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
            kill_process_tree(process)
            self._process = None
            raise RvcNextError("The live worker did not start; see the server log")
        return result["conn"]

    def _drain_output(self, process: subprocess.Popen[str]) -> None:
        """The worker's stdout and stderr (library warnings, tracebacks) go to the server log."""
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            if text:
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

    def _read(self, conn: Connection) -> None:
        try:
            while True:
                data = conn.recv_bytes()
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
            self._on_exit(expected)

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
        if self._request_path is not None:
            self._request_path.unlink(missing_ok=True)
            self._request_path = None
