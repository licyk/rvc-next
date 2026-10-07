"""SQLite storage with numbered migrations."""

import json
import logging
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Each entry is one migration. Append only; never edit a released entry.
MIGRATIONS: list[str] = [
    # 1
    """
    CREATE TABLE voice_models (
        id             TEXT PRIMARY KEY,
        name           TEXT NOT NULL,
        description    TEXT NOT NULL DEFAULT '',
        tags           TEXT NOT NULL DEFAULT '[]',
        location       TEXT NOT NULL,
        model_path     TEXT NOT NULL,
        sample_rate    INTEGER NOT NULL,
        version        TEXT NOT NULL,
        pitch_guidance INTEGER NOT NULL,
        speakers       TEXT NOT NULL,
        speaker_slots  INTEGER NOT NULL,
        indexes        TEXT NOT NULL,
        size           INTEGER NOT NULL,
        mtime_ns       INTEGER NOT NULL,
        sha256         TEXT,
        info           TEXT NOT NULL DEFAULT '',
        hidden         INTEGER NOT NULL DEFAULT 0,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL
    );
    CREATE TABLE presets (
        id         TEXT PRIMARY KEY,
        name       TEXT NOT NULL,
        model_id   TEXT REFERENCES voice_models(id) ON DELETE CASCADE,
        is_default INTEGER NOT NULL DEFAULT 0,
        voice      TEXT NOT NULL,
        stream     TEXT,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX presets_model ON presets(model_id);
    CREATE TABLE jobs (
        id          TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,
        title       TEXT NOT NULL,
        state       TEXT NOT NULL,
        request     TEXT NOT NULL,
        progress    REAL,
        step        TEXT,
        steps       TEXT NOT NULL,
        error       TEXT,
        result      TEXT,
        created_at  TEXT NOT NULL,
        started_at  TEXT,
        finished_at TEXT
    );
    CREATE INDEX jobs_created ON jobs(created_at DESC);
    CREATE TABLE outputs (
        id          TEXT PRIMARY KEY,
        job_id      TEXT REFERENCES jobs(id) ON DELETE SET NULL,
        path        TEXT NOT NULL,
        kind        TEXT NOT NULL,
        label       TEXT NOT NULL,
        source_path TEXT,
        model_id    TEXT,
        voice       TEXT,
        duration    REAL,
        sample_rate INTEGER,
        channels    INTEGER,
        size        INTEGER,
        created_at  TEXT NOT NULL
    );
    CREATE INDEX outputs_job ON outputs(job_id);
    CREATE INDEX outputs_created ON outputs(created_at DESC);
    CREATE TABLE audio_files (
        id          TEXT PRIMARY KEY,
        path        TEXT NOT NULL,
        name        TEXT NOT NULL,
        origin      TEXT NOT NULL,
        duration    REAL,
        sample_rate INTEGER,
        channels    INTEGER,
        size        INTEGER NOT NULL,
        mtime_ns    INTEGER NOT NULL,
        created_at  TEXT NOT NULL
    );
    CREATE INDEX audio_files_path ON audio_files(path);
    CREATE TABLE client_state (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    # 2
    """
    ALTER TABLE voice_models ADD COLUMN catalog_id TEXT;
    """,
    # 3: what a voice file says about its training (author, epoch, ...); every row is read again.
    """
    ALTER TABLE voice_models ADD COLUMN meta TEXT NOT NULL DEFAULT '{}';
    UPDATE voice_models SET mtime_ns = 0;
    """,
]


class Database:
    """One shared connection guarded by a lock. Fine for a single local process."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path) if path != ":memory:" else path
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            # The command line reads the same file while a server writes it.
            self._conn.execute("PRAGMA busy_timeout=5000")
        self.created = self.schema_version == 0
        self.migrate()

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> None:
        with self._lock:
            current = self.schema_version
            for number, script in enumerate(MIGRATIONS, start=1):
                if number <= current:
                    continue
                self._conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def execute(self, sql: str, params: tuple[Any, ...] | list[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple[Any, ...] | list[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple[Any, ...] | list[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- client state --------------------------------------------------------

    def get_client_state(self, key: str) -> Any:
        row = self.fetchone("SELECT value FROM client_state WHERE key = ?", (key,))
        return None if row is None else json.loads(row["value"])

    def set_client_state(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO client_state(key, value, updated_at) VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, json.dumps(value)),
        )
