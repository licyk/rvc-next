"""Record a live session to a WAV file, off the processing thread.

``AudioSession`` hands every processed block (input and converted, both at the session rate) to
``Recorder.write``, which only queues it; a writer thread encodes. ``source`` picks what is kept:
``converted`` or ``input`` (mono), or ``both`` (stereo: input left, converted right, for comparing).
If the disk falls behind by more than the queue holds, blocks are dropped and counted, never waited for.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCES = ("converted", "input", "both")
QUEUE_BLOCKS = 512


@dataclass
class RecordingResult:
    path: str
    source: str
    sample_rate: int
    seconds: float
    dropped_blocks: int


class Recorder:
    def __init__(self, path: Path | str, sample_rate: int, source: str = "both") -> None:
        import soundfile as sf

        if source not in SOURCES:
            raise ValueError(f"Unknown recording source: {source}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.sample_rate = int(sample_rate)
        self.started = time.time()
        self._file = sf.SoundFile(str(self.path), "w", self.sample_rate, 2 if source == "both" else 1, subtype="PCM_16")
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=QUEUE_BLOCKS)
        self.frames = 0
        self.dropped = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="rvc-live-recorder", daemon=True)
        self._thread.start()

    def write(self, x: np.ndarray, y: np.ndarray) -> None:
        """Queue one block: ``x`` the input, ``y`` the converted audio, mono, the same length."""
        if self._closed:
            return
        block = y if self.source == "converted" else x if self.source == "input" else np.stack([x, y[: x.shape[0]]], axis=1)
        try:
            self._queue.put_nowait(np.clip(block, -1.0, 1.0).astype(np.float32, copy=False))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            block = self._queue.get()
            if block is None:
                return
            self._file.write(block)
            self.frames += block.shape[0]

    def close(self) -> RecordingResult:
        """Finish writing and close the file."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
            self._file.close()
        return RecordingResult(str(self.path), self.source, self.sample_rate, self.frames / self.sample_rate, self.dropped)
