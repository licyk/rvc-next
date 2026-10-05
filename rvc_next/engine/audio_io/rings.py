"""A single-producer, single-consumer ring of float32 samples shared by an audio callback and the processing thread."""

from __future__ import annotations

import threading

import numpy as np


class Ring:
    """Fixed capacity; writing past it drops the excess (an overrun), reading past the fill returns less (an underrun).

    The lock is held only for index arithmetic and two slice copies, never across user code, so
    an audio callback never waits on the processing thread for longer than a copy.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._read = 0
        self._fill = 0
        self._lock = threading.Lock()
        self._data_ready = threading.Condition(self._lock)
        self.overruns = 0

    @property
    def available(self) -> int:
        return self._fill

    @property
    def space(self) -> int:
        return self.capacity - self._fill

    def clear(self) -> None:
        with self._lock:
            self._read = 0
            self._fill = 0

    def write(self, data: np.ndarray) -> int:
        """Append ``data``; return how many samples fit. Counts an overrun when some did not."""
        n = int(data.shape[0])
        with self._lock:
            n_fit = min(n, self.capacity - self._fill)
            if n_fit < n:
                self.overruns += 1
            if n_fit:
                start = (self._read + self._fill) % self.capacity
                first = min(n_fit, self.capacity - start)
                self._buf[start : start + first] = data[:first]
                if n_fit > first:
                    self._buf[: n_fit - first] = data[first:n_fit]
                self._fill += n_fit
                self._data_ready.notify_all()
            return n_fit

    def read_into(self, out: np.ndarray, n: int | None = None) -> int:
        """Copy up to ``n`` (default ``len(out)``) samples into ``out``; return how many were copied."""
        n = int(out.shape[0] if n is None else n)
        with self._lock:
            n_read = min(n, self._fill)
            if n_read:
                first = min(n_read, self.capacity - self._read)
                out[:first] = self._buf[self._read : self._read + first]
                if n_read > first:
                    out[first:n_read] = self._buf[: n_read - first]
                self._read = (self._read + n_read) % self.capacity
                self._fill -= n_read
            return n_read

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        got = self.read_into(out, n)
        return out[:got]

    def wait_for(self, n: int, timeout: float) -> bool:
        """Block until ``n`` samples are available or ``timeout`` seconds pass."""
        with self._data_ready:
            return self._data_ready.wait_for(lambda: self._fill >= n, timeout=timeout)
