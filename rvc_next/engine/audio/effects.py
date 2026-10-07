"""Effects over converted audio: a chain of pedalboard plugins, in the order given.

pedalboard is optional (``rvc-next[effects]``) and imported only here, inside functions, and only
after ``unavailable_reason`` has imported it once in a child process: some of its wheels are built
for newer CPUs and stop the importing process with an illegal instruction, which would take the
server or the live worker down with it. ``EFFECTS``
is the catalog — every kind with its parameters' defaults and ranges, pedalboard's own names — and
the one place those numbers live: the core validates against it and the web UI draws it.

``apply`` processes a whole file and lets a reverb or delay ring out past the end; ``EffectsChain``
processes a stream block by block, keeping the plugins' state between blocks.
"""

from __future__ import annotations

import functools
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_EFFECTS = 16
TAIL_SECONDS = 3.0
"""Silence appended before a reverb or delay, so its tail is kept; trailing silence is cut after."""
TAIL_FLOOR = 10 ** (-80 / 20)


@dataclass(frozen=True)
class EffectParam:
    name: str
    default: float
    min: float
    max: float
    unit: str = ""


@dataclass(frozen=True)
class EffectSpec:
    kind: str
    plugin: str
    """The pedalboard class."""
    params: tuple[EffectParam, ...]
    tail: bool = False
    """Whether it rings on after the input ends (reverb, delay)."""
    streams: bool = True
    """Whether it suits a live stream: pedalboard's pitch shift holds back about a second of audio there."""


def _spec(kind: str, plugin: str, *params: tuple[Any, ...], tail: bool = False, streams: bool = True) -> EffectSpec:
    return EffectSpec(kind, plugin, tuple(EffectParam(*p) for p in params), tail, streams)


EFFECTS: dict[str, EffectSpec] = {
    s.kind: s
    for s in (
        _spec("highpass", "HighpassFilter", ("cutoff_frequency_hz", 80.0, 20.0, 2000.0, "Hz")),
        _spec("lowpass", "LowpassFilter", ("cutoff_frequency_hz", 12000.0, 1000.0, 20000.0, "Hz")),
        _spec(
            "noise_gate",
            "NoiseGate",
            ("threshold_db", -50.0, -100.0, 0.0, "dB"),
            ("ratio", 10.0, 1.0, 100.0),
            ("attack_ms", 1.0, 0.1, 100.0, "ms"),
            ("release_ms", 100.0, 1.0, 1000.0, "ms"),
        ),
        _spec(
            "compressor",
            "Compressor",
            ("threshold_db", -18.0, -60.0, 0.0, "dB"),
            ("ratio", 3.0, 1.0, 20.0),
            ("attack_ms", 5.0, 0.1, 200.0, "ms"),
            ("release_ms", 100.0, 1.0, 1000.0, "ms"),
        ),
        _spec("pitch_shift", "PitchShift", ("semitones", 0.0, -12.0, 12.0, "st"), streams=False),
        _spec("distortion", "Distortion", ("drive_db", 15.0, 0.0, 60.0, "dB")),
        _spec("bitcrush", "Bitcrush", ("bit_depth", 8.0, 1.0, 32.0, "bit")),
        _spec("clipping", "Clipping", ("threshold_db", -6.0, -60.0, 0.0, "dB")),
        _spec(
            "chorus",
            "Chorus",
            ("rate_hz", 1.0, 0.01, 10.0, "Hz"),
            ("depth", 0.25, 0.0, 1.0),
            ("centre_delay_ms", 7.0, 1.0, 50.0, "ms"),
            ("feedback", 0.0, -0.95, 0.95),
            ("mix", 0.5, 0.0, 1.0),
        ),
        _spec(
            "phaser",
            "Phaser",
            ("rate_hz", 1.0, 0.01, 10.0, "Hz"),
            ("depth", 0.5, 0.0, 1.0),
            ("centre_frequency_hz", 1300.0, 100.0, 10000.0, "Hz"),
            ("feedback", 0.0, -0.95, 0.95),
            ("mix", 0.5, 0.0, 1.0),
        ),
        _spec("delay", "Delay", ("delay_seconds", 0.25, 0.01, 2.0, "s"), ("feedback", 0.3, 0.0, 0.95), ("mix", 0.3, 0.0, 1.0), tail=True),
        _spec(
            "reverb",
            "Reverb",
            ("room_size", 0.4, 0.0, 1.0),
            ("damping", 0.5, 0.0, 1.0),
            ("wet_level", 0.25, 0.0, 1.0),
            ("dry_level", 0.8, 0.0, 1.0),
            ("width", 1.0, 0.0, 1.0),
            tail=True,
        ),
        _spec("gain", "Gain", ("gain_db", 0.0, -24.0, 24.0, "dB")),
        _spec("limiter", "Limiter", ("threshold_db", -1.0, -24.0, 0.0, "dB"), ("release_ms", 100.0, 1.0, 1000.0, "ms")),
    )
}


@dataclass(frozen=True)
class Effect:
    kind: str
    params: tuple[tuple[str, float], ...] = ()
    """The parameters given; the others take their defaults."""

    def values(self) -> dict[str, float]:
        spec = EFFECTS[self.kind]
        given = dict(self.params)
        return {p.name: float(given.get(p.name, p.default)) for p in spec.params}


ILLEGAL_INSTRUCTION = {-4, 132, 0xC000001D}
"""SIGILL as a child's return code (POSIX, a shell's 128 + 4, Windows' STATUS_ILLEGAL_INSTRUCTION)."""


@functools.cache
def unavailable_reason() -> str | None:
    """None when pedalboard is installed and loads on this computer; else why effects cannot run.
    Found once per process, by importing it in a child process."""
    if importlib.util.find_spec("pedalboard") is None:
        return "Effects need pedalboard: pip install rvc-next[effects]"
    try:
        result = subprocess.run([sys.executable, "-c", "import pedalboard"], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"pedalboard could not be loaded: {e}"
    if result.returncode == 0:
        return None
    if result.returncode in ILLEGAL_INSTRUCTION:
        return "This pedalboard build needs a newer processor (it stops with an illegal instruction); effects stay off until another pedalboard release is installed"
    detail = result.stderr.decode(errors="replace").strip().splitlines()
    return f"pedalboard could not be loaded: {detail[-1] if detail else f'exit code {result.returncode}'}"


def available() -> bool:
    """Whether pedalboard is installed and loads here."""
    return unavailable_reason() is None


def effect(kind: str, params: dict[str, float] | None = None) -> Effect:
    """A checked ``Effect``; raises ``ValueError`` for an unknown kind or parameter, or a value out of range."""
    spec = EFFECTS.get(kind)
    if spec is None:
        raise ValueError(f"Unknown effect {kind!r}; one of {', '.join(EFFECTS)}")
    names = {p.name: p for p in spec.params}
    for name, value in (params or {}).items():
        p = names.get(name)
        if p is None:
            raise ValueError(f"{kind} has no parameter {name!r}; it has {', '.join(names)}")
        if not p.min <= float(value) <= p.max:
            raise ValueError(f"{kind} {name} must be between {p.min:g} and {p.max:g}, not {value:g}")
    return Effect(kind, tuple((name, float(value)) for name, value in (params or {}).items()))


def from_dicts(items: Any) -> tuple[Effect, ...]:
    """Effects from ``[{"kind": ..., "params": {...}}]`` (the protocol's and the settings' form)."""
    return tuple(i if isinstance(i, Effect) else effect(i["kind"], i.get("params")) for i in items or ())


def board(effects: tuple[Effect, ...]) -> Any:
    reason = unavailable_reason()
    if reason is not None:
        raise RuntimeError(reason)
    import pedalboard

    return pedalboard.Pedalboard([getattr(pedalboard, EFFECTS[e.kind].plugin)(**e.values()) for e in effects])


def _planar(audio: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(audio.reshape(1, -1) if audio.ndim == 1 else audio, dtype=np.float32)


def apply(audio: np.ndarray, sample_rate: int, effects: tuple[Effect, ...]) -> np.ndarray:
    """``audio`` (float, mono ``[T]`` or ``[C, T]``) through the chain, in the input's shape.

    With a reverb or delay the result is longer than the input by its audible tail.
    """
    if not effects:
        return audio
    x = _planar(audio)
    n = x.shape[1]
    tail = any(EFFECTS[e.kind].tail for e in effects)
    if tail:
        x = np.concatenate([x, np.zeros((x.shape[0], int(TAIL_SECONDS * sample_rate)), dtype=np.float32)], axis=1)
    y = board(effects)(x, sample_rate)
    if tail:
        loud = np.flatnonzero(np.abs(y[:, n:]).max(axis=0) > TAIL_FLOOR)
        y = y[:, : n + (int(loud[-1]) + 1 if loud.size else 0)]
    return y[0] if audio.ndim == 1 else y


class EffectsChain:
    """The chain over a stream: plugin state (a reverb's tail, a compressor's envelope) carries over between blocks.

    Every block comes out as long as it went in. A plugin with latency returns less at first; the
    shortfall is silence at the start, and that latency stays.
    """

    def __init__(self, effects: tuple[Effect, ...], sample_rate: int) -> None:
        self.effects = effects
        self.sample_rate = sample_rate
        self._board = board(effects) if effects else None
        self._pending: np.ndarray | None = None

    def process(self, block: np.ndarray) -> np.ndarray:
        if self._board is None:
            return block
        x = _planar(block)
        y = self._board(x, self.sample_rate, reset=False)
        if self._pending is not None:
            y = np.concatenate([self._pending, y], axis=1)
        n = x.shape[1]
        if y.shape[1] < n:
            y = np.concatenate([np.zeros((y.shape[0], n - y.shape[1]), dtype=np.float32), y], axis=1)
        self._pending = y[:, n:] if y.shape[1] > n else None
        y = y[:, :n]
        return y[0] if block.ndim == 1 else y

    def reset(self) -> None:
        self._pending = None
        if self._board is not None:
            self._board.reset()
