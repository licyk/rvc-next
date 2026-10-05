"""Decode, encode and probe audio files.

Decoding goes through PyAV (libavformat and libswresample, the same libraries the original reached
through the ``ffmpeg`` command), with the ``ffmpeg`` command as a fallback for files PyAV rejects.
Any path the operating system accepts works; nothing strips quotes or spaces.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rvc_next.engine.errors import AudioError

AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm", ".aif", ".aiff"})
OUTPUT_FORMATS = ("wav", "flac", "mp3", "m4a")


@dataclass(frozen=True)
class AudioInfo:
    duration: float
    sample_rate: int
    channels: int
    codec: str | None = None


def is_audio_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def probe(path: Path | str) -> AudioInfo:
    """Duration, sample rate and channels of the first audio stream."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioError(f"{Path(path).name} has no audio stream")
            stream = container.streams.audio[0]
            rate = int(stream.codec_context.sample_rate or stream.rate or 0)
            channels = int(stream.codec_context.channels or 1)
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / 1_000_000
            else:
                duration = 0.0
            if duration <= 0:
                # Some containers (raw ADTS, some WAVs) carry no duration: count the samples.
                samples = sum(frame.samples for frame in container.decode(stream))
                duration = samples / rate if rate else 0.0
            return AudioInfo(duration=duration, sample_rate=rate, channels=channels, codec=stream.codec_context.name)
    except AudioError:
        raise
    except Exception as e:
        info = _probe_ffprobe(path)
        if info is None:
            raise AudioError(f"Cannot read {Path(path).name}: {e}") from e
        return info


def _probe_ffprobe(path: Path | str) -> AudioInfo | None:
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    import json

    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels,codec_name:format=duration", "-of", "json", str(path)],
            capture_output=True,
            check=True,
        ).stdout
        data = json.loads(out)
        stream = data["streams"][0]
        return AudioInfo(float(data["format"]["duration"]), int(stream["sample_rate"]), int(stream.get("channels", 1)), stream.get("codec_name"))
    except Exception:
        return None


def decode(path: Path | str, sample_rate: int | None = None, mono: bool = True, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Decode to float32. Mono returns shape ``[T]``; otherwise ``[C, T]``.

    ``sample_rate`` resamples in libswresample, as ``ffmpeg -ar`` does; None keeps the file's rate.
    ``max_seconds`` stops early, for previews.
    """
    try:
        return _decode_av(path, sample_rate, mono, max_seconds)
    except Exception as e:
        if shutil.which("ffmpeg") is None:
            raise AudioError(f"Cannot decode {Path(path).name}: {e}") from e
        return _decode_ffmpeg(path, sample_rate, mono, max_seconds)


def _decode_av(path: Path | str, sample_rate: int | None, mono: bool, max_seconds: float | None) -> tuple[np.ndarray, int]:
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise AudioError(f"{Path(path).name} has no audio stream")
        stream = container.streams.audio[0]
        source_rate = int(stream.codec_context.sample_rate or stream.rate)
        rate = int(sample_rate or source_rate)
        layout = "mono" if mono else None
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        chunks: list[np.ndarray] = []
        total = 0
        limit = int(max_seconds * rate) if max_seconds else None
        for frame in container.decode(stream):
            frame.pts = None
            for out in resampler.resample(frame):
                arr = out.to_ndarray()
                chunks.append(arr)
                total += arr.shape[-1]
            if limit is not None and total >= limit:
                break
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    if not chunks:
        raise AudioError(f"{Path(path).name} holds no audio")
    audio = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)
    if limit is not None:
        audio = audio[..., :limit]
    if mono:
        return np.ascontiguousarray(audio[0]), rate
    return np.ascontiguousarray(audio), rate


def _decode_ffmpeg(path: Path | str, sample_rate: int | None, mono: bool, max_seconds: float | None) -> tuple[np.ndarray, int]:
    info = probe(path) if (sample_rate is None or not mono) else None
    rate = int(sample_rate or (info.sample_rate if info else 44100))
    channels = 1 if mono else (info.channels if info else 2)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(rate), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise AudioError(f"Cannot decode {Path(path).name}: {e.stderr.decode(errors='replace').strip()}") from e
    samples = np.frombuffer(out, np.float32)
    if mono:
        return samples.copy(), rate
    usable = samples.size - samples.size % channels
    return samples[:usable].reshape(-1, channels).T.copy(), rate


def encode(path: Path | str, audio: np.ndarray, sample_rate: int, fmt: str | None = None) -> Path:
    """Write ``audio`` (``[T]`` or ``[C, T]``, float or int16) as wav, flac, mp3 or m4a.

    The file is written beside the target and renamed into place, so a failure leaves nothing.
    """
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".") or "wav").lower()
    if fmt not in OUTPUT_FORMATS:
        raise AudioError(f"Unsupported output format: {fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.T if audio.ndim == 2 else audio
    tmp = path.with_name(f".{path.name}.part")
    try:
        if fmt in ("wav", "flac"):
            import soundfile as sf

            sf.write(str(tmp), data, sample_rate, format=fmt.upper())
        else:
            import soundfile as sf

            with io.BytesIO() as wav:
                sf.write(wav, data, sample_rate, format="WAV")
                wav.seek(0)
                _transcode(wav, tmp, fmt)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise AudioError(f"Encoding produced no output: {path.name}")
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        if isinstance(e, AudioError):
            raise
        raise AudioError(f"Cannot write {path.name}: {e}") from e
    return path


def encoder_rate(codec_name: str, rate: int) -> int:
    """``rate`` if the encoder takes it, else the closest supported rate at or above it (40 kHz → 44.1 kHz)."""
    import av

    rates = sorted(av.codec.Codec(codec_name, "w").audio_rates or [])
    if not rates or rate in rates:
        return rate
    higher = [r for r in rates if r >= rate]
    return higher[0] if higher else rates[-1]


def _transcode(src: io.BytesIO, dst: Path, fmt: str) -> None:
    """WAV bytes to mp3 or m4a through PyAV (the original's ``wav2``).

    MP3 and AAC take a fixed set of rates; a model's 40 kHz output is resampled to 44.1 kHz, which
    the original's ``wav2`` did not do (its mp3 and m4a export of 40k models failed).
    """
    import av
    from av.audio.stream import AudioStream

    container_format = "mp4" if fmt == "m4a" else fmt
    codec = "aac" if fmt == "m4a" else "libmp3lame"
    with av.open(src, "r") as inp, av.open(str(dst), "w", format=container_format) as out:
        in_stream = inp.streams.audio[0]
        channels = in_stream.codec_context.channels
        layout = "mono" if channels == 1 else "stereo"
        rate = encoder_rate(codec, int(in_stream.codec_context.sample_rate))
        ostream = out.add_stream(codec, rate=rate)
        # Older PyAV (the last for Python 3.10) types add_stream() as any kind of stream.
        assert isinstance(ostream, AudioStream)
        ostream.layout = layout
        fmt_name = "fltp" if codec == "aac" else "s16p"
        resampler = av.AudioResampler(format=fmt_name, layout=layout, rate=rate)
        for frame in inp.decode(in_stream):
            frame.pts = None
            for converted in resampler.resample(frame):
                for packet in ostream.encode(converted):
                    out.mux(packet)
        for converted in resampler.resample(None):
            for packet in ostream.encode(converted):
                out.mux(packet)
        for packet in ostream.encode(None):
            out.mux(packet)


def to_int16(audio: np.ndarray) -> np.ndarray:
    """Peak-normalise to 0.99 when needed and convert to int16 (the original's output rule)."""
    audio_max = np.abs(audio).max() / 0.99 if audio.size else 0.0
    max_int16 = 32768.0
    if audio_max > 1:
        max_int16 /= audio_max
    return (audio * max_int16).astype(np.int16)
