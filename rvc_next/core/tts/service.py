"""Text to speech as an audio source: Microsoft Edge's online voices through edge-tts (optional, ``rvc-next[tts]``).

The text goes to Microsoft's speech service; the speech comes back as MP3 and is stored as an upload,
so it is an input like any dropped file. A test gives ``backend`` instead of the network.
"""

from __future__ import annotations

import importlib.util
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from rvc_next.core.audio.models import AudioFile
from rvc_next.core.audio.service import AudioFileService
from rvc_next.core.errors import RvcNextError, ValidationError
from rvc_next.core.files import slugify
from rvc_next.core.settings import SettingsService
from rvc_next.core.tts.models import TtsRequest, TtsVoice, TtsVoices

VOICES_MAX_AGE = 7 * 24 * 3600
"""The voice list is cached on disk this long."""


class TtsBackend(Protocol):
    def voices(self) -> list[dict[str, Any]]: ...

    def synthesize(self, text: str, voice: str, rate: int, pitch: int) -> bytes: ...


class EdgeTts:
    """edge-tts, run to completion on the calling thread (a request handler's worker thread)."""

    @staticmethod
    def _proxy() -> str | None:
        import urllib.request

        proxies = urllib.request.getproxies()
        return proxies.get("https") or proxies.get("http")

    def voices(self) -> list[dict[str, Any]]:
        import asyncio

        import edge_tts

        return [dict(v) for v in asyncio.run(edge_tts.list_voices(proxy=self._proxy()))]

    def synthesize(self, text: str, voice: str, rate: int, pitch: int) -> bytes:
        import asyncio

        import edge_tts

        async def run() -> bytes:
            communicate = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz", proxy=self._proxy())
            data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    data += chunk["data"]
            return bytes(data)

        return asyncio.run(run())


def _voice(entry: dict[str, Any]) -> TtsVoice:
    short = str(entry["ShortName"])
    name = re.sub(r"Neural$", "", short.split("-", 2)[-1]) or short
    return TtsVoice(id=short, name=name, locale=str(entry.get("Locale", "")), language=str(entry.get("LocaleName", entry.get("Locale", ""))), gender=str(entry.get("Gender", "")))


class TtsService:
    def __init__(self, settings: SettingsService, audio: AudioFileService, backend: TtsBackend | None = None) -> None:
        self._settings = settings
        self._audio = audio
        self.backend = backend
        self._voices: list[TtsVoice] | None = None
        self._lock = threading.Lock()

    @property
    def _cache(self) -> Path:
        return self._settings.data_dir / "cache" / "tts-voices.json"

    def available(self) -> bool:
        return self.backend is not None or importlib.util.find_spec("edge_tts") is not None

    def _backend(self) -> TtsBackend:
        if not self.available():
            raise ValidationError("Text to speech needs edge-tts: pip install rvc-next[tts]", {"reason": "tts_unavailable"})
        return self.backend or EdgeTts()

    def voices(self, refresh: bool = False) -> TtsVoices:
        """The voices, from memory, the disk cache (a week) or the service."""
        if not self.available():
            return TtsVoices(available=False, voices=[])
        with self._lock:
            if self._voices is None or refresh:
                self._voices = None if refresh else self._read_cache()
                if self._voices is None:
                    try:
                        raw = self._backend().voices()
                    except Exception as e:
                        raise RvcNextError(f"Cannot list the text-to-speech voices: {e}", {"reason": "tts_failed"}) from e
                    self._voices = sorted((_voice(v) for v in raw if v.get("ShortName")), key=lambda v: (v.language, v.name))
                    self._write_cache(self._voices)
            return TtsVoices(available=True, voices=self._voices)

    def _read_cache(self) -> list[TtsVoice] | None:
        try:
            if time.time() - self._cache.stat().st_mtime > VOICES_MAX_AGE:
                return None
            return [TtsVoice.model_validate(v) for v in json.loads(self._cache.read_text(encoding="utf-8"))]
        except (OSError, ValueError):
            return None

    def _write_cache(self, voices: list[TtsVoice]) -> None:
        try:
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            self._cache.write_text(json.dumps([v.model_dump() for v in voices]), encoding="utf-8")
        except OSError:
            pass

    def synthesize(self, request: TtsRequest) -> AudioFile:
        """Speak ``request.text`` and store it as an uploaded audio file."""
        backend = self._backend()
        text = request.text.strip()
        if not text:
            raise ValidationError("There is no text to speak")
        try:
            data = backend.synthesize(text, request.voice, request.rate, request.pitch)
        except Exception as e:
            if type(e).__name__ != "NoAudioReceived":
                raise RvcNextError(f"Text to speech failed: {e}", {"reason": "tts_failed"}) from e
            data = b""
        if not data:
            # What the service does with text in a language the voice does not speak, or with an unknown voice.
            raise RvcNextError(f"No speech came back from {request.voice}: does it speak the language of the text?", {"reason": "tts_no_audio"})
        name = f"{slugify(text[:40], fallback='speech')}.mp3"
        return self._audio.upload(name, [data])
