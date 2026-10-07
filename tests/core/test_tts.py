import pytest

from rvc_next.core.errors import RvcNextError, ValidationError
from rvc_next.core.tts.models import TtsRequest


class FakeTts:
    def __init__(self, audio: bytes) -> None:
        self.audio = audio
        self.calls: list[tuple] = []
        self.listed = 0

    def voices(self):
        self.listed += 1
        return [
            {"ShortName": "zh-CN-XiaoxiaoNeural", "Locale": "zh-CN", "LocaleName": "Chinese (Mainland)", "Gender": "Female"},
            {"ShortName": "en-US-AndrewMultilingualNeural", "Locale": "en-US", "LocaleName": "English (United States)", "Gender": "Male"},
        ]

    def synthesize(self, text, voice, rate, pitch):
        self.calls.append((text, voice, rate, pitch))
        if voice == "broken":
            raise OSError("connection reset")
        return b"" if voice == "silent" else self.audio


def test_voices_are_listed_once_and_cached_on_disk(services, wav_file):
    fake = FakeTts(wav_file.read_bytes())
    services.tts.backend = fake
    voices = services.tts.voices().voices
    assert [(v.id, v.name, v.locale) for v in voices] == [("zh-CN-XiaoxiaoNeural", "Xiaoxiao", "zh-CN"), ("en-US-AndrewMultilingualNeural", "AndrewMultilingual", "en-US")]
    services.tts._voices = None  # a restarted server reads the disk cache
    assert services.tts.voices().voices == voices and fake.listed == 1
    services.tts.voices(refresh=True)
    assert fake.listed == 2


def test_speech_becomes_an_uploaded_input(services, wav_file):
    fake = FakeTts(wav_file.read_bytes())
    services.tts.backend = fake
    f = services.tts.synthesize(TtsRequest(text="  Hello there, world!  ", voice="en-US-AndrewMultilingualNeural", rate=10, pitch=-5))
    assert fake.calls == [("Hello there, world!", "en-US-AndrewMultilingualNeural", 10, -5)]
    assert f.name == "hello-there-world.mp3" and f.origin == "upload" and abs(f.duration - 1.5) < 0.05
    assert services.audio.get_file(f.id).path == f.path
    with pytest.raises(RvcNextError) as e:
        services.tts.synthesize(TtsRequest(text="x", voice="broken"))
    assert e.value.detail["reason"] == "tts_failed" and "connection reset" in e.value.message
    with pytest.raises(RvcNextError) as e:
        services.tts.synthesize(TtsRequest(text="你好", voice="silent"))
    assert e.value.detail["reason"] == "tts_no_audio"


def test_without_edge_tts(services, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    assert services.tts.voices().available is False
    with pytest.raises(ValidationError) as e:
        services.tts.synthesize(TtsRequest(text="x", voice="en-US-AriaNeural"))
    assert e.value.detail["reason"] == "tts_unavailable"
