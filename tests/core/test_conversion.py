from pathlib import Path

import pytest

from rvc_next.core.audio.models import AudioRef
from rvc_next.core.conversion.models import ConvertRequest
from rvc_next.core.errors import AssetMissingError, ValidationError
from rvc_next.core.params import VoiceParamsModel


def test_convert_job_writes_output_with_provenance(services, tiny_voice_file, wav_file, no_asset_checks):
    voice = services.models.import_paths([tiny_voice_file], name="Tiny").voices[0]
    upload = services.audio.upload("speech.wav", [wav_file.read_bytes()])
    request = ConvertRequest(inputs=[AudioRef(kind="upload", id=upload.id)], voice_id=voice.id, params=VoiceParamsModel(f0_method="pm", pitch=2), output_format="flac")
    job = services.conversion.convert(request, foreground=True)
    assert job.state == "completed", job.error
    outputs = services.audio.list_outputs(job_id=job.id).items
    assert len(outputs) == 1
    out = outputs[0]
    assert out.name == "speech.tiny.flac" and out.kind == "converted" and out.model_id == voice.id
    assert out.voice.pitch == 2 and out.sample_rate == 40000 and abs(out.duration - 1.5) < 0.05
    assert Path(out.source_path) == Path(upload.path)
    peaks = services.audio.peaks(Path(out.path), 128)
    assert peaks.points == 128 and len(peaks.data) == 256


def test_preview_and_folder_inputs(services, tiny_voice_file, wav_file, no_asset_checks, tmp_path):
    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    out_dir = tmp_path / "out"
    request = ConvertRequest(
        inputs=[AudioRef(kind="path", path=str(wav_file.parent))], voice_id=voice.id, params=VoiceParamsModel(f0_method="pm"), preview_seconds=1.0, output_dir=str(out_dir)
    )
    job = services.conversion.convert(request, foreground=True)
    assert job.state == "completed", job.error
    files = list(out_dir.iterdir())
    assert [f.name for f in files] == ["speech.tiny-voice.preview.wav"]
    again = services.conversion.convert(request, foreground=True)
    assert len(list(out_dir.iterdir())) == 2, "never overwrites"
    assert again.state == "completed"


def test_validation_before_queueing(services, tiny_voice_file, wav_file):
    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    ref = [AudioRef(kind="path", path=str(wav_file))]
    with pytest.raises(AssetMissingError) as e:
        services.conversion.convert(ConvertRequest(inputs=ref, voice_id=voice.id))
    assert "hubert" in e.value.detail["assets"]
    with pytest.raises(ValidationError):
        services.conversion.validate(ConvertRequest(inputs=ref, voice_id=voice.id, params=VoiceParamsModel(speaker_id=5)))


def test_each_pitch_method_requires_its_model(services, tiny_voice_file):
    """FCPE is an asset like RMVPE: choosing it asks for the ``fcpe`` download."""
    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    assert services.conversion.required_assets(voice, "pm") == ["hubert"]
    assert services.conversion.required_assets(voice, "rmvpe") == ["hubert", "rmvpe"]
    assert services.conversion.required_assets(voice, "fcpe") == ["hubert", "fcpe"]
    assert services.assets.spec("fcpe").files[0].path == "fcpe/fcpe_c_v001.pt"


def test_automatic_key_denoise_and_ogg(services, tiny_voice_file, wav_file, no_asset_checks):
    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    params = VoiceParamsModel(f0_method="pm", auto_pitch="semitone", autotune=0.5)
    assert params.to_engine().auto_pitch_target == 155.0 and params.to_engine(210.0).auto_pitch_target == 210.0
    assert params.model_copy(update={"auto_pitch_target": 255.0}).to_engine(210.0).auto_pitch_target == 255.0
    request = ConvertRequest(inputs=[AudioRef(kind="path", path=str(wav_file))], voice_id=voice.id, params=params, output_denoise=0.6, output_format="ogg", preview_seconds=1.0)
    job = services.conversion.convert(request, foreground=True)
    assert job.state == "completed", job.error
    assert any("Automatic key" in line for line in services.jobs.read_log(job.id).lines)
    out = services.audio.list_outputs(job_id=job.id).items[0]
    assert out.path.endswith(".ogg") and out.sample_rate == 48000


def test_analysis_of_an_output_and_its_source(services, tiny_voice_file, wav_file, no_asset_checks):
    import base64

    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    request = ConvertRequest(inputs=[AudioRef(kind="path", path=str(wav_file))], voice_id=voice.id, params=VoiceParamsModel(f0_method="pm"), preview_seconds=1.0)
    job = services.conversion.convert(request, foreground=True)
    out = services.audio.list_outputs(job_id=job.id).items[0]
    result = services.audio.analyse(Path(out.path), columns=200)
    spec = result.spectrogram
    assert len(base64.b64decode(spec.data)) == spec.rows * spec.cols and 150 <= spec.cols <= 260 and spec.rows == 128
    assert abs(len(result.pitch) * result.pitch_hop - result.duration) < 0.05
    assert services.audio.analyse(Path(out.path), columns=200) == result  # cached
    source = services.audio.analyse(Path(out.source_path))
    assert source.duration > result.duration  # the preview is the first second


def test_effects_over_the_converted_voice(services, tiny_voice_file, wav_file, no_asset_checks, monkeypatch):
    pytest.importorskip("pedalboard")
    from pydantic import ValidationError as PydanticError

    from rvc_next.core.params import EffectModel

    catalog = services.conversion.effects_catalog()
    assert catalog.available and {"reverb", "delay", "limiter"} <= {e.kind for e in catalog.effects}
    with pytest.raises(PydanticError):
        EffectModel(kind="reverb", params={"room_size": 2.0})
    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    effects = [EffectModel(kind="reverb", params={"room_size": 0.8, "wet_level": 0.5}), EffectModel(kind="limiter", params={})]
    request = ConvertRequest(inputs=[AudioRef(kind="path", path=str(wav_file))], voice_id=voice.id, params=VoiceParamsModel(f0_method="pm"), effects=effects, preview_seconds=1.0)
    job = services.conversion.convert(request, foreground=True)
    assert job.state == "completed", job.error
    out = services.audio.list_outputs(job_id=job.id).items[0]
    assert out.duration > 1.2  # the reverb's tail
    monkeypatch.setattr("rvc_next.engine.audio.effects.available", lambda: False)
    with pytest.raises(ValidationError) as e:
        services.conversion.validate(request)
    assert e.value.detail["reason"] == "effects_unavailable"
