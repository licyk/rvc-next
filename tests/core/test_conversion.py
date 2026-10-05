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
