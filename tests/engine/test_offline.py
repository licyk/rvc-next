from pathlib import Path

import numpy as np
import pytest

from rvc_next.engine.convert.offline import OfflineConverter
from rvc_next.engine.convert.params import VoiceParams
from tests.tiny import make_tiny_voice, tone


@pytest.mark.parametrize(("version", "f0"), [("v1", True), ("v2", True), ("v1", False), ("v2", False)])
def test_convert_shapes(tmp_path: Path, tiny_runtime, version: str, f0: bool) -> None:
    voice = tiny_runtime.voice(make_tiny_voice(tmp_path / "v.pth", version=version, f0=f0))
    out, sr = OfflineConverter(tiny_runtime).convert(tone(1.5), voice, VoiceParams(f0_method="pm"))
    assert sr == 40000
    assert out.dtype == np.int16
    # The original drops up to two 10 ms frames at the end (frame counts are floored).
    assert 60000 - 800 <= out.shape[0] <= 60000


def test_convert_with_fcpe(tmp_path: Path, tiny_runtime) -> None:
    """FCPE runs from the ``fcpe`` asset through the ported model (no torchfcpe)."""
    voice = tiny_runtime.voice(make_tiny_voice(tmp_path / "v.pth"))
    out, sr = OfflineConverter(tiny_runtime).convert(tone(1.0), voice, VoiceParams(f0_method="fcpe"))
    assert sr == 40000 and 40000 - 800 <= out.shape[0] <= 40000


def test_formant_keeps_length(tmp_path: Path, tiny_runtime) -> None:
    voice = tiny_runtime.voice(make_tiny_voice(tmp_path / "v.pth"))
    out, _ = OfflineConverter(tiny_runtime).convert(tone(1.0), voice, VoiceParams(f0_method="pm", formant=1.5))
    assert 40000 - 800 <= out.shape[0] <= 40000


def test_resample_and_cancel(tmp_path: Path, tiny_runtime) -> None:
    from rvc_next.engine.cancel import CancelToken
    from rvc_next.engine.errors import Cancelled

    voice = tiny_runtime.voice(make_tiny_voice(tmp_path / "v.pth"))
    out, sr = OfflineConverter(tiny_runtime).convert(tone(1.0), voice, VoiceParams(f0_method="pm"), resample_to=16000)
    assert sr == 16000 and 16000 - 320 <= out.shape[0] <= 16000
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        OfflineConverter(tiny_runtime).convert(tone(1.0), voice, VoiceParams(f0_method="pm"), cancel=token)


def test_long_audio_is_chunked(tmp_path: Path, tiny_runtime) -> None:
    voice = tiny_runtime.voice(make_tiny_voice(tmp_path / "v.pth"))
    conv = OfflineConverter(tiny_runtime)
    seconds = conv.t_max / 16000 + 3
    out, _ = conv.convert(tone(seconds), voice, VoiceParams(f0_method="pm"))
    assert abs(out.shape[0] - int(seconds * 40000)) <= 800


def _voiced_then_noise() -> np.ndarray:
    rng = np.random.default_rng(1)
    return np.concatenate([tone(0.6), 0.05 * rng.standard_normal(9600).astype(np.float32), tone(0.6)])


def test_protect_acts_on_unvoiced_frames_unless_original(tmp_path: Path, tiny_runtime) -> None:
    """Protect keeps the input's own features where the index would replace them, on unvoiced frames."""
    import torch

    from tests.tiny import make_tiny_index

    voice = tiny_runtime.voice(make_tiny_voice(tmp_path / "v.pth"))
    index = tiny_runtime.index(make_tiny_index(tmp_path / "v.index", 768))
    conv = OfflineConverter(tiny_runtime)
    x = _voiced_then_noise()

    def run(**kw) -> np.ndarray:
        torch.manual_seed(0)
        return conv.convert(x, voice, VoiceParams(f0_method="pm", index_rate=1.0, **kw), index=index)[0]

    run()  # loads HuBERT and pm, whose construction draws from the RNG
    original = run(protect=0.0, unvoiced="original")
    assert not np.array_equal(original, run(protect=0.0, unvoiced="protect"))
    # Protect off: "protect" fills the frames as the original does, so nothing changes.
    assert np.array_equal(run(protect=0.5, unvoiced="original"), run(protect=0.5, unvoiced="protect"))
    assert run(protect=0.0, unvoiced="zero").shape == original.shape
