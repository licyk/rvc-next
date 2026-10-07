"""Block-by-block parity of ``StreamEngine`` with the original realtime maths (marker ``golden``).

The original side is ``rvc:infer/rtrvc.py`` (``RVC.infer``) imported from the checkout, driven by a
copy of ``rvc:realtime_gui.py`` ``start_vc``/``audio_callback`` (they live inside the GUI's
``__main__`` block, so they cannot be imported). Both run on the CPU in fp32 with the same seed per
block.

Environment: ``RVC_NEXT_GOLDEN_ORIGINAL`` (default ``~/code_workspace/Retrieval-based-Voice-Conversion-WebUI``),
``RVC_NEXT_TEST_ASSETS`` (assets folder), ``RVC_NEXT_GOLDEN_VOICES`` (``os.pathsep``-separated
``.pth`` files) and ``RVC_NEXT_GOLDEN_CLIP`` (an audio file).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.golden

ORIGINAL = Path(os.environ.get("RVC_NEXT_GOLDEN_ORIGINAL", "~/code_workspace/Retrieval-based-Voice-Conversion-WebUI")).expanduser()
TOLERANCE = 1e-4


def _inputs() -> tuple[Path, list[Path], Path]:
    assets = Path(os.environ.get("RVC_NEXT_TEST_ASSETS", ""))
    voices = [Path(p) for p in os.environ.get("RVC_NEXT_GOLDEN_VOICES", "").split(os.pathsep) if p]
    clip = Path(os.environ.get("RVC_NEXT_GOLDEN_CLIP", ""))
    if not (ORIGINAL / "infer" / "rtrvc.py").is_file():
        pytest.skip("the original checkout is missing")
    if not (assets / "hubert_base" / "config.json").is_file() or not voices or not clip.is_file():
        pytest.skip("RVC_NEXT_TEST_ASSETS, RVC_NEXT_GOLDEN_VOICES and RVC_NEXT_GOLDEN_CLIP are needed")
    pytest.importorskip("torchaudio", reason="the original's realtime code imports torchaudio; install it next to the original checkout")
    return assets, voices, clip


class _Config:
    device = "cpu"
    is_half = False


class OriginalStream:
    """``start_vc`` and ``audio_callback`` from the original GUI, minus the GUI."""

    def __init__(
        self,
        rvc: Any,
        samplerate: int,
        block_time: float,
        crossfade_time: float,
        extra_time: float,
        threshold: float,
        i_noise: bool,
        o_noise: bool,
        rms_mix_rate: float,
        f0method: str,
    ) -> None:
        import torch

        # The original's side: torchaudio, which the original checkout's environment provides (rvc-next does
        # not use it). Imported by name, so type-checking passes whether or not it is installed.
        tat = importlib.import_module("torchaudio.transforms")

        from rvc_next.engine.audio.denoise import TorchGate

        self.rvc = rvc
        self.device = "cpu"
        self.samplerate = samplerate
        self.threhold = threshold
        self.I_noise_reduce = i_noise
        self.O_noise_reduce = o_noise
        self.rms_mix_rate = rms_mix_rate
        self.f0method = f0method
        self.zc = samplerate // 100
        self.block_frame = int(np.round(block_time * samplerate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(crossfade_time * samplerate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(extra_time * samplerate / self.zc)) * self.zc
        self.input_wav = torch.zeros(self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame, dtype=torch.float32)
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(160 * self.input_wav.shape[0] // self.zc, dtype=torch.float32)
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.fade_in_window = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, dtype=torch.float32)) ** 2
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(orig_freq=samplerate, new_freq=16000, dtype=torch.float32)
        self.resampler2 = tat.Resample(orig_freq=rvc.tgt_sr, new_freq=samplerate, dtype=torch.float32) if rvc.tgt_sr != samplerate else None
        self.tg = TorchGate(sr=samplerate, n_fft=4 * self.zc, prop_decrease=0.9)

    def callback(self, indata: np.ndarray) -> np.ndarray:
        import librosa
        import torch
        import torch.nn.functional as F

        if self.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(y=indata, frame_length=4 * self.zc, hop_length=self.zc)[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.threhold
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        if self.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[self.block_frame :].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(self.input_wav_denoise[-self.block_frame - 2 * self.zc :])[160:]
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[160:]
        infer_wav = self.rvc.infer(self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, self.f0method)
        if self.resampler2 is not None:
            infer_wav = self.resampler2(infer_wav)
        if self.O_noise_reduce:
            self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
        if self.rms_mix_rate < 1:
            input_wav = self.input_wav_denoise[self.extra_frame :] if self.I_noise_reduce else self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(y=input_wav[: infer_wav.shape[0]].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc)
            rms1 = F.interpolate(torch.from_numpy(rms1).unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = librosa.feature.rms(y=infer_wav[:].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc)
            rms2 = F.interpolate(torch.from_numpy(rms2).unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, 1.0 - self.rms_mix_rate)
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        return infer_wav[: self.block_frame].cpu().numpy()


def _original_rvc(assets: Path, voice: Path, pitch: float, formant: float, index_rate: float) -> Any:
    os.environ.pop("RVC_CUDA_GRAPH", None)
    if str(ORIGINAL) not in sys.path:
        sys.path.insert(0, str(ORIGINAL))
    hubert_mod = importlib.import_module("infer.hubert")
    setattr(hubert_mod, "HUBERT_MODEL_PATH", (assets / "hubert_base").resolve())  # noqa: B010
    rtrvc = importlib.import_module("infer.rtrvc")
    rmvpe = importlib.import_module("infer.rmvpe")
    rvc = rtrvc.RVC(pitch, formant, str(voice), "", index_rate, _Config())
    rvc.model_rmvpe = rmvpe.RMVPE(str(assets / "rmvpe" / "rmvpe.pt"), is_half=False, device="cpu")
    return rvc


CASES = [
    # samplerate, block, crossfade, extra, threshold, in-noise, out-noise, rms mix, f0, pitch, formant
    (48000, 0.25, 0.05, 2.5, -60, False, False, 0.0, "rmvpe", 0, 0.0),
    (40000, 0.2, 0.05, 1.0, -45, False, False, 0.5, "pm", 3, 0.0),
    (48000, 0.25, 0.08, 2.0, -60, True, True, 0.0, "rmvpe", -2, 1.0),
]


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-{c[8]}-f{c[10]}-nr{int(c[5])}" for c in CASES])
def test_stream_matches_original(case: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from rvc_next.engine.audio.io import decode
    from rvc_next.engine.convert.params import VoiceParams
    from rvc_next.engine.runtime import Runtime
    from rvc_next.engine.stream.engine import StreamEngine
    from rvc_next.engine.stream.params import StreamParams

    assets, voices, clip = _inputs()
    # The original's i18n reads its locale files relative to the working directory.
    monkeypatch.chdir(ORIGINAL)
    sr, block, xfade, extra, threshold, i_noise, o_noise, rms_mix, f0, pitch, formant = case
    runtime = Runtime(assets, device="cpu", precision="fp32")
    runtime.hubert()
    runtime.f0(case[8])  # model init draws from the RNG; load before any seed is set
    audio, _ = decode(clip, sr)
    worst = 0.0
    for voice_path in voices:
        rvc = _original_rvc(assets, voice_path, pitch, formant, 0.0)
        original = OriginalStream(rvc, sr, block, xfade, extra, threshold, i_noise, o_noise, rms_mix, f0)
        voice = runtime.voice(voice_path)
        params = VoiceParams(pitch=pitch, formant=formant, f0_method=f0, index_rate=0.0, rms_mix_rate=rms_mix)
        stream = StreamParams(
            block_ms=int(block * 1000), crossfade_ms=int(xfade * 1000), context_ms=int(extra * 1000), threshold_db=threshold, input_denoise=i_noise, output_denoise=o_noise
        )
        engine = StreamEngine(runtime, voice, params, stream, sr, None)
        assert engine.block_size == original.block_frame
        n = engine.block_size
        diffs = []
        for i in range(min(12, len(audio) // n)):
            x = audio[i * n : (i + 1) * n].astype(np.float32)
            torch.manual_seed(1000 + i)
            a = original.callback(x.copy())
            torch.manual_seed(1000 + i)
            b = engine.process(x.copy())
            diffs.append(float(np.sqrt(np.mean((a.astype(np.float64) - b) ** 2))))
        worst = max(worst, max(diffs))
        print(f"{voice_path.name} {case}: worst block rms {max(diffs):.2e}, mean {np.mean(diffs):.2e}")
    assert worst <= TOLERANCE
