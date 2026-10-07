"""The streaming engine: one block in, one block out, no I/O.

The maths is ported from the original ``infer/rtrvc.py`` (``RVC.infer``) and ``realtime_gui.py``
(``start_vc``, ``audio_callback``): the rolling device-rate and 16 kHz buffers, the RMS gate,
input and output TorchGate, the resample to 16 kHz, HuBERT over the whole context with retrieval on
the new frames only, pitch on the tail rolled into a 1024-frame cache, ``skip_head`` /
``return_length`` generation, formant resample, RMS mix and the SOLA crossfade. The original's
three copies (GUI callback, ``rtrvc``, VST worker) become this one class. Three changes: the speaker
id is honoured (the original fixed ``sid = 0``), changing block, crossfade or context rebuffers
without reloading the voice, and ``protect`` applies when ``unvoiced`` is not "original" (the
original has no protect step in realtime).
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from rvc_next.engine.audio.sola import Sola, fade_windows
from rvc_next.engine.convert.params import VoiceParams
from rvc_next.engine.features.hubert import extract_features
from rvc_next.engine.graph import cuda_graph_enabled, run_cuda_graph
from rvc_next.engine.stream.params import StreamParams

if TYPE_CHECKING:
    from rvc_next.engine.index.retrieval import LoadedIndex
    from rvc_next.engine.models.loader import LoadedVoice
    from rvc_next.engine.runtime import Runtime

F0_MIN = 50
F0_MAX = 1100
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)
PITCH_CACHE = 1024


def _frames(ms: float, sample_rate: int, zc: int) -> int:
    """Milliseconds rounded to whole 10 ms frames, in samples (the original's rounding)."""
    return int(np.round(ms / 1000 * sample_rate / zc)) * zc


def prepare_voice(voice: LoadedVoice) -> None:
    """Fold weight norm into the weights once, as the original realtime loader does."""
    net_g = voice.net_g
    if getattr(net_g, "_rvc_stream_ready", False):
        return
    try:
        net_g.remove_weight_norm()
    except ValueError:
        pass  # already removed
    net_g._rvc_stream_ready = True


class StreamEngine:
    def __init__(
        self,
        runtime: Runtime,
        voice: LoadedVoice,
        params: VoiceParams,
        stream: StreamParams,
        sample_rate: int,
        index: LoadedIndex | None = None,
    ) -> None:
        import torch

        from rvc_next.engine.audio.denoise import TorchGate

        self.runtime = runtime
        self.device = runtime.device
        self.is_half = runtime.is_half
        self.sample_rate = int(sample_rate)
        self.zc = self.sample_rate // 100
        self.params = params
        self.stream = stream
        self.passthrough = False
        """Skip the voice and return the (gated, denoised) input, like the original's ``im`` mode."""
        self._lock = threading.RLock()
        self._index: LoadedIndex | None = index
        self._voice: LoadedVoice = voice
        self.resample_kernel: dict[int, Any] = {}
        self.cache_pitch = torch.zeros(PITCH_CACHE, device=self.device, dtype=torch.long)
        self.cache_pitchf = torch.zeros(PITCH_CACHE, device=self.device, dtype=torch.float32)
        self.cache_voiced = torch.zeros(PITCH_CACHE, device=self.device, dtype=torch.bool)
        self.tg = TorchGate(sr=self.sample_rate, n_fft=4 * self.zc, prop_decrease=0.9).to(self.device)
        self._set_voice(voice)
        self._build()

    # -- properties -----------------------------------------------------------------------

    @property
    def block_size(self) -> int:
        """Samples per block at ``sample_rate``."""
        return self.block_frame

    @property
    def voice(self) -> LoadedVoice:
        return self._voice

    @property
    def index(self) -> LoadedIndex | None:
        return self._index

    # -- configuration -------------------------------------------------------------------------

    def _set_voice(self, voice: LoadedVoice) -> None:
        import torch

        from rvc_next.engine.audio.sinc_resample import Resample

        prepare_voice(voice)
        self._voice = voice
        self.tgt_sr = voice.tgt_sr
        self.resampler2: Any = None
        if self.tgt_sr != self.sample_rate:
            self.resampler2 = Resample(orig_freq=self.tgt_sr, new_freq=self.sample_rate, dtype=torch.float32).to(self.device)
        self.resample_kernel = {}

    def _build(self) -> None:
        """Allocate the buffers for the current stream parameters (``start_vc``)."""
        import torch

        from rvc_next.engine.audio.sinc_resample import Resample

        sr, zc, s = self.sample_rate, self.zc, self.stream
        dev = self.device
        self.block_frame = max(zc, _frames(s.block_ms, sr, zc))
        self.block_frame_16k = 160 * self.block_frame // zc
        self.crossfade_frame = _frames(s.crossfade_ms, sr, zc)
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * zc)
        self.sola_search_frame = zc
        self.extra_frame = _frames(s.context_ms, sr, zc)
        n = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(n, device=dev, dtype=torch.float32)
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(160 * n // zc, device=dev, dtype=torch.float32)
        self.rms_buffer = np.zeros(4 * zc, dtype="float32")
        self.sola = Sola(self.sola_buffer_frame, self.sola_search_frame, dev)
        self.nr_buffer = self.sola.buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc
        self.fade_in_window, self.fade_out_window = fade_windows(self.sola_buffer_frame, dev)
        self.resampler = Resample(orig_freq=sr, new_freq=16000, dtype=torch.float32).to(dev)
        self.cache_pitch.zero_()
        self.cache_pitchf.zero_()
        self.cache_voiced.zero_()

    def update(self, params: VoiceParams) -> None:
        """Hot: applies from the next block."""
        with self._lock:
            self.params = params

    def reconfigure(self, stream: StreamParams) -> None:
        """Change stream parameters. Threshold and denoise are hot; block, crossfade and context rebuild the buffers (about one block of silence) while the voice stays loaded."""
        with self._lock:
            old, self.stream = self.stream, stream
            if not old.same_buffers(stream):
                self._build()

    def set_voice(self, voice: LoadedVoice, index: LoadedIndex | None = None) -> None:
        """Swap the voice between two blocks. HuBERT and the pitch models are shared."""
        with self._lock:
            if voice is not self._voice:
                self._set_voice(voice)
            self._index = index

    def set_index(self, index: LoadedIndex | None) -> None:
        with self._lock:
            self._index = index

    def reset(self) -> None:
        """Clear every buffer, as after a device change."""
        with self._lock:
            for t in (self.input_wav, self.input_wav_denoise, self.input_wav_res, self.output_buffer, self.nr_buffer, self.cache_pitch, self.cache_pitchf, self.cache_voiced):
                t.zero_()
            self.sola.reset()
            self.rms_buffer[:] = 0

    def prewarm(self) -> None:
        """Run test blocks through the whole of ``process`` before the audio starts, then clear the buffers.

        A first block pays for everything done for the first time: librosa's lazily imported
        ``feature.rms``, kernel selection, CUDA Graph capture for each input shape. Paid inside a
        session, that cost makes the first block late, which shows as a load spike and leaves its
        backlog in the output buffer. So every path a running session can switch to is taken here:
        the gate off and on (it lengthens the resampler's input by two frames), input and output
        denoise, and the loudness match.
        """
        import torch

        with self._lock:
            stream, params, passthrough = self.stream, self.params, self.passthrough
            try:
                self.passthrough = False
                t = np.arange(self.block_frame) / self.sample_rate
                block = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)  # -23 dB, above the gate
                self.params = replace(params, rms_mix_rate=min(params.rms_mix_rate, 0.5))
                for gate, denoise in ((-60.0, False), (-50.0, False), (-50.0, True)):
                    self.stream = replace(stream, threshold_db=gate, input_denoise=denoise, output_denoise=denoise)
                    with torch.no_grad():
                        self._process(block)
                if cuda_graph_enabled(self.device):
                    torch.cuda.synchronize(self.device)
            finally:
                self.stream, self.params, self.passthrough = stream, params, passthrough
                self.reset()

    # -- processing ------------------------------------------------------------------------------

    def process(self, block: np.ndarray) -> np.ndarray:
        """Convert one block (mono float32 at ``sample_rate``, ``block_size`` samples)."""
        import torch

        with self._lock, torch.no_grad():
            return self._process(np.asarray(block, dtype=np.float32).reshape(-1))

    def _process(self, indata: np.ndarray) -> np.ndarray:
        import librosa
        import torch
        import torch.nn.functional as F

        zc = self.zc
        s = self.stream
        params = self.params
        if indata.shape[0] != self.block_frame:
            fixed = np.zeros(self.block_frame, dtype=np.float32)
            fixed[-min(indata.shape[0], self.block_frame) :] = indata[-self.block_frame :]
            indata = fixed
        if s.threshold_db > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(y=indata, frame_length=4 * zc, hop_length=zc)[:, 2:]
            self.rms_buffer[:] = indata[-4 * zc :]
            indata = indata[2 * zc - zc // 2 :]
            db_threshold = librosa.amplitude_to_db(rms, ref=1.0)[0] < s.threshold_db
            for i in range(db_threshold.shape[0]):
                if db_threshold[i]:
                    indata[i * zc : (i + 1) * zc] = 0
            indata = indata[zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        resampler = self.resampler
        if s.input_denoise:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[self.block_frame :].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            resample_input = self.input_wav_denoise[-self.block_frame - 2 * zc :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = run_cuda_graph(resampler, "realtime-input-resample", lambda audio: resampler(audio), resample_input)[160:]
        else:
            resample_input = self.input_wav[-indata.shape[0] - 2 * zc :]
            self.input_wav_res[-160 * (indata.shape[0] // zc + 1) :] = run_cuda_graph(resampler, "realtime-input-resample", lambda audio: resampler(audio), resample_input)[160:]

        if not self.passthrough:
            infer_wav = self._infer()
            if self.resampler2 is not None:
                resampler2 = self.resampler2
                infer_wav = run_cuda_graph(resampler2, "realtime-output-resample", lambda audio: resampler2(audio), infer_wav)
        elif s.input_denoise:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()

        if s.output_denoise and not self.passthrough:
            self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)

        if params.rms_mix_rate < 1 and not self.passthrough:
            input_wav = self.input_wav_denoise[self.extra_frame :] if s.input_denoise else self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(y=input_wav[: infer_wav.shape[0]].cpu().numpy(), frame_length=4 * zc, hop_length=zc)
            rms1 = torch.from_numpy(rms1).to(self.device)
            rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = librosa.feature.rms(y=infer_wav[:].cpu().numpy(), frame_length=4 * zc, hop_length=zc)
            rms2 = torch.from_numpy(rms2).to(self.device)
            rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, 1.0 - params.rms_mix_rate)

        out = self.sola.apply(infer_wav, self.block_frame)
        return out.detach().cpu().numpy().astype(np.float32, copy=True)

    # -- inference (``rtrvc.RVC.infer``) ------------------------------------------------------

    def _infer(self) -> Any:
        import torch
        import torch.nn.functional as F

        input_wav = self.input_wav_res
        params = self.params
        voice = self._voice
        net_g = voice.net_g
        hubert = self.runtime.hubert()
        feats = input_wav.half().view(1, -1) if self.is_half else input_wav.float().view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        feats = extract_features(hubert, feats, voice.version, padding_mask=padding_mask)
        feats = torch.cat((feats, feats[:, -1:, :]), 1)
        protect = voice.if_f0 and params.unvoiced != "original" and params.protect < 0.5
        feats0 = feats.clone() if protect else None
        skip_head, return_length = self.skip_head, self.return_length
        index = self._index
        if index is not None and params.index_rate != 0:
            npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
            score, ix = index.index.search(npy, k=8)
            if (ix >= 0).all():
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(index.vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
                if self.is_half:
                    npy = npy.astype("float16")
                feats[0][skip_head // 2 :] = torch.from_numpy(npy).unsqueeze(0).to(self.device) * params.index_rate + (1 - params.index_rate) * feats[0][skip_head // 2 :]
        p_len = input_wav.shape[0] // 160
        factor = pow(2, params.formant / 12)
        return_length2 = int(np.ceil(return_length * factor))
        cache_pitch = cache_pitchf = None
        if voice.if_f0:
            f0_extractor_frame = self.block_frame_16k + 800
            if params.f0_method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            pitch, pitchf, voiced = self._get_f0(input_wav[-f0_extractor_frame:], params.pitch - params.formant, params.f0_method, params.unvoiced)
            shift = self.block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_voiced[:-shift] = self.cache_voiced[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            self.cache_voiced[4 - pitch.shape[0] :] = voiced[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:] * return_length2 / return_length
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        if feats0 is not None:
            # The offline protect blend, on the frames the detector found unvoiced.
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)[:, :p_len, :]
            mask = torch.where(self.cache_voiced[-p_len:], 1.0, params.protect).to(feats.dtype)[None, :, None]
            feats = feats * mask + feats0 * (1 - mask)
        p_len_tensor = torch.LongTensor([p_len]).to(self.device)
        speaker = min(max(int(params.speaker_id), 0), max(voice.n_spk - 1, 0))
        sid = torch.LongTensor([speaker]).to(self.device)
        skip, ret, ret2 = int(skip_head), int(return_length), int(return_length2)
        if cache_pitch is not None and cache_pitchf is not None:
            infered = run_cuda_graph(
                net_g,
                f"rvc-realtime-f0-{skip}-{ret}-{ret2}",
                lambda phone, lengths, coarse, continuous, speaker_t: net_g.infer(phone, lengths, coarse, continuous, speaker_t, skip, ret, ret2)[0],
                feats,
                p_len_tensor,
                cache_pitch,
                cache_pitchf,
                sid,
            )
        else:
            infered = run_cuda_graph(
                net_g,
                f"rvc-realtime-no-f0-{skip}-{ret}-{ret2}",
                lambda phone, lengths, speaker_t: net_g.infer(phone, lengths, speaker_t, skip, ret, ret2)[0],
                feats,
                p_len_tensor,
                sid,
            )
        infered = infered.squeeze(1).float()
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                from rvc_next.engine.audio.sinc_resample import Resample

                self.resample_kernel[upp_res] = Resample(orig_freq=upp_res, new_freq=self.tgt_sr // 100, dtype=torch.float32).to(self.device)
            infered = self.resample_kernel[upp_res](infered[:, : return_length * upp_res])
        return infered.squeeze()

    def _get_f0(self, x: Any, semitones: float, method: str, unvoiced: str = "original") -> tuple[Any, Any, Any]:
        """Pitch for the tail of the 16 kHz buffer, interpolated (unless ``unvoiced`` is "zero") and
        shifted, as (coarse, Hz, voiced) on the device."""
        # The providers' models take the tensor already on the device; their ``compute`` would copy it to numpy.
        if method == "pm":
            f0 = self._pm(x.cpu().numpy())
        elif method == "rmvpe":
            rmvpe: Any = self.runtime.f0("rmvpe")
            f0 = rmvpe.model.infer_from_audio(x, thred=0.03)
        elif method == "fcpe":
            fcpe: Any = self.runtime.f0("fcpe")
            f0 = fcpe.model.infer(x.unsqueeze(0).float(), sr=16000, decoder_mode="local_argmax", threshold=0.006).squeeze().detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported F0 method: {method}")
        import torch

        uv = f0 == 0
        voiced = torch.from_numpy(~np.asarray(uv)).to(self.device)
        if np.any(~uv) and unvoiced != "zero":
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0 *= pow(2, semitones / 12)
        return (*self._f0_post(f0), voiced)

    @staticmethod
    def _pm(x: np.ndarray) -> np.ndarray:
        """Realtime pm: padded for a 65 Hz floor, cut to the block's frames (unlike the offline pm)."""
        import parselmouth  # ty: ignore[unresolved-import]  # a compiled extension without stubs

        p_len = x.shape[0] // 160 + 1
        f0_min = 65
        l_pad = int(np.ceil(1.5 / f0_min * 16000))
        r_pad = l_pad + 1
        s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), 16000).to_pitch_ac(time_step=0.01, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=1100)
        f0 = s.selected_array["frequency"]
        if len(f0) < p_len:
            f0 = np.pad(f0, (0, p_len - len(f0)))
        return f0[:p_len]

    def _f0_post(self, f0: Any) -> tuple[Any, Any]:
        import torch

        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(np.asarray(f0))
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return torch.round(f0_mel).long(), f0
