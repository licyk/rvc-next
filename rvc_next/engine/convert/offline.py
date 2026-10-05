"""Offline conversion, ported from the original ``infer/vc/pipeline.py``.

Long audio is split at the quietest point near every ``x_center`` seconds; F0 is computed once over
the whole padded file; each chunk gets HuBERT features, optional index retrieval and the protect
blend, then the synthesizer runs and the padding is trimmed. Two changes: the index is loaded once
by the runtime instead of once per file, and formant shift (from the realtime engine) is added. With
``formant = 0`` the arithmetic is the original's.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from rvc_next.engine.audio.dsp import change_rms, highpass_16k
from rvc_next.engine.audio.io import to_int16
from rvc_next.engine.audio.resample import resample
from rvc_next.engine.cancel import CancelToken
from rvc_next.engine.convert.params import VoiceParams
from rvc_next.engine.f0.base import offline_f0
from rvc_next.engine.features.hubert import extract_features
from rvc_next.engine.graph import cuda_graph_enabled, run_cuda_graph
from rvc_next.engine.index.retrieval import LoadedIndex, blend

if TYPE_CHECKING:
    from rvc_next.engine.models.loader import LoadedVoice
    from rvc_next.engine.runtime import Runtime

logger = logging.getLogger(__name__)

SR_16K = 16000
WINDOW = 160


def prepare_input(audio: np.ndarray) -> np.ndarray:
    """Scale down to a 0.95 peak when louder, as the original does before converting."""
    audio_max = np.abs(audio).max() / 0.95 if audio.size else 0.0
    if audio_max > 1:
        audio = audio / audio_max
    return audio.astype(np.float32, copy=False)


class OfflineConverter:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.device = runtime.device
        self.is_half = runtime.is_half
        chunk = runtime.chunk
        self.t_pad = SR_16K * chunk.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = SR_16K * chunk.x_query
        self.t_center = SR_16K * chunk.x_center
        self.t_max = SR_16K * chunk.x_max
        self.x_pad = chunk.x_pad

    def convert(
        self,
        audio_16k: np.ndarray,
        voice: LoadedVoice,
        params: VoiceParams,
        *,
        index: LoadedIndex | None = None,
        resample_to: int | None = None,
        progress: Callable[[float], None] | None = None,
        cancel: CancelToken | None = None,
    ) -> tuple[np.ndarray, int]:
        """Convert 16 kHz mono audio; return int16 samples and their rate."""
        import torch

        audio = prepare_input(audio_16k)
        hubert = self.runtime.hubert()
        tgt_sr = voice.tgt_sr
        t_pad_tgt = tgt_sr * self.x_pad
        index_rate = params.index_rate if index is not None else 0.0
        if index is None or params.index_rate == 0:
            index = None

        audio = highpass_16k(audio)
        audio_pad = np.pad(audio, (WINDOW // 2, WINDOW // 2), mode="reflect")
        opt_ts: list[int] = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(WINDOW):
                audio_sum += np.abs(audio_pad[i : i - WINDOW])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                window = audio_sum[t - self.t_query : t + self.t_query]
                opt_ts.append(t - self.t_query + int(np.where(window == window.min())[0][0]))
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // WINDOW
        sid = torch.tensor(params.speaker_id, device=self.device).unsqueeze(0).long()
        pitch = pitchf = None
        if voice.if_f0:
            if cancel:
                cancel.check()
            provider = self.runtime.f0(params.f0_method)
            coarse, hz = offline_f0(provider, audio_pad, p_len, params.pitch - params.formant)
            pitch = torch.tensor(coarse[:p_len], device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(hz[:p_len].astype(np.float32), device=self.device).unsqueeze(0).float()

        total = len(opt_ts) + 1
        audio_opt: list[np.ndarray] = []
        s = 0
        t: int | None = None
        for n, t in enumerate(opt_ts):
            if cancel:
                cancel.check()
            t = t // WINDOW * WINDOW
            seg_pitch = pitch[:, s // WINDOW : (t + self.t_pad2) // WINDOW] if pitch is not None else None
            seg_pitchf = pitchf[:, s // WINDOW : (t + self.t_pad2) // WINDOW] if pitchf is not None else None
            out = self._vc(hubert, voice, sid, audio_pad[s : t + self.t_pad2 + WINDOW], seg_pitch, seg_pitchf, index, index_rate, params)
            audio_opt.append(out[t_pad_tgt:-t_pad_tgt])
            s = t
            if progress:
                progress((n + 1) / total)
        if cancel:
            cancel.check()
        seg_pitch = (pitch[:, t // WINDOW :] if t is not None else pitch) if pitch is not None else None
        seg_pitchf = (pitchf[:, t // WINDOW :] if t is not None else pitchf) if pitchf is not None else None
        out = self._vc(hubert, voice, sid, audio_pad[t:] if t is not None else audio_pad, seg_pitch, seg_pitchf, index, index_rate, params)
        audio_opt.append(out[t_pad_tgt:-t_pad_tgt])
        if progress:
            progress(1.0)
        result = np.concatenate(audio_opt)
        if params.rms_mix_rate != 1:
            result = change_rms(audio, SR_16K, result, tgt_sr, params.rms_mix_rate)
        out_sr = tgt_sr
        if resample_to and resample_to >= 16000 and resample_to != tgt_sr:
            result = resample(result, tgt_sr, resample_to)
            out_sr = resample_to
        self._empty_cache()
        return to_int16(result), out_sr

    def _vc(
        self,
        hubert: Any,
        voice: LoadedVoice,
        sid: Any,
        audio0: np.ndarray,
        pitch: Any,
        pitchf: Any,
        index: LoadedIndex | None,
        index_rate: float,
        params: VoiceParams,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        protect = params.protect
        net_g = voice.net_g
        feats = torch.from_numpy(audio0)
        feats = feats.half() if self.is_half else feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        with torch.no_grad():
            feats = extract_features(hubert, feats.to(self.device), voice.version, padding_mask=padding_mask)
        use_protect = protect < 0.5 and pitch is not None and pitchf is not None
        feats0 = feats.clone() if use_protect else None
        if index is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")
            npy = blend(npy, index)
            if self.is_half:
                npy = npy.astype("float16")
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if feats0 is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = audio0.shape[0] // WINDOW
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        if feats0 is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        factor = pow(2, params.formant / 12)
        frames = p_len
        n_res = math.ceil(frames * factor) if params.formant else None
        p_len_t = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                if n_res is None:
                    synthesized = run_cuda_graph(
                        net_g,
                        "rvc-synth-f0",
                        lambda phone, lengths, coarse, continuous, speaker: net_g.infer(phone, lengths, coarse, continuous, speaker)[0],
                        feats,
                        p_len_t,
                        pitch,
                        pitchf,
                        sid,
                    )
                else:
                    scaled = pitchf * (n_res / frames)
                    synthesized = run_cuda_graph(
                        net_g,
                        f"rvc-synth-f0-formant-{n_res}",
                        lambda phone, lengths, coarse, continuous, speaker: net_g.infer(phone, lengths, coarse, continuous, speaker, None, None, n_res)[0],
                        feats,
                        p_len_t,
                        pitch,
                        scaled,
                        sid,
                    )
            else:
                synthesized = run_cuda_graph(
                    net_g, f"rvc-synth-no-f0-{n_res}", lambda phone, lengths, speaker: net_g.infer(phone, lengths, speaker, None, None, n_res)[0], feats, p_len_t, sid
                )
            audio1 = synthesized[0, 0].data.cpu().float().numpy()
        if n_res is not None:
            audio1 = self._formant_resample(audio1, voice.tgt_sr, frames, factor)
        del feats, p_len_t, padding_mask
        self._empty_cache()
        return audio1

    def _formant_resample(self, audio: np.ndarray, tgt_sr: int, frames: int, factor: float) -> np.ndarray:
        """Squeeze audio generated at ``factor`` times the length back to ``frames`` frames, raising formants by ``factor``."""
        import torch

        from rvc_next.engine.audio.sinc_resample import Resample

        hop = tgt_sr // 100
        upp_res = int(np.floor(factor * tgt_sr // 100))
        if upp_res == hop:
            return audio[: frames * hop]
        x = torch.from_numpy(np.ascontiguousarray(audio[: frames * upp_res])).unsqueeze(0)
        return Resample(orig_freq=upp_res, new_freq=hop, dtype=torch.float32)(x).squeeze(0).numpy()

    def _empty_cache(self) -> None:
        import torch

        if torch.cuda.is_available() and not cuda_graph_enabled(self.device):
            torch.cuda.empty_cache()
