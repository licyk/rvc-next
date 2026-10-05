"""Load a small model into a synthesizer (the original's ``VC.get_vc``, without the UI updates)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rvc_next.engine.models.checkpoint import SmallModel, read_small_model


@dataclass
class LoadedVoice:
    path: Path
    net_g: Any
    small: SmallModel

    @property
    def tgt_sr(self) -> int:
        return self.small.target_sample_rate

    @property
    def if_f0(self) -> bool:
        return self.small.pitch_guidance

    @property
    def version(self) -> str:
        return self.small.version

    @property
    def n_spk(self) -> int:
        return self.small.speaker_slots


def synthesizer_class(version: str, f0: bool) -> type:
    from rvc_next.engine.models.synthesizer import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    return {
        ("v1", True): SynthesizerTrnMs256NSFsid,
        ("v1", False): SynthesizerTrnMs256NSFsid_nono,
        ("v2", True): SynthesizerTrnMs768NSFsid,
        ("v2", False): SynthesizerTrnMs768NSFsid_nono,
    }.get((version, f0), SynthesizerTrnMs256NSFsid)


def build_synthesizer(small: SmallModel, device: Any, is_half: bool) -> Any:
    config = list(small.config)
    config[-3] = small.speaker_slots
    net_g = synthesizer_class(small.version, small.pitch_guidance)(*config, is_half=is_half)
    del net_g.enc_q
    net_g.load_state_dict(small.weight, strict=False)
    net_g.eval().to(device)
    return net_g.half() if is_half else net_g.float()


def load_voice(path: Path | str, device: Any, is_half: bool) -> LoadedVoice:
    path = Path(path)
    small = read_small_model(path)
    return LoadedVoice(path=path, net_g=build_synthesizer(small, device, is_half), small=small)
