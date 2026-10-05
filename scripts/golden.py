"""Run the original RVC repository and rvc-next on the same inputs, and compare.

    python scripts/golden.py --original ~/code_workspace/Retrieval-based-Voice-Conversion-WebUI \
        --assets <assets dir> --voice a.pth [--voice b.pth] --clip speech.wav [--clip ...] [--f0 pm --f0 rmvpe]

Both sides run on the CPU in fp32 with the same seed and the same 16 kHz input, decoded once.
The original's ``infer.vc.pipeline.Pipeline`` and ``infer.module.models`` are imported from its
checkout; nothing of its UI is loaded. Prints one line per case and exits 1 if any case exceeds
the tolerance (RMS of the per-sample difference, on the -1..1 scale).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TOLERANCE = 1e-4


@dataclass
class Case:
    voice: Path
    clip: Path
    f0: str
    pitch: float = 0.0
    index_rate: float = 0.0
    index: Path | None = None
    speaker_id: int = 0
    protect: float = 0.33
    rms_mix_rate: float = 0.25


class _OriginalConfig:
    """The fields ``Pipeline`` reads, with the original's CPU values."""

    x_pad, x_query, x_center, x_max = 1, 6, 38, 41
    is_half = False
    device = "cpu"


def load_original(original: Path, assets: Path) -> dict[str, Any]:
    os.environ.pop("RVC_CUDA_GRAPH", None)
    os.environ["rmvpe_root"] = str(assets / "rmvpe")  # noqa: SIM112 - the original reads this lower-case name
    sys.path.insert(0, str(original))
    hubert_mod = importlib.import_module("infer.hubert")
    vars(hubert_mod)["HUBERT_MODEL_PATH"] = (assets / "hubert_base").resolve()
    pipeline_mod = importlib.import_module("infer.vc.pipeline")
    models_mod = importlib.import_module("infer.module.models")
    return {"hubert": hubert_mod, "pipeline": pipeline_mod, "models": models_mod}


def original_convert(orig: dict[str, Any], hubert: Any, case: Case, audio: np.ndarray, shared: dict[str, Any], assets_dir: Path) -> tuple[np.ndarray, int]:
    import torch

    cpt = torch.load(str(case.voice), map_location="cpu", weights_only=False)
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    m = orig["models"]
    cls = {
        ("v1", 1): m.SynthesizerTrnMs256NSFsid,
        ("v1", 0): m.SynthesizerTrnMs256NSFsid_nono,
        ("v2", 1): m.SynthesizerTrnMs768NSFsid,
        ("v2", 0): m.SynthesizerTrnMs768NSFsid_nono,
    }[(version, if_f0)]
    net_g = cls(*cpt["config"], is_half=False)
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().float()
    pipeline = orig["pipeline"].Pipeline(tgt_sr, _OriginalConfig())
    # The original builds RMVPE inside the pipeline; its random init would draw from the seeded RNG.
    if case.f0 == "rmvpe":
        if "rmvpe" not in shared:
            from infer.rmvpe import RMVPE  # ty: ignore[unresolved-import]

            shared["rmvpe"] = RMVPE(str(assets_dir / "rmvpe" / "rmvpe.pt"), is_half=False, device="cpu")
        pipeline.model_rmvpe = shared["rmvpe"]
    if case.f0 == "fcpe":
        if "fcpe" not in shared:
            from infer.fcpe import FCPEInfer  # ty: ignore[unresolved-import]

            shared["fcpe"] = FCPEInfer("cpu")
        pipeline.model_fcpe = shared["fcpe"]
    torch.manual_seed(0)
    out = pipeline.pipeline(
        hubert,
        net_g,
        case.speaker_id,
        audio.copy(),
        [0, 0, 0],
        case.pitch,
        case.f0,
        str(case.index or ""),
        case.index_rate,
        if_f0,
        tgt_sr,
        0,
        case.rms_mix_rate,
        version,
        case.protect,
    )
    return out, tgt_sr


def next_convert(runtime: Any, case: Case, audio: np.ndarray) -> tuple[np.ndarray, int]:
    import torch

    from rvc_next.engine.convert.offline import OfflineConverter
    from rvc_next.engine.convert.params import VoiceParams

    voice = runtime.voice(case.voice)
    index = runtime.index(case.index) if case.index else None
    params = VoiceParams(speaker_id=case.speaker_id, pitch=case.pitch, f0_method=case.f0, index_rate=case.index_rate, protect=case.protect, rms_mix_rate=case.rms_mix_rate)
    torch.manual_seed(0)
    return OfflineConverter(runtime).convert(audio, voice, params, index=index)


def rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    d = a.astype(np.float64) / 32768 - b.astype(np.float64) / 32768
    return float(np.sqrt(np.mean(d * d)))


def run(original: Path, assets: Path, cases: list[Case]) -> list[tuple[Case, float, bool]]:
    from rvc_next.engine.audio.io import decode
    from rvc_next.engine.convert.offline import prepare_input
    from rvc_next.engine.runtime import Runtime

    orig = load_original(original, assets)
    hubert = orig["hubert"].load_hubert_model("cpu", False)
    runtime = Runtime(assets, device="cpu", precision="fp32")
    # Load models before any seed is set: their weight initialisation draws from the RNG.
    runtime.hubert()
    for method in {c.f0 for c in cases}:
        runtime.f0(method)
    shared: dict[str, Any] = {}
    results = []
    for case in cases:
        audio, _ = decode(case.clip, 16000)
        a, sr_a = original_convert(orig, hubert, case, prepare_input(audio), shared, assets)
        b, sr_b = next_convert(runtime, case, audio)
        diff = rms_diff(a, b) if sr_a == sr_b else float("inf")
        results.append((case, diff, bool(np.array_equal(a, b))))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--voice", type=Path, action="append", required=True)
    parser.add_argument("--clip", type=Path, action="append", required=True)
    parser.add_argument("--f0", action="append", default=None)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--speaker-id", type=int, default=0)
    args = parser.parse_args(argv)
    cases = [
        Case(voice, clip, f0, pitch=args.pitch, index=args.index, index_rate=args.index_rate, speaker_id=args.speaker_id)
        for voice in args.voice
        for clip in args.clip
        for f0 in (args.f0 or ["pm", "rmvpe"])
    ]
    failed = 0
    for case, diff, identical in run(args.original, args.assets, cases):
        ok = diff <= TOLERANCE
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {case.voice.name} {case.clip.name} {case.f0}: rms {diff:.2e}{' (identical)' if identical else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
