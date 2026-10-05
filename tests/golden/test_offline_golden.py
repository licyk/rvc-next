"""Offline conversion against the original repository, through ``scripts/golden.py``.

Same environment as ``test_stream_golden.py``: ``RVC_NEXT_GOLDEN_ORIGINAL``, ``RVC_NEXT_TEST_ASSETS``,
``RVC_NEXT_GOLDEN_VOICES`` and ``RVC_NEXT_GOLDEN_CLIP``. Formant 0 must be bit-identical; every
case must stay within the plan's 1e-4 RMS.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.golden

ORIGINAL = Path(os.environ.get("RVC_NEXT_GOLDEN_ORIGINAL", "~/code_workspace/Retrieval-based-Voice-Conversion-WebUI")).expanduser()
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "golden.py"


def _golden():
    spec = importlib.util.spec_from_file_location("golden_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_matches_original() -> None:
    assets = Path(os.environ.get("RVC_NEXT_TEST_ASSETS", ""))
    voices = [Path(p) for p in os.environ.get("RVC_NEXT_GOLDEN_VOICES", "").split(os.pathsep) if p]
    clip = Path(os.environ.get("RVC_NEXT_GOLDEN_CLIP", ""))
    if not (ORIGINAL / "infer" / "vc" / "pipeline.py").is_file():
        pytest.skip("the original checkout is missing")
    if not (assets / "hubert_base").is_dir() or not voices or not clip.is_file():
        pytest.skip("RVC_NEXT_TEST_ASSETS, RVC_NEXT_GOLDEN_VOICES and RVC_NEXT_GOLDEN_CLIP are needed")
    pytest.importorskip("torchfcpe", reason="the original's FCPE imports torchfcpe; install it next to the original checkout")
    if not (assets / "fcpe" / "fcpe_c_v001.pt").is_file():
        # The original reads torchfcpe's bundled copy; rvc-next reads the same file as its fcpe asset.
        pytest.skip("RVC_NEXT_TEST_ASSETS needs fcpe/fcpe_c_v001.pt (rvc-next assets download fcpe)")
    golden = _golden()
    cases = [golden.Case(voice, clip, f0, pitch=2, speaker_id=1) for voice in voices for f0 in ("pm", "rmvpe", "fcpe")]
    results = golden.run(ORIGINAL, assets, cases)
    failures = [f"{c.voice.name} {c.f0}: {d:.2e}" for c, d, _ in results if d > golden.TOLERANCE]
    assert not failures, failures
    assert all(identical for _, _, identical in results), "formant 0 must be bit-identical"
