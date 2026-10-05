"""HuBERT/ContentVec features through ``transformers`` (the original's ``infer/hubert.py``).

v1 voices use layer 9 followed by ``final_proj`` (256 dimensions); v2 voices use the last layer
(768). Whether audio is normalised first is read once from the asset's
``preprocessor_config.json`` and applied the same way in training and inference.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from rvc_next.engine.errors import MissingAssetError
from rvc_next.engine.graph import run_cuda_graph

logger = logging.getLogger(__name__)


def _model_class() -> Any:
    from torch import nn
    from transformers import HubertModel

    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    return HubertModelWithFinalProj


def load_hubert(model_dir: Path, device: Any, is_half: bool = False) -> Any:
    import torch

    if not (model_dir / "config.json").is_file() or not any((model_dir / f).is_file() for f in ("pytorch_model.bin", "model.safetensors")):
        raise MissingAssetError(f"HuBERT is not installed in {model_dir}", ["hubert"])
    dtype = torch.float16 if is_half else torch.float32
    options: dict[str, Any] = {"local_files_only": True, "dtype": dtype}
    # DirectML does not implement every SDPA kernel transformers uses.
    if "privateuseone" in str(device):
        options["attn_implementation"] = "eager"
    logger.info("Loading HuBERT (%s on %s)", str(dtype).removeprefix("torch."), device)
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    model = _model_class().from_pretrained(str(model_dir), **options)
    model.hubert_normalize = requires_normalization(model_dir)
    return model.to(device).eval()


@lru_cache(maxsize=8)
def requires_normalization(model_dir: Path) -> bool:
    try:
        return bool(json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8")).get("do_normalize", False))
    except (OSError, ValueError):
        return False


def normalize_input(model: Any, wav: Any) -> Any:
    """Zero-mean, unit-variance per utterance when the feature extractor asks for it."""
    if not getattr(model, "hubert_normalize", False):
        return wav
    import torch.nn.functional as F

    return F.layer_norm(wav, wav.shape[-1:])


def extract_features(model: Any, source: Any, version: str, padding_mask: Any = None) -> Any:
    """Return ``[1, frames, 256]`` (v1) or ``[1, frames, 768]`` (v2) features for 16 kHz ``source``."""
    import torch

    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported RVC feature version: {version!r}")
    source = normalize_input(model, source)
    attention_mask = None
    if padding_mask is not None and bool(torch.any(padding_mask).item()):
        attention_mask = (~padding_mask.bool()).long()

    if version == "v1":
        if attention_mask is None:

            def forward_v1(input_values: Any) -> Any:
                outputs = model(input_values=input_values, attention_mask=None, output_hidden_states=True, return_dict=True)
                return model.final_proj(outputs.hidden_states[9])

            return run_cuda_graph(model, "hubert-v1-no-mask", forward_v1, source)

        def forward_v1_mask(input_values: Any, mask: Any) -> Any:
            outputs = model(input_values=input_values, attention_mask=mask, output_hidden_states=True, return_dict=True)
            return model.final_proj(outputs.hidden_states[9])

        return run_cuda_graph(model, "hubert-v1-mask", forward_v1_mask, source, attention_mask)

    if attention_mask is None:

        def forward_v2(input_values: Any) -> Any:
            return model(input_values=input_values, attention_mask=None, output_hidden_states=False, return_dict=True).last_hidden_state

        return run_cuda_graph(model, "hubert-v2-no-mask", forward_v2, source)

    def forward_v2_mask(input_values: Any, mask: Any) -> Any:
        return model(input_values=input_values, attention_mask=mask, output_hidden_states=False, return_dict=True).last_hidden_state

    return run_cuda_graph(model, "hubert-v2-mask", forward_v2_mask, source, attention_mask)
