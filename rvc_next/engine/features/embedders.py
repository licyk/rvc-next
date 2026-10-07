"""Content-feature models ("embedders"): RVC's HuBERT base (ContentVec), and the ones Applio voices
may be trained on, whose name an Applio ``.pth`` carries as ``embedder_model``.

Each is a transformers HuBERT folder (``config.json`` + ``pytorch_model.bin``). ContentVec is the
``hubert`` asset in ``hubert_base/``; the others are ``embedder-<name>`` assets in
``embedders/<name>/``. Only ContentVec and SPIN carry ``final_proj``, which v1 voices read; the
others are for v2 voices only.
"""

from __future__ import annotations

CONTENTVEC = "contentvec"
EMBEDDERS = ("contentvec", "spin", "spin-v2", "chinese-hubert-base", "japanese-hubert-base", "korean-hubert-base")
WITH_FINAL_PROJ = ("contentvec", "spin", "spin-v2")
"""Embedders a v1 voice can use: v1 reads ``final_proj``, which the others do not ship."""


def asset_id(name: str) -> str:
    """The asset that holds an embedder."""
    return "hubert" if name == CONTENTVEC else f"embedder-{name}"


def folder(name: str) -> str:
    """Its folder inside the assets directory."""
    return "hubert_base" if name == CONTENTVEC else f"embedders/{name}"


def known(name: str | None) -> bool:
    return (name or CONTENTVEC) in EMBEDDERS
