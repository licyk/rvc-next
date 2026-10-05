"""Retrieval index: loaded once per file, top-8 blend (the original's ``Pipeline.vc`` search)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LoadedIndex:
    path: Path
    index: Any
    vectors: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0


def load_index(path: Path | str) -> LoadedIndex:
    """Read a faiss index and reconstruct its vectors once. A ``trained_`` index is swapped for ``added_``, as the original does."""
    import faiss

    path = Path(path)
    if "trained" in path.name:
        added = path.with_name(path.name.replace("trained", "added"))
        if added.is_file():
            path = added
    index = faiss.read_index(str(path))
    return LoadedIndex(path=path, index=index, vectors=index.reconstruct_n(0, index.ntotal))


def blend(feats: np.ndarray, index: LoadedIndex, k: int = 8) -> np.ndarray:
    """Replace each frame by the inverse-square-distance weighted mean of its ``k`` nearest neighbours."""
    score, ix = index.index.search(feats, k=k)
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    return np.sum(index.vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
