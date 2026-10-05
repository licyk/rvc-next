"""Read what an index file is without loading it into a voice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The feature width of each voice version: v1 voices read HuBERT layer 9 + final_proj, v2 the last layer.
DIM_VERSION = {256: "v1", 768: "v2"}


@dataclass(frozen=True)
class IndexInfo:
    dim: int
    ntotal: int
    description: str

    @property
    def version(self) -> str | None:
        """The voice version this index fits, or None for a width no voice uses."""
        return DIM_VERSION.get(self.dim)

    @property
    def empty(self) -> bool:
        """A ``trained_*`` index: trained, but no vectors were ever added."""
        return self.ntotal == 0


def inspect_index(path: Path | str) -> IndexInfo:
    """The dimension and vector count of a faiss index; raises ``ValueError`` for anything else."""
    import faiss

    try:
        try:
            index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
        except RuntimeError:
            index = faiss.read_index(str(path))
    except Exception as e:
        raise ValueError(f"{Path(path).name} is not a faiss index: {e}") from e
    return IndexInfo(dim=int(index.d), ntotal=int(index.ntotal), description=type(index).__name__)
