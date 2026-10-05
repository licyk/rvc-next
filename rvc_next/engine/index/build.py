"""Index stage: a faiss ``IVF<n>,Flat`` index over the experiment's features (the original ``train/train_index.py``).

Changes from the original:
- the index is rebuilt when its feature files changed (a fingerprint is stored in
  ``index_fingerprint.json``), instead of being skipped whenever any ``added_*.index`` exists;
- above 200,000 vectors the features are reduced to 10,000 centres with faiss k-means rather than
  scikit-learn's mini-batch k-means;
- nothing is linked into a global index folder: the core attaches the index to the exported voice.
In multi-speaker mode each speaker gets its own index (``…_spkid<N>.index``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rvc_next.engine.cancel import CancelToken

FINGERPRINT_FILE = "index_fingerprint.json"
KMEANS_THRESHOLD = 200_000
KMEANS_CENTRES = 10_000


@dataclass(frozen=True)
class IndexRequest:
    exp_dir: str
    version: str = "v2"
    name: str = ""
    """Used in the index file names; default the experiment folder's name."""
    multi_speaker: bool | None = None
    """None: per speaker when the experiment has a multi-speaker manifest."""
    n_cpu: int = 0
    force: bool = False
    seed: int | None = None


def _groups(exp: Path, version: str, multi: bool | None) -> dict[int | None, list[Path]]:
    from rvc_next.engine.train import multispeaker
    from rvc_next.engine.train.feature_extract import feature_dir_name

    feature_dir = exp / feature_dir_name(version)
    paths = sorted(p for p in feature_dir.glob("*.npy")) if feature_dir.is_dir() else []
    if not paths:
        raise ValueError("Extract features first: the feature folder is empty")
    manifest = None
    if multi is not False:
        try:
            manifest = multispeaker.load_manifest(exp, check_files=False)
        except multispeaker.ManifestError:
            if multi:
                raise
    groups: dict[int | None, list[Path]] = {}
    if manifest is None:
        groups[None] = paths
        return groups
    for p in paths:
        entry = multispeaker.speaker_of(p.stem, manifest)
        if entry is not None:
            groups.setdefault(int(entry["speaker_id"]), []).append(p)
    if not groups:
        raise ValueError("No feature file belongs to a speaker in the manifest")
    return groups


def _read_fingerprints(exp: Path) -> dict[str, str]:
    try:
        data = json.loads((exp / FINGERPRINT_FILE).read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_fingerprints(exp: Path, data: dict[str, str]) -> None:
    (exp / FINGERPRINT_FILE).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _existing(exp: Path, kind: str, name: str, version: str, suffix: str) -> list[Path]:
    tail = f"_{name}_{version}{suffix}.index"
    return [p for p in exp.glob(f"{kind}_IVF*_Flat_nprobe_*{tail}") if p.name.endswith(tail)]


def build_one(exp: Path, name: str, version: str, speaker_id: int | None, paths: list[Path], n_cpu: int, seed: int | None, log: Any) -> dict[str, Any]:
    import faiss

    suffix = "" if speaker_id is None else f"_spkid{speaker_id}"
    rng = np.random.default_rng(seed) if seed is not None else np.random
    big_npy = np.concatenate([np.load(p) for p in paths], 0)
    big_npy = big_npy[rng.permutation(big_npy.shape[0])]
    np.save(exp / f"total_fea{suffix}.npy", big_npy)
    if big_npy.shape[0] > KMEANS_THRESHOLD:
        log(f"Reducing {big_npy.shape[0]} vectors to {KMEANS_CENTRES} centres")
        kmeans = faiss.Kmeans(big_npy.shape[1], KMEANS_CENTRES, niter=20, seed=seed if seed is not None else 1234, verbose=False)
        kmeans.train(np.ascontiguousarray(big_npy, dtype=np.float32))
        big_npy = kmeans.centroids
    big_npy = np.ascontiguousarray(big_npy, dtype=np.float32)
    n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))
    log(f"Index{' for speaker ' + str(speaker_id) if speaker_id is not None else ''}: {big_npy.shape[0]} vectors, IVF{n_ivf}")
    if n_cpu > 0:
        faiss.omp_set_num_threads(n_cpu)
    index = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    trained = exp / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{name}_{version}{suffix}.index"
    faiss.write_index(index, str(trained))
    for start in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[start : start + 8192])
    added = exp / f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{name}_{version}{suffix}.index"
    faiss.write_index(index, str(added))
    return {"speaker_id": speaker_id, "added": str(added), "trained": str(trained), "vectors": int(big_npy.shape[0]), "n_ivf": n_ivf}


def run(request: IndexRequest, progress: Any = None, log: Any = None, cancel: CancelToken | None = None) -> dict[str, Any]:
    """Build or reuse one index per group; return ``{"indexes": [{speaker_id, added, …, reused}], "built", "reused"}``."""
    from rvc_next.engine.train.fingerprint import of_paths

    exp = Path(request.exp_dir)
    name = request.name or exp.name
    log = log or (lambda message: None)
    groups = _groups(exp, request.version, request.multi_speaker)
    stored = _read_fingerprints(exp)
    results: list[dict[str, Any]] = []
    order = sorted(groups, key=lambda v: -1 if v is None else v)
    for n, speaker_id in enumerate(order):
        if cancel is not None:
            cancel.check()
        suffix = "" if speaker_id is None else f"_spkid{speaker_id}"
        fp = of_paths(groups[speaker_id], extra=(request.version, name))
        key = suffix or "default"
        added = _existing(exp, "added", name, request.version, suffix)
        if not request.force and added and stored.get(key) == fp:
            newest = max(added, key=os.path.getmtime)
            log(f"Index{' for speaker ' + str(speaker_id) if speaker_id is not None else ''} is up to date: {newest.name}")
            results.append({"speaker_id": speaker_id, "added": str(newest), "reused": True})
        else:
            for p in added + _existing(exp, "trained", name, request.version, suffix):
                p.unlink(missing_ok=True)
            result = build_one(exp, name, request.version, speaker_id, groups[speaker_id], request.n_cpu, request.seed, log)
            result["reused"] = False
            results.append(result)
            stored[key] = fp
            _write_fingerprints(exp, stored)
        if progress:
            progress((n + 1) / len(order), "index")
    return {"indexes": results, "built": sum(not r["reused"] for r in results), "reused": sum(bool(r["reused"]) for r in results)}
