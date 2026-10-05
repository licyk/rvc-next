"""Pairing files that belong together, when their names may say nothing.

Pure functions: they see what inspection found (versions, dimensions, names, folders) and return
proposals with the reasons behind them. The import session applies them; the user can change any.

Indexes and voices: an index's width fixes the voice version it fits (256 → v1, 768 → v2) and an
index without vectors fits nothing. Within that, the only evidence is the names and where the files
came from, so a pairing is automatic only when it is the only one possible, or when the names agree.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

DIM_VERSION = {256: "v1", 768: "v2"}

# How far ahead the best candidate must be to be proposed, and how good to need no confirmation.
PROPOSE_SCORE = 0.5
PROPOSE_MARGIN = 0.25
CONFIDENT_SCORE = 0.8

_ADDED = re.compile(r"^(?:added|trained)_ivf\d+_flat_nprobe_\d+_(?P<exp>.+?)_(?P<ver>v1|v2)(?:_spkid(?P<spk>\d+))?$")
_SPK = re.compile(r"_spkid(\d+)$")
_VER = re.compile(r"(?:^|[_\-. ])(v1|v2)(?:$|[_\-. ])")
_EPOCH = re.compile(r"_e\d+_s\d+$")
_NOISE = re.compile(r"^(?:added|trained|ivf\d+|flat|nprobe|\d+|index|model|pth|v1|v2|spkid\d+|e\d+|s\d+|g|d|f0g\d+k|f0d\d+k|g\d+k|d\d+k|config|ckpt|yaml)$")


def _stem(name: str) -> str:
    return re.sub(r"\.[^./\\]+$", "", name.replace("\\", "/").rsplit("/", 1)[-1]).lower()


def tokens(name: str) -> set[str]:
    """Meaningful words of a file name: separators, numbers and index boilerplate removed."""
    parts = re.split(r"[^0-9a-z぀-ヿ一-鿿]+", _stem(name))
    return {p for p in parts if p and not _NOISE.match(p)}


def experiment_of_index(name: str) -> str | None:
    """The experiment name RVC writes into ``added_IVF<n>_Flat_nprobe_1_<exp>_<v>.index``."""
    m = _ADDED.match(_stem(name))
    return m.group("exp") if m else None


def experiment_of_voice(name: str) -> str:
    """A voice's experiment name: its file stem without ``_e<N>_s<N>``."""
    return _EPOCH.sub("", _stem(name))


def speaker_of_index(name: str) -> int | None:
    m = _SPK.search(_stem(name))
    return int(m.group(1)) if m else None


def version_in_name(name: str) -> str | None:
    m = _VER.search(_stem(name))
    return m.group(1) if m else None


def name_score(index_name: str, voice_name: str) -> tuple[float, str | None]:
    """How strongly the two names say they belong together, 0..1, and the reason code."""
    voice_exp = experiment_of_voice(voice_name)
    exp = experiment_of_index(index_name)
    if exp is not None and exp == voice_exp:
        return 1.0, "experiment_name"
    stem = _stem(index_name)
    if voice_exp and len(voice_exp) >= 3 and (voice_exp in stem or (exp and exp in voice_exp and len(exp) >= 3)):
        return 0.8, "name_contains"
    a, b = tokens(index_name), tokens(voice_name)
    if not a or not b:
        return 0.0, None
    overlap = len(a & b) / len(a | b)
    return (round(0.7 * overlap, 3), "similar_name") if overlap else (0.0, None)


@dataclass(frozen=True)
class VoiceCandidate:
    ref: str
    """``file:<id>`` for a voice in this import, ``voice:<id>`` for a library voice."""
    name: str
    version: str
    speaker_ids: tuple[int, ...] = ()
    group: str | None = None
    """The folder or archive the file came in; None for library voices."""

    @property
    def staged(self) -> bool:
        return self.ref.startswith("file:")


@dataclass(frozen=True)
class IndexCandidate:
    ref: str
    name: str
    dim: int
    ntotal: int
    group: str | None = None


@dataclass
class Candidate:
    voice: str
    score: float
    reasons: list[str]


Status = Literal["unique", "evidence", "choose", "incompatible", "empty", "duplicate"]


@dataclass
class IndexProposal:
    index: str
    voice: str | None
    key: str
    status: Status
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    """Compatible voices, best first; the only ones the user may choose from."""
    confident: bool = False


def _key(index: IndexCandidate, voice: VoiceCandidate | None) -> str:
    spk = speaker_of_index(index.name)
    if spk is not None and voice is not None and spk in voice.speaker_ids:
        return f"spk{spk}"
    return "default"


def _score(index: IndexCandidate, voice: VoiceCandidate) -> Candidate:
    score, reason = name_score(index.name, voice.name)
    reasons = ["version"] + ([reason] if reason else [])
    if voice.staged and index.group is not None and index.group == voice.group:
        score += 0.2
        reasons.append("same_folder")
    named = version_in_name(index.name)
    if named == voice.version:
        score += 0.1
    spk = speaker_of_index(index.name)
    if spk is not None and spk in voice.speaker_ids:
        score += 0.1
        reasons.append("speaker")
    return Candidate(voice.ref, round(min(score, 1.5), 3), reasons)


def pair_indexes(indexes: Iterable[IndexCandidate], voices: Iterable[VoiceCandidate]) -> list[IndexProposal]:
    """A proposal for every index: which voice it goes with, under which key, and how sure that is."""
    indexes = list(indexes)
    voices = list(voices)
    staged = [v for v in voices if v.staged]
    out: list[IndexProposal] = []
    for idx in indexes:
        version = DIM_VERSION.get(idx.dim)
        if idx.ntotal == 0:
            out.append(IndexProposal(idx.ref, None, "default", "empty", reasons=["empty"]))
            continue
        if version is None:
            out.append(IndexProposal(idx.ref, None, "default", "incompatible", reasons=[f"dimension:{idx.dim}"]))
            continue
        pool = [v for v in staged if v.version == version] or [v for v in voices if not v.staged and v.version == version]
        if not pool:
            out.append(IndexProposal(idx.ref, None, "default", "incompatible", reasons=[f"no_{version}_voice"]))
            continue
        cands = sorted((_score(idx, v) for v in pool), key=lambda c: -c.score)
        best = cands[0]
        best_voice = next(v for v in pool if v.ref == best.voice)
        rivals = [i for i in indexes if i.ntotal > 0 and DIM_VERSION.get(i.dim) == version and _key(i, best_voice) == _key(idx, best_voice)]
        if best_voice.staged and len(pool) == 1 and len(rivals) == 1:
            out.append(IndexProposal(idx.ref, best.voice, _key(idx, best_voice), "unique", best.score, ["only_match", *best.reasons[1:]], cands, confident=True))
            continue
        second = cands[1].score if len(cands) > 1 else 0.0
        if best.score >= PROPOSE_SCORE and best.score - second >= PROPOSE_MARGIN:
            confident = best_voice.staged and best.score >= CONFIDENT_SCORE
            out.append(IndexProposal(idx.ref, best.voice, _key(idx, best_voice), "evidence", best.score, best.reasons, cands, confident))
            continue
        out.append(IndexProposal(idx.ref, None, "default", "choose", best.score, ["ambiguous"], cands))
    # Two indexes for the same voice and key: the weaker one is left for the user.
    taken: dict[tuple[str, str], IndexProposal] = {}
    for p in sorted((p for p in out if p.voice), key=lambda p: -p.score):
        slot = (p.voice or "", p.key)
        if slot in taken:
            p.voice, p.status, p.confident = None, "duplicate", False
            p.reasons = ["duplicate"]
        else:
            taken[slot] = p
    return out


@dataclass(frozen=True)
class Named:
    ref: str
    name: str
    version: str | None = None
    group: str | None = None


@dataclass
class PairProposal:
    left: str
    right: str | None
    status: Literal["unique", "evidence", "choose", "none"]
    candidates: list[str] = field(default_factory=list)


# A generator's name with the G turned into a D, as RVC names its pairs: f0G40k/f0D40k, G_2333333/D_2333333.
G_TO_D: list[tuple[str, str]] = [(r"(^|[_\-. ]|f0)g(?=\d|_|\.|$)", r"\1d")]


def _partner_score(left: Named, right: Named, swap: list[tuple[str, str]]) -> float:
    a, b = _stem(left.name), _stem(right.name)
    for pattern, repl in swap:
        a = re.sub(pattern, repl, a)
    if a == b:
        return 1.0
    score = 0.0
    ta, tb = tokens(left.name), tokens(right.name)
    if ta and tb:
        score = 0.7 * len(ta & tb) / len(ta | tb)
    if left.group is not None and left.group == right.group:
        score += 0.2
    return score


def pair_partners(lefts: Iterable[Named], rights: Iterable[Named], swap: list[tuple[str, str]] | None = None, same_version: bool = False) -> list[PairProposal]:
    """Pair each left file with at most one right file (a G with its D, a checkpoint with its YAML).

    One of each (of the same version, when asked) pairs at once; otherwise names decide, with the same
    thresholds as indexes; what stays unclear is left for the user.
    """
    lefts, rights = list(lefts), list(rights)
    out: list[PairProposal] = []
    used: set[str] = set()
    for left in lefts:
        pool = [r for r in rights if not same_version or r.version is None or left.version is None or r.version == left.version]
        if not pool:
            out.append(PairProposal(left.ref, None, "none"))
            continue
        ranked = sorted(pool, key=lambda r: -_partner_score(left, r, swap or []))
        if len(lefts) == 1 and len(pool) == 1:
            out.append(PairProposal(left.ref, pool[0].ref, "unique", [r.ref for r in ranked]))
            used.add(pool[0].ref)
            continue
        best = _partner_score(left, ranked[0], swap or [])
        second = _partner_score(left, ranked[1], swap or []) if len(ranked) > 1 else 0.0
        if best >= PROPOSE_SCORE and best - second >= PROPOSE_MARGIN and ranked[0].ref not in used:
            out.append(PairProposal(left.ref, ranked[0].ref, "evidence", [r.ref for r in ranked]))
            used.add(ranked[0].ref)
        else:
            out.append(PairProposal(left.ref, None, "choose", [r.ref for r in ranked]))
    return out
