"""Deterministic subject-level cross-fitting and learning-curve subsampling for V3.

Pure functions only. Folds hash the subject id, so every cutoff row of a stay
(one stay per subject) always lands in the same fold.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union

V3_CROSSFIT_VERSION = "performance_v3_dev_crossfit_v1"


def crossfit_fold_for_subject(subject_id: str, n_folds: int = 5, version: str = V3_CROSSFIT_VERSION) -> int:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    digest = hashlib.sha256((version + "||" + subject_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % n_folds


def subject_rank_key(subject_id: str, salt: str) -> str:
    return hashlib.sha256((salt + "||" + subject_id).encode("utf-8")).hexdigest()


def _unique_sorted_by_rank(subject_ids: Iterable[str], salt: str) -> Tuple[str, ...]:
    ids = list(subject_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("subject_ids must be unique")
    return tuple(sorted(ids, key=lambda s: (subject_rank_key(s, salt), s)))


def nested_subject_subsets(
    subject_ids: Iterable[str],
    sizes: Sequence[int],
    *,
    salt: str,
    allowed_prefix_sizes: Optional[Iterable[int]] = None,
) -> Dict[int, Tuple[str, ...]]:
    """Nested subsets: every smaller subset is a prefix of every larger one.

    ``allowed_prefix_sizes`` (optional) is a closed grid of permitted sizes; a
    requested size outside it raises, so a driver cannot silently drift off the
    pre-registered learning-curve grid.
    """

    ranked = _unique_sorted_by_rank(subject_ids, salt)
    allowed = None if allowed_prefix_sizes is None else set(int(s) for s in allowed_prefix_sizes)
    out: Dict[int, Tuple[str, ...]] = {}
    for size in sorted(set(int(s) for s in sizes)):
        if size < 0 or size > len(ranked):
            raise ValueError(f"size {size} outside 0..{len(ranked)}")
        if allowed is not None and size not in allowed:
            raise ValueError(f"size {size} is not in the allowed prefix grid {sorted(allowed)}")
        out[size] = ranked[:size]
    return out


def learning_curve_subsample(subject_ids: Iterable[str], size: int, repeat: int, *, salt: str) -> Tuple[str, ...]:
    """Independent seeded draw per ``repeat`` (different repeats are different permutations)."""

    ranked = _unique_sorted_by_rank(subject_ids, f"{salt}||repeat={int(repeat)}")
    if size < 0 or size > len(ranked):
        raise ValueError(f"size {size} outside 0..{len(ranked)}")
    return ranked[: int(size)]


def clone_safe_folds(
    subject_ids: Iterable[str],
    fingerprint_groups: Union[Mapping[str, Hashable], Iterable[Iterable[str]]],
    n_folds: int = 5,
    version: str = V3_CROSSFIT_VERSION,
) -> Dict[str, int]:
    """Every member of a fingerprint (clone) group gets the fold of its lexicographically first member.

    ``fingerprint_groups`` is either subject_id -> group key, or an iterable of
    groups (each an iterable of subject ids). Subjects in no group fold alone.
    """

    ids = list(subject_ids)
    known = set(ids)
    if isinstance(fingerprint_groups, Mapping):
        buckets: Dict[Hashable, list] = {}
        for sid, key in fingerprint_groups.items():
            buckets.setdefault(key, []).append(sid)
        groups = list(buckets.values())
    else:
        groups = [list(g) for g in fingerprint_groups]
    anchor: Dict[str, str] = {}
    for group in groups:
        members = sorted(m for m in group if m in known)
        if not members:
            continue
        for member in members:
            if member in anchor and anchor[member] != members[0]:
                raise ValueError(f"subject {member} appears in more than one fingerprint group")
            anchor[member] = members[0]
    return {sid: crossfit_fold_for_subject(anchor.get(sid, sid), n_folds, version) for sid in ids}
