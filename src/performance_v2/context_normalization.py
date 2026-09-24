"""Train-only affine normalization for the V2 context vector.

DEVELOPMENT_VALIDATION_ONLY. Fit exclusively on TRAIN rows (optionally
restricted to a set of TRAIN-fold subjects, for CV-safe fold-local fitting);
never fit on VALIDATION or TEST. Persists mean/std/transform version/a hash
of the exact fit-subject set, so any downstream consumer can verify what
the transform was actually fit on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

TRANSFORM_VERSION = "performance_v2_context_normalization_v1"


def fit_normalization(rows: Sequence[Mapping[str, object]], variant: str) -> Mapping[str, object]:
    from performance_v2.context_v2 import context_feature_names, context_vector

    names = context_feature_names(variant)
    if not names:
        return {"transform_version": TRANSFORM_VERSION, "variant": variant, "feature_names": [], "mean": [], "std": [], "fit_subject_hash": _subject_hash(rows)}
    matrix = np.asarray([context_vector(row, variant) for row in rows], dtype=np.float64)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {
        "transform_version": TRANSFORM_VERSION, "variant": variant,
        "feature_names": list(names), "mean": mean.tolist(), "std": std.tolist(),
        "fit_subject_hash": _subject_hash(rows), "n_rows_fit": len(rows),
    }


def _subject_hash(rows: Sequence[Mapping[str, object]]) -> str:
    subjects = sorted({str(row["subject_id"]) for row in rows})
    return hashlib.sha256("||".join(subjects).encode("utf-8")).hexdigest()


def apply_normalization(rows: Sequence[Mapping[str, object]], normalization: Mapping[str, object]) -> np.ndarray:
    from performance_v2.context_v2 import context_vector

    variant = normalization["variant"]
    if not normalization["feature_names"]:
        return np.zeros((len(rows), 0), dtype=np.float32)
    matrix = np.asarray([context_vector(row, variant) for row in rows], dtype=np.float64)
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    std = np.asarray(normalization["std"], dtype=np.float64)
    normalized = (matrix - mean) / std
    return np.nan_to_num(normalized, nan=0.0).astype(np.float32)
