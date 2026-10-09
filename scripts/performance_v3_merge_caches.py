#!/usr/bin/env python3
"""Concatenate per-cohort V3 feature caches (V1 dev + D1..D3) into the development-pool cache."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from performance_v3.features import FeatureCache, _identity_sha256, load_feature_cache, save_feature_cache  # noqa: E402

CACHE = ROOT / "artifacts/performance_v3/cache"


def main() -> None:
    parts = [load_feature_cache(CACHE / f"features_{c}") for c in ("V1", "D1", "D2", "D3")]
    names = parts[0].names
    assert all(p.names == names for p in parts), "column universe mismatch"
    meta = {k: np.concatenate([p.meta[k] for p in parts]) for k in parts[0].meta}
    pool = FeatureCache(names=names, matrix=np.concatenate([p.matrix for p in parts]), meta=meta)
    pool.input_sha256 = _identity_sha256(pool)
    subjects = set(pool.meta["subject_id"].tolist())
    assert len(subjects) == 8000, len(subjects)
    save_feature_cache(pool, CACHE / "dev_pool")
    print("dev_pool rows", pool.n_rows, "cols", len(pool.names), "subjects", len(subjects))


if __name__ == "__main__":
    main()
