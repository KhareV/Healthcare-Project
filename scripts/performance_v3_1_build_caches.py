#!/usr/bin/env python3
"""Feature caches + frozen-V3 apply-only scores for the V3.1 development cohorts."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from performance_v3 import data as v3data, features as v3feat
from performance_v3.serving import V3Bundle
from performance_v3_1 import protocol as P

CACHE = ROOT / "artifacts/performance_v3_1/cache"
NAMES = v3feat.load_feature_cache(ROOT / "artifacts/performance_v3/cache/dev_pool").names

def build(cid):
    spec = P.SPECS[cid]; cdir = P.cohort_dir(ROOT, cid)
    rows = list(v3data.iter_cohort_rows(ROOT, cdir, cohort_id=spec["namespace"], role="dev"))
    base = f"artifacts/performance_v3_1/cohorts/{cid}"
    with v3feat.cohort_scoring_context(ROOT, raw_manifest_ref=f"{base}/raw/synthetic_dataset_manifest_v1.json",
                                       timeline_manifest_ref=f"{base}/timeline/synthetic_processed_manifest_v1.json") as ctx:
        trends, comps = v3feat.compute_sofa_trends_and_components(rows, ctx)
    cache = v3feat.build_feature_cache(rows, sofa_trends=trends, sofa_components=comps, n_jobs=8, names=NAMES)
    v3feat.save_feature_cache(cache, CACHE / f"features_{cid}")
    bundle = V3Bundle.load(ROOT, "artifacts/performance_v3/phase3/selected_models_v3.json")
    out = bundle.predict_rows(rows, sofa_trends=trends, sofa_components=comps)
    np.savez_compressed(CACHE / f"v3_scores_{cid}.npz", stay=np.asarray([r["stay_id"] for r in rows]), pt=np.asarray([r["prediction_time"] for r in rows]),
                        cal=np.asarray([o["support_calibrated_probability"] for o in out]), raw=np.asarray([o["support_raw_probability"] for o in out]))
    print(cid, cache.n_rows, "rows", flush=True)

if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    for c in (sys.argv[1:] or ["V31_DEV1", "V31_DEV2", "V31_HOLD"]):
        build(c)
