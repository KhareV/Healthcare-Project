#!/usr/bin/env python3
"""Build Performance-V3 feature caches (all feature groups, float32) per development cohort.

Stages (run per cohort id in {V1, D1, D2, D3, LCE}):
  sofa      past-cutoff SOFA trend + six component scores/observed flags at t, via the frozen
            sofa_at (data.synthetic.sofa) -- no second SOFA implementation. Also writes the cohort's
            feature-name universe.
  features  streams rows through performance_v2.diagnostic_features.build_all_groups into a float32
            matrix over the UNION name universe, saves npz + sidecar.

Only roles 'dev' (V1 train+validation, D1-D3) and 'lce' are loadable; sealed/exposed test cohorts
are refused by performance_v3.data.
"""

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from performance_v3 import data as v3data  # noqa: E402
from performance_v3 import features as v3feat  # noqa: E402

CACHE_DIR = ROOT / "artifacts/performance_v3/cache"
COHORTS = {
    "V1": {"kind": "v1", "role": "dev"},
    "D1": {"kind": "v3", "role": "dev"},
    "D2": {"kind": "v3", "role": "dev"},
    "D3": {"kind": "v3", "role": "dev"},
    "LCE": {"kind": "v3", "role": "lce"},
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cohort_path(cohort: str) -> Path:
    return ROOT / "artifacts/performance_v3/cohorts" / cohort


def _rows_callable(cohort: str):
    spec = COHORTS[cohort]
    if spec["kind"] == "v1":
        return lambda: v3data.iter_v1_dev_rows(ROOT)
    return lambda: v3data.iter_cohort_rows(ROOT, _cohort_path(cohort), cohort_id=cohort, role=spec["role"])


def _input_files(cohort: str):
    if COHORTS[cohort]["kind"] == "v1":
        from performance_v2 import data_loading as dl

        return [ROOT / dl.PRE_SPLIT_PATH, ROOT / dl.STATICS_PATH, ROOT / dl.SPLIT_PATH]
    cd = v3data.CohortDir(_cohort_path(cohort))
    return [cd.pre_split_path, cd.statics_path]


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS: bytes


def stage_sofa(cohort: str) -> None:
    started = time.time()
    out_json = CACHE_DIR / f"sofa_{cohort}.json"
    rows = list(_rows_callable(cohort)())
    if COHORTS[cohort]["kind"] == "v1":
        from experiments.stage5_final_test import build_sofa_scoring_context

        context = build_sofa_scoring_context(ROOT)
        trends, components = v3feat.compute_sofa_trends_and_components(rows, context)
    else:
        base = f"artifacts/performance_v3/cohorts/{cohort}"
        with v3feat.cohort_scoring_context(
            ROOT, raw_manifest_ref=f"{base}/raw/synthetic_dataset_manifest_v1.json",
            timeline_manifest_ref=f"{base}/timeline/synthetic_processed_manifest_v1.json",
        ) as context:
            trends, components = v3feat.compute_sofa_trends_and_components(rows, context)
    names = v3feat._collect_names(rows, trends, components)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "trends": {f"{k[0]}||{k[1]}": v for k, v in trends.items()},
        "components": {f"{k[0]}||{k[1]}": {c: [int(s), bool(o)] for c, (s, o) in v.items()} for k, v in components.items()},
        "names": list(names),
        "n_rows": len(rows),
    }
    out_json.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[sofa {cohort}] rows={len(rows)} trends={len(trends)} comps={len(components)} names={len(names)} "
          f"{time.time()-started:.0f}s peak_rss_mb={_peak_rss_mb():.0f}", flush=True)


def _load_sofa(cohort: str):
    payload = json.loads((CACHE_DIR / f"sofa_{cohort}.json").read_text(encoding="utf-8"))
    split = lambda key: tuple(key.split("||", 1))
    trends = {split(k): v for k, v in payload["trends"].items()}
    components = {split(k): {c: (s, o) for c, (s, o) in v.items()} for k, v in payload["components"].items()}
    return trends, components, tuple(payload["names"])


def stage_features(cohorts, n_jobs: int) -> None:
    union: set = set()
    for cohort in cohorts:
        union.update(_load_sofa(cohort)[2])
    names = tuple(sorted(union))
    print(f"[features] union universe: {len(names)} columns", flush=True)
    for cohort in cohorts:
        started = time.time()
        trends, components, _ = _load_sofa(cohort)
        cache = v3feat.build_feature_cache(
            _rows_callable(cohort), sofa_trends=trends, sofa_components=components, n_jobs=n_jobs, names=names,
        )
        npz, js = v3feat.save_feature_cache(cache, CACHE_DIR / f"features_{cohort}")
        record = {
            "cohort": cohort, "role": COHORTS[cohort]["role"], "n_rows": cache.n_rows, "n_columns": len(cache.names),
            "matrix_bytes": int(cache.matrix.nbytes), "seconds": round(time.time() - started, 1),
            "input_sha256": {str(p.relative_to(ROOT)): _sha256_file(p) for p in _input_files(cohort)},
            "cache_input_identity_sha256": cache.input_sha256, "npz": str(npz.relative_to(ROOT)),
            "builder_version": cache.builder_version, "peak_rss_mb": round(_peak_rss_mb()),
            "n_subjects": int(len(set(cache.meta["subject_id"]))),
        }
        (CACHE_DIR / f"features_{cohort}.build_record.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[features {cohort}] rows={cache.n_rows} cols={len(cache.names)} {record['seconds']}s "
              f"peak_rss_mb={record['peak_rss_mb']}", flush=True)
        del cache


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("sofa"); p.add_argument("cohort", choices=sorted(COHORTS))
    p = sub.add_parser("features"); p.add_argument("cohorts", nargs="*", default=list(COHORTS)); p.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "sofa":
        stage_sofa(args.cohort)
    else:
        stage_features(args.cohorts, args.n_jobs)


if __name__ == "__main__":
    main()
