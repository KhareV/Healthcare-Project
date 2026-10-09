#!/usr/bin/env python3
"""Support calibration + threshold stage (development evidence only).

Takes the selected organ-support candidate (candidate_selection_v1.json), produces 5-fold
stay-grouped OOF raw scores on the development pool, compares none / platt / isotonic / beta
with nested cross-fitting, applies the pre-registered selection rule, fits the chosen
calibrator on all OOF scores and selects the stay-balanced F1 threshold on the cross-fitted
held-out calibrated probabilities. Writes development artifacts under phase2/.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from performance_v3 import calibration as C  # noqa: E402
from performance_v3 import search as S  # noqa: E402
from performance_v3 import task_metrics as tm  # noqa: E402
from performance_v3.features import load_feature_cache  # noqa: E402

OUT = ROOT / "artifacts/performance_v3/phase2"
METHODS = ("none", "platt", "isotonic", "beta")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="artifacts/performance_v3/cache/dev_pool")
    parser.add_argument("--selection", default=str(OUT / "candidate_selection_v1.json"))
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text())
    winner = selection["tasks"]["organ_support"]["winner"]
    cache = load_feature_cache(ROOT / args.cache)
    include_s = any(n.startswith("S__") for n in cache.names)
    td = S.prepare_task_data(cache, "organ_support", winner["variant"], include_s=include_s)
    result = S.run_cv_trial(td, winner["family"], winner["config"], folds=(0, 1, 2, 3, 4), seed=args.seed, nthread=args.nthread, keep_oof=True)
    oof = result.pop("oof")
    assert np.isfinite(oof).all()

    nested = C.nested_crossfit_calibration(oof, td.y, td.w, td.subject_id, td.fold, METHODS)
    audit = nested.pop("_audit")
    chosen = C.select_calibrator(nested, tie_tolerance=0.002, order=("none", "platt", "beta", "isotonic"), metric="log_loss")
    frozen = C.fit_calibrator(chosen, oof, td.y, td.w)
    held = nested[chosen]["heldout_prob"]
    threshold = C.select_threshold_weighted_f1(held, td.y, td.w)

    methods_report = {}
    for method in METHODS:
        p = nested[method]["heldout_prob"]
        methods_report[method] = {
            "metrics": nested[method]["metrics"],
            "ties_fraction": float(1.0 - len(np.unique(p)) / len(p)),
            "auprc_heldout": tm.fast_weighted_auprc(td.y, p, td.w),
            "auroc_heldout": tm.fast_weighted_auroc(td.y, p, td.w),
        }
    report = {
        "artifact_version": "performance_v3_support_calibration_v1",
        "stage": "development_only_no_test_access",
        "candidate": {k: winner[k] for k in ("family", "variant", "config")},
        "oof_raw": {
            "n_rows": int(td.n_rows), "n_stays": td.n_stays,
            "auprc": tm.fast_weighted_auprc(td.y, oof, td.w), "auroc": tm.fast_weighted_auroc(td.y, oof, td.w),
            "per_fold_auprc": result["per_fold_primary"], "best_iterations": result["best_iterations"],
            "oof_sha256": hashlib.sha256(np.ascontiguousarray(oof).tobytes()).hexdigest(),
        },
        "methods": methods_report,
        "chosen_calibrator": chosen,
        "selection_rule": "lowest nested cross-fitted stay-balanced log loss; tie tolerance 0.002; order none<platt<beta<isotonic",
        "frozen_calibrator": frozen,
        "threshold": threshold,
        "nested_audit": audit,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "support_calibration_v1.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    np.savez_compressed(ROOT / "artifacts/performance_v3/cache/support_oof_v1.npz", oof=oof, y=td.y, w=td.w, stay_id=td.stay_id, subject_id=td.subject_id, fold=td.fold)
    print(json.dumps({k: report[k] for k in ("chosen_calibrator", "threshold")}, indent=1, default=str))
    for m, v in methods_report.items():
        print(m, {k: round(v["metrics"][k], 4) for k in ("brier", "log_loss", "ece") if k in v["metrics"]}, "ties", round(v["ties_fraction"], 3), "AUPRC", round(v["auprc_heldout"], 4))


if __name__ == "__main__":
    main()
