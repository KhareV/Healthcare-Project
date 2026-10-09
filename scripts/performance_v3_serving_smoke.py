#!/usr/bin/env python3
"""Development smoke for the V3 serving sidecar (no sealed cohort, nothing written to artifacts/).

 1. V2Comparator identity: frozen V2 models reproduce stored V2 phase-4 predictions on N rows
    (read-only use of the consumed, exposed V2 cohort).
 2. V3Bundle on a throw-away tiny bundle trained on V1 DEV rows (tmp dir): predict_rows vs
    predict_stay_at through the real V2ServingRuntime for one demo stay (same canonical row => same output).
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-rows", type=int, default=200)
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()

    from performance_v3 import serving as sv

    t0 = time.time()
    comparator = sv.V2Comparator.load(ROOT)
    from performance_v2.fresh_test_access import load_fresh_test_rows

    fresh = load_fresh_test_rows(ROOT)[: args.identity_rows]
    out = comparator.predict_rows(fresh)
    stored = {}
    for line in (ROOT / "artifacts/performance_v2/phase4/predictions/organ_support_predictions_v1.jsonl").read_text().splitlines():
        r = json.loads(line)
        stored[(r["stay_id"], r["prediction_time"])] = r
    diffs = [abs(p["support_calibrated_probability"] - stored[(p["stay_id"], p["prediction_time"])]["calibrated_probability"])
             for p in out if (p["stay_id"], p["prediction_time"]) in stored]
    print(f"[identity] rows={len(out)} compared={len(diffs)} max|dp|={max(diffs):.3e} ({time.time()-t0:.1f}s)")

    if args.skip_runtime:
        return

    from performance_v2.data_loading import load_dev_rows
    from performance_v3 import final_models as fm
    from performance_v3.calibration import fit_calibrator
    from performance_v2.diagnostic_features import build_all_groups
    from serving.v2.runtime import V2ServingRuntime

    dev = list(load_dev_rows(ROOT, splits=("train",)))[:600]
    feats = [build_all_groups(r, baseline_sofa=r.get("baseline_sofa"), sofa_trend={}) for r in dev]
    names = tuple(sorted({n for f in feats for n in f if n.split("__", 1)[0] in ("A", "B", "F")}))
    index = {n: i for i, n in enumerate(names)}
    X = np.full((len(dev), len(names)), np.nan, dtype=np.float32)
    X[:, [("==" in n) for n in names]] = 0.0
    for r, f in enumerate(feats):
        for k, v in f.items():
            if k in index:
                X[r, index[k]] = v
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as tmp:
        tmp = Path(tmp)
        entries = {}
        for task, family in (("recovery24", "xgboost"), ("recovery48", "xgboost"), ("icu_stay_time", "ridge"), ("organ_support", "xgboost")):
            y = (rng.random(len(dev)) < 0.4).astype(float) if task == "organ_support" else rng.normal(size=len(dev))
            cfg = {"objective": "binary" if task == "organ_support" else "squarederror", "weight_mode": "none", "max_depth": 3, "learning_rate": 0.2, "alpha": 1.0}
            model = fm.fit_final(family, cfg, X, y, np.ones(len(dev)), task=task, feature_columns=names, n_estimators=10, nthread=2)
            d = tmp / task
            model.save(d)
            (d / "feature_names.json").write_text(json.dumps(list(names)))
            entries[task] = {"family": family, "feature_variant": "V3_BF", "model_ref": str((d / fm.META_NAME).relative_to(ROOT)),
                             "model_sha256": sv._sha256_file(d / fm.META_NAME), "feature_names_ref": str((d / "feature_names.json").relative_to(ROOT)),
                             "feature_names_sha256": sv._sha256_file(d / "feature_names.json"), "n_estimators": model.n_estimators,
                             "postprocess": sv.ICU_POSTPROCESS if task == "icu_stay_time" else "none"}
        raw = fm.load_fitted_model(tmp / "organ_support").predict(X)
        params = fit_calibrator("platt", raw, (rng.random(len(dev)) < 0.4).astype(float))
        manifest = {"artifact_version": "performance_v3_selected_models_v1", "bundle_id": "smoke", "tasks": entries,
                    "calibrator": {"method": "platt", "params": params, "sha256": sv.params_sha256(params)},
                    "threshold": {"value": 0.4, "comparator": ">="}}
        (tmp / "manifest.json").write_text(json.dumps(manifest))
        bundle = sv.V3Bundle.load(ROOT, tmp / "manifest.json")
        t1 = time.time()
        runtime = V2ServingRuntime(ROOT)
        print(f"[runtime] loaded in {time.time()-t1:.1f}s")
        demo = json.loads((ROOT / "configs/performance_v2/v2_demo_manifest_v1.json").read_text())
        from datetime import datetime

        subject = demo["demo_subjects"][0]
        stay_id = subject["stay_id"]
        cutoff = subject["legal_cutoffs"][1]
        via_runtime = bundle.predict_stay_at(runtime, stay_id, cutoff)
        parse = lambda s: datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        full = load_dev_rows(ROOT, splits=("train", "validation"))
        row = next((r for r in full if r["stay_id"] == stay_id and parse(r["prediction_time"]) == parse(cutoff)), None)
        print("[v3 via runtime]", {k: via_runtime[k] for k in ("recovery24_delta", "recovery48_delta", "icu_remaining_hours", "support_calibrated_probability")})
        if row is not None:
            direct = bundle.predict_rows([row])[0]
            print("[parity] max abs diff", max(abs(direct[k] - via_runtime[k]) for k in ("recovery24_delta", "recovery48_delta", "support_raw_probability")))
        else:
            print("[parity] matching dev row not found; direct comparison skipped")


if __name__ == "__main__":
    main()
