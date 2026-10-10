#!/usr/bin/env python3
"""Final development refit -> serving-compatible bundle manifest (development data only).

  --kind v3       the four selected V3 models (candidate_selection_v1.json + support_calibration_v1.json)
  --kind control  the V2 recipe (hyperparameters, variants, isotonic, F1-threshold policy) refit on the
                  8,000-subject pool with corrected features (secondary comparator)

Writes models + feature-name lists + a V3Bundle-schema manifest under artifacts/performance_v3/phase3/.
Never reads a sealed or exposed cohort: the feature cache is the development pool only.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from performance_v3 import calibration as C  # noqa: E402
from performance_v3 import search as S  # noqa: E402
from performance_v3.features import load_feature_cache  # noqa: E402
from performance_v3.final_models import fit_final, n_estimators_rule, names_sha256  # noqa: E402
from performance_v3.serving import params_sha256  # noqa: E402

PHASE2 = ROOT / "artifacts/performance_v3/phase2"
PHASE3 = ROOT / "artifacts/performance_v3/phase3"
ICU_POST = "expm1(clamp_min(raw,0))"
SEED = 11


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def v2_recipe(task):
    sys.path.insert(0, str(ROOT / "scripts"))
    import performance_v3_search as search_cli  # noqa: E402
    return search_cli.v2_recipe_config(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("v3", "control"), required=True)
    parser.add_argument("--cache", default="artifacts/performance_v3/cache/dev_pool")
    parser.add_argument("--nthread", type=int, default=8)
    args = parser.parse_args()

    cache = load_feature_cache(ROOT / args.cache)
    include_s = any(n.startswith("S__") for n in cache.names)
    out_dir = PHASE3 / args.kind
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_manifest = {}
    started = time.time()
    plan = {}
    if args.kind == "v3":
        selection = json.loads((PHASE2 / "candidate_selection_v1.json").read_text())
        for task in S.TASKS:
            win = selection["tasks"][task]["winner"]
            cfg = dict(win["config"])
            n_est = 0 if win["family"] in ("ridge", "logistic") else n_estimators_rule(win["best_iterations"])
            plan[task] = (win["family"], win["variant"], cfg, n_est)
    else:
        for task in S.TASKS:
            cfg, _ = v2_recipe(task)
            plan[task] = ("xgboost", "V3_CTRL", cfg, int(cfg["n_estimators"]))

    for task in S.TASKS:
        family, variant, cfg, n_est = plan[task]
        td = S.prepare_task_data(cache, task, variant, include_s=include_s)
        cfg = dict(cfg)
        cfg["use_early_stopping"] = False
        model = fit_final(family, cfg, td.X, td.y, td.w, task=task, feature_columns=td.names,
                          n_estimators=n_est if n_est else None, nthread=args.nthread, seed=SEED)
        task_dir = out_dir / "models" / task
        meta = model.save(task_dir)
        names_path = task_dir / "feature_names.json"
        names_path.write_text(json.dumps(list(td.names)), encoding="utf-8")
        meta_path = task_dir / "model_meta.json"
        tasks_manifest[task] = {
            "family": family, "feature_variant": variant, "n_estimators": int(model.n_estimators),
            "model_ref": str(meta_path.relative_to(ROOT)), "model_sha256": sha256_file(meta_path),
            "feature_names_ref": str(names_path.relative_to(ROOT)), "feature_names_sha256": sha256_file(names_path),
            "postprocess": ICU_POST if task == "icu_stay_time" else "none",
            "training_rows": td.n_rows, "training_stays": td.n_stays, "n_features": len(td.names),
            "config_hash": cfg.get("config_hash"), "names_sha256": names_sha256(td.names),
        }
        print(task, family, variant, "n_est", model.n_estimators, "rows", td.n_rows, flush=True)

    if args.kind == "v3":
        report = json.loads((PHASE2 / "support_calibration_v1.json").read_text())
        frozen = report["frozen_calibrator"]
        threshold = float(report["threshold"]["threshold"])
        method = report["chosen_calibrator"]
    else:
        family, variant, cfg, n_est = plan["organ_support"]
        td = S.prepare_task_data(cache, "organ_support", variant)
        oof = S.run_cv_trial(td, "xgboost", dict(cfg, use_early_stopping=False), folds=(0, 1, 2, 3, 4), seed=SEED, nthread=args.nthread, keep_oof=True)["oof"]
        frozen = C.fit_calibrator("isotonic", oof, td.y, td.w)
        cal_oof = C.nested_crossfit_calibration(oof, td.y, td.w, td.subject_id, td.fold, ("isotonic",))["isotonic"]["heldout_prob"]
        threshold = float(C.select_threshold_weighted_f1(cal_oof, td.y, td.w)["threshold"])
        method = "isotonic"
    manifest = {
        "artifact_version": "performance_v3_selected_models_v1",
        "bundle_id": "performance_v3_" + args.kind,
        "kind": args.kind,
        "status": "DEVELOPMENT_REFIT_PRE_FREEZE",
        "tasks": tasks_manifest,
        "calibrator": {"method": method, "params": frozen, "sha256": params_sha256(frozen)},
        "threshold": {"value": threshold, "comparator": ">="},
        "development_pool": {"cache_ref": args.cache, "cache_matrix_sha256": json.loads((ROOT / args.cache).with_suffix(".json").read_text())["matrix_sha256"],
                             "subjects": 8000},
        "code_commit": git_head(),
        "protocol_sha256": sha256_file(ROOT / "configs/performance_v3/scientific_protocol_v1.yaml"),
        "amendment_sha256": sha256_file(ROOT / "configs/performance_v3/protocol_amendment_v1_1.yaml"),
        "seed": SEED, "fit_wall_seconds": round(time.time() - started, 1),
    }
    path = out_dir / "bundle_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print("wrote", path.relative_to(ROOT), sha256_file(path))


if __name__ == "__main__":
    main()
