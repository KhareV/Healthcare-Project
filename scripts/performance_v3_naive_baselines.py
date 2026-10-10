#!/usr/bin/env python3
"""Master Plan section 19 naive floors on the V3 test predictions (constants from the DEVELOPMENT pool only)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from evaluation.weighted_stats import weighted_median
from performance_v3 import task_metrics as tm
from performance_v3.features import load_feature_cache, task_mask, weights_stay_balanced

cache = load_feature_cache(ROOT / "artifacts/performance_v3/cache/dev_pool")
m = task_mask(cache.meta, "icu_time"); w = weights_stay_balanced(cache.meta, "icu_time")[m]
icu_const = float(weighted_median(tuple(cache.meta["icu_time_log1p"][m]), tuple(w)))
ms = task_mask(cache.meta, "organ_support"); ws = weights_stay_balanced(cache.meta, "organ_support")[ms]
prev = float(np.sum(ws * cache.meta["organ_support_label"][ms]) / np.sum(ws))
rows = [json.loads(l) for l in open(ROOT / "artifacts/performance_v3/phase4/predictions/test_predictions_v1.jsonl")]
out = {"artifact_version": "performance_v3_naive_baselines_v1", "constants_source": "development pool (8000 subjects) only",
       "icu_constant_log1p_weighted_median": icu_const, "support_prevalence_stay_balanced": prev, "tasks": {}}
for task, el, lab, h in (("recovery24", "recovery24_eligible", "delta_sofa_24", "24h"), ("recovery48", "recovery48_eligible", "delta_sofa_48", "48h")):
    sel = [r for r in rows if r["labels"][el] and r["labels"][lab] is not None]
    y = np.array([r["labels"][lab] for r in sel]); st = [r["stay_id"] for r in sel]
    out["tasks"][task] = {k: float(v) for k, v in tm.recovery_metrics(y, np.zeros(len(y)), st, horizon=h).items() if k in ("mae", "rmse", "median_ae")}
sel = [r for r in rows if r["labels"]["icu_time_eligible"] and r["labels"]["icu_time_log1p"] is not None]
y = np.expm1([r["labels"]["icu_time_log1p"] for r in sel]); st = [r["stay_id"] for r in sel]
out["tasks"]["icu_stay_time"] = {k: float(v) for k, v in tm.icu_metrics(y, np.full(len(y), icu_const), st).items() if k.startswith(("weighted", "mae", "rmse"))}
sel = [r for r in rows if r["labels"]["organ_support_eligible"] and r["labels"]["organ_support_label"] is not None]
y = np.array([r["labels"]["organ_support_label"] for r in sel]); st = [r["stay_id"] for r in sel]; wt = tm.stay_weights(st)
out["tasks"]["organ_support"] = {"auprc": tm.fast_weighted_auprc(y, np.full(len(y), prev), wt), "brier": float(np.sum(wt * (prev - y) ** 2) / np.sum(wt)),
                                  "prevalence_test_stay_balanced": float(np.sum(wt * y) / np.sum(wt))}
(ROOT / "artifacts/performance_v3/phase4/naive_baselines_v1.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out["tasks"], indent=1))
