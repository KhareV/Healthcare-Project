#!/usr/bin/env python3
"""Quantify the D1 leak on DEVELOPMENT data: frozen V2 models on LCE as-evaluated vs as-served.

V2_AS_EVALUATED: pre-split baseline_sofa with nulls preserved (V2 offline harness).
V2_AS_SERVED:    SOFA(t) from the frozen sofa_at for every row (V2 serving behaviour).
Clean subset:    rows with non-null baseline_sofa (>=24h follow-up); both modes coincide there.
Reads only LCE (development) rows; never an exposed or sealed cohort.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from performance_v3 import task_metrics as tm  # noqa: E402
from performance_v3.data import CohortDir, load_cohort_rows  # noqa: E402
from performance_v3.serving import V2Comparator  # noqa: E402

LCE = ROOT / "artifacts/performance_v3/cohorts/LCE"
OUT = ROOT / "artifacts/performance_v3/data_characterization/v2_modes_on_lce_v1.json"


def metrics(task, rows, preds, key_pred, mask=None):
    sel = []
    for i, r in enumerate(rows):
        if mask is not None and not mask[i]:
            continue
        if task == "recovery24" and r.get("recovery24_eligible") and r.get("delta_sofa_24") is not None:
            sel.append((i, r["delta_sofa_24"]))
        elif task == "recovery48" and r.get("recovery48_eligible") and r.get("delta_sofa_48") is not None:
            sel.append((i, r["delta_sofa_48"]))
        elif task == "icu_stay_time" and r.get("icu_time_eligible") and r.get("icu_time_log1p") is not None:
            sel.append((i, float(np.expm1(r["icu_time_log1p"]))))
        elif task == "organ_support" and r.get("organ_support_eligible") and r.get("organ_support_label") is not None:
            sel.append((i, float(r["organ_support_label"])))
    if not sel:
        return None
    idx = [i for i, _ in sel]
    y = np.asarray([v for _, v in sel])
    stay = [rows[i]["stay_id"] for i in idx]
    pt = [rows[i]["prediction_time"] for i in idx]
    if task in ("recovery24", "recovery48"):
        pred = np.asarray([preds[i][key_pred[task]] for i in idx])
        m = tm.recovery_metrics(y, pred, stay, pt, horizon="24h" if task == "recovery24" else "48h")
    elif task == "icu_stay_time":
        raw = np.asarray([preds[i]["icu_raw_log_prediction"] for i in idx])
        m = tm.icu_metrics(y, raw, stay, pt)
    else:
        w = tm.stay_weights(stay)
        raw = np.asarray([preds[i]["support_raw_probability"] for i in idx])
        cal = np.asarray([preds[i]["support_calibrated_probability"] for i in idx])
        m = {"raw_auprc": tm.fast_weighted_auprc(y, raw, w), "raw_auroc": tm.fast_weighted_auroc(y, raw, w),
             "cal_auprc": tm.fast_weighted_auprc(y, cal, w), "cal_auroc": tm.fast_weighted_auroc(y, cal, w),
             "n_positive_rows": int(y.sum())}
    m = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()}
    m["n_rows"] = len(idx); m["n_stays"] = len(set(stay))
    return m


def main() -> None:
    rows = list(load_cohort_rows(ROOT, CohortDir(LCE), cohort_id="SYN-V3-LCE", role="lce"))
    trends = json.loads((ROOT / "artifacts/performance_v3/cache/sofa_LCE.json").read_text())["trends"]
    trend_map = {tuple(k.split("||", 1)): v for k, v in trends.items()}
    v2 = V2Comparator.load(ROOT)
    evaluated = v2.predict_rows(rows)
    served = v2.predict_rows(rows, current_sofa_trends=trend_map)
    clean = np.asarray([r.get("baseline_sofa") is not None for r in rows])
    keys = {"recovery24": "recovery24_delta", "recovery48": "recovery48_delta"}
    report = {"artifact_version": "performance_v3_v2_modes_on_lce_v1", "cohort": "LCE (development)", "rows": len(rows),
              "fraction_rows_null_baseline_sofa": float(1 - clean.mean()), "modes": {}}
    for name, preds in (("V2_AS_EVALUATED", evaluated), ("V2_AS_SERVED", served)):
        report["modes"][name] = {
            "all_eligible_rows": {t: metrics(t, rows, preds, keys) for t in ("recovery24", "recovery48", "icu_stay_time", "organ_support")},
            "clean_subset_nonnull_baseline": {t: metrics(t, rows, preds, keys, clean) for t in ("icu_stay_time", "organ_support")},
        }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for mode, d in report["modes"].items():
        print(mode)
        for t, m in d["all_eligible_rows"].items():
            print("  ", t, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
