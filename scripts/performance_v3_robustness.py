#!/usr/bin/env python3
"""Development-only robustness / stratification study (protocol F10) on the LCE cohort."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from performance_v3 import robustness as R  # noqa: E402
from performance_v3 import search as S  # noqa: E402
from performance_v3 import task_metrics as tm  # noqa: E402
from performance_v3.data import CohortDir, load_cohort_rows  # noqa: E402
from performance_v3.serving import V2Comparator, V3Bundle  # noqa: E402

OUT = ROOT / "artifacts/performance_v3/phase2"
SEED = 20261013
TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")


def _sel(task, r):
    if task == "recovery24":
        return r.get("recovery24_eligible") and r.get("delta_sofa_24") is not None, r.get("delta_sofa_24")
    if task == "recovery48":
        return r.get("recovery48_eligible") and r.get("delta_sofa_48") is not None, r.get("delta_sofa_48")
    if task == "icu_stay_time":
        ok = r.get("icu_time_eligible") and r.get("icu_time_log1p") is not None
        return ok, (float(np.expm1(r["icu_time_log1p"])) if ok else None)
    ok = r.get("organ_support_eligible") and r.get("organ_support_label") is not None
    return ok, (float(r["organ_support_label"]) if ok else None)


def task_metrics(rows, preds, mask=None):
    out = {}
    for task in TASKS:
        idx, y = [], []
        for i, r in enumerate(rows):
            if mask is not None and not mask[i]:
                continue
            ok, v = _sel(task, r)
            if ok:
                idx.append(i); y.append(v)
        if not idx:
            out[task] = None
            continue
        stay = [rows[i]["stay_id"] for i in idx]
        pt = [rows[i]["prediction_time"] for i in idx]
        key = {"recovery24": "recovery24_delta", "recovery48": "recovery48_delta", "icu_stay_time": "icu_raw_log_prediction",
               "organ_support": "support_raw_probability"}[task]
        pred = np.asarray([preds[i][key] for i in idx])
        m = S.fold_metrics(task, np.asarray(y), pred, stay, pt)
        if task == "organ_support":
            cal = np.clip(np.asarray([preds[i]["support_calibrated_probability"] for i in idx]), 1e-12, 1 - 1e-12)
            w = tm.stay_weights(stay)
            m["calibrated_brier"] = float(np.sum(w * (cal - np.asarray(y)) ** 2) / np.sum(w))
        m = {k: float(v) for k, v in m.items()}
        m["n_rows"] = len(idx); m["n_stays"] = len(set(stay))
        out[task] = m
    return out


def primary(task, m):
    return None if m is None else m[S.PRIMARY_NAME[task]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stays", type=int, default=500)
    args = parser.parse_args()
    started = time.time()
    rows = list(load_cohort_rows(ROOT, CohortDir(ROOT / "artifacts/performance_v3/cohorts/LCE"), cohort_id="SYN-V3-LCE", role="lce"))
    rows = R.stay_subsample(rows, args.stays, SEED)
    cache = json.loads((ROOT / "artifacts/performance_v3/cache/sofa_LCE.json").read_text())
    keyed = lambda d: {tuple(k.split("||", 1)): v for k, v in d.items()}
    trends, comps = keyed(cache["trends"]), keyed(cache["components"])

    v3 = V3Bundle.load(ROOT, "artifacts/performance_v3/phase3/v3/bundle_manifest.json")
    ctrl = V3Bundle.load(ROOT, "artifacts/performance_v3/phase3/control/bundle_manifest.json")
    v2 = V2Comparator.load(ROOT)
    models = {
        "V3": lambda rs: v3.predict_rows(rs, sofa_trends=trends, sofa_components=comps),
        "CONTROL": lambda rs: ctrl.predict_rows(rs, sofa_trends=trends, sofa_components=comps),
        "V2_AS_SERVED": lambda rs: v2.predict_rows(rs, current_sofa_trends=trends),
    }
    report = {"artifact_version": "performance_v3_robustness_v1", "cohort": "LCE (development), stay-subsample",
              "seed": SEED, "n_rows": len(rows), "n_stays": len({r["stay_id"] for r in rows}),
              "limitation": "SOFA(t)/trend/components from the frozen scorer on the unperturbed record are not re-derived under perturbation",
              "scenarios": {}, "strata": {}}
    base_preds = {}
    for sc in R.scenario_list():
        prt = [R.apply_scenario(r, sc, SEED) for r in rows]
        entry = {"spec": sc, "models": {}}
        for name, fn in models.items():
            preds = fn(prt)
            if sc["kind"] == "none":
                base_preds[name] = preds
            m = task_metrics(rows, preds)
            entry["models"][name] = m
        report["scenarios"][sc["name"]] = entry
        print(sc["name"], f"{time.time() - started:.0f}s", flush=True)
    base = report["scenarios"]["unperturbed"]["models"]
    for name, entry in report["scenarios"].items():
        for model, m in entry["models"].items():
            entry.setdefault("delta_vs_unperturbed", {})[model] = {
                t: (None if m[t] is None else primary(t, m[t]) - primary(t, base[model][t])) for t in TASKS}
    for label, keyf in (("cardiac_subtype", lambda r: str(r["cardiac_condition_group"])),
                        ("cutoff_band", lambda r: R.cutoff_band(int(r["grid_index"])))):
        groups = sorted({keyf(r) for r in rows})
        report["strata"][label] = {
            g: {model: task_metrics(rows, base_preds[model], [keyf(r) == g for r in rows]) for model in models} for g in groups}
    report["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "robustness_v1.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [s["name"] for s in R.scenario_list()]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, task in zip(axes, TASKS):
            for model in models:
                ax.plot(range(len(names)), [primary(task, report["scenarios"][n]["models"][model][task]) for n in names], marker="o", label=model)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
            ax.set_title(f"{task} ({S.PRIMARY_NAME[task]})", fontsize=9)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "figures/robustness_primary_metrics.png", dpi=130)
    except Exception as error:  # plotting must never invalidate numbers
        print("plot skipped:", error)
    print("wrote robustness_v1.json", report["wall_seconds"], "s")


if __name__ == "__main__":
    main()
