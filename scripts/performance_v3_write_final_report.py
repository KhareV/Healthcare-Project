#!/usr/bin/env python3
"""Render docs/performance_v3/FINAL_V3_EVALUATION.md strictly from frozen result JSON (no hand-typed metrics)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "artifacts/performance_v3/phase4"
fm = json.loads((P4 / "final_metrics_v3.json").read_text())
pv = json.loads((P4 / "paired_v2_v3_metrics.json").read_text())
nv = json.loads((P4 / "naive_baselines_v1.json").read_text())
fz = json.loads((ROOT / "artifacts/performance_v3/governance/v3_final_evaluation_freeze_v1.json").read_text())
M = fm["models"]; D = fm["denominators"]; PD = pv["point_differences"]; PR = pv["promotion"]
order = ("V3", "V2_AS_SERVED", "V2_AS_EVALUATED", "CONTROL")
def ci(model, task):
    c = M[model][task].get("primary_ci", {}).get("0.95")
    return f"[{c[0]:.4f}, {c[1]:.4f}]" if c else ""
rows = []
prim = {"recovery24": ("mae", "Recovery24 MAE"), "recovery48": ("mae", "Recovery48 MAE"), "icu_stay_time": ("weighted_median_ae_hours", "ICU weighted median AE (h)")}
lines = ["# Performance V3 — Final Evaluation (SYN-V3-T, sealed cohort, one-time access)", "",
 "All values are rendered from `artifacts/performance_v3/phase4/*.json` by `scripts/performance_v3_write_final_report.py`.",
 f"Predictions SHA-256 `{fm['predictions_sha256']}`; evaluation freeze `artifacts/performance_v3/governance/v3_final_evaluation_freeze_v1.json`.", "",
 "## Denominators", "", "| Task | eligible rows |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in D.items()]
lines += ["", "## Primary metrics (stay-balanced, 95% stay-cluster bootstrap CI, 2,000 replicates)", "",
 "| Task | Naive | V3 | V2 as served (primary comparator) | V2 as evaluated (historical harness) | V2-recipe refit (control) |", "|---|---:|---|---|---|---|"]
for task, (metric, label) in prim.items():
    naive = nv["tasks"][task][metric if task != "icu_stay_time" else "weighted_median_ae_hours"]
    cells = [f"{M[m][task][metric]:.4f} {ci(m, task)}" for m in order]
    lines.append(f"| {label} | {naive:.4f} | " + " | ".join(cells) + " |")
cal = {m: M[m]["organ_support"] for m in order}
lines.append("| Support calibrated AUPRC | %.4f | " % nv["tasks"]["organ_support"]["auprc"] + " | ".join(f"{cal[m]['auprc']:.4f} {ci(m,'organ_support')}" for m in order) + " |")
lines += ["", "## Support detail", "", "| Model | cal AUPRC | cal AUROC | Brier | F1 @ frozen threshold | raw AUPRC | raw AUROC |", "|---|---:|---:|---:|---:|---:|---:|"]
for m in order:
    c, r = M[m]["organ_support"], M[m]["organ_support_raw_scores"]
    lines.append(f"| {m} | {c['auprc']:.4f} | {c['auroc']:.4f} | {c['brier']:.4f} | {c['f1']:.4f} | {r['auprc']:.4f} | {r['auroc']:.4f} |")
lines += ["", "## Paired differences, V3 minus V2 as served (primary)", "", "| Task | metric | diff | 95% CI | 98.75% CI | decision |", "|---|---|---:|---|---|---|"]
for task, key in (("recovery24", "recovery24|mae"), ("recovery48", "recovery48|mae"), ("icu_stay_time", "icu_stay_time|weighted_median_ae_hours"), ("organ_support", "organ_support|auprc")):
    d = PD[key + "|V3_vs_V2_AS_SERVED"]
    lines.append(f"| {task} | {key.split('|')[1]} | {d['diff_point']:+.4f} | [{d['ci']['0.95'][0]:+.4f}, {d['ci']['0.95'][1]:+.4f}] | [{d['ci']['0.9875'][0]:+.4f}, {d['ci']['0.9875'][1]:+.4f}] | {PR[task]['decision']} |")
lines += ["", "## Paired differences vs the other comparators", "", "| Comparison | diff | 98.75% CI |", "|---|---:|---|"]
for key, d in sorted(PD.items()):
    if any(s in key for s in ("|mae|", "weighted_median_ae_hours|", "organ_support|auprc|")) and not key.endswith("V2_AS_SERVED") and "raw" not in key:
        lines.append(f"| {key} | {d['diff_point']:+.4f} | [{d['ci']['0.9875'][0]:+.4f}, {d['ci']['0.9875'][1]:+.4f}] |")
lines += ["", "## Reading these results", "",
 "* **Defect D1.** The frozen V2 models learned to use the *missingness* of `current_sofa` (null exactly when a stay ends within 24 h of the cutoff). The V2 offline harness preserved that pattern, which inflates its ICU-time and support numbers (`V2 as evaluated`); the V2 serving runtime always supplies SOFA(t), so the deployed behaviour is `V2 as served`. See `configs/performance_v3/protocol_amendment_v1_1.yaml` and `docs/performance_v3/DEFECT_D1.md`.",
 "* V3 is *not* better than `V2 as evaluated` on ICU time or support; that comparator contains the leak and is reported only for continuity and attribution.",
 "* The `clean subset` (>=24 h follow-up) is selected on the outcome for ICU time and support, so it is not leak-free for those tasks; it is reported in the JSON for recovery tasks only as a like-for-like check.",
 "* V3 ICU time equals the V2-recipe control by the pre-registered replacement rule (no candidate beat the repaired V2 recipe beyond tolerance).",
 "* Promotion decisions follow the frozen plan: " + "; ".join(f"{t}: {PR[t]['decision']}" for t in PR) + ".",
 "* Absolute gains over the naive floor are modest, as predicted by the generator's irreducible noise; negative or null findings would have been reported identically.", ""]
(ROOT / "docs/performance_v3/FINAL_V3_EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:60]))
