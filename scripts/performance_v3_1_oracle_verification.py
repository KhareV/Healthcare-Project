#!/usr/bin/env python3
"""The single bounded verification (addendum): Oracle B refit on a larger privileged sample."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import performance_v3_1_oracle_diagnostic as M
from performance_v3 import data as v3data, paired_bootstrap as pb, task_metrics as tm
from performance_v3_1 import diagnostic as D

t0 = time.time()
stored = np.load(ROOT / "artifacts/performance_v3_1/cache/diag_eval_predictions.npz")
ev_ctx = D.CohortContext(ROOT, "DIAG_EVAL")
ev_rows = M.eligible(M.rows_of("DIAG_EVAL"))
ev_ent = D.snapshot_table(ev_ctx, ev_rows)
assert len(ev_rows) == len(stored["y"]) and all(r["stay_id"] == s for r, s in zip(ev_rows, stored["stay"]))
Xe = M.matrix(ev_ent, ev_rows, D.B_NAMES)
Xs, ys = [], []
sources = [("DIAG_FIT", None, None)] + [(f"D{i}", ROOT / f"artifacts/performance_v3/cohorts/D{i}", ROOT / f"configs/performance_v3/cohorts/d{i}_v1.json") for i in (1, 2, 3)]
n_sub = 0
for cid, cdir, cfg in sources:
    if cdir is None:
        ctx = D.CohortContext(ROOT, "DIAG_FIT"); rows = M.eligible(M.rows_of("DIAG_FIT"))
    else:
        ctx = D.CohortContext(ROOT, cid, cohort_dir=cdir, config_path=cfg)
        rows = M.eligible(list(v3data.iter_cohort_rows(ROOT, cdir, cohort_id=f"SYN-V3-{cid}", role="dev")))
    ent = D.snapshot_table(ctx, rows)
    Xs.append(M.matrix(ent, rows, D.B_NAMES)); ys.append(np.asarray([r["organ_support_label"] for r in rows], dtype=float)); n_sub += ctx.config.n_subjects
    print(cid, len(rows), round(time.time() - t0), "s", flush=True)
X = np.vstack(Xs); y = np.concatenate(ys)
y_ev = stored["y"]; st = list(stored["stay"]); pt = [r["prediction_time"] for r in ev_rows]
ref = tm.make_records(y_ev, stored["REFERENCE_V3"], st, pt)
out = {"status": "DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY", "fit_subjects": n_sub, "fit_rows": int(len(y)), "models": {}}
for kind in ("logistic", "xgboost"):
    p = M.fit_predict(kind, X, y, Xe)
    m = M.metrics(y_ev, p, st)
    b = pb.paired_stay_cluster_bootstrap(ref, tm.make_records(y_ev, p, st, pt), tm.record_metric_fn("organ_support", "auprc"), n=M.BOOT_N, seed=M.BOOT_SEED, ci_levels=(0.95,), metric_name="auprc", task="organ_support")
    out["models"][f"ORACLE_B_LARGE_{kind.upper()}"] = {"metrics": m, "diff_vs_reference": b["diff_point"], "ci95": list(b["ci"]["0.95"])}
    print(kind, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, "diff", round(b["diff_point"], 4), [round(x, 4) for x in b["ci"]["0.95"]], flush=True)
best = max(out["models"].items(), key=lambda kv: kv[1]["metrics"]["auprc"])
gap, lb = best[1]["diff_vs_reference"], best[1]["ci95"][0]
primary = json.loads((ROOT / "artifacts/performance_v3_1/diagnostic/oracle_comparison.json").read_text())["gate_a"]
c_pass = primary["c_pass"]; audits = primary["audits_pass"]
out["gate_a_after_verification"] = {"best": best[0], "gap": gap, "lb95": lb, "b_pass": bool(gap >= 0.03 and lb > 0.01), "c_pass": c_pass, "audits_pass": audits,
                                    "passes_gate_a": bool(gap >= 0.03 and lb > 0.01 and c_pass and audits)}
out["seconds"] = round(time.time() - t0, 1)
(ROOT / "artifacts/performance_v3_1/diagnostic/oracle_verification_v1.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=float) + "\n")
print(json.dumps(out["gate_a_after_verification"], indent=1, default=float))
