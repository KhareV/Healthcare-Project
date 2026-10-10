#!/usr/bin/env python3
"""V3.1 Work Package A: privileged latent-state diagnostic (DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY)."""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from performance_v3 import calibration as C3  # noqa: E402
from performance_v3 import data as v3data  # noqa: E402
from performance_v3 import features as v3feat  # noqa: E402
from performance_v3 import paired_bootstrap as pb  # noqa: E402
from performance_v3 import task_metrics as tm  # noqa: E402
from performance_v3.serving import V3Bundle  # noqa: E402
from performance_v3_1 import diagnostic as D  # noqa: E402

OUT = ROOT / "artifacts/performance_v3_1/diagnostic"
CACHE = ROOT / "artifacts/performance_v3_1/cache"
PROTO = ROOT / "configs/performance_v3_1/diagnostic_protocol_v1.yaml"
BOOT_N, BOOT_SEED, MC_M = 2000, 20261101, 200
XGB = dict(max_depth=4, learning_rate=0.05, n_estimators=300, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0)
_G = {}


def mc_worker(i):
    ctx, keys, entries = _G["ctx"], _G["keys"], _G["entries"]
    key = keys[i]
    p, pos, neg = D.oracle_c_probability(ctx, entries[key], _G['M'], (20261102, int(hashlib.sha256(f"{key[0]}|{key[1]}".encode()).hexdigest()[:8], 16)))
    return i, p, pos, neg


def rows_of(cid):
    return list(v3data.iter_cohort_rows(ROOT, ROOT / "artifacts/performance_v3_1/cohorts" / cid, cohort_id={"DIAG_FIT": "SYN-V31-DF", "DIAG_EVAL": "SYN-V31-DE"}[cid], role="dev"))


def sofa_inputs(cid, rows):
    base = f"artifacts/performance_v3_1/cohorts/{cid}"
    with v3feat.cohort_scoring_context(ROOT, raw_manifest_ref=f"{base}/raw/synthetic_dataset_manifest_v1.json",
                                       timeline_manifest_ref=f"{base}/timeline/synthetic_processed_manifest_v1.json") as context:
        return v3feat.compute_sofa_trends_and_components(rows, context)


def eligible(rows):
    return [r for r in rows if r.get("organ_support_eligible") and r.get("organ_support_label") is not None]


def metrics(y, p, stays):
    w = tm.stay_weights(stays)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    cal = C3.calibration_metrics(p, np.asarray(y, dtype=float), w)
    return {"auprc": tm.fast_weighted_auprc(y, p, w), "auroc": tm.fast_weighted_auroc(y, p, w),
            "brier": float(np.sum(w * (p - np.asarray(y)) ** 2) / np.sum(w)), "log_loss": tm.weighted_log_loss(y, p, w),
            "ece": float(cal["ece_10_equal_mass"]), "slope": float(cal["calibration_slope"]), "intercept": float(cal["calibration_intercept"]),
            "prevalence": float(np.sum(w * np.asarray(y)) / np.sum(w)), "n_rows": int(len(y)), "n_stays": int(len(set(stays)))}


def fit_predict(kind, Xtr, ytr, Xev):
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xev))[:, 1]
    import xgboost as xgb
    m = xgb.XGBClassifier(**XGB, random_state=11, n_jobs=8, tree_method="hist", verbosity=0)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xev)[:, 1]


def matrix(entries, rows, names):
    return np.asarray([[entries[(r["stay_id"], r["prediction_time"])]["features"][n] for n in names] for r in rows], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mc-m", type=int, default=MC_M); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit-eval-rows", type=int, default=0); a = ap.parse_args()
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    runtime = {"stages": {}}
    ctx = {c: D.CohortContext(ROOT, c) for c in ("DIAG_FIT", "DIAG_EVAL")}
    rows = {c: rows_of(c) for c in ctx}
    audits = {"replay_bitwise_identity": {c: D.verify_replay(ctx[c]) for c in ctx}}
    runtime["stages"]["replay_verify"] = round(time.time() - t0, 1)
    elig = {c: eligible(rows[c]) for c in ctx}
    if a.limit_eval_rows:
        elig["DIAG_EVAL"] = elig["DIAG_EVAL"][: a.limit_eval_rows]
    ent = {c: D.snapshot_table(ctx[c], elig[c]) for c in ctx}
    runtime["stages"]["snapshots"] = round(time.time() - t0, 1)

    # truncation invariance audit: replay truncated at hour h+1 must give the same snapshot features
    trunc_bad = 0; trunc_n = 0
    import math
    for key, e in list(ent["DIAG_EVAL"].items())[:60]:
        stay = key[0]; intime, _ = ctx["DIAG_EVAL"].intime_outtime(stay); h = e["h"]
        from performance_v3_1.replay import replay_stay
        _, _, snaps, _ = replay_stay(ctx["DIAG_EVAL"].config, ctx["DIAG_EVAL"].contract, e["cand"], intime, h + 1, snapshot_hours={h})
        f2 = D._state_features(snaps[h], e["cand"], h); trunc_n += 1
        if any(abs(f2[k] - e["features"][k]) > 0 for k in e["features"]):
            trunc_bad += 1
    audits["truncation_invariance"] = {"checked": trunc_n, "mismatched": trunc_bad, "pass": trunc_bad == 0}
    audits["forbidden_token_scan"] = {"flagged_A": D.forbidden_token_scan(D.A_NAMES), "flagged_B": D.forbidden_token_scan(D.B_NAMES)}

    y = {c: np.asarray([r["organ_support_label"] for r in elig[c]], dtype=float) for c in ctx}
    st = {c: [r["stay_id"] for r in elig[c]] for c in ctx}
    pt = {c: [r["prediction_time"] for r in elig[c]] for c in ctx}
    pred = {}

    # REFERENCE_V3 (apply-only) and observed-feature control
    trends, comps = {}, {}
    for c in ctx:
        tr, co = sofa_inputs(c, rows[c]); trends[c], comps[c] = tr, co
    bundle = V3Bundle.load(ROOT, "artifacts/performance_v3/phase3/selected_models_v3.json")
    out = bundle.predict_rows(elig["DIAG_EVAL"], sofa_trends=trends["DIAG_EVAL"], sofa_components=comps["DIAG_EVAL"])
    pred["REFERENCE_V3"] = np.asarray([o["support_calibrated_probability"] for o in out])
    pred["REFERENCE_V3_RAW"] = np.asarray([o["support_raw_probability"] for o in out])
    names = tuple(json.loads((ROOT / "artifacts/performance_v3/phase3/v3/models/organ_support/feature_names.json").read_text()))
    cols = {}
    for c in ctx:
        cache = v3feat.build_feature_cache(elig[c], sofa_trends=trends[c], sofa_components=comps[c], n_jobs=8,
                                           names=v3feat.load_feature_cache(ROOT / "artifacts/performance_v3/cache/dev_pool").names)
        idx = [cache.names.index(n) for n in names]
        cols[c] = cache.matrix[:, idx]
    pred["OBSERVED_REFIT_CONTROL_XGB"] = fit_predict("xgboost", cols["DIAG_FIT"], y["DIAG_FIT"], cols["DIAG_EVAL"])
    runtime["stages"]["reference_and_control"] = round(time.time() - t0, 1)

    # privileged learners
    for tag, nm in (("A", D.A_NAMES), ("B", D.B_NAMES)):
        Xf, Xe = matrix(ent["DIAG_FIT"], elig["DIAG_FIT"], nm), matrix(ent["DIAG_EVAL"], elig["DIAG_EVAL"], nm)
        for kind in ("logistic", "xgboost"):
            pred[f"ORACLE_{tag}_{kind.upper()}"] = fit_predict(kind, Xf, y["DIAG_FIT"], Xe)
    runtime["stages"]["privileged_learners"] = round(time.time() - t0, 1)

    # Oracle C Monte Carlo
    _G['M'] = a.mc_m
    keys = [(r["stay_id"], r["prediction_time"]) for r in elig["DIAG_EVAL"]]
    _G.update({"ctx": ctx["DIAG_EVAL"], "keys": keys, "entries": ent["DIAG_EVAL"]})
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(mc_worker, range(len(keys)), chunksize=16)
    pc = np.full(len(keys), np.nan); npos = np.zeros(len(keys)); nneg = np.zeros(len(keys))
    for i, p, pos, neg in res:
        pc[i], npos[i], nneg[i] = p, pos, neg
    ok = ~np.isnan(pc)
    runtime["stages"]["oracle_c_mc"] = round(time.time() - t0, 1)
    pred["ORACLE_C_MC"] = np.where(ok, pc, float(np.nanmean(pc)))
    mc_info = {"M": a.mc_m, "rows": len(keys), "rows_with_zero_eligible_continuations": int((~ok).sum()),
               "mean_eligible_continuations": float(np.mean(npos + nneg)), "mean_binomial_se": float(np.nanmean(np.sqrt(pc * (1 - pc) / np.maximum(npos + nneg, 1))[ok]))}

    ev_y, ev_s = y["DIAG_EVAL"], st["DIAG_EVAL"]
    results = {k: metrics(ev_y, v, ev_s) for k, v in pred.items()}
    for k, v in results.items():
        print(f"{k:30s} AUPRC {v['auprc']:.4f} AUROC {v['auroc']:.4f} Brier {v['brier']:.4f} LL {v['log_loss']:.4f}", flush=True)
    guard = {k: {"auroc": v["auroc"], "auprc": v["auprc"], "suspicious": bool(v["auroc"] > 0.98 or v["auprc"] > 0.95)} for k, v in results.items() if k.startswith("ORACLE")}
    audits["suspicious_score_guard"] = guard
    c_rel = results["ORACLE_C_MC"]
    c_valid = bool(0.85 <= c_rel["slope"] <= 1.15 and c_rel["ece"] <= 0.04) if not np.isnan(c_rel["slope"]) else False
    audits["mc_reliability"] = {"slope": c_rel["slope"], "ece": c_rel["ece"], "valid": c_valid}

    # paired bootstrap vs REFERENCE_V3
    fn = tm.record_metric_fn("organ_support", "auprc")
    fn_auroc = tm.record_metric_fn("organ_support", "auroc")
    ref = tm.make_records(ev_y, pred["REFERENCE_V3"], ev_s, pt["DIAG_EVAL"])
    boot = {}
    for k in pred:
        if k == "REFERENCE_V3":
            continue
        rec = tm.make_records(ev_y, pred[k], ev_s, pt["DIAG_EVAL"])
        b = pb.paired_stay_cluster_bootstrap(ref, rec, fn, n=BOOT_N, seed=BOOT_SEED, ci_levels=(0.95,), metric_name="auprc", task="organ_support")
        boot[k] = {"diff_point": b["diff_point"], "ci95": list(b["ci"]["0.95"]) if b["ci"]["0.95"] else None, "point_reference": b["point_a"], "point_model": b["point_b"],
                   "n_valid_joint": b["n_valid_joint"], "n_stays": b["n_stays"], "n_rows": b["n_rows"]}
    gate_cfg = {"b_threshold": 0.03, "lb_threshold": 0.01}
    def gap(name):
        d = boot[name]["diff_point"]
        lo = boot[name]["ci95"][0] if boot[name]["ci95"] else None
        return d, lo
    best_b = max(("ORACLE_B_XGBOOST", "ORACLE_B_LOGISTIC"), key=lambda n: results[n]["auprc"])
    gb, lb = gap(best_b)
    gc, lc = gap("ORACLE_C_MC")
    audits_pass = bool(audits["truncation_invariance"]["pass"] and all(v["bitwise_identical"] for v in audits["replay_bitwise_identity"].values())
                       and not audits["forbidden_token_scan"]["flagged_A"] and not audits["forbidden_token_scan"]["flagged_B"]
                       and not any(g["suspicious"] for g in guard.values()))
    gate = {"best_oracle_b": best_b, "b_gap": gb, "b_gap_lb95": lb, "c_valid": c_valid, "c_gap": gc, "c_gap_lb95": lc, "audits_pass": audits_pass,
            "b_pass": bool(lb is not None and gb >= 0.03 and lb > 0.01),
            "c_pass": (True if not c_valid else bool(lc is not None and gc >= 0.03 and lc > 0.01))}
    gate["passes_gate_a"] = bool(audits_pass and gate["b_pass"] and gate["c_pass"])

    cohort_manifest = {c: {"subjects": ctx[c].config.n_subjects, "namespace": ctx[c].config.subject_id_namespace, "seed": ctx[c].config.seed,
                           "rows": len(rows[c]), "eligible_support_rows": len(elig[c]), "eligible_stays": len(set(st[c])),
                           "prevalence_stay_balanced": float(np.sum(tm.stay_weights(st[c]) * y[c]) / np.sum(tm.stay_weights(st[c]))),
                           "status": "DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY"} for c in ctx}
    runtime.update({"total_seconds": round(time.time() - t0, 1)})
    (OUT / "cohort_manifest.json").write_text(json.dumps(cohort_manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "information_set_contract.json").write_text(json.dumps({"ORACLE_A_FIVE_STATE": list(D.A_NAMES), "ORACLE_B_AUGMENTED": list(D.B_NAMES),
        "OBSERVED_REFIT_CONTROL": list(names), "forbidden": ["realized outtime/duration noise", "labels", "eligibility", "future states/support/shocks", "RNG state"],
        "status": "DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY"}, indent=2) + "\n")
    (OUT / "oracle_metrics.json").write_text(json.dumps({"status": "DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY", "metrics": results, "oracle_c": mc_info}, indent=2, sort_keys=True, default=float) + "\n")
    (OUT / "oracle_bootstrap.json").write_text(json.dumps({"n": BOOT_N, "seed": BOOT_SEED, "reference": "REFERENCE_V3", "comparisons": boot}, indent=2, sort_keys=True, default=float) + "\n")
    (OUT / "oracle_comparison.json").write_text(json.dumps({"audits": audits, "gate_a": gate, "protocol_sha256": hashlib.sha256(PROTO.read_bytes()).hexdigest()}, indent=2, sort_keys=True, default=float) + "\n")
    (OUT / "oracle_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    np.savez_compressed(CACHE / "diag_eval_predictions.npz", y=ev_y, stay=np.asarray(ev_s), **{k: np.asarray(v) for k, v in pred.items()})
    print(json.dumps(gate, indent=1, default=float))


if __name__ == "__main__":
    main()
