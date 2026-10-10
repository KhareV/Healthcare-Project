#!/usr/bin/env python3
"""V3.1 Work Package B/C: causal-filter candidates (search) and ensemble/calibration/refit (final).

  search : 4 feature families x 8 configs, 5-fold grouped CV on v3 pool + V31_DEV, selection per frozen rule
  final  : ensemble weight / calibrator / threshold on V31_DEV only, refit, freeze, THEN score V31_HOLD once
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from performance_v3 import calibration as C
from performance_v3 import paired_bootstrap as pb
from performance_v3 import search as S
from performance_v3 import task_metrics as tm
from performance_v3.features import FeatureCache, load_feature_cache, task_mask, weights_stay_balanced
from performance_v3.final_models import fit_final, n_estimators_rule
from performance_v3.models import sample_family_configs
from performance_v3_1 import filters as F

OUT = ROOT / "artifacts/performance_v3_1/phase2"
CACHE = ROOT / "artifacts/performance_v3_1/cache"
FOLD_VERSION = "performance_v3_1_crossfit_v1"
FAMILIES = ("F0", "F1", "F2", "F3")
SEEDS = (11, 23, 47)
BOOT_N, BOOT_SEED = 2000, 20261102


def load_pool():
    parts = [load_feature_cache(ROOT / "artifacts/performance_v3/cache/dev_pool"), load_feature_cache(CACHE / "features_V31_DEV1"), load_feature_cache(CACHE / "features_V31_DEV2")]
    names = parts[0].names
    assert all(p.names == names for p in parts)
    meta = {k: np.concatenate([p.meta[k] for p in parts]) for k in parts[0].meta}
    return FeatureCache(names=names, matrix=np.concatenate([p.matrix for p in parts]), meta=meta)


def f0_names():
    return tuple(json.loads((ROOT / "artifacts/performance_v3/phase3/v3/models/organ_support/feature_names.json").read_text()))


def task_data(cache, params, family, folds):
    mask = task_mask(cache.meta, "organ_support")
    X0 = cache.matrix[mask]
    f0 = f0_names(); idx0 = [cache.names.index(n) for n in f0]
    parts, cols = [X0[:, idx0]], list(f0)
    if family != "F0":
        FM, fnames = F.build_filter_features(X0, cache.names, params)
        keep = [fnames.index(n) for n in F.family_columns(fnames, family)]
        parts.append(FM[:, keep]); cols += [fnames[i] for i in keep]
    X = np.ascontiguousarray(np.hstack(parts), dtype=np.float32)
    subj = np.asarray([str(s) for s in cache.meta["subject_id"][mask]])
    fold = np.asarray([folds[s] for s in subj], dtype=np.int32)
    y = np.asarray(cache.meta["organ_support_label"][mask], dtype=np.float64)
    w = weights_stay_balanced(cache.meta, "organ_support")[mask]
    return S.TaskData(task="organ_support", variant=family, X=X, names=tuple(cols), y=y, y_eval=y, w=w,
                      stay_id=np.asarray([str(s) for s in cache.meta["stay_id"][mask]]), subject_id=subj,
                      prediction_time=np.asarray([str(s) for s in cache.meta["prediction_time"][mask]]), fold=fold, n_folds=5), mask


def configs():
    sel = json.loads((ROOT / "artifacts/performance_v3/phase2/candidate_selection_v1.json").read_text())["tasks"]["organ_support"]["winner"]["config"]
    first = dict(sel); first["config_hash"] = "v3frozen-" + str(sel.get("config_hash", ""))[:8]
    return [first] + sample_family_configs("xgboost", 7, 20261201, "organ_support")


def stage_search(args):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = load_pool()
    subjects = sorted(set(str(s) for s in cache.meta["subject_id"]))
    folds = S.assign_folds(subjects, n_folds=5, version=FOLD_VERSION)
    mask = task_mask(cache.meta, "organ_support")
    params = F.fit_kalman_params(cache.matrix[mask], cache.names)
    (OUT / "kalman_params_v1.json").write_text(json.dumps(params, indent=1, sort_keys=True))
    ledger = S.TrialLedger(OUT / "trial_ledger_v1.jsonl")
    t0 = time.time()
    for fam in FAMILIES:
        td, _ = task_data(cache, params, fam, folds)
        print(fam, "rows", td.n_rows, "stays", td.n_stays, "features", td.X.shape[1], flush=True)
        for cfg in configs():
            if (time.time() - t0) > args.budget_hours * 3600:
                print("budget exhausted"); break
            rec = S.run_trial_logged(ledger, td, "V31", "xgboost", cfg, folds=(0, 1, 2, 3, 4), seed=11, nthread=8)
            print(fam, cfg["config_hash"], rec["status"], round(rec.get("primary_mean", float("nan")), 4), f"{rec['wall_seconds']}s", flush=True)
    recs = [r for r in ledger.records() if r["status"] == "ok"]
    best = max(recs, key=lambda r: r["primary_mean"])
    tol = [r for r in recs if best["primary_mean"] - r["primary_mean"] <= 0.003]
    winner = min(tol, key=lambda r: (r["n_features"], r["primary_sd"], -r["primary_mean"]))
    base = max([r for r in recs if r["variant"] == "F0"], key=lambda r: r["primary_mean"])
    sel = {"rule": "highest OOF stay-balanced AUPRC; tie 0.003 -> fewer features -> lower fold sd", "best_raw": {k: best[k] for k in ("trial_id", "variant", "primary_mean", "primary_sd")},
           "winner": {k: winner[k] for k in ("trial_id", "variant", "config", "config_hash", "primary_mean", "primary_sd", "per_fold_primary", "best_iterations", "n_features")},
           "best_F0_baseline": {k: base[k] for k in ("trial_id", "primary_mean", "primary_sd")}, "tie_group": len(tol), "n_trials": len(recs),
           "by_family_best": {f: max([r["primary_mean"] for r in recs if r["variant"] == f]) for f in FAMILIES}}
    (OUT / "candidate_selection_v1.json").write_text(json.dumps(sel, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps(sel, indent=1, default=float)[:2500])


def metrics(y, p, stays):
    w = tm.stay_weights(stays); p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    cal = C.calibration_metrics(p, np.asarray(y, float), w)
    return {"auprc": tm.fast_weighted_auprc(y, p, w), "auroc": tm.fast_weighted_auroc(y, p, w), "brier": cal["brier"], "log_loss": cal["log_loss"],
            "ece": cal["ece_10_equal_mass"], "slope": cal["calibration_slope"], "n_rows": len(y), "n_stays": len(set(stays))}


def load_v3_scores(cid):
    z = np.load(CACHE / f"v3_scores_{cid}.npz")
    return {(s, t): (c, r) for s, t, c, r in zip(z["stay"], z["pt"], z["cal"], z["raw"])}


def stage_final(args):
    sel = json.loads((OUT / "candidate_selection_v1.json").read_text())
    win = sel["winner"]
    cache = load_pool(); subjects = sorted(set(str(s) for s in cache.meta["subject_id"]))
    folds = S.assign_folds(subjects, n_folds=5, version=FOLD_VERSION)
    mask = task_mask(cache.meta, "organ_support")
    params = F.fit_kalman_params(cache.matrix[mask], cache.names)
    td, mk = task_data(cache, params, win["variant"], folds)
    cfg = dict(win["config"])
    # OOF scores of the selected candidate (seed 11), for ensemble weight / calibrator on V31_DEV rows only
    r = S.run_cv_trial(td, "xgboost", cfg, folds=(0, 1, 2, 3, 4), seed=11, nthread=8, keep_oof=True)
    oof = r.pop("oof")
    cid = np.asarray([str(c) for c in cache.meta["cohort_id"][mk]])
    dev_rows = np.asarray([c.startswith("SYN-V31-D") for c in cid])
    scores = {}
    for c in ("V31_DEV1", "V31_DEV2"):
        scores.update(load_v3_scores(c))
    v3_cal = np.asarray([scores[(s, t)][0] for s, t in zip(td.stay_id[dev_rows], td.prediction_time[dev_rows])])
    y, w, sub, fo = td.y[dev_rows], td.w[dev_rows], td.subject_id[dev_rows], td.fold[dev_rows]
    oof_d = oof[dev_rows]
    nested = C.nested_crossfit_calibration(oof_d, y, w, sub, fo, ("none", "platt", "isotonic", "beta"))
    nested.pop("_audit")
    method = C.select_calibrator(nested, tie_tolerance=0.002, order=("none", "platt", "beta", "isotonic"), metric="log_loss")
    frozen_cal = C.fit_calibrator(method, oof_d, y, w)
    new_cal = C.apply_calibrator(frozen_cal, oof_d)
    grid = {}
    for wt in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = wt * new_cal + (1 - wt) * v3_cal
        grid[str(wt)] = {"log_loss": tm.weighted_log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), w), "auprc": tm.fast_weighted_auprc(y, p, w), "auroc": tm.fast_weighted_auroc(y, p, w)}
    w_star = float(min(grid, key=lambda k: grid[k]["log_loss"]))
    ens_dev = w_star * new_cal + (1 - w_star) * v3_cal
    thr = C.select_threshold_weighted_f1(ens_dev, y, w)
    n_est = n_estimators_rule(win["best_iterations"])
    # refit on all training rows (v3 pool + V31_DEV) with 3 seeds
    models = []
    out_dir = ROOT / "artifacts/performance_v3_1/phase3/candidate"
    for sd in SEEDS:
        m = fit_final("xgboost", dict(cfg, use_early_stopping=False), td.X, td.y, td.w, task="organ_support", feature_columns=td.names, n_estimators=n_est, nthread=8, seed=sd)
        meta = m.save(out_dir / f"seed{sd}")
        models.append((m, meta["model_sha256"]))
    freeze = {"artifact_version": "performance_v3_1_candidate_freeze_v1", "status": "FROZEN_BEFORE_HOLD_SCORING",
              "selected": {"variant": win["variant"], "config_hash": win["config_hash"], "n_estimators": n_est, "n_features": len(td.names), "seeds": list(SEEDS),
                           "seed_model_sha256": [h for _, h in models], "feature_names_sha256": hashlib.sha256("|".join(td.names).encode()).hexdigest()},
              "kalman_params_ref": "artifacts/performance_v3_1/phase2/kalman_params_v1.json", "kalman_params_sha256": hashlib.sha256((OUT / "kalman_params_v1.json").read_bytes()).hexdigest(),
              "calibrator": {"method": method, "params": frozen_cal, "nested_log_loss": {k: nested[k]["metrics"]["log_loss"] for k in nested}},
              "ensemble": {"weight_new": w_star, "grid_on_v31_dev": grid, "rows": int(dev_rows.sum()), "stays": int(len(set(td.stay_id[dev_rows])))},
              "threshold": thr["threshold"], "threshold_f1_dev": thr["f1"],
              "frozen_v3_reference": "artifacts/performance_v3/phase3/selected_models_v3.json", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_freeze_v1.json").write_text(json.dumps(freeze, indent=2, sort_keys=True, default=float) + "\n")
    print("FROZEN candidate:", win["variant"], "method", method, "w_new", w_star, "threshold", round(thr["threshold"], 4), flush=True)

    # ---- held-out development decision set (single comparison) ----
    hold = load_feature_cache(CACHE / "features_V31_HOLD")
    hm = task_mask(hold.meta, "organ_support")
    X0 = hold.matrix[hm]
    f0 = f0_names(); parts = [X0[:, [hold.names.index(n) for n in f0]]]
    if win["variant"] != "F0":
        FM, fn = F.build_filter_features(X0, hold.names, params)
        keep = [fn.index(n) for n in td.names[len(f0):]]
        parts.append(FM[:, keep])
    XH = np.ascontiguousarray(np.hstack(parts), dtype=np.float32)
    assert XH.shape[1] == len(td.names)
    yH = np.asarray(hold.meta["organ_support_label"][hm], float)
    stH = [str(s) for s in hold.meta["stay_id"][hm]]; ptH = [str(s) for s in hold.meta["prediction_time"][hm]]
    sub_h = np.asarray([str(s) for s in hold.meta["subject_id"][hm]])
    v3h = load_v3_scores("V31_HOLD")
    p_v3 = np.asarray([v3h[(s, t)][0] for s, t in zip(stH, ptH)])
    per_seed = [m.predict(XH, td.names) for m, _ in models]
    raw_new = np.mean(per_seed, axis=0)
    cal_new = C.apply_calibrator(frozen_cal, raw_new)
    p_cand = w_star * cal_new + (1 - w_star) * p_v3
    res = {"V3_FROZEN": metrics(yH, p_v3, stH), "CANDIDATE": metrics(yH, p_cand, stH), "NEW_MODEL_ONLY_CALIBRATED": metrics(yH, cal_new, stH)}
    fn_ = tm.record_metric_fn("organ_support", "auprc")
    ref = tm.make_records(yH, p_v3, stH, ptH)
    b = pb.paired_stay_cluster_bootstrap(ref, tm.make_records(yH, p_cand, stH, ptH), fn_, n=BOOT_N, seed=BOOT_SEED, ci_levels=(0.95,), metric_name="auprc", task="organ_support")
    seed_gaps = []
    for ps in per_seed:
        pc = w_star * C.apply_calibrator(frozen_cal, ps) + (1 - w_star) * p_v3
        seed_gaps.append(tm.fast_weighted_auprc(yH, pc, tm.stay_weights(stH)) - res["V3_FROZEN"]["auprc"])
    ranks = np.asarray([int(hashlib.sha256(("thirds|" + s).encode()).hexdigest(), 16) % 3 for s in sub_h])
    thirds = []
    for k in range(3):
        m_ = ranks == k
        st_k = [s for s, keep in zip(stH, m_) if keep]
        w_k = tm.stay_weights(st_k)
        thirds.append(float(tm.fast_weighted_auprc(yH[m_], p_cand[m_], w_k) - tm.fast_weighted_auprc(yH[m_], p_v3[m_], w_k)))
    d = {"diff_auprc": b["diff_point"], "ci95": list(b["ci"]["0.95"]), "diff_auroc": res["CANDIDATE"]["auroc"] - res["V3_FROZEN"]["auroc"],
         "diff_brier": res["CANDIDATE"]["brier"] - res["V3_FROZEN"]["brier"], "diff_log_loss": res["CANDIDATE"]["log_loss"] - res["V3_FROZEN"]["log_loss"],
         "per_seed_auprc_gaps": seed_gaps, "thirds_auprc_gaps": thirds}
    gate = {"auprc_gap_ge_0.015": bool(d["diff_auprc"] >= 0.015), "lb95_gt_0": bool(d["ci95"][0] > 0.0),
            "brier_ok": bool(d["diff_brier"] <= 0.003), "log_loss_ok": bool(d["diff_log_loss"] <= 0.01),
            "stable_thirds_ge_0.005": bool(all(t >= 0.005 for t in thirds))}
    gate["passes_gate_c"] = bool(all(gate.values()))
    report = {"artifact_version": "performance_v3_1_hold_evaluation_v1", "cohort": "V31_HOLD (development decision set)", "metrics": res, "paired_vs_v3": d, "gate_c": gate}
    (ROOT / "artifacts/performance_v3_1/phase2/hold_evaluation_v1.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=("search", "final")); ap.add_argument("--budget-hours", type=float, default=6.0)
    a = ap.parse_args()
    {"search": stage_search, "final": stage_final}[a.stage](a)
