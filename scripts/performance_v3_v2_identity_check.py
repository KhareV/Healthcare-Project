#!/usr/bin/env python3
"""Support-AUPRC discrepancy check (development data + outcome-free structure of the exposed V2 cohort).

(a) score the UNMODIFIED frozen V2 models on V1-dev, LCE, D1 (development data only)
(b) outcome-free structural comparison V2 fresh cohort vs LCE / D1-D3 / V1-dev
    (reads ONLY features/canonical_feature_inputs.jsonl + timeline/canonical_statics.jsonl of the V2 cohort;
     never pre_split labels, never phase4 predictions)
(c) manifest/config/code identity comparison of the two build chains
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

from performance_v3 import serving as sv  # noqa: E402
from performance_v3 import task_metrics as tm  # noqa: E402
from performance_v3.data import load_cohort_rows  # noqa: E402
from performance_v2.data_loading import load_dev_rows, load_jsonl  # noqa: E402

V2F = ROOT / "artifacts/performance_v2/phase3/fresh_test_cohort"
OUT = ROOT / "artifacts/performance_v3/data_characterization/v2_support_discrepancy_check_v1.json"


def main():
    comp = sv.V2Comparator.load(ROOT)
    cohorts = {
        "V1_DEV": list(load_dev_rows(ROOT, splits=("train", "validation"))),
        "LCE": list(load_cohort_rows(ROOT, "artifacts/performance_v3/cohorts/LCE", cohort_id="LCE", role="lce")),
        "D1": list(load_cohort_rows(ROOT, "artifacts/performance_v3/cohorts/D1", cohort_id="D1", role="dev")),
    }
    sample_out = comp.predict_rows(cohorts["LCE"][:3])
    print("prediction keys:", sorted(sample_out[0].keys()))
    report = {"artifact_version": "performance_v3_v2_support_discrepancy_check_v1", "scores": {}, "structure": {}, "identity": {}}
    for name, rows in cohorts.items():
        out = comp.predict_rows(rows)
        entry = {"n_rows": len(rows), "n_subjects": len({r["subject_id"] for r in rows})}
        for task, elig, tgt, okey in (("recovery24", "recovery24_eligible", "delta_sofa_24", "recovery24_delta"),
                                      ("recovery48", "recovery48_eligible", "delta_sofa_48", "recovery48_delta")):
            idx = [i for i, r in enumerate(rows) if r.get(elig) and r.get(tgt) is not None]
            pred = [out[i][okey] for i in idx]
            m = tm.recovery_metrics([rows[i][tgt] for i in idx], pred, [rows[i]["stay_id"] for i in idx], None,
                                    horizon="24h" if task == "recovery24" else "48h")
            entry[task] = {"mae": m["mae"], "n_rows": m["n_examples"], "n_stays": m["n_stays"]}
        idx = [i for i, r in enumerate(rows) if r.get("icu_time_eligible") and r.get("icu_time_log1p") is not None]
        m = tm.icu_metrics([float(np.expm1(rows[i]["icu_time_log1p"])) for i in idx],
                           [out[i]["icu_raw_log_prediction"] for i in idx],
                           [rows[i]["stay_id"] for i in idx])
        entry["icu_stay_time"] = {"weighted_median_ae_hours": m["weighted_median_ae_hours"], "n_rows": m["n_examples"]}
        idx = [i for i, r in enumerate(rows) if r.get("organ_support_eligible") and r.get("organ_support_label") is not None]
        y = np.asarray([rows[i]["organ_support_label"] for i in idx], dtype=float)
        sid = [rows[i]["stay_id"] for i in idx]
        w = tm.stay_weights(sid)
        raw = np.asarray([out[i]["support_raw_probability"] for i in idx])
        cal = np.asarray([out[i]["support_calibrated_probability"] for i in idx])
        thr = comp.threshold
        pred = (cal >= thr).astype(int)
        tp = float(np.sum(w[(y == 1) & (pred == 1)])); fp = float(np.sum(w[(y == 0) & (pred == 1)])); fn = float(np.sum(w[(y == 1) & (pred == 0)]))
        entry["organ_support"] = {
            "n_rows": len(idx), "n_stays": len(set(sid)), "prevalence_rows": float(y.mean()),
            "prevalence_stay_balanced": float(np.sum(w * y) / np.sum(w)),
            "raw_auprc": tm.fast_weighted_auprc(y, raw, w), "raw_auroc": tm.fast_weighted_auroc(y, raw, w),
            "raw_brier": float(np.sum(w * (raw - y) ** 2) / np.sum(w)),
            "cal_auprc": tm.fast_weighted_auprc(y, cal, w), "cal_auroc": tm.fast_weighted_auroc(y, cal, w),
            "cal_brier": float(np.sum(w * (cal - y) ** 2) / np.sum(w)),
            "f1_at_frozen_threshold": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "threshold": thr,
        }
        report["scores"][name] = entry
        print(name, json.dumps(entry["organ_support"], default=float)[:400], flush=True)

    # (b) outcome-free structure ------------------------------------------------
    def structure(feature_rows, statics):
        rows = list(feature_rows)
        names = rows[0]["temporal_feature_names"]
        obs = np.zeros(len(names)); denom = 0
        bin_obs = np.zeros(8); bin_den = np.zeros(8)
        support_on = Counter(); n = 0
        grid = Counter()
        for r in rows:
            m = np.asarray(r["observation_mask"], dtype=bool)  # [8, C]
            pad = np.asarray(r["padding_mask"], dtype=bool).reshape(-1)
            valid = ~pad
            obs += m[valid].sum(axis=0); denom += int(valid.sum())
            bin_obs[valid] += m[valid].mean(axis=1); bin_den[valid] += 1
            last = r["history_values"][-1]
            for ch in ("vasopressor_on", "invasive_ventilation_on"):
                if ch in names:
                    v = last[names.index(ch)]
                    support_on[ch] += int(v is not None and float(v) >= 0.5)
            grid[int(r["grid_index"])] += 1; n += 1
        st = list(statics)
        dur = np.asarray([(np.datetime64(s["outtime"].replace("Z", "")) - np.datetime64(s["intime"].replace("Z", ""))) / np.timedelta64(1, "h") for s in st], dtype=float)
        age = np.asarray([s["age_years"] for s in st], dtype=float)
        return {
            "n_rows": n, "n_stays": len(st), "rows_per_stay": n / len(st),
            "grid_index_share": {str(k): grid[k] / n for k in sorted(grid)},
            "subtype_share": {k: v / len(st) for k, v in Counter(s["cardiac_condition_group"] for s in st).items()},
            "sex_share": {k: v / len(st) for k, v in Counter(s["sex_category"] for s in st).items()},
            "age_mean": float(age.mean()), "age_sd": float(age.std()),
            "episode_hours_mean": float(dur.mean()), "episode_hours_median": float(np.median(dur)), "episode_hours_p10_p90": [float(np.percentile(dur, 10)), float(np.percentile(dur, 90))],
            "obs_fraction_by_channel": {c: float(o / denom) for c, o in zip(names, obs)},
            "obs_fraction_by_bin": [float(a / b) if b else None for a, b in zip(bin_obs, bin_den)],
            "support_on_at_t_fraction": {k: v / n for k, v in support_on.items()},
        }

    def feats(cdir):
        return load_jsonl(cdir / "features/canonical_feature_inputs.jsonl"), load_jsonl(cdir / "timeline/canonical_statics.jsonl")

    struct = {"V2_FRESH(exposed; outcome-free fields only)": structure(*feats(V2F))}
    for c in ("LCE", "D1", "D2", "D3"):
        struct[c] = structure(*feats(ROOT / "artifacts/performance_v3/cohorts" / c))
    v1 = ROOT / "artifacts/data/synthetic"
    v1_feat = next(iter(v1.glob("features/final/phase9_final_v1/canonical_feature_inputs.jsonl")), None)
    v1_stat = ROOT / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_statics.jsonl"
    if v1_feat is not None and v1_stat.exists():
        split = {r["subject_id"] for r in cohorts["V1_DEV"]}
        fr = (r for r in load_jsonl(v1_feat) if r["subject_id"] in split)
        st = (s for s in load_jsonl(v1_stat) if s["subject_id"] in split)
        struct["V1_DEV"] = structure(fr, st)
    report["structure"] = struct
    ref = struct["V2_FRESH(exposed; outcome-free fields only)"]
    comp_tbl = {}
    for c, s in struct.items():
        if c.startswith("V2_FRESH"):
            continue
        d = [abs(ref["obs_fraction_by_channel"][k] - s["obs_fraction_by_channel"][k]) for k in ref["obs_fraction_by_channel"]]
        comp_tbl[c] = {"max_abs_obs_fraction_diff": float(max(d)), "mean_abs_obs_fraction_diff": float(np.mean(d)),
                       "rows_per_stay_diff": ref["rows_per_stay"] - s["rows_per_stay"],
                       "episode_hours_mean_diff": ref["episode_hours_mean"] - s["episode_hours_mean"],
                       "age_mean_diff": ref["age_mean"] - s["age_mean"],
                       "support_on_at_t_diff": {k: ref["support_on_at_t_fraction"][k] - s["support_on_at_t_fraction"].get(k, 0) for k in ref["support_on_at_t_fraction"]}}
    report["structure_vs_v2_fresh"] = comp_tbl

    # (c) build-chain identity -----------------------------------------------
    def flat(x, p=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from flat(v, p + "/" + k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from flat(v, p + f"/{i}")
        else:
            yield p, x

    ident = {}
    lce = ROOT / "artifacts/performance_v3/cohorts/LCE"
    for f in ("raw/synthetic_dataset_manifest_v1.json", "cohort/synthetic_cohort_manifest_v2.json", "timeline/synthetic_processed_manifest_v1.json",
              "features/synthetic_feature_input_manifest_v1.json", "pre_split/phase9_pre_split_manifest_v1.json"):
        a = dict(flat(json.loads((V2F / f).read_text())))
        b = dict(flat(json.loads((lce / f).read_text())))
        diff = {}
        for k in sorted(set(a) | set(b)):
            if a.get(k) == b.get(k):
                continue
            if any(t in k for t in ("repository_relative_path", "/path", "record_count", "sha256", "semantic_sha", "_path", "/seed", "count", "row_", "requested", "output", "generation_command", "dependencies/", "/value", "primary_seed", "subject", "stay", "byte", "size")):
                continue
            diff[k] = {"v2_fresh": str(a.get(k))[:100], "lce": str(b.get(k))[:100]}
        ident[f] = diff
    # spec/config hashes referenced by both chains (identical => same spec)
    spec_hash_keys = {}
    for f in ("raw/synthetic_dataset_manifest_v1.json", "cohort/synthetic_cohort_manifest_v2.json", "timeline/synthetic_processed_manifest_v1.json",
              "features/synthetic_feature_input_manifest_v1.json", "pre_split/phase9_pre_split_manifest_v1.json"):
        a = dict(flat(json.loads((V2F / f).read_text())))
        b = dict(flat(json.loads((lce / f).read_text())))
        for k in a:
            if ("sha256" in k.lower() or "hash" in k.lower()) and not any(t in k for t in ("artifacts/", "output", "count", "seed", "source_manifest", "cohort_manifest", "processed_manifest", "feature_manifest", "dataset_manifest", "phase3_manifest")):
                spec_hash_keys[f + k] = {"same": a[k] == b.get(k), "v2": str(a[k])[:16], "lce": str(b.get(k))[:16]}
    report["identity"] = {"non_count_field_differences": ident, "hash_fields": spec_hash_keys,
                          "v2_fresh_code_commit": "393414f987161a8f2f4cb9728329b2b21021badd (from manifests)",
                          "lce_code_commit": "360185e120d758f468a5d78f9e8d374ec8b7168d"}

    # (d) eligibility-null leak diagnostics (development data only) ---------------
    import xgboost as xgb
    from performance_v2.targets import support_rows
    from performance_v2.v2_features import build_xgb_matrices

    diag = {}
    for name in ("V1_DEV", "LCE"):
        rows = support_rows(cohorts[name])
        out = comp.predict_rows(rows)
        p = np.asarray([o["support_raw_probability"] for o in out]); y = np.asarray([r["organ_support_label"] for r in rows], float)
        null = np.asarray([r.get("baseline_sofa") is None for r in rows])
        sid = np.asarray([r["stay_id"] for r in rows])
        nn = ~null
        diag[name] = {
            "support_eligible_rows": len(rows), "baseline_sofa_null_fraction": float(null.mean()),
            "positive_rate_when_null": float(y[null].mean()), "positive_rate_when_not_null": float(y[nn].mean()),
            "frozen_v2_mean_raw_p_null": float(p[null].mean()), "frozen_v2_mean_raw_p_not_null": float(p[nn].mean()),
            "frozen_v2_raw_auprc_all": tm.fast_weighted_auprc(y, p, tm.stay_weights(sid)),
            "frozen_v2_raw_auprc_not_null_only": tm.fast_weighted_auprc(y[nn], p[nn], tm.stay_weights(sid[nn])),
        }
    sel = json.loads((ROOT / "artifacts/performance_v2/phase3/selected_models_v2.json").read_text())["tasks"]["organ_support"]
    hp = {k: v for k, v in sel["hyperparameters"].items() if k != "feature_variant"}
    dev_s, lce_s = support_rows(cohorts["V1_DEV"]), support_rows(cohorts["LCE"])
    Xt, Xe, _, _ = build_xgb_matrices(dev_s, lce_s, "B_FULL")
    m = xgb.XGBClassifier(objective="binary:logistic", n_estimators=int(sel["final_training_budget"]), missing=np.nan, random_state=20260925, n_jobs=1, **hp)
    m.fit(Xt, np.asarray([r["organ_support_label"] for r in dev_s], float))
    pe = m.predict_proba(Xe)[:, 1]
    ye = np.asarray([r["organ_support_label"] for r in lce_s], float); we = tm.stay_weights([r["stay_id"] for r in lce_s])
    diag["v2_recipe_retrained_with_v2_matrix_builders_on_v1_dev__scored_on_LCE"] = {
        "raw_auprc": tm.fast_weighted_auprc(ye, pe, we), "raw_auroc": tm.fast_weighted_auroc(ye, pe, we)}
    report["eligibility_null_leak_diagnostics"] = diag
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
