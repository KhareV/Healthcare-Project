#!/usr/bin/env python3
"""Development-only TreeSHAP evidence for the V3 bundle (and V2-control routing check).

Reads only the LCE (development) feature cache. Attributions explain the RAW model margin
(pre-calibration, pre-expm1); they are NOT a decomposition of the calibrated support probability.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from performance_v3.features import forbidden_name_tokens, load_feature_cache  # noqa: E402
from performance_v3.serving import OUTPUT_DOMAIN, TASKS, V3Bundle, explain_route, tree_shap_values  # noqa: E402

OUT = ROOT / "artifacts/performance_v3/phase2"
WORDING = "contributed to the prediction"
SUBTYPES = ("SYNTHETIC_HF", "SYNTHETIC_ISCHEMIC", "SYNTHETIC_ARRHYTHMIA")


def readable(name: str) -> str:
    m = re.match(r"A__bin(\d)__(.+?)__(value|mask|tslo)$", name)
    if m:
        return f"{m.group(2)} [{m.group(3)}] @ bin {m.group(1)} (0=oldest, 7=newest 6h bin)"
    return name


def sample_rows(n_rows: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=min(n, n_rows), replace=False))


def select_columns(cache, names):
    index = cache.column_index()
    return np.asarray([index[n] for n in names])


def run(bundle_path: Path, cache_prefix: Path, n: int = 1500, seed: int = 11, make_figures: bool = True):
    bundle = V3Bundle.load(ROOT, bundle_path)
    cache = load_feature_cache(cache_prefix)
    idx = sample_rows(cache.n_rows, n, seed)
    forbidden = forbidden_name_tokens()
    report = {"artifact_version": "performance_v3_explainability_v1", "stage": "development_only_LCE",
              "output_domain": OUTPUT_DOMAIN, "interpretation_wording": WORDING,
              "note": "TreeSHAP of the raw margin before calibration/postprocess; not a decomposition of the calibrated probability; non-causal.",
              "sample": {"rows": int(len(idx)), "seed": seed, "cohort": "LCE"}, "tasks": {}}
    max_err = 0.0
    figures = []
    for task in TASKS:
        model = bundle.models[task]
        cols = select_columns(cache, model.feature_columns)
        X = cache.matrix[idx][:, cols]
        values, base, diag = tree_shap_values(model, X)
        margin_err = float(np.max(np.abs(values.sum(axis=1) + base - np.asarray(_margin(model, X)))))
        max_err = max(max_err, margin_err)
        bad = [c for c in model.feature_columns if any(t in c.lower() for t in forbidden)]
        mean_abs = np.abs(values).mean(axis=0)
        top = np.argsort(-mean_abs)[:20]
        entry = {"family": model.family, "route": explain_route(task, model.family), "n_features": len(model.feature_columns),
                 "additivity_max_abs_error": margin_err, "additivity_passed": bool(diag["additivity_check_passed"]),
                 "forbidden_token_feature_names": bad, "base_value": base,
                 "global_top20": [{"feature": model.feature_columns[i], "readable": readable(model.feature_columns[i]),
                                   "mean_abs_shap": float(mean_abs[i])} for i in top], "local": []}
        # local examples: one stay per subtype, first and last cutoff present in the sample-independent full cache
        group_cols = {s: cache.column_index().get(f"A__static__cardiac_condition_group=={s}") for s in SUBTYPES}
        for s in SUBTYPES:
            gcol = group_cols[s]
            if gcol is None:
                continue
            rows = np.where(cache.matrix[:, gcol] == 1.0)[0]
            if not len(rows):
                continue
            stay = sorted(set(cache.meta["stay_id"][rows].tolist()))[0]
            srows = rows[cache.meta["stay_id"][rows] == stay]
            srows = srows[np.argsort(cache.meta["grid_index"][srows])]
            for label, r in (("first_cutoff", srows[0]), ("last_cutoff", srows[-1])):
                xr = cache.matrix[r:r + 1][:, cols]
                v, b, d = tree_shap_values(model, xr)
                order = np.argsort(-np.abs(v[0]))[:10]
                entry["local"].append({
                    "subtype": s, "stay_id": str(stay), "which": label, "grid_index": int(cache.meta["grid_index"][r]),
                    "raw_margin": float(b + v[0].sum()), "additivity_passed": bool(d["additivity_check_passed"]),
                    "top10": [{"feature": model.feature_columns[i], "readable": readable(model.feature_columns[i]),
                               "shap": float(v[0][i]), "direction": "raised" if v[0][i] > 0 else "lowered",
                               "statement": f"{model.feature_columns[i]} {WORDING} (raw-margin shift {v[0][i]:+.4f})"} for i in order]})
        report["tasks"][task] = entry
        if make_figures:
            figures.append(_figure(task, entry))
    report["additivity_max_abs_error_overall"] = max_err
    report["figures"] = [str(p.relative_to(ROOT)) for p in figures if p]
    return report


def _margin(model, X):
    from performance_v3.serving import _raw_margin
    return _raw_margin(model, X)


def _figure(task, entry):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    top = entry["global_top20"][:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([t["feature"][:46] for t in top], [t["mean_abs_shap"] for t in top], color="#2b7a9e")
    ax.set_xlabel("mean |SHAP| (raw margin; pre-calibration)")
    ax.set_title(f"{task}: {entry['family']} global contributions (LCE dev sample)")
    fig.tight_layout()
    path = out / f"shap_{task}_global.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/performance_v3/phase3/v3/bundle_manifest.json")
    parser.add_argument("--cache", default="artifacts/performance_v3/cache/features_LCE")
    parser.add_argument("--control", default="artifacts/performance_v3/phase3/control/bundle_manifest.json")
    parser.add_argument("--rows", type=int, default=1500)
    args = parser.parse_args()
    report = run(ROOT / args.bundle, ROOT / args.cache, n=args.rows)
    control = V3Bundle.load(ROOT, ROOT / args.control)
    report["v2_control_routes"] = {t: explain_route(t, m.family) for t, m in control.models.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explainability_v1.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for t, e in report["tasks"].items():
        print(t, e["family"], "additivity_err=%.2e" % e["additivity_max_abs_error"], [x["feature"] for x in e["global_top20"][:5]])
    print("max additivity error", report["additivity_max_abs_error_overall"], "control routes", report["v2_control_routes"])


if __name__ == "__main__":
    main()
