#!/usr/bin/env python3
"""Run the frozen-V2-recipe learning curve on the LCE cohort (Performance V3, WP1).

Requires the feature caches from scripts/performance_v3_build_feature_caches.py.
Outputs under artifacts/performance_v3/data_characterization/:
  learning_curve_metrics.json, learning_curve_bootstrap.json, learning_curve_table.csv,
  learning_curve_<task>.png
"""

import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.bootstrap import grouped_bootstrap  # noqa: E402
from performance_v3 import features as v3feat  # noqa: E402
from performance_v3 import learning_curve as lc  # noqa: E402
from performance_v3.models import postprocess_icu  # noqa: E402
from performance_v3.task_metrics import make_records, record_metric_fn  # noqa: E402

CACHE = ROOT / "artifacts/performance_v3/cache"
OUT = ROOT / "artifacts/performance_v3/data_characterization"


def concat_caches(caches):
    names = caches[0].names
    if any(c.names != names for c in caches):
        raise SystemExit("feature caches have different column universes")
    meta = {k: np.concatenate([c.meta[k] for c in caches]) for k in caches[0].meta}
    return v3feat.FeatureCache(names=names, matrix=np.concatenate([c.matrix for c in caches]), meta=meta)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    recipes = lc.load_v2_recipes(ROOT)
    v1 = v3feat.load_feature_cache(CACHE / "features_V1")
    chunks = [v3feat.load_feature_cache(CACHE / f"features_{c}") for c in ("D1", "D2", "D3")]
    pool = concat_caches([v1] + chunks)
    ev = v3feat.load_feature_cache(CACHE / "features_LCE")
    v1_ids = sorted(set(map(str, v1.meta["subject_id"])))
    pool_ids = sorted(set(map(str, pool.meta["subject_id"])))
    ev_ids = sorted(set(map(str, ev.meta["subject_id"])))
    lc.assert_disjoint(pool_ids, ev_ids)
    assert len(v1_ids) == 1700 and len(pool_ids) == 8000, (len(v1_ids), len(pool_ids))
    print(f"pool rows={pool.n_rows} subjects={len(pool_ids)} | LCE rows={ev.n_rows} subjects={len(ev_ids)}", flush=True)

    subsets = {r: lc.training_subsets(v1_ids, pool_ids, r) for r in range(len(lc.MODEL_SEEDS))}
    results, boot = [], {}
    for task, recipe in recipes.items():
        ev_idx = lc.eligible_index(ev, task)
        stay_ids = [str(s) for s in ev.meta["stay_id"][ev_idx]]
        times = [str(t) for t in ev.meta["prediction_time"][ev_idx]]
        y_ev = lc.target_vector(ev, task, ev_idx)
        for size in lc.SIZES:
            for repeat, seed in enumerate(lc.MODEL_SEEDS):
                subject_ids = subsets[repeat][size]
                lc.assert_disjoint(subject_ids, ev_ids)
                tr_idx = lc.eligible_index(pool, task, subject_ids)
                metrics, pred = lc.fit_and_score(recipe, pool, tr_idx, ev, ev_idx, seed=seed)
                metrics.update({
                    "task": task, "size_subjects": size, "repeat": repeat, "model_seed": seed,
                    "n_train_subjects_requested": len(subject_ids),
                    "peak_rss_mb_process": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)),
                    "n_eval_eligible_rows": int(len(ev_idx)),
                })
                results.append(metrics)
                primary = metrics[lc.PRIMARY_NAME[task]]
                print(f"{task:14s} size={size:5d} rep={repeat} primary={primary:.4f} train={metrics['train_seconds']}s", flush=True)
                if repeat == 0:
                    target = np.expm1(y_ev) if task == "icu_stay_time" else y_ev
                    score = postprocess_icu(pred) if task == "icu_stay_time" else pred
                    records = make_records(target, score, stay_ids, times)
                    fn = record_metric_fn(task, lc.PRIMARY_NAME[task])
                    result = grouped_bootstrap(
                        records, metric_fn=fn, metric_name=lc.PRIMARY_NAME[task], task=task,
                        n_bootstrap=lc.BOOTSTRAP_REPLICATES, seed=lc.BOOTSTRAP_SEED,
                    )
                    boot[f"{task}|{size}"] = {
                        "task": task, "size_subjects": size, "metric": lc.PRIMARY_NAME[task], "point": result.point_estimate,
                        "ci_lower": result.ci_lower, "ci_upper": result.ci_upper, "ci_level": 0.95,
                        "n_valid_replicates": result.n_valid_replicates, "n_requested": result.n_requested_replicates,
                        "seed": lc.BOOTSTRAP_SEED, "n_stays": result.n_original_stays, "n_examples": result.n_original_examples,
                    }

    summary = {}
    for task in recipes:
        primary = lc.PRIMARY_NAME[task]
        per_size = {}
        for size in lc.SIZES:
            vals = [r[primary] for r in results if r["task"] == task and r["size_subjects"] == size]
            per_size[size] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "values": vals}
        means = [per_size[s]["mean"] for s in lc.SIZES]
        summary[task] = {
            "primary_metric": primary, "per_size": {str(k): v for k, v in per_size.items()},
            "trend": lc.classify_trend(task, lc.SIZES, means),
            "recipe": {k: recipes[task][k] for k in ("variant", "objective", "weight_mode", "n_estimators", "params")},
        }

    payload = {
        "artifact_version": lc.LEARNING_CURVE_VERSION, "status": "DEVELOPMENT_EVIDENCE_NOT_FINAL_VALIDATION",
        "evaluation_cohort": "LCE", "n_eval_subjects": len(ev_ids), "training_pool_subjects": len(pool_ids),
        "sizes": list(lc.SIZES), "model_seeds": list(lc.MODEL_SEEDS), "v2_recipe_ref": lc.SELECTED_V2_REL,
        "caveat": "Fixed holdout reused across all points; at size 1700 all repeats use the same subjects (V1 dev) and differ only by model seed; at 8000 the training set is the full pool.",
        "wall_seconds": round(time.time() - started, 1), "summary": summary, "fits": results,
    }
    (OUT / "learning_curve_metrics.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    (OUT / "learning_curve_bootstrap.json").write_text(json.dumps({"replicates": lc.BOOTSTRAP_REPLICATES, "seed": lc.BOOTSTRAP_SEED, "repeat": 0, "entries": boot}, indent=1, sort_keys=True) + "\n")
    with (OUT / "learning_curve_table.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["task", "metric", "size_subjects", "mean", "sd", "ci_lower_repeat0", "ci_upper_repeat0", "n_valid_replicates"])
        for task in recipes:
            for size in lc.SIZES:
                b = boot[f"{task}|{size}"]
                s = summary[task]["per_size"][str(size)]
                w.writerow([task, lc.PRIMARY_NAME[task], size, f"{s['mean']:.6f}", f"{s['sd']:.6f}", f"{b['ci_lower']:.6f}", f"{b['ci_upper']:.6f}", b["n_valid_replicates"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for task in recipes:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=160)
        s = summary[task]["per_size"]
        xs = list(lc.SIZES)
        ys = [s[str(x)]["mean"] for x in xs]
        sd = [s[str(x)]["sd"] for x in xs]
        ax.errorbar(xs, ys, yerr=sd, marker="o", capsize=3, color="#2b6f9e", label="mean ± SD over 3 repeats")
        ax.fill_between(xs, [boot[f"{task}|{x}"]["ci_lower"] for x in xs], [boot[f"{task}|{x}"]["ci_upper"] for x in xs], alpha=0.15, color="#2b6f9e", label="95% stay-cluster CI (repeat 0)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("training subjects (frozen V2 recipe)")
        ax.set_ylabel(lc.PRIMARY_NAME[task])
        ax.set_title(f"{task}: learning curve on LCE")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"learning_curve_{task}.png")
        plt.close(fig)
    print("done", round(time.time() - started), "s", flush=True)


if __name__ == "__main__":
    main()
