#!/usr/bin/env python3
"""Performance-V3 governed development search (stages S0, S1, S2, S4) + candidate selection.

Development evidence only: reads a feature cache built from DEVELOPMENT rows and writes
an append-only trial ledger. Resumable. Never opens any sealed cohort.

  s0     control: frozen V2 recipe per task, 5-fold cross-fit on the development pool
  s1     family screen (3 screening folds, V3_CTRL variant)
  s2     feature-family ablation (best S1 boosted config per task x variants x 2 seeds)
  s4     focused search (top-3 family/variant per task x 20 configs, 5 folds)
  select apply the pre-registered selection rule and write phase2 candidate summary
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from performance_v3 import search as S  # noqa: E402
from performance_v3.features import load_feature_cache, variant_columns  # noqa: E402
from performance_v3.models import sample_family_configs  # noqa: E402

OUT = ROOT / "artifacts/performance_v3/phase2"
LEDGER = OUT / "trial_ledger_v1.jsonl"
MASTER_SEED = 20261011
MODEL_SEEDS = (11, 23, 47)
SCREEN_FOLDS = (0, 1, 2)
ALL_FOLDS = (0, 1, 2, 3, 4)
BOOSTED = ("xgboost", "lightgbm", "catboost")
VARIANTS = ("V3_CTRL", "V3_B", "V3_BF", "V3_BC", "V3_BD", "V3_BE", "V3_BFS", "V3_FULL")
COMPLEXITY = {"ridge": 0, "logistic": 0, "xgboost": 1, "lightgbm": 1, "catboost": 1, "gru": 2}
TIE = {"recovery24": 0.005, "recovery48": 0.005, "icu_stay_time": 0.005, "organ_support": 0.003}
V2_OBJECTIVE = {"reg:absoluteerror": "absoluteerror", "reg:pseudohubererror": "pseudohubererror", "reg:squarederror": "squarederror"}


def families_for(task, linear=True):
    out = list(BOOSTED)
    if linear:
        out.append("logistic" if task == "organ_support" else "ridge")
    return out


def cfg_seed(task, family, stage_offset):
    return MASTER_SEED + 1000 * S.TASKS.index(task) + 100 * (list(COMPLEXITY).index(family)) + stage_offset


def v2_recipe_config(task):
    sel = json.loads((ROOT / "artifacts/performance_v2/phase3/selected_models_v2.json").read_text())["tasks"][task]
    hp = dict(sel["hyperparameters"])
    config = {k: hp[k] for k in ("max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda", "gamma")}
    config.update({
        "n_estimators": int(sel["final_training_budget"]),
        "use_early_stopping": False,
        "objective": "binary" if task == "organ_support" else V2_OBJECTIVE[hp["objective"]],
        "weight_mode": "stay_balanced" if hp.get("weight_mode") == "stay_balanced" else "none",
        "class_weight": "none",
        "config_hash": "v2recipe-" + task,
    })
    return config, sel["feature_variant"]


def ctrl_variant_label(task):
    return {"recovery24": "B_MIN", "recovery48": "B_MIN", "icu_stay_time": "B_PLUS_F", "organ_support": "B_FULL"}[task]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("s0", "s1", "s2", "s4", "select"))
    parser.add_argument("--cache", default="artifacts/performance_v3/cache/dev_pool")
    parser.add_argument("--tasks", nargs="*", default=list(S.TASKS))
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--budget-hours", type=float, default=6.0)
    parser.add_argument("--configs-boosted", type=int, default=10)
    parser.add_argument("--configs-linear", type=int, default=6)
    parser.add_argument("--configs-s4", type=int, default=20)
    parser.add_argument("--include-s", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ledger = S.TrialLedger(LEDGER)
    if args.stage == "select":
        return select(ledger)
    cache = load_feature_cache(ROOT / args.cache)
    include_s = args.include_s or any(n.startswith("S__") for n in cache.names)
    started = time.time()
    over = lambda: (time.time() - started) > args.budget_hours * 3600
    print(f"cache rows={cache.n_rows} cols={len(cache.names)} include_s={include_s}", flush=True)

    for task in args.tasks:
        if over():
            print("budget exhausted before", task); break
        if args.stage == "s0":
            config, variant_label = v2_recipe_config(task)
            td = S.prepare_task_data(cache, task, "V3_CTRL", include_s=include_s)
            rec = S.run_trial_logged(ledger, td, "S0", "xgboost", config, folds=ALL_FOLDS, seed=MODEL_SEEDS[0], nthread=args.nthread)
            print("S0", task, rec["status"], rec.get("primary_mean"), flush=True)
        elif args.stage == "s1":
            td = S.prepare_task_data(cache, task, "V3_CTRL", include_s=include_s)
            for family in families_for(task):
                n = args.configs_boosted if family in BOOSTED else args.configs_linear
                for config in sample_family_configs(family, n, cfg_seed(task, family, 1), task):
                    if over():
                        break
                    rec = S.run_trial_logged(ledger, td, "S1", family, config, folds=SCREEN_FOLDS, seed=MODEL_SEEDS[0], nthread=args.nthread)
                    print("S1", task, family, config["config_hash"], rec["status"], round(rec.get("primary_mean", float("nan")), 4), f"{rec['wall_seconds']}s", flush=True)
        elif args.stage == "s2":
            best = S.best_trial([r for r in ledger.records() if r["family"] in BOOSTED], task, stage="S1")
            if best is None:
                print("no S1 boosted result for", task); continue
            for variant in VARIANTS:
                try:
                    variant_columns(cache.names, variant, "recovery24" if task != "icu_stay_time" else "icu_time", include_s=include_s)
                    td = S.prepare_task_data(cache, task, variant, include_s=include_s)
                except NotImplementedError as error:
                    print("skip", variant, task, "unavailable:", str(error)[:80]); continue
                for seed in MODEL_SEEDS[:2]:
                    if over():
                        break
                    rec = S.run_trial_logged(ledger, td, "S2", best["family"], best["config"], folds=SCREEN_FOLDS, seed=seed, nthread=args.nthread)
                    print("S2", task, variant, seed, rec["status"], round(rec.get("primary_mean", float("nan")), 4), f"{rec['wall_seconds']}s", flush=True)
        elif args.stage == "s4":
            records = [r for r in ledger.records() if r["status"] == "ok" and r["task"] == task and r["stage"] in ("S1", "S2", "S3")]
            groups = {}
            for r in records:
                key = (r["family"], r["variant"])
                if key not in groups or S.better(task, r["primary_mean"], groups[key]["primary_mean"]):
                    groups[key] = r
            sign = 1 if S.LOWER_IS_BETTER[task] else -1
            top = sorted(groups.values(), key=lambda r: sign * r["primary_mean"])[:3]
            for cand in top:
                td = S.prepare_task_data(cache, task, cand["variant"], include_s=include_s)
                configs = [dict(cand["config"])] + sample_family_configs(cand["family"], args.configs_s4, cfg_seed(task, cand["family"], 4), task)
                for config in configs:
                    if over():
                        break
                    rec = S.run_trial_logged(ledger, td, "S4", cand["family"], config, folds=ALL_FOLDS, seed=MODEL_SEEDS[0], nthread=args.nthread)
                    print("S4", task, cand["family"], cand["variant"], config["config_hash"], rec["status"], round(rec.get("primary_mean", float("nan")), 4), f"{rec['wall_seconds']}s", flush=True)
        del_td = None


def select(ledger):
    """Pre-registered rule: lowest mean cross-fitted primary; ties (<=tolerance of best) -> lower
    complexity, then lower fold SD. Only S4 (5-fold) trials are eligible; V2 recipe (S0) is the control."""

    summary = {"rule": "lowest mean primary; tie tolerance relative 0.5% (AUPRC abs 0.003) -> complexity -> fold sd", "tasks": {}}
    records = ledger.records()
    for task in S.TASKS:
        pool = [r for r in records if r["status"] == "ok" and r["task"] == task and r["stage"] == "S4"]
        control = S.best_trial(records, task, stage="S0")
        if not pool:
            summary["tasks"][task] = {"status": "NO_S4_TRIALS", "control": control and control["primary_mean"]}
            continue
        sign = 1 if S.LOWER_IS_BETTER[task] else -1
        best = min(pool, key=lambda r: sign * r["primary_mean"])
        tol = TIE[task] if task == "organ_support" else abs(best["primary_mean"]) * TIE[task]
        near = [r for r in pool if abs(r["primary_mean"] - best["primary_mean"]) <= tol]
        winner = min(near, key=lambda r: (COMPLEXITY[r["family"]], r["primary_sd"], sign * r["primary_mean"]))
        replace_v2 = control is None or (S.better(task, winner["primary_mean"], control["primary_mean"]) and abs(winner["primary_mean"] - control["primary_mean"]) > tol)
        summary["tasks"][task] = {
            "winner": {k: winner[k] for k in ("trial_id", "family", "variant", "config", "primary_mean", "primary_sd", "per_fold_primary", "best_iterations", "secondary_means")},
            "best_raw": {"trial_id": best["trial_id"], "primary_mean": best["primary_mean"]},
            "tie_group_size": len(near), "tolerance": tol,
            "v2_recipe_control": None if control is None else {"primary_mean": control["primary_mean"], "primary_sd": control["primary_sd"]},
            "beats_v2_recipe_beyond_tolerance": bool(replace_v2),
        }
    path = OUT / "candidate_selection_v1.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
