#!/usr/bin/env python3
"""Protocol stage S3 (bounded GRU search). Appends 'S3'/'gru' trials to the shared V3 trial ledger.

Development pool only (V1 train+validation + D1..D3); resumable; CPU.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from performance_v3 import gru_search as G  # noqa: E402
from performance_v3.search import TASKS, TrialLedger  # noqa: E402

OUT = ROOT / "artifacts/performance_v3/phase2"
LEDGER = OUT / "trial_ledger_v1.jsonl"
MASTER_SEED = 20261011
MODEL_SEED = 11
SCREEN_FOLDS = (0, 1, 2)
CHUNKS = [("SYN-V3-D1", "artifacts/performance_v3/cohorts/D1"), ("SYN-V3-D2", "artifacts/performance_v3/cohorts/D2"),
          ("SYN-V3-D3", "artifacts/performance_v3/cohorts/D3")]


def best_variant(records, task):
    pool = [r for r in records if r["status"] == "ok" and r["task"] == task and r["stage"] == "S2"]
    if not pool:
        return "V3_CTRL"
    lower = task != "organ_support"
    agg = {}
    for r in pool:
        agg.setdefault(r["variant"], []).append(r["primary_mean"])
    score = {v: sum(x) / len(x) for v, x in agg.items()}
    return (min if lower else max)(score, key=score.get)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--configs", type=int, default=8)
    parser.add_argument("--budget-hours", type=float, default=4.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit-rows", type=int, default=None, help="smoke only")
    parser.add_argument("--max-epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--folds", nargs="*", type=int, default=list(SCREEN_FOLDS))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = TrialLedger(LEDGER)
    started = time.time()
    pool = G.load_pool(ROOT, [(c, ROOT / p) for c, p in CHUNKS], limit=args.limit_rows)
    print(f"pool rows={pool.n_rows}", flush=True)
    for task in args.tasks:
        gt = G.prepare_gru_task(pool, task)
        variant = best_variant(ledger.records(), task)
        offset = MASTER_SEED + 1000 * TASKS.index(task) + 500 + 3
        for config in G.sample_gru_configs(args.configs, offset, task):
            if time.time() - started > args.budget_hours * 3600:
                print("budget exhausted"); return
            rec = G.run_gru_trial_logged(ledger, pool, gt, variant, config, folds=args.folds, seed=MODEL_SEED,
                                         oof_dir=OUT / "gru_oof", max_epochs=args.max_epochs)
            print("S3", task, variant, config["config_hash"], rec["status"], round(rec.get("primary_mean", float("nan")), 4),
                  f"{rec.get('wall_seconds')}s", f"{rec.get('seconds_per_epoch', float('nan')):.2f}s/epoch", flush=True)


if __name__ == "__main__":
    main()
