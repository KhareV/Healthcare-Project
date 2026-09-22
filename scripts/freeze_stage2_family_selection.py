#!/usr/bin/env python3
"""Freeze validation-only XGBoost-versus-GRU selection before LSTM execution."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from experiments.stage2_finalization import (
    EXPECTED_G1_SHA256, EXPECTED_GRU_BEST_SHA256, EXPECTED_GRU_WINNERS,
    EXPECTED_PHASE14_SHA256, EXPECTED_XGB_WINNERS, STAGE2_VERSION, TASKS,
    assert_prediction_parity, canonical_sha256, choose_family, selection_key,
)


OUTPUT = ROOT / "artifacts/selection/validation_family_selection_v1.json"
XGB_BEST = ROOT / "artifacts/search/xgb/phase12/best_xgb_candidates_v1.json"
XGB_MASTER = ROOT / "artifacts/search/xgb/phase12/manifests/phase12_search_manifest_v1.json"
GRU_BEST = ROOT / "artifacts/search/gru/final_v2/best_gru_candidates_v1.json"
GRU_MASTER = ROOT / "artifacts/search/gru/final_v2/manifests/search_manifest_v1.json"
G1 = ROOT / "artifacts/acceptance/g1_synthetic_data_freeze_v1.json"
HANDOFF = ROOT / "artifacts/handoffs/sanskruti_phase14_handoff_v1.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(canonical_json_bytes(value)); temp.replace(path)


def git(*args):
    return subprocess.check_output(("git",) + args, cwd=ROOT, text=True).strip()


def require_absent():
    forbidden = (
        ROOT / "artifacts/models/selected_models_v1.json",
        ROOT / "artifacts/governance/g3_freeze.json",
        ROOT / "artifacts/acceptance/g3_model_selection_freeze_v1.json",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("Stage-2 hard gate failed: selected-model or G3 freeze already exists")


def run_audit(script):
    subprocess.run((sys.executable, str(ROOT / script)), cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


def prediction_paths(task, family, candidate):
    if family == "gru":
        return [ROOT / candidate["metrics_path"].replace("validation_metrics.json", "validation_predictions.jsonl")]
    base = ROOT / candidate["metrics"]["metrics_path"].replace("validation_metrics.json", "")
    if task == "recovery":
        return [base / "validation_predictions_24h.jsonl", base / "validation_predictions_48h.jsonl"]
    return [base / "validation_predictions.jsonl"]


def decisive_component(task, xkey, gkey):
    labels = {
        "recovery": ("mae24", "mae48", "rmse24", "rmse48", "family_name"),
        "icu_time": ("weighted_median_absolute_error_hours", "mae_hours", "rmse_hours", "family_name"),
        "organ_support": ("auprc", "brier", "auroc", "family_name"),
    }[task]
    index = next(i for i, pair in enumerate(zip(xkey, gkey)) if pair[0] != pair[1])
    return {"component": labels[index], "component_index": index, "tie_break_used": index > 0}


def main():
    if OUTPUT.exists():
        raise RuntimeError("immutable Stage-2 family-selection artifact already exists")
    require_absent()
    if sha256_file(G1) != EXPECTED_G1_SHA256:
        raise RuntimeError("Stage-2 hard gate failed: G1 hash mismatch")
    if sha256_file(HANDOFF) != EXPECTED_PHASE14_SHA256:
        raise RuntimeError("Stage-2 hard gate failed: Phase-14 handoff hash mismatch")
    if sha256_file(GRU_BEST) != EXPECTED_GRU_BEST_SHA256:
        raise RuntimeError("Stage-2 hard gate failed: best-GRU hash mismatch")
    run_audit("scripts/audit_phase12_xgb_search.py")
    run_audit("scripts/audit_final_gru_search.py")
    xgb, gru, xm, gm, handoff = map(load, (XGB_BEST, GRU_BEST, XGB_MASTER, GRU_MASTER, HANDOFF))
    if ({task: xgb["tasks"][task]["candidate_id"] for task in TASKS} != EXPECTED_XGB_WINNERS
            or {task: gru["tasks"][task]["candidate_id"] for task in TASKS} != EXPECTED_GRU_WINNERS):
        raise RuntimeError("Stage-2 hard gate failed: within-family winner identity mismatch")
    if (xm["status"] != "COMPLETE" or gm["status"] != "COMPLETE"
            or xm["test_accessed"] is not False or gm["test_accessed"] is not False
            or any(gm["terminal_candidate_counts"][task] != {"complete": 30, "failed": 0} for task in TASKS)):
        raise RuntimeError("Stage-2 hard gate failed: source search incomplete or test-accessed")
    common = ("split_sha256", "feature_schema_sha256", "preprocessor_sha256")
    mismatched = [field for field in common if xm[field] != gm[field]]
    if mismatched:
        raise RuntimeError("Stage-2 compatibility mismatch: " + ", ".join(mismatched))
    if (xm["recovery_target_scaler_sha256"] != gm["recovery_target_scaler_sha256"]
            or xm["support_class_weight_sha256"] != gm["support_class_weight_sha256"]):
        raise RuntimeError("Stage-2 task preprocessing mismatch")
    if (handoff["parents"]["phase12_best_xgb"]["sha256"] != sha256_file(XGB_BEST)
            or handoff["g1_sha256"] != EXPECTED_G1_SHA256):
        raise RuntimeError("Phase-14 does not bind the authoritative XGB/G1 parents")

    tasks = {}
    for task in TASKS:
        x, g = xgb["tasks"][task], gru["tasks"][task]
        parity = assert_prediction_parity(task, prediction_paths(task, "xgboost", x), prediction_paths(task, "gru", g)[0])
        xmetrics, gmetrics = dict(x["metrics"]), dict(g["metrics"])
        if task == "organ_support":
            xmetrics["probability_type"] = gmetrics["probability_type"] = "raw_uncalibrated"
        selected = choose_family(task, xmetrics, gmetrics)
        winner = x if selected == "xgboost" else g
        xkey, gkey = selection_key(task, "xgboost", xmetrics), selection_key(task, "gru", gmetrics)
        bundle_path = x["metrics"]["bundle_path"]
        sources = {
            "xgboost": {
                "candidate_id": x["candidate_id"], "run_id": x["run_id"], "metrics": xmetrics,
                "metrics_path": x["metrics"]["metrics_path"],
                "metrics_sha256": sha256_file(ROOT / x["metrics"]["metrics_path"]),
                "bundle_path": bundle_path, "bundle_sha256": sha256_file(ROOT / bundle_path),
                "model_hashes": x["metrics"]["model_hashes"],
            },
            "gru": {
                "candidate_id": g["candidate_id"], "run_id": g["run_id"], "metrics": gmetrics,
                "metrics_path": g["metrics_path"], "metrics_sha256": sha256_file(ROOT / g["metrics_path"]),
                "checkpoint_path": g["checkpoint_path"], "checkpoint_sha256": g["checkpoint_sha256"],
            },
        }
        tasks[task] = {
            "selected_family": selected, "selected_candidate_id": winner["candidate_id"],
            "selected_run_id": winner["run_id"],
            "selected_model_hash": canonical_sha256(x["metrics"]["model_hashes"]) if selected == "xgboost" else g["checkpoint_sha256"],
            "primary_selection_metric": {"recovery": "mae24", "icu_time": "weighted_median_absolute_error_hours", "organ_support": "auprc"}[task],
            "deterministic_decision": decisive_component(task, xkey, gkey),
            "ranking_keys": {"xgboost": list(xkey), "gru": list(gkey)},
            "sources": sources, "compatibility": parity,
        }
    payload = {
        "manifest_version": "validation_family_selection_v1", "stage_version": STAGE2_VERSION,
        "status": "VALIDATION_FAMILY_SELECTION_FROZEN", "frozen_before_lstm_interpretation": True,
        "serving_competitors": ["xgboost", "gru"], "lstm_eligible_for_selection": False,
        "g1_path": str(G1.relative_to(ROOT)), "g1_sha256": sha256_file(G1),
        "phase14_handoff_path": str(HANDOFF.relative_to(ROOT)), "phase14_handoff_sha256": sha256_file(HANDOFF),
        "split_sha256": gm["split_sha256"], "feature_schema_sha256": gm["feature_schema_sha256"],
        "preprocessor_sha256": gm["preprocessor_sha256"],
        "recovery_target_scaler_sha256": gm["recovery_target_scaler_sha256"],
        "support_class_weight_sha256": gm["support_class_weight_sha256"],
        "source_search_manifests": {
            "xgboost": {"path": str(XGB_MASTER.relative_to(ROOT)), "sha256": sha256_file(XGB_MASTER)},
            "gru": {"path": str(GRU_MASTER.relative_to(ROOT)), "sha256": sha256_file(GRU_MASTER)},
        },
        "best_source_manifests": {
            "xgboost": {"path": str(XGB_BEST.relative_to(ROOT)), "sha256": sha256_file(XGB_BEST)},
            "gru": {"path": str(GRU_BEST.relative_to(ROOT)), "sha256": sha256_file(GRU_BEST)},
        },
        "implementation_commit": git("rev-parse", "HEAD"), "tasks": tasks,
        "support_calibrated": False, "support_threshold": None,
        "selected_models_v1_created": False, "g3_created": False, "test_accessed": False,
    }
    write(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "selected": {t: tasks[t]["selected_family"] for t in TASKS},
                      "sha256": sha256_file(OUTPUT), "test_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
