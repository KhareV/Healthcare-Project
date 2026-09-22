#!/usr/bin/env python3
"""Freeze selected models, validation calibration/threshold, and real G3."""
from __future__ import annotations

from dataclasses import replace
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes
from evaluation.calibrate import apply_support_calibrator, fit_support_calibrator, save_calibrator_artifact
from evaluation.calibration_validation import IsotonicPolicy, SelectedSupportModel, load_validation_predictions
from evaluation.threshold import ThresholdSearchPolicy, choose_support_threshold, save_threshold_artifact
from evaluation.weights import compute_stay_weights
from experiments.lineage import ArtifactRecord, read_artifact_index, read_run_registry, validate_artifact_lineage, write_artifact_index
from experiments.search_governance import canonical_sha256
from vedant_infra.g3 import audit_g3, freeze_g3, validate_g3_marker, write_audit_report
from vedant_infra.hashing import sha256_file


EXPECTED = {
    "g1": "1fbcc1b0859c0052eb867aec15e89bd3f7df503ac318b30f92dd98ff9eeccf90",
    "handoff": "3c4fa9a9efd25145ab1daa23c03632aa0cfd0b04d3ee0b4254c1cb180fff63d4",
    "selection": "0ccc7296618be5c6db1f93bc8b9f22f48c664f6b7abc5c80072828eedb8c5016",
    "lstm": "0040a296876613cb714dac96c9a57d930750484e359be4d393d899ebee84ad8e",
}
REFS = {
    "g1": "artifacts/acceptance/g1_synthetic_data_freeze_v1.json",
    "handoff": "artifacts/handoffs/sanskruti_phase14_handoff_v1.json",
    "selection": "artifacts/selection/validation_family_selection_v1.json",
    "lstm": "artifacts/models/lstm_sensitivity_v1.json",
    "feature_schema": "configs/synthetic/feature_schema_v2.json",
    "split": "artifacts/splits/synthetic_split_v2.csv",
    "preprocessor": "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json",
    "recovery_scaler": "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json",
    "support_weight": "artifacts/preprocessors/support_class_weight_synthetic_v1.json",
}
CAL_REF = "artifacts/calibration/isotonic_support_v1.json"
CAL_PRED_REF = "artifacts/calibration/support_validation_calibrated_predictions_v1.jsonl"
THRESHOLD_REF = "artifacts/thresholds/support_threshold_v1.json"
SELECTED_REF = "artifacts/models/selected_models_v1.json"


def load(ref):
    return json.loads((ROOT / ref).read_text(encoding="utf-8"))


def write_json(ref, value):
    path = ROOT / ref; path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp"); temp.write_bytes(canonical_json_bytes(value)); temp.replace(path)


def write_jsonl(ref, rows):
    path = ROOT / ref; path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        for row in rows: handle.write(canonical_json_bytes(row) + b"\n")
    temp.replace(path)


def git(*args):
    return subprocess.check_output(("git",) + args, cwd=ROOT, text=True).strip()


def registry_rows():
    with (ROOT / "experiments/registry.csv").open(newline="", encoding="utf-8") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def ensure_prerequisites():
    for name in ("g1", "handoff", "selection", "lstm"):
        if sha256_file(ROOT / REFS[name]) != EXPECTED[name]:
            raise RuntimeError("Stage-3 frozen parent mismatch: " + name)
    selection, lstm = load(REFS["selection"]), load(REFS["lstm"])
    chosen = {task: item["selected_candidate_id"] for task, item in selection["tasks"].items()}
    if chosen != {"recovery": "xgb-recovery-014", "icu_time": "gru-icu-time-026", "organ_support": "xgb-support-024"}:
        raise RuntimeError("Stage-2 selected-family identities changed")
    if (selection["test_accessed"] is not False or lstm["test_accessed"] is not False
            or lstm["run_count"] != 3 or any(row["sensitivity_only"] is not True for row in lstm["runs"])):
        raise RuntimeError("Stage-2/LSTM hard prerequisite failed")
    if any((ROOT / ref).exists() for ref in (CAL_REF, THRESHOLD_REF, SELECTED_REF, "artifacts/governance/g3_freeze.json")):
        raise RuntimeError("immutable Stage-3 output already exists")
    head = git("rev-parse", "HEAD")
    if subprocess.run(("git", "diff", "--quiet", head, "--", "src", "scripts", "configs", "tests"), cwd=ROOT).returncode:
        raise RuntimeError("science-affecting Stage-3 code is dirty")
    return selection, lstm, head


def selected_support(selection, runs):
    source = selection["tasks"]["organ_support"]["sources"]["xgboost"]
    run = runs[source["run_id"]]
    raw_ref = source["metrics_path"].replace("validation_metrics.json", "validation_predictions.jsonl")
    model_ref = run["model_artifact_ref"]
    return SelectedSupportModel(task="organ_support", family="xgboost", run_id=run["run_id"],
        candidate_id=run["candidate_id"], config_hash=run["config_hash"], artifact_ref=model_ref,
        artifact_sha256=run["model_sha256"], split_hash=run["split_hash"], feature_version=run["feature_version"],
        label_version=run["label_version"], preprocessing_hash=run["preprocessor_sha256"], code_commit=run["code_commit"],
        selection_metric="validation_auprc", probability_type="raw_uncalibrated",
        validation_prediction_ref=raw_ref, validation_prediction_sha256=sha256_file(ROOT / raw_ref),
        selection_mode="real", test_accessed=False, selection_manifest_ref=REFS["selection"],
        selection_manifest_sha256=EXPECTED["selection"])


def governance_bindings():
    return {
        "g1": {"ref": REFS["g1"], "sha256": sha256_file(ROOT / REFS["g1"])},
        "phase14_handoff": {"ref": REFS["handoff"], "sha256": sha256_file(ROOT / REFS["handoff"])},
        "stage2_family_selection": {"ref": REFS["selection"], "sha256": sha256_file(ROOT / REFS["selection"])},
        "lstm_sensitivity": {"ref": REFS["lstm"], "sha256": sha256_file(ROOT / REFS["lstm"])},
        "feature_schema": {"ref": REFS["feature_schema"], "sha256": sha256_file(ROOT / REFS["feature_schema"])},
        "split": {"ref": REFS["split"], "sha256": sha256_file(ROOT / REFS["split"])},
        "preprocessor": {"ref": REFS["preprocessor"], "sha256": sha256_file(ROOT / REFS["preprocessor"])},
        "recovery_target_scaler": {"ref": REFS["recovery_scaler"], "sha256": sha256_file(ROOT / REFS["recovery_scaler"])},
        "support_class_weight": {"ref": REFS["support_weight"], "sha256": sha256_file(ROOT / REFS["support_weight"])},
    }


def fit_calibration(selection, support, commit):
    policy = IsotonicPolicy(policy_version="stage3_support_isotonic_v1", out_of_bounds="clip",
        sample_weighting="stay_balanced", increasing=True, y_min=0.0, y_max=1.0, approved_for_real=True)
    calibrator = fit_support_calibrator(support, policy=policy, mode="real")
    records = load_validation_predictions(support)
    calibrated = apply_support_calibrator(calibrator, [row.raw_probability for row in records])
    cal_rows = [{"row_id": row.row_id, "stay_id": row.stay_id, "label": row.label,
                 "calibrated_probability": probability, "partition": "validation", "eligible": True}
                for row, probability in zip(records, calibrated)]
    write_jsonl(CAL_PRED_REF, cal_rows)
    weights = compute_stay_weights([row.stay_id for row in records], [True] * len(records)).weights
    raw_brier = math.fsum(w * (row.raw_probability - row.label) ** 2 for row, w in zip(records, weights)) / math.fsum(weights)
    cal_brier = math.fsum(w * (probability - row.label) ** 2 for row, probability, w in zip(records, calibrated, weights)) / math.fsum(weights)
    governance = {"g1_sha256": EXPECTED["g1"], "phase14_handoff_sha256": EXPECTED["handoff"],
        "stage2_family_selection_sha256": EXPECTED["selection"], "feature_schema_sha256": sha256_file(ROOT / REFS["feature_schema"]),
        "preprocessor_sha256": sha256_file(ROOT / REFS["preprocessor"]), "split_sha256": sha256_file(ROOT / REFS["split"]),
        "validation_row_key_hash": calibrator.eligible_population_hash, "weighting_definition": "eligible row weight = 1 / eligible rows in stay",
        "calibrated_prediction_ref": CAL_PRED_REF, "calibrated_prediction_sha256": sha256_file(ROOT / CAL_PRED_REF),
        "raw_validation_brier": raw_brier, "calibrated_validation_brier_resubstitution": cal_brier,
        "code_commit": commit, "test_accessed": False}
    calibrator = replace(calibrator, code_commit=commit, governance=governance)
    cal_artifact = save_calibrator_artifact(CAL_REF, calibrator)
    threshold_policy = ThresholdSearchPolicy(policy_version="stage3_support_threshold_v1",
        candidate_method="unique_calibrated_probabilities_plus_endpoints",
        tie_policy="closest_to_half_then_higher", comparator="greater_than_or_equal", approved_for_real=True)
    threshold = choose_support_threshold(support, calibrator, calibrator_ref=CAL_REF,
        calibrator_sha256=cal_artifact["artifact_sha256"], policy=threshold_policy, mode="real")
    threshold = replace(threshold, code_commit=commit, calibrated_prediction_ref=CAL_PRED_REF,
        calibrated_prediction_sha256=sha256_file(ROOT / CAL_PRED_REF), governance=governance)
    threshold_artifact = save_threshold_artifact(THRESHOLD_REF, threshold)
    return calibrator, cal_artifact, threshold, threshold_artifact


def build_manifest(selection, lstm, runs, support, cal_artifact, threshold, threshold_artifact, commit):
    split_hash = sha256_file(ROOT / REFS["split"]); pre_hash = sha256_file(ROOT / REFS["preprocessor"])
    flat_ref = "artifacts/features/synthetic_xgb_flat_feature_map_v1.json"; flat_hash = sha256_file(ROOT / flat_ref)
    winners = {
        "recovery": runs["phase12-xgb-recovery-014-attempt-1"],
        "icu_stay_time": runs["final-v2-gru-icu-time-026-attempt-1"],
        "organ_support": runs["phase12-xgb-support-024-attempt-1"],
    }
    gru_parents = {row["task"]: row["parent_run_id"] for row in runs.values() if row["run_type"] == "scientific_sensitivity"}
    tasks = {}
    for task, run in winners.items():
        family = run["model_family"]
        entry = {"family": family, "candidate_id": run["candidate_id"], "run_id": run["run_id"],
            "config_hash": run["config_hash"], "artifact_ref": run["model_artifact_ref"], "artifact_sha256": run["model_sha256"],
            "artifact_metadata_ref": run["model_artifact_ref"] + ".metadata.json" if family == "gru" else run["model_artifact_ref"],
            "feature_version": run["feature_version"], "feature_schema_ref": REFS["feature_schema"],
            "feature_schema_sha256": sha256_file(ROOT / REFS["feature_schema"]), "label_version": run["label_version"],
            "split_ref": REFS["split"], "split_hash": split_hash, "preprocessor_ref": REFS["preprocessor"],
            "preprocessor_sha256": pre_hash, "preprocessing_hash": pre_hash, "preprocessing_fit_partition": "train",
            "code_commit": run["code_commit"], "explanation_method": "tree_shap" if family == "xgboost" else "integrated_gradients",
            "selected_gru_run_id": gru_parents[task], "test_accessed": False}
        if task == "recovery":
            bundle = load(run["model_artifact_ref"])
            entry.update(selection_metric="validation_recovery24_mae", model_artifacts=bundle["models"],
                flat_feature_map_ref=flat_ref, flat_feature_map_sha256=flat_hash,
                recovery_target_scaler_ref=REFS["recovery_scaler"], recovery_target_scaler_sha256=sha256_file(ROOT / REFS["recovery_scaler"]))
        elif task == "icu_stay_time":
            sidecar = load(run["model_artifact_ref"] + ".metadata.json")
            entry.update(selection_metric="validation_median_absolute_error_hours", target_scaler=None,
                         icu_postprocess_version=sidecar["postprocess_version"])
        else:
            bundle = load(run["model_artifact_ref"])
            entry.update(selection_metric="validation_auprc", model_artifacts=bundle["models"],
                flat_feature_map_ref=flat_ref, flat_feature_map_sha256=flat_hash,
                support_class_weight_ref=REFS["support_weight"], support_class_weight_sha256=sha256_file(ROOT / REFS["support_weight"]),
                probability_type="raw_uncalibrated", validation_prediction_ref=support.validation_prediction_ref,
                validation_prediction_sha256=support.validation_prediction_sha256, calibration_status="FROZEN_VALIDATION_ONLY",
                calibrator={**cal_artifact, "selected_model_sha256": support.artifact_sha256, "partition": "validation", "method": "isotonic"},
                threshold={**threshold_artifact, "calibrator_sha256": cal_artifact["artifact_sha256"],
                    "criterion": "validation_f1", "partition": "validation", "value": threshold.threshold,
                    "comparator": "greater_than_or_equal"})
        tasks[task] = entry
    content = {"manifest_version": "selected_models_calibrated_v1", "selection_mode": "real",
        "status": "SELECTED_MODELS_FROZEN_PRE_TEST", "serving_ready": False, "test_accessed": False,
        "code_commit": commit, "stage2_family_selection": {"ref": REFS["selection"], "sha256": EXPECTED["selection"]},
        "lstm_sensitivity": {"ref": REFS["lstm"], "sha256": EXPECTED["lstm"]},
        "g1": {"ref": REFS["g1"], "sha256": EXPECTED["g1"]},
        "phase14_handoff": {"ref": REFS["handoff"], "sha256": EXPECTED["handoff"]},
        "governance_bindings": governance_bindings(), "tasks": tasks}
    manifest = {**content, "manifest_sha256": canonical_sha256(content)}
    write_json(SELECTED_REF, manifest)
    return manifest


def register_artifacts(support, cal_artifact, threshold, threshold_artifact, commit):
    existing = list(read_artifact_index(ROOT / "experiments/artifacts.csv")); by_path = {row.artifact_path: row for row in existing}
    raw = by_path[support.validation_prediction_ref]; model = by_path[support.artifact_ref]
    common = dict(producing_run_id=support.run_id, task="organ_support", model_family="xgboost", split_hash=support.split_hash,
        feature_version=support.feature_version, label_version=support.label_version, config_hash=support.config_hash,
        creation_commit=commit, generating_script="scripts/finalize_stage3.py", creation_date_utc="2026-09-22T00:00:00Z",
        preprocessor_sha256=support.preprocessing_hash, run_type="scientific", status="registered")
    cal_pred_id = "stage3-support-calibrated-validation-predictions"
    additions = [
        ArtifactRecord(artifact_id=cal_pred_id, artifact_path=CAL_PRED_REF, artifact_type="prediction",
            artifact_version="support_calibrated_validation_predictions_v1", artifact_sha256=sha256_file(ROOT / CAL_PRED_REF),
            parent_artifact_ids=model.artifact_id, model_sha256=support.artifact_sha256, partition="validation",
            probability_type="calibrated", prediction_population_hash=threshold.calibrated_prediction_hash, **common),
        ArtifactRecord(artifact_id="stage3-support-isotonic-calibrator", artifact_path=CAL_REF, artifact_type="calibrator",
            artifact_version="isotonic_support_v1", artifact_sha256=cal_artifact["artifact_sha256"],
            parent_artifact_ids=model.artifact_id + ";" + raw.artifact_id, model_sha256=support.artifact_sha256,
            calibration_method="isotonic", partition="validation", **common),
        ArtifactRecord(artifact_id="stage3-support-threshold", artifact_path=THRESHOLD_REF, artifact_type="threshold",
            artifact_version="threshold_support_v1", artifact_sha256=threshold_artifact["artifact_sha256"],
            parent_artifact_ids="stage3-support-isotonic-calibrator;" + cal_pred_id,
            calibrator_sha256=cal_artifact["artifact_sha256"], threshold_criterion="validation_f1",
            threshold_value=str(threshold.threshold), partition="validation", **common),
    ]
    model_parent_ids = ["phase12-xgb-recovery-014-attempt-1:model:recovery24", "phase12-xgb-recovery-014-attempt-1:model:recovery48",
                        "final-v2-gru-icu-time-026-attempt-1:model", "phase12-xgb-support-024-attempt-1:model:organ_support"]
    additions.append(ArtifactRecord(artifact_id="stage3-selected-models-manifest", artifact_path=SELECTED_REF,
        artifact_type="selected_model_manifest", artifact_version="selected_models_calibrated_v1",
        artifact_sha256=sha256_file(ROOT / SELECTED_REF), producing_run_id=support.run_id,
        parent_artifact_ids=";".join(model_parent_ids), split_hash=support.split_hash,
        feature_version=support.feature_version, creation_commit=commit, generating_script="scripts/finalize_stage3.py",
        creation_date_utc="2026-09-22T00:00:00Z", run_type="scientific", status="registered"))
    combined = existing + additions
    validate_artifact_lineage(combined, read_run_registry(ROOT / "experiments/registry.csv"), repository_root=ROOT)
    write_artifact_index(ROOT / "experiments/artifacts.csv", combined)


def write_governance_inputs(commit):
    write_json("artifacts/governance/explanation_adapters_review.json", {"scope": "real",
        "status": "METHOD_ROUTING_FROZEN_ONLY", "tree_shap_implemented": True,
        "integrated_gradients_implemented": True, "reviewed": True, "production_explanations_generated": False,
        "code_commit": commit, "test_accessed": False})
    write_json("artifacts/governance/g3_review_signoffs.json", {"scope": "real",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION", "authorization": "STAGE3_G3_FREEZE_REQUESTED",
        "authorized_roles": ["project_owner"], "human_member_signoff_claimed": False,
        "stage2_family_selection_sha256": EXPECTED["selection"], "test_accessed": False})


def main():
    selection, lstm, commit = ensure_prerequisites(); runs = registry_rows(); support = selected_support(selection, runs)
    calibrator, cal_artifact, threshold, threshold_artifact = fit_calibration(selection, support, commit)
    manifest = build_manifest(selection, lstm, runs, support, cal_artifact, threshold, threshold_artifact, commit)
    register_artifacts(support, cal_artifact, threshold, threshold_artifact, commit); write_governance_inputs(commit)
    report = audit_g3(ROOT, scope="real"); write_audit_report(ROOT, report)
    if report.overall != "PASS": raise RuntimeError("pre-G3 audit blocked: " + "; ".join(report.blockers))
    marker = freeze_g3(ROOT, scope="real")
    validate_g3_marker(marker, ROOT, expected_scope="real")
    print(json.dumps({"status": "G3_ACTIVE", "selected_models_sha256": sha256_file(ROOT / SELECTED_REF),
        "calibrator_sha256": cal_artifact["artifact_sha256"], "threshold_sha256": threshold_artifact["artifact_sha256"],
        "threshold": threshold.threshold, "weighted_f1": threshold.validation_metrics["f1"],
        "candidate_thresholds": threshold.candidate_threshold_count, "g3_sha256": sha256_file(marker),
        "test_accessed": False}, sort_keys=True))


if __name__ == "__main__": main()
