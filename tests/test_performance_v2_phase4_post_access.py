"""Performance-v2 Phase 4 focused tests: POST-ACCESS portion. These tests
read only already-saved Phase-4 artifacts -- never trigger fresh-test row
access or model inference."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PHASE4_DIR = ROOT / "artifacts/performance_v2/phase4"
PRED_DIR = PHASE4_DIR / "predictions"
GOV_DIR = ROOT / "artifacts/performance_v2/governance"


def _load(path):
    return json.loads((ROOT / path).read_text())


def _load_jsonl(path):
    rows = []
    with (ROOT / path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# --- Access state history ----------------------------------------------------

def test_access_state_final_run_completed_exactly_once():
    state = _load("artifacts/performance_v2/governance/v2_fresh_test_access_state.json")
    assert state["status"] == "FINAL_V2_RUN_COMPLETED"
    history = state["history"]
    transitions = [(h["from"], h["to"]) for h in history]
    assert transitions == [
        ("SEALED_NOT_ACCESSED", "AUTHORIZED_FOR_ONE_FINAL_RUN"),
        ("AUTHORIZED_FOR_ONE_FINAL_RUN", "FINAL_V2_TEST_ACCESS_CONSUMED"),
        ("FINAL_V2_TEST_ACCESS_CONSUMED", "FINAL_V2_RUN_COMPLETED"),
    ]


def test_second_run_guard_blocked():
    guard = _load("artifacts/performance_v2/governance/v2_second_run_guard_v1.json")
    assert guard["status"] == "BLOCKED_ALREADY_CONSUMED"
    assert guard["second_authorize_attempt_blocked"] is True
    assert guard["model_inference_invoked"] is False
    assert guard["fresh_test_rows_loaded"] is False


# --- Prediction schema / lineage ---------------------------------------------

@pytest.mark.parametrize("task,file,required_fields", [
    ("recovery24", "recovery24_predictions_v1.jsonl", {"subject_id", "stay_id", "prediction_time", "grid_index", "target_delta24", "prediction_delta24", "eligible", "model_sha256", "feature_contract_sha256", "fresh_cohort_freeze_sha256"}),
    ("recovery48", "recovery48_predictions_v1.jsonl", {"subject_id", "stay_id", "prediction_time", "grid_index", "target_delta48", "prediction_delta48", "eligible", "model_sha256", "feature_contract_sha256", "fresh_cohort_freeze_sha256"}),
    ("icu_stay_time", "icu_stay_time_predictions_v1.jsonl", {"subject_id", "stay_id", "prediction_time", "grid_index", "true_remaining_hours", "raw_log_prediction", "prediction_hours", "eligible", "model_sha256", "feature_contract_sha256", "fresh_cohort_freeze_sha256"}),
    ("organ_support", "organ_support_predictions_v1.jsonl", {"subject_id", "stay_id", "prediction_time", "grid_index", "label", "raw_probability", "calibrated_probability", "frozen_threshold", "predicted_alert", "eligible", "model_sha256", "feature_contract_sha256", "fresh_cohort_freeze_sha256"}),
])
def test_prediction_schema_complete(task, file, required_fields):
    rows = _load_jsonl(PRED_DIR / file)
    assert len(rows) > 0
    for row in rows[:50]:
        assert required_fields <= set(row.keys())


def test_prediction_files_hash_bound_in_manifest():
    manifest = _load("artifacts/performance_v2/phase4/predictions/predictions_manifest_v1.json")
    assert manifest["status"] == "FINAL_V2_PREDICTIONS_WRITTEN"
    for task, entry in manifest["files"].items():
        path = ROOT / entry["ref"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_prediction_lineage_matches_frozen_model_and_cohort_hashes():
    manifest = _load("artifacts/performance_v2/phase4/predictions/predictions_manifest_v1.json")
    expected_model_hashes = {
        "recovery24": "b355c26339271f0b84c123eb46cfb61c8df0cc9c9c6fcfcbb86c29047d10ad3c",
        "recovery48": "5dda6bb0830428b44b3a7cf554dc7f6d2adc278c7b218045d1df8ff960f6c701",
        "icu_stay_time": "918c81eef86bd89e91e02bb06c254774375f3f2371c4f3c5200c270dea24f7e6",
        "organ_support": "2fa653a0d2c644d8a1de806146f8e8cba46df87e6bd037868122276b4a440671",
    }
    assert manifest["model_hashes"] == expected_model_hashes
    assert manifest["fresh_cohort_freeze_sha256"] == "f70d4cc7fca69981861c8d679101176529f036579ba1849f742e6c3fc17916ce"


def test_no_v1_test_rows_or_dev_rows_in_predictions():
    rec24 = _load_jsonl(PRED_DIR / "recovery24_predictions_v1.jsonl")
    subject_ids = {row["subject_id"] for row in rec24}
    assert all(sid.startswith("SYN-V2-S-") for sid in subject_ids)
    with (ROOT / "artifacts/splits/synthetic_split_v2.csv").open() as handle:
        v1_subject_ids = {line.split(",")[0] for line in handle.readlines()[1:]}
    assert not (subject_ids & v1_subject_ids)


# --- Metric reproducibility from saved predictions ---------------------------

def test_final_metrics_reproducible_from_saved_predictions():
    from evaluation.metrics import PredictionRecord, evaluate_recovery_horizon

    rows = _load_jsonl(PRED_DIR / "recovery24_predictions_v1.jsonl")
    records = tuple(PredictionRecord(stay_id=r["stay_id"], prediction_time=r["prediction_time"], target=r["target_delta24"], prediction=r["prediction_delta24"], eligible=True) for r in rows)
    recomputed = evaluate_recovery_horizon(records, horizon="24h").metrics["mae"]
    metrics = _load("artifacts/performance_v2/phase4/metrics/final_metrics_v2.json")
    assert abs(recomputed - metrics["recovery24"]["mae"]) < 1e-12


def test_support_calibrated_metrics_reproducible():
    from evaluation.metrics import PredictionRecord, evaluate_organ_support

    rows = _load_jsonl(PRED_DIR / "organ_support_predictions_v1.jsonl")
    records = tuple(PredictionRecord(stay_id=r["stay_id"], prediction_time=r["prediction_time"], target=float(r["label"]), prediction=r["calibrated_probability"], eligible=True) for r in rows)
    recomputed = evaluate_organ_support(records, fixed_threshold=0.39781983118092895, threshold_identifier="support_threshold_v2", probability_type="calibrated").metrics["auprc"]
    metrics = _load("artifacts/performance_v2/phase4/metrics/final_metrics_v2.json")
    assert abs(recomputed - metrics["organ_support_calibrated"]["auprc"]) < 1e-12


# --- Bootstrap reproducibility ------------------------------------------------

def test_bootstrap_config_frozen_values():
    bootstrap = _load("artifacts/performance_v2/phase4/bootstrap/final_bootstrap_v2.json")
    assert bootstrap["config"]["n_bootstrap"] == 2000
    assert bootstrap["config"]["seed"] == 20260924
    assert bootstrap["config"]["resampling_unit"] == "stay_id"


def test_bootstrap_reproducible_deterministically():
    from evaluation.bootstrap import grouped_bootstrap
    from evaluation.metrics import PredictionRecord, evaluate_recovery_horizon

    rows = _load_jsonl(PRED_DIR / "recovery24_predictions_v1.jsonl")
    records = tuple(PredictionRecord(stay_id=r["stay_id"], prediction_time=r["prediction_time"], target=r["target_delta24"], prediction=r["prediction_delta24"], eligible=True) for r in rows)

    def metric_fn(recs):
        return evaluate_recovery_horizon(recs, horizon="24h").metrics["mae"]

    result_a = grouped_bootstrap(records, metric_fn=metric_fn, metric_name="mae", task="recovery", n_bootstrap=50, seed=20260924, horizon="24h")
    result_b = grouped_bootstrap(records, metric_fn=metric_fn, metric_name="mae", task="recovery", n_bootstrap=50, seed=20260924, horizon="24h")
    assert result_a.bootstrap_distribution == result_b.bootstrap_distribution
    assert result_a.ci_lower == result_b.ci_lower
    assert result_a.ci_upper == result_b.ci_upper


def test_bootstrap_replicate_counts_accounted():
    bootstrap = _load("artifacts/performance_v2/phase4/bootstrap/final_bootstrap_v2.json")
    for key in ("mae", "rmse"):
        entry = bootstrap["recovery24"][key]
        assert entry["n_requested_replicates"] == 2000
        assert entry["n_valid_replicates"] + entry["n_invalid_replicates"] == 2000


# --- Evaluation-freeze integrity ---------------------------------------------

def test_v2_final_evaluation_freeze_status_and_flags():
    freeze = _load("artifacts/performance_v2/governance/v2_final_evaluation_freeze_v1.json")
    assert freeze["status"] == "V2_FINAL_EVALUATION_FROZEN"
    assert freeze["fresh_test_accessed"] is True
    assert freeze["successful_scientific_evaluations"] == 1
    assert freeze["post_test_tuning"] is False


def test_v2_final_evaluation_freeze_binds_required_parents():
    freeze = _load("artifacts/performance_v2/governance/v2_final_evaluation_freeze_v1.json")
    required = (
        "v2_model_freeze", "v2_fresh_test_freeze", "final_evaluation_plan", "naive_baselines_v2",
        "access_state_history", "selected_models_v2", "recovery24_model", "recovery48_model", "icu_model",
        "support_model", "isotonic_support_v2", "support_threshold_v2", "predictions_manifest",
        "recovery24_predictions", "recovery48_predictions", "icu_predictions", "support_predictions",
        "final_metrics", "naive_comparison", "calibration_evidence", "error_analysis",
        "generalization_comparison", "final_bootstrap", "second_run_guard",
    )
    for key in required:
        assert key in freeze["parents"], key


# --- No post-test tuning: models/artifacts unchanged since pre-access commit --

def test_frozen_model_hashes_unchanged_since_pre_access():
    from vedant_infra.hashing import sha256_file

    expected = {
        "artifacts/performance_v2/phase3/models/recovery24_xgb_v2.json": "b355c26339271f0b84c123eb46cfb61c8df0cc9c9c6fcfcbb86c29047d10ad3c",
        "artifacts/performance_v2/phase3/models/recovery48_xgb_v2.json": "5dda6bb0830428b44b3a7cf554dc7f6d2adc278c7b218045d1df8ff960f6c701",
        "artifacts/performance_v2/phase3/models/icu_time_xgb_v2.json": "918c81eef86bd89e91e02bb06c254774375f3f2371c4f3c5200c270dea24f7e6",
        "artifacts/performance_v2/phase3/models/organ_support_xgb_v2.json": "2fa653a0d2c644d8a1de806146f8e8cba46df87e6bd037868122276b4a440671",
        "artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json": "7d14ac249c3ee6b7dcb04110b6aa7ba274b27c73f3ced6bb7aae614d68a9d77c",
        "artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json": "1ef311fd8a81a3e392db8a4cb7668eb4f404b0d21d818a94fc5ce308fdfe0b4f",
    }
    for path, expected_hash in expected.items():
        assert sha256_file(ROOT / path) == expected_hash
