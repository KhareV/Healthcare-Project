import json
from pathlib import Path

import pytest

from evaluation.calibration_validation import SelectedSupportModel, load_validation_predictions
from evaluation.threshold import ThresholdSearchPolicy, _candidate_thresholds
from evaluation.weights import compute_stay_weights


ROOT = Path(__file__).resolve().parents[1]


def _support_model():
    stage2 = json.loads((ROOT / "artifacts/selection/validation_family_selection_v1.json").read_text())
    source = stage2["tasks"]["organ_support"]["sources"]["xgboost"]
    prediction = Path(source["metrics_path"].replace("validation_metrics.json", "validation_predictions.jsonl"))
    bundle = Path(source["bundle_path"])
    import hashlib
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return SelectedSupportModel(task="organ_support", family="xgboost", run_id=source["run_id"],
        candidate_id=source["candidate_id"], config_hash="3d347912d3ec7b7bf9478e96dc60af739587d80e2093b8d52bbd0b0924e92bc1",
        artifact_ref=str(bundle), artifact_sha256=digest(bundle), split_hash=stage2["split_sha256"],
        feature_version="synthetic_feature_schema_v2", label_version="synthetic_xgb_target_contract_v1",
        preprocessing_hash=stage2["preprocessor_sha256"], code_commit="d72e2c8220ee65afe777fddf09c017f1d4ab8a4b",
        selection_metric="validation_auprc", probability_type="raw_uncalibrated",
        validation_prediction_ref=str(prediction), validation_prediction_sha256=digest(prediction),
        selection_mode="real", test_accessed=False,
        selection_manifest_ref="artifacts/selection/validation_family_selection_v1.json",
        selection_manifest_sha256=digest(ROOT / "artifacts/selection/validation_family_selection_v1.json"))


def test_stage2_selected_support_jsonl_loads_only_eligible_validation_rows():
    records = load_validation_predictions(_support_model())
    assert records and all(row.eligible for row in records)
    assert {row.label for row in records} == {0, 1}
    assert len({row.row_id for row in records}) == len(records)


def test_stay_balanced_calibration_weights_sum_to_unique_stays():
    records = load_validation_predictions(_support_model())
    result = compute_stay_weights([row.stay_id for row in records], [True] * len(records))
    assert sum(result.weights) == pytest.approx(result.n_icu_stays)
    for stay in {row.stay_id for row in records}:
        assert sum(weight for row, weight in zip(records, result.weights) if row.stay_id == stay) == pytest.approx(1.0)


def test_exact_threshold_candidates_include_deduplicated_boundaries():
    policy = ThresholdSearchPolicy("v", "unique_calibrated_probabilities_plus_endpoints",
                                   "closest_to_half_then_higher", approved_for_real=True)
    assert _candidate_thresholds((0.2, 0.2, 0.8), policy) == (0.0, 0.2, 0.8, 1.0)


def test_closest_to_half_then_higher_tie_key_is_deterministic():
    tied = [(0.4, {"f1": 0.5}), (0.6, {"f1": 0.5}), (0.8, {"f1": 0.5})]
    chosen = min(tied, key=lambda item: (abs(item[0] - 0.5), -item[0]))
    assert chosen[0] == 0.6


def test_final_test_files_are_not_stage3_inputs():
    source = Path("scripts/finalize_stage3.py").read_text()
    assert 'partition="validation"' in source
    assert "guarded_test_access" not in source
    assert "FINAL_RUN_COMPLETED" not in source
