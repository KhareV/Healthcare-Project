"""Performance-v2 Phase 4 focused tests: PRE-ACCESS portion only. Never
reads fresh-test row content; only structural/hash/load checks plus the
frozen pre-access plan/baseline artifacts."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PHASE3_DIR = ROOT / "artifacts/performance_v2/phase3"
PHASE4_DIR = ROOT / "artifacts/performance_v2/phase4"
GOV_DIR = ROOT / "artifacts/performance_v2/governance"


def _load(path):
    return json.loads((ROOT / path).read_text())


# --- Frozen upstream hashes (authoritative task contract) -------------------

EXPECTED_HASHES = {
    "artifacts/performance_v2/phase3/selected_models_v2.json": "6fed05f3c71ce648658b2f864e69fa9f80f3ec8538f53a4e370fa8a4eaf176a8",
    "artifacts/performance_v2/governance/v2_model_freeze_v1.json": "f5feb29cab2e25d6e8bff860af52ac505f40951578aa73547ec420ba8fcda570",
    "artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json": "1ef311fd8a81a3e392db8a4cb7668eb4f404b0d21d818a94fc5ce308fdfe0b4f",
    "artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json": "7d14ac249c3ee6b7dcb04110b6aa7ba274b27c73f3ced6bb7aae614d68a9d77c",
    "artifacts/performance_v2/phase3/fresh_test_cohort_spec_v1.json": "b25789da0af513e71e9ca03040f305cae4e09b52b8ea1a2fee1abab916219377",
    "artifacts/performance_v2/governance/v2_fresh_test_freeze_v1.json": "f70d4cc7fca69981861c8d679101176529f036579ba1849f742e6c3fc17916ce",
    "artifacts/performance_v2/governance/generator_equivalence_audit_v1.json": "bd95e38f16eba014ef881864005ce3936a91518204e9a502080f6b74458578f8",
}

MODEL_HASHES = {
    "artifacts/performance_v2/phase3/models/recovery24_xgb_v2.json": "b355c26339271f0b84c123eb46cfb61c8df0cc9c9c6fcfcbb86c29047d10ad3c",
    "artifacts/performance_v2/phase3/models/recovery48_xgb_v2.json": "5dda6bb0830428b44b3a7cf554dc7f6d2adc278c7b218045d1df8ff960f6c701",
    "artifacts/performance_v2/phase3/models/icu_time_xgb_v2.json": "918c81eef86bd89e91e02bb06c254774375f3f2371c4f3c5200c270dea24f7e6",
    "artifacts/performance_v2/phase3/models/organ_support_xgb_v2.json": "2fa653a0d2c644d8a1de806146f8e8cba46df87e6bd037868122276b4a440671",
}


@pytest.mark.parametrize("path,expected", list(EXPECTED_HASHES.items()) + list(MODEL_HASHES.items()))
def test_frozen_artifact_hash_matches(path, expected):
    from vedant_infra.hashing import sha256_file

    assert sha256_file(ROOT / path) == expected


def test_recovery48_b_min_correction_honored():
    sm = _load("artifacts/performance_v2/phase3/selected_models_v2.json")
    assert sm["tasks"]["recovery48"]["feature_variant"] == "B_MIN"
    assert sm["tasks"]["recovery24"]["feature_variant"] == "B_MIN"
    assert sm["tasks"]["icu_stay_time"]["feature_variant"] == "B_PLUS_F"
    assert sm["tasks"]["organ_support"]["feature_variant"] == "B_FULL"


def test_model_freeze_and_fresh_test_freeze_status():
    assert _load("artifacts/performance_v2/governance/v2_model_freeze_v1.json")["status"] == "V2_MODEL_FREEZE_ACTIVE"
    assert _load("artifacts/performance_v2/governance/v2_fresh_test_freeze_v1.json")["status"] == "V2_FRESH_TEST_FROZEN"
    assert _load("artifacts/performance_v2/governance/generator_equivalence_audit_v1.json")["status"] == "GENERATOR_EQUIVALENCE_AUDIT_PASS"


# --- Model / preprocessing / calibrator / threshold loading -----------------

def test_all_four_models_load():
    from performance_v2.final_evaluation import load_xgb_model

    for task in ("recovery24", "recovery48", "icu_stay_time", "organ_support"):
        model = load_xgb_model(ROOT, task)
        assert model is not None


def test_feature_dimension_reconstruction_matches_frozen_models():
    from performance_v2.final_evaluation import verify_feature_dimension

    for task in ("recovery24", "recovery48", "icu_stay_time", "organ_support"):
        result = verify_feature_dimension(ROOT, task)
        assert result["total"] == result["model_num_feature"]


def test_preprocessing_artifacts_load():
    for variant in ("B_MIN", "B_FULL", "B_PLUS_F"):
        payload = _load(f"artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_{variant}.json")
        assert "mean" in payload and "std" in payload and payload["feature_names"]


def test_calibrator_and_threshold_load():
    cal = _load("artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json")
    knots = cal["calibration_knots"]
    assert len(knots["x_thresholds"]) == len(knots["y_thresholds"]) > 0
    thr = _load("artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json")
    assert abs(thr["threshold_value"] - 0.39781983118092895) < 1e-12


# --- Bulk-cohort hash validation ---------------------------------------------

def test_fresh_cohort_bulk_files_hash_match_pipeline_manifests():
    from vedant_infra.hashing import sha256_file

    fresh_dir = PHASE3_DIR / "fresh_test_cohort"
    manifests = [
        fresh_dir / "raw/synthetic_dataset_manifest_v1.json",
        fresh_dir / "cohort/synthetic_cohort_manifest_v2.json",
        fresh_dir / "timeline/synthetic_processed_manifest_v1.json",
        fresh_dir / "features/synthetic_feature_input_manifest_v1.json",
        fresh_dir / "pre_split/phase9_pre_split_manifest_v1.json",
    ]
    n_checked = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        for artifact in manifest["artifacts"]:
            relative = artifact.get("path") or artifact.get("repository_relative_path")
            full = ROOT / relative
            assert full.is_file(), relative
            assert sha256_file(full) == artifact["sha256"], relative
            n_checked += 1
    assert n_checked >= 15


# --- Access state -------------------------------------------------------------

def test_access_state_sealed_not_accessed_and_never_consumed_at_preflight_time():
    """This pre-access suite documents the state AS OF the Phase-4A
    pre-access commit. The live access state has since legitimately (and
    permanently) progressed past SEALED_NOT_ACCESSED via the one governed
    Phase-4 run -- so this checks the frozen, immutable preflight report's
    recorded result for that moment, not the live mutable state."""

    report = _load("artifacts/performance_v2/phase4/preflight_report_v1.json")
    items = {item["name"]: item for item in report["items"]}
    assert items["access_state_sealed_not_accessed"]["status"] == "PASS"


def test_no_prior_final_outputs_exist_at_preflight_time():
    report = _load("artifacts/performance_v2/phase4/preflight_report_v1.json")
    items = {item["name"]: item for item in report["items"]}
    assert items["no_prior_final_v2_prediction_artifact"]["status"] == "PASS"
    assert items["no_prior_final_v2_metric_artifact"]["status"] == "PASS"


# --- Evaluator / bootstrap initialization ------------------------------------

def test_evaluation_modules_initialize():
    import evaluation.bootstrap  # noqa: F401
    import evaluation.metrics  # noqa: F401
    import evaluation.weighted_stats  # noqa: F401
    import evaluation.weights  # noqa: F401


def test_bootstrap_dummy_fixture():
    from evaluation.bootstrap import grouped_bootstrap
    from evaluation.metrics import PredictionRecord, evaluate_recovery_horizon

    dummy = tuple(
        PredictionRecord(stay_id=f"DUMMY-{i}", prediction_time=f"2026-01-01T00:0{i}:00Z", target=float(i % 3), prediction=float(i % 3) + 0.1, eligible=True)
        for i in range(12)
    )

    def metric_fn(rows):
        return evaluate_recovery_horizon(rows, horizon="24h").metrics["mae"]

    result = grouped_bootstrap(dummy, metric_fn=metric_fn, metric_name="mae", task="recovery", n_bootstrap=25, seed=1, horizon="24h")
    assert result.n_valid_replicates > 0


def test_output_directories_writable():
    for sub in ("predictions", "metrics", "bootstrap"):
        d = PHASE4_DIR / sub
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".pytest_write_probe"
        probe.write_text("ok")
        probe.unlink()


# --- One-run guard design (state machine cannot re-require SEALED_NOT_ACCESSED after consumption) --

def test_consume_access_does_not_re_check_sealed_state():
    import inspect

    import performance_v2.fresh_test_access as fta

    source = inspect.getsource(fta.consume_access)
    assert "SEALED_NOT_ACCESSED" not in source


def test_second_authorize_call_blocked_after_consumption_design():
    """Structural check: authorize_one_final_run only accepts SEALED_NOT_ACCESSED
    as the expected_current state, so a second call after any transition away
    from it is guaranteed to raise -- verified without touching real state."""

    from performance_v2.fresh_test_access import (
        AUTHORIZED_FOR_ONE_FINAL_RUN,
        FINAL_V2_RUN_COMPLETED,
        FINAL_V2_TEST_ACCESS_CONSUMED,
        SEALED_NOT_ACCESSED,
        _ALLOWED_TRANSITIONS,
    )

    assert _ALLOWED_TRANSITIONS[SEALED_NOT_ACCESSED] == AUTHORIZED_FOR_ONE_FINAL_RUN
    assert _ALLOWED_TRANSITIONS[AUTHORIZED_FOR_ONE_FINAL_RUN] == FINAL_V2_TEST_ACCESS_CONSUMED
    assert _ALLOWED_TRANSITIONS[FINAL_V2_TEST_ACCESS_CONSUMED] == FINAL_V2_RUN_COMPLETED
    assert FINAL_V2_RUN_COMPLETED not in _ALLOWED_TRANSITIONS


# --- Evaluation plan / naive baselines (Parts 1 and 3) -----------------------

def test_final_evaluation_plan_frozen():
    plan = _load("artifacts/performance_v2/phase4/final_evaluation_plan_v1.json")
    assert plan["status"] == "V2_FINAL_EVALUATION_PLAN_FROZEN_PRE_ACCESS"
    assert plan["bootstrap_config"]["n_bootstrap"] == 2000
    assert plan["bootstrap_config"]["seed"] == 20260924
    assert plan["bootstrap_config"]["resampling_unit"] == "stay_id"
    assert plan["model_contract"]["recovery48"]["feature_variant"] == "B_MIN"


def test_naive_baselines_frozen_dev_only():
    baselines = _load("artifacts/performance_v2/phase4/naive_baselines_v2.json")
    assert baselines["status"] == "V2_NAIVE_BASELINES_FROZEN_PRE_ACCESS"
    assert baselines["recovery24"]["constant_prediction"] == 0.0
    assert baselines["recovery48"]["constant_prediction"] == 0.0
    assert baselines["icu_stay_time"]["constant_prediction_hours"] > 0
    assert 0.0 < baselines["organ_support"]["constant_probability"] < 1.0


def test_prespecified_slices_frozen_with_dev_derived_boundaries():
    plan = _load("artifacts/performance_v2/phase4/final_evaluation_plan_v1.json")
    slices = plan["prespecified_error_analysis_slices"]
    assert slices["status"] == "PREDECLARED_BEFORE_ACCESS"
    assert slices["icu_true_remaining_time_quartile"]["dev_derived_quartile_boundaries"]["q50"] > 0
    assert set(slices["cardiac_subtype"]["categories"]) == {"SYNTHETIC_ARRHYTHMIA", "SYNTHETIC_HF", "SYNTHETIC_ISCHEMIC"}


def test_preflight_report_pass():
    report = _load("artifacts/performance_v2/phase4/preflight_report_v1.json")
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["n_blocked"] == 0
    assert report["fresh_test_rows_read"] is False


# --- Fresh-row enrichment structural check (dummy fixture, not real fresh data) --

def test_enrich_derives_required_context_fields_on_a_dummy_row():
    """performance_v2.context_v2's Group-B features (needed by every feature
    variant, since even B_MIN uses elapsed_episode_hours_at_t) require
    elapsed_episode_hours_at_t/cutoff_index/hours_since_first_eligible_cutoff
    on the row -- fields the raw pre-split package does not carry. This
    checks the accessor's enrichment step derives them correctly, using a
    synthetic dummy row (no fresh-cohort content) so it is safe to run
    before access."""

    from performance_v2.fresh_test_access import _enrich

    dummy_row = {"stay_id": "DUMMY-STAY-1", "prediction_time": "2026-01-02T06:00:00Z", "grid_index": 3}
    dummy_statics = {
        "DUMMY-STAY-1": {
            "intime": "2026-01-01T00:00:00Z",
            "age_years": 55,
            "sex_category": "SYNTHETIC_MALE",
            "cardiac_condition_group": "SYNTHETIC_HF",
        }
    }
    (enriched,) = _enrich((dummy_row,), dummy_statics)
    assert enriched["elapsed_episode_hours_at_t"] == 30.0
    assert enriched["cutoff_index"] == 3
    assert enriched["hours_since_first_eligible_cutoff"] == 6.0
    assert enriched["age_years"] == 55
    assert enriched["cardiac_condition_group"] == "SYNTHETIC_HF"
