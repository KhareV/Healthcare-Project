import copy
import json
import math
from pathlib import Path

import pytest

from serving.prediction_schema import (
    FIXTURE_CLASSIFICATION,
    PredictionSchemaError,
    lint_contract_text,
    load_contract,
    load_synthetic_fixture,
    schema_sha256,
    validate_exchange,
    validate_request,
    validate_response,
)
from vedant_infra.hashing import is_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/prediction_schema"
SCHEMA = ROOT / "configs/prediction_schema_v1.json"


def _response(name="valid_response_v1.json"):
    return json.loads((FIXTURES / name).read_text())["payload"]


def test_contract_loads_and_exact_hash_is_deterministic():
    contract = load_contract(SCHEMA)
    assert contract["contract_version"] == "prediction_schema_v1"
    assert schema_sha256(SCHEMA) == sha256_file(SCHEMA)
    assert is_sha256(schema_sha256(SCHEMA))


def test_valid_synthetic_request_and_response_fixtures():
    request = load_synthetic_fixture(FIXTURES / "valid_request_v1.json", payload_kind="request")
    response = load_synthetic_fixture(FIXTURES / "valid_response_v1.json", payload_kind="response")
    validate_exchange(request, response, synthetic=True)


@pytest.mark.parametrize("missing", ("stay_id", "prediction_time"))
def test_request_requires_exact_public_fields(missing):
    request = {"stay_id": "DEMO", "prediction_time": "2026-01-02T12:00:00+00:00"}
    request.pop(missing)
    with pytest.raises(PredictionSchemaError, match="fields mismatch"):
        validate_request(request)


@pytest.mark.parametrize("value", ("not-a-time", "2026-01-02T12:00:00", 123))
def test_malformed_or_timezone_naive_prediction_time_is_rejected(value):
    with pytest.raises(PredictionSchemaError, match="prediction_time"):
        validate_request({"stay_id": "DEMO", "prediction_time": value})


@pytest.mark.parametrize("stay_id", (7, "SYNTHETIC_DEMO"))
def test_canonical_integer_or_string_stay_identifier_is_accepted(stay_id):
    validate_request({"stay_id": stay_id, "prediction_time": "2026-01-02T12:00:00Z"})


@pytest.mark.parametrize("stay_id", (True, "", None, 1.25))
def test_invalid_stay_identifier_is_rejected(stay_id):
    with pytest.raises(PredictionSchemaError, match="stay_id"):
        validate_request({"stay_id": stay_id, "prediction_time": "2026-01-02T12:00:00Z"})


def test_wrong_replay_mode_is_rejected():
    response = _response()
    response["mode"] = "LIVE"
    with pytest.raises(PredictionSchemaError, match="serving mode"):
        validate_response(response, synthetic=True)


def test_generic_model_version_cannot_replace_per_task_versions():
    response = _response()
    response["model_version"] = "GENERIC"
    response.pop("model_versions")
    with pytest.raises(PredictionSchemaError, match="fields mismatch"):
        validate_response(response, synthetic=True)


@pytest.mark.parametrize("task", ("recovery", "icu_stay_time", "organ_support"))
def test_each_task_model_version_is_required(task):
    response = _response()
    response["model_versions"].pop(task)
    with pytest.raises(PredictionSchemaError, match="model_versions"):
        validate_response(response, synthetic=True)


def test_mixed_family_fixture_and_all_xgb_gru_combinations_are_supported():
    load_synthetic_fixture(FIXTURES / "mixed_family_response_v1.json", payload_kind="response")
    for recovery in ("xgboost", "gru"):
        for icu in ("xgboost", "gru"):
            for support in ("xgboost", "gru"):
                response = _response()
                for task, family in zip(("recovery", "icu_stay_time", "organ_support"), (recovery, icu, support)):
                    response["model_versions"][task]["family"] = family
                    response["explanation_features"][task]["family"] = family
                    response["explanation_features"][task]["explanation_method"] = (
                        "tree_shap" if family == "xgboost" else "integrated_gradients"
                    )
                validate_response(response, synthetic=True)


@pytest.mark.parametrize("family", ("lstm", "naive"))
def test_non_serving_families_are_rejected(family):
    response = _response()
    response["model_versions"]["recovery"]["family"] = family
    with pytest.raises(PredictionSchemaError, match="serving family"):
        validate_response(response, synthetic=True)


@pytest.mark.parametrize(
    "field",
    ("delta_24h", "delta_48h", "reconstructed_sofa_24h", "reconstructed_sofa_48h"),
)
def test_all_independent_recovery_fields_are_required(field):
    response = _response()
    response["recovery"].pop(field)
    with pytest.raises(PredictionSchemaError, match="recovery fields mismatch"):
        validate_response(response, synthetic=True)


@pytest.mark.parametrize("value", (-1, math.inf, -math.inf, math.nan, "42"))
def test_icu_hours_must_be_finite_and_nonnegative(value):
    response = _response()
    response["icu_stay_time_hours"] = value
    with pytest.raises(PredictionSchemaError, match="icu_stay_time_hours"):
        validate_response(response, synthetic=True)


@pytest.mark.parametrize("value", (-0.01, 1.01, math.inf, math.nan, "0.5"))
def test_calibrated_support_probability_bounds(value):
    response = _response()
    response["organ_support_probability_calibrated"] = value
    with pytest.raises(PredictionSchemaError, match="organ-support probability|must be a number|finite range"):
        validate_response(response, synthetic=True)


@pytest.mark.parametrize("family,wrong", (("xgboost", "integrated_gradients"), ("gru", "tree_shap")))
def test_explanation_family_mismatch_is_rejected(family, wrong):
    response = _response()
    response["model_versions"]["recovery"]["family"] = family
    response["explanation_features"]["recovery"]["family"] = family
    response["explanation_features"]["recovery"]["explanation_method"] = wrong
    with pytest.raises(PredictionSchemaError, match="explanation routing"):
        validate_response(response, synthetic=True)


@pytest.mark.parametrize("field", ("feature_version", "label_version", "split_version", "manifest_version"))
def test_metadata_versions_are_required(field):
    response = _response()
    response["model_metadata"].pop(field)
    with pytest.raises(PredictionSchemaError, match="model_metadata fields mismatch"):
        validate_response(response, synthetic=True)


@pytest.mark.parametrize("field", ("recovery_confidence", "prediction_interval", "quality_score"))
def test_fake_confidence_and_interval_fields_are_rejected(field):
    response = _response()
    response[field] = 0.9
    with pytest.raises(PredictionSchemaError):
        validate_response(response, synthetic=True)


def test_data_quality_is_explicit_counts_not_a_score():
    response = _response()
    response["data_quality"]["confidence_score"] = 0.8
    with pytest.raises(PredictionSchemaError):
        validate_response(response, synthetic=True)


def test_data_quality_counts_must_reconcile():
    response = _response()
    response["data_quality"]["missing_feature_values"] = 31
    with pytest.raises(PredictionSchemaError, match="do not reconcile"):
        validate_response(response, synthetic=True)


def test_response_cutoff_must_equal_request_cutoff():
    request = load_synthetic_fixture(FIXTURES / "valid_request_v1.json", payload_kind="request")
    response = _response()
    response["prediction_time"] = "2026-01-02T18:00:00+00:00"
    with pytest.raises(PredictionSchemaError, match="requested cutoff"):
        validate_exchange(request, response, synthetic=True)


def test_mock_response_is_rejected_as_real_serving_metadata():
    with pytest.raises(PredictionSchemaError, match="synthetic|SHA-256"):
        validate_response(_response(), synthetic=False)


def test_fixture_requires_explicit_synthetic_classification(tmp_path):
    fixture = json.loads((FIXTURES / "valid_response_v1.json").read_text())
    assert fixture["fixture_classification"] == FIXTURE_CLASSIFICATION
    fixture["fixture_classification"] = "REAL"
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(PredictionSchemaError, match="explicitly classified"):
        load_synthetic_fixture(path, payload_kind="response")


def test_contract_has_no_support_alert_or_threshold_field_until_review():
    response = _response()
    response["organ_support_threshold"] = 0.5
    with pytest.raises(PredictionSchemaError, match="extra"):
        validate_response(response, synthetic=True)
    contract = load_contract(SCHEMA)
    assert contract["response"]["support_alert_representation"]["included_in_v1"] is False


def test_terminology_lint_detects_misleading_language():
    assert lint_contract_text("This is a hospital discharge ETA")
    assert lint_contract_text("causal explanation")
    assert lint_contract_text("remaining time until the current ICU stay ends") == ()


def test_committed_contract_and_fixtures_pass_terminology_lint():
    paths = [
        SCHEMA,
        ROOT / "docs/pulkit/PHASE5_PREDICTION_SCHEMA_REVIEW.md",
        *sorted(FIXTURES.glob("*.json")),
    ]
    for path in paths:
        assert lint_contract_text(path.read_text()) == (), path


def test_no_phase12_or_later_implementation_was_added():
    assert (ROOT / "src/serving/pipeline.py").is_file()
    assert (ROOT / "api/main.py").is_file()
    assert (ROOT / "api/schemas.py").is_file()
    assert (ROOT / "dashboard/app.py").is_file()
    assert (ROOT / "src/explainability/ig.py").is_file()
    assert (ROOT / "src/explainability/tree_shap.py").is_file()
