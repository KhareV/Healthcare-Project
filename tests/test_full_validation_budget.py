import pytest

from experiments.validation_suite import (
    ValidationSuiteError,
    audit_search_budget,
    resume_action,
    validate_scientific_candidate_manifest,
)
from vedant_infra.hashing import sha256_file
from validation_suite_helpers import candidate_manifest


@pytest.mark.parametrize("count", (29, 31))
def test_scientific_budget_must_be_exactly_30(count):
    with pytest.raises(ValidationSuiteError, match="exactly 30"):
        validate_scientific_candidate_manifest(candidate_manifest(count=count))


def test_retries_do_not_inflate_candidate_count():
    manifest = candidate_manifest()
    identifier = manifest["candidates"][0]["candidate_id"]
    attempts = [
        {"candidate_id": identifier, "status": "failed"},
        {"candidate_id": identifier, "status": "completed"},
    ]
    audit = audit_search_budget(manifest, attempts)
    assert audit["scientific_candidate_count"] == 30
    assert audit["execution_attempts"] == 2
    assert audit["retry_attempts"] == 1


def test_candidate_31_is_rejected():
    manifest = candidate_manifest()
    with pytest.raises(ValidationSuiteError, match="outside frozen"):
        audit_search_budget(
            manifest,
            ({"candidate_id": "candidate-31", "status": "completed"},),
        )


def test_resume_skips_only_hash_verified_complete_candidate(tmp_path):
    candidate = candidate_manifest()["candidates"][0]
    model = tmp_path / "model.bin"
    predictions = tmp_path / "validation_predictions.json"
    model.write_bytes(b"model")
    predictions.write_bytes(b"predictions")
    attempt = {
        "candidate_id": candidate["candidate_id"],
        "status": "completed",
        "config_hash": candidate["config_hash"],
        "model_artifact_ref": "model.bin",
        "model_sha256": sha256_file(model),
        "validation_prediction_ref": "validation_predictions.json",
        "validation_prediction_sha256": sha256_file(predictions),
    }

    assert resume_action(candidate, (attempt,), tmp_path) == "SKIP_COMPLETE"

    model.write_bytes(b"changed")
    with pytest.raises(ValidationSuiteError, match="missing or incompatible"):
        resume_action(candidate, (attempt,), tmp_path)


def test_resume_retries_same_failed_candidate_without_new_identity(tmp_path):
    candidate = candidate_manifest()["candidates"][0]
    failed = {
        "candidate_id": candidate["candidate_id"],
        "status": "failed",
    }

    assert resume_action(candidate, (failed,), tmp_path) == "RETRY_SAME_CANDIDATE"
    assert resume_action(candidate, (), tmp_path) == "RUN_SAME_CANDIDATE"
