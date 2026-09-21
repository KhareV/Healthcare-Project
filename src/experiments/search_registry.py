"""Search-attempt recording in the existing experiment registry."""

import csv
from pathlib import Path

from experiments.search_governance import GovernanceError
from vedant_infra.registry import make_registry_record, upsert_registry_record


ATTEMPT_DETAILS = {
    "PLANNED",
    "RUNNING",
    "COMPLETE",
    "FAILED_SOFTWARE",
    "FAILED_RESOURCE",
    "FAILED_SCIENTIFIC",
    "INVALID",
    "CANCELLED",
}


def _registry_status(detail):
    if detail == "COMPLETE":
        return "completed"
    if detail == "RUNNING":
        return "running"
    if detail == "PLANNED":
        return "planned"
    if detail == "CANCELLED":
        return "aborted"
    return "failed"


def _rows(path):
    registry = Path(path)
    if not registry.exists():
        return []
    with registry.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def record_search_attempt(
    registry_path,
    *,
    manifest,
    candidates,
    candidate_id,
    run_id,
    attempt_number,
    attempt_status_detail,
    timestamp_utc,
    seed,
    retry_of_run_id="",
    artifact_ref="",
    artifact_hash="",
    metrics_ref="",
    metrics_hash="",
    config_ref="",
    run_type="synthetic",
    preprocessor_ref="",
    preprocessor_sha256="",
    derived_feature_hash="",
    environment_ref="",
    environment_sha256="",
    notes="",
    max_attempts=3,
):
    if attempt_status_detail not in ATTEMPT_DETAILS:
        raise GovernanceError("unsupported search attempt status")
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if candidate_id not in by_id:
        raise GovernanceError("candidate is outside the frozen exact-30 manifest")
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int) or attempt_number < 1:
        raise GovernanceError("attempt_number must be a positive integer")
    if attempt_number > max_attempts:
        raise GovernanceError("maximum search-attempt policy exceeded")
    existing = _rows(registry_path)
    if any(row.get("run_id") == run_id for row in existing):
        raise GovernanceError("run_id already exists")
    same_candidate = [row for row in existing if row.get("candidate_id") == candidate_id]
    if attempt_number != len(same_candidate) + 1:
        raise GovernanceError("attempt_number must be sequential for one candidate")
    if attempt_number > 1:
        prior = {row.get("run_id"): row for row in same_candidate}
        if retry_of_run_id not in prior or prior[retry_of_run_id].get("status") != "failed":
            raise GovernanceError("retry must reference a failed attempt for the same candidate")
    elif retry_of_run_id:
        raise GovernanceError("first attempt cannot be a retry")
    candidate = by_id[candidate_id]
    if run_type not in ("synthetic", "scientific"):
        raise GovernanceError("unsupported search registry run type")
    if run_type == "scientific" and manifest.get("run_type") != "scientific":
        raise GovernanceError("scientific registry row requires a scientific manifest")
    if run_type == "scientific" and candidate.get("model_seed") != seed:
        raise GovernanceError("scientific retry/model seed differs from frozen candidate")
    record = make_registry_record(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        task=manifest["task"],
        model_family=manifest["family"],
        seed=str(seed),
        code_commit=manifest["code_commit"],
        config_ref=config_ref or manifest["search_space_ref"],
        config_hash=candidate["config_hash"],
        search_space_hash=manifest["search_space_hash"],
        split_hash=manifest["split_hash"],
        feature_version=manifest["feature_version"],
        label_version=manifest["label_version"],
        model_artifact_ref=artifact_ref,
        model_sha256=artifact_hash,
        metrics_ref=metrics_ref,
        metrics_sha256=metrics_hash,
        status=_registry_status(attempt_status_detail),
        parent_run_id=retry_of_run_id,
        notes=notes or (
            "RUN_TYPE=SCIENTIFIC_SEARCH; VALIDATION_ONLY; TEST_ACCESS=false"
            if run_type == "scientific" else
            "RUN_TYPE=SYNTHETIC_DRY_RUN; NOT SCIENTIFIC SEARCH; TEST_ACCESS=false"
        ),
        candidate_id=candidate_id,
        search_version=manifest["search_version"],
        candidate_list_hash=manifest["candidate_list_hash"],
        attempt_number=str(attempt_number),
        retry_of_run_id=retry_of_run_id,
        attempt_status_detail=attempt_status_detail,
        validation_objective=manifest["validation_objective"],
        run_type=run_type,
        preprocessor_ref=preprocessor_ref,
        preprocessor_sha256=preprocessor_sha256,
        derived_feature_hash=derived_feature_hash,
        environment_ref=environment_ref,
        environment_sha256=environment_sha256,
        dirty_worktree="false" if run_type == "scientific" else "",
    )
    upsert_registry_record(registry_path, record)
    return record
