from registry_helpers import run, write_runs
from vedant_infra.registry import validate_registry


def test_failed_attempt_is_retained_and_retry_does_not_inflate_candidate_count(tmp_path):
    path = tmp_path / "registry.csv"
    failed = run(
        "attempt-A", status="failed", finalized="true", candidate_id="candidate-12",
        attempt_number="1", attempt_status_detail="FAILED_SOFTWARE",
        model_artifact_ref="", model_sha256="", metrics_ref="",
    )
    retry = run(
        "attempt-B", candidate_id="candidate-12", attempt_number="2",
        retry_of_run_id="attempt-A", parent_run_id="attempt-A",
        attempt_status_detail="COMPLETE",
    )
    write_runs(path, (failed, retry))
    assert validate_registry(path) == 2
    assert len({failed["candidate_id"], retry["candidate_id"]}) == 1
