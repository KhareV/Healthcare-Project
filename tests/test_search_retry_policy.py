import tempfile
import unittest
from pathlib import Path

from experiments.search_governance import GovernanceError
from experiments.search_registry import record_search_attempt
from search_helpers import load_manifest, manifest_paths


class SearchRetryPolicyTests(unittest.TestCase):
    def test_config_31_is_rejected_and_failed_candidate_is_not_replaced(self):
        manifest, _, candidates = load_manifest(manifest_paths()[0])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(GovernanceError):
                record_search_attempt(
                    Path(directory) / "registry.csv",
                    manifest=manifest,
                    candidates=candidates,
                    candidate_id="candidate_31",
                    run_id="forbidden",
                    attempt_number=1,
                    attempt_status_detail="PLANNED",
                    timestamp_utc="2026-09-18T00:00:00Z",
                    seed=1,
                )

    def test_retry_must_reference_failed_same_candidate_attempt(self):
        manifest, _, candidates = load_manifest(manifest_paths()[0])
        candidate_id = candidates[0]["candidate_id"]
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "registry.csv"
            record_search_attempt(
                registry,
                manifest=manifest,
                candidates=candidates,
                candidate_id=candidate_id,
                run_id="running-1",
                attempt_number=1,
                attempt_status_detail="RUNNING",
                timestamp_utc="2026-09-18T00:00:00Z",
                seed=1,
            )
            with self.assertRaises(GovernanceError):
                record_search_attempt(
                    registry,
                    manifest=manifest,
                    candidates=candidates,
                    candidate_id=candidate_id,
                    run_id="bad-retry",
                    attempt_number=2,
                    retry_of_run_id="running-1",
                    attempt_status_detail="RUNNING",
                    timestamp_utc="2026-09-18T00:01:00Z",
                    seed=1,
                )


if __name__ == "__main__":
    unittest.main()
