import tempfile
import unittest
from pathlib import Path

from experiments.audit import audit_search, read_registry_rows
from experiments.search_registry import record_search_attempt
from search_helpers import load_manifest, manifest_paths
from vedant_infra.registry import validate_registry


class SearchRegistryTests(unittest.TestCase):
    def test_retry_and_early_stop_history_do_not_inflate_candidate_count(self):
        manifest, _, candidates = load_manifest(manifest_paths()[0])
        candidate_id = candidates[6]["candidate_id"]
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "registry.csv"
            record_search_attempt(
                registry,
                manifest=manifest,
                candidates=candidates,
                candidate_id=candidate_id,
                run_id="attempt-1",
                attempt_number=1,
                attempt_status_detail="FAILED_SOFTWARE",
                timestamp_utc="2026-09-18T00:00:00Z",
                seed=1,
            )
            record_search_attempt(
                registry,
                manifest=manifest,
                candidates=candidates,
                candidate_id=candidate_id,
                run_id="attempt-2",
                attempt_number=2,
                retry_of_run_id="attempt-1",
                attempt_status_detail="RUNNING",
                timestamp_utc="2026-09-18T00:01:00Z",
                seed=1,
            )
            summary = audit_search(manifest, candidates, read_registry_rows(registry))
            self.assertEqual(validate_registry(registry), 2)
            self.assertEqual(summary["attempted_candidates"], 1)
            self.assertEqual(summary["execution_attempts"], 2)
            self.assertEqual(summary["retry_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
