import csv
import tempfile
import unittest
from pathlib import Path

import vedant_infra
from vedant_infra.governance import G3FreezeError, assert_g3_freeze
from vedant_infra.hashing import is_sha256, sha256_bytes, sha256_file
from vedant_infra.registry import (
    REGISTRY_COLUMNS,
    RegistryValidationError,
    make_registry_record,
    validate_registry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPOSITORY_ROOT / "experiments" / "registry.csv"
G3_MARKER_PATH = REPOSITORY_ROOT / "artifacts" / "governance" / "g3_freeze.json"


class HashingTests(unittest.TestCase):
    def test_identical_content_has_stable_sha256(self):
        content = b"phase-0-deterministic-content\n"
        self.assertEqual(sha256_bytes(content), sha256_bytes(content))
        self.assertTrue(is_sha256(sha256_bytes(content)))

    def test_changed_content_changes_sha256(self):
        self.assertNotEqual(sha256_bytes(b"before"), sha256_bytes(b"after"))

    def test_file_hash_matches_exact_bytes(self):
        content = b"exact\r\nbytes\x00"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(content)
            self.assertEqual(sha256_file(path), sha256_bytes(content))


class RegistryTests(unittest.TestCase):
    def _write_registry(self, path, records):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
            writer.writeheader()
            writer.writerows(records)

    def test_repository_registry_preserves_smoke_baseline_and_authorized_phase12_search(self):
        self.assertEqual(validate_registry(REGISTRY_PATH), 97)
        with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_run = {row["run_id"]: row for row in rows}
        smoke_ids = {
                "phase4_synthetic_smoke_v1",
                "phase5_synthetic_recovery_gru_smoke_v1",
                "phase6_synthetic_icu_time_gru_smoke_v1",
                "phase7_synthetic_support_gru_smoke_v1",
                "phase8_synthetic_recovery_lstm_smoke_v1",
                "phase8_synthetic_icu_time_lstm_smoke_v1",
                "phase8_synthetic_support_lstm_smoke_v1",
            }
        self.assertEqual(smoke_ids, set(by_run) - {run_id for run_id in by_run if run_id.startswith("phase12-xgb-")})
        phase12 = [row for row in rows if row["run_id"].startswith("phase12-xgb-")]
        self.assertEqual(len(phase12), 90)
        self.assertTrue(all(row["run_type"] == "scientific" for row in phase12))
        self.assertTrue(all(row["search_version"] == "synthetic_xgb_phase12_validation_search_v1" for row in phase12))
        self.assertTrue(all(row["status"] == "completed" for row in phase12))
        self.assertEqual(by_run["phase4_synthetic_smoke_v1"]["model_family"], "smoke_test")
        self.assertIn("SYNTHETIC SMOKE TEST ONLY", by_run["phase4_synthetic_smoke_v1"]["notes"])
        recovery = by_run["phase5_synthetic_recovery_gru_smoke_v1"]
        self.assertEqual(recovery["model_family"], "gru")
        self.assertEqual(recovery["task"], "recovery")
        self.assertIn("NOT SEARCH", recovery["notes"])
        self.assertIn("NOT SCIENTIFIC PERFORMANCE", recovery["notes"])
        icu_time = by_run["phase6_synthetic_icu_time_gru_smoke_v1"]
        self.assertEqual(icu_time["model_family"], "gru")
        self.assertEqual(icu_time["task"], "icu_stay_time")
        self.assertIn("NOT SEARCH", icu_time["notes"])
        self.assertIn("NOT SCIENTIFIC PERFORMANCE", icu_time["notes"])
        support = by_run["phase7_synthetic_support_gru_smoke_v1"]
        self.assertEqual(support["model_family"], "gru")
        self.assertEqual(support["task"], "organ_support")
        self.assertIn("UNCALIBRATED", support["notes"])
        self.assertIn("NO THRESHOLD", support["notes"])
        self.assertIn("NOT SEARCH", support["notes"])
        expected_lstm_parents = {
            "phase8_synthetic_recovery_lstm_smoke_v1": "phase5_synthetic_recovery_gru_smoke_v1",
            "phase8_synthetic_icu_time_lstm_smoke_v1": "phase6_synthetic_icu_time_gru_smoke_v1",
            "phase8_synthetic_support_lstm_smoke_v1": "phase7_synthetic_support_gru_smoke_v1",
        }
        for run_id, parent_run_id in expected_lstm_parents.items():
            row = by_run[run_id]
            self.assertEqual(row["model_family"], "lstm")
            self.assertEqual(row["parent_run_id"], parent_run_id)
            self.assertIn("SENSITIVITY_SMOKE", row["notes"])
            self.assertIn("NOT SCIENTIFIC", row["notes"])
            self.assertIn("NOT SERVING CANDIDATE", row["notes"])

    def test_completed_record_with_lineage_is_valid(self):
        digest = "a" * 64
        record = make_registry_record(
            run_id="test-run-only",
            timestamp_utc="2026-09-17T00:00:00Z",
            task="synthetic-test-task",
            model_family="synthetic-test-family",
            seed="test-only-not-project-seed",
            code_commit="test-commit",
            config_ref="configs/test-only.yaml",
            config_hash=digest,
            split_hash=digest,
            feature_version="test-only",
            label_version="test-only",
            model_artifact_ref="artifacts/test-only.bin",
            model_sha256=digest,
            metrics_ref="artifacts/test-only-metrics.json",
            status="completed",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.csv"
            self._write_registry(path, [record])
            self.assertEqual(validate_registry(path), 1)

    def test_completed_record_missing_metadata_is_rejected(self):
        record = make_registry_record(run_id="incomplete", status="completed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.csv"
            self._write_registry(path, [record])
            with self.assertRaises(RegistryValidationError):
                validate_registry(path)

    def test_absolute_artifact_reference_is_rejected(self):
        record = make_registry_record(
            run_id="planned-test",
            model_artifact_ref="/machine-specific/artifact.bin",
            status="planned",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.csv"
            self._write_registry(path, [record])
            with self.assertRaises(RegistryValidationError):
                validate_registry(path)


class GovernanceTests(unittest.TestCase):
    def test_no_g3_freeze_marker_exists(self):
        self.assertFalse(G3_MARKER_PATH.exists())
        self.assertFalse(vedant_infra.is_g3_freeze_valid(G3_MARKER_PATH))

    def test_absent_g3_marker_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "absent.json"
            with self.assertRaises(G3FreezeError):
                assert_g3_freeze(marker)

    def test_incomplete_g3_marker_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "g3_freeze.json"
            marker.write_text('{"gate": "G3", "status": "frozen"}', encoding="utf-8")
            with self.assertRaises(G3FreezeError):
                assert_g3_freeze(marker)


class ScopeTests(unittest.TestCase):
    def test_package_import_surface_is_available(self):
        self.assertTrue(callable(vedant_infra.sha256_file))
        self.assertTrue(callable(vedant_infra.validate_registry))

    def test_no_fabricated_later_phase_artifacts_exist(self):
        prohibited = (
            REPOSITORY_ROOT / "artifacts" / "splits" / "split_v1.csv",
            REPOSITORY_ROOT / "artifacts" / "models" / "selected_models_v1.json",
            REPOSITORY_ROOT / "configs" / "vedant" / "timestamp_spec_v1.yaml",
            G3_MARKER_PATH,
        )
        self.assertEqual([path for path in prohibited if path.exists()], [])


if __name__ == "__main__":
    unittest.main()
