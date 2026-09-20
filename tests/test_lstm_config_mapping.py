import json
import unittest
from pathlib import Path

from training.lstm_comparison import (
    LSTMComparisonConfigError,
    MAPPED_TRAINING_FIELDS,
    assert_no_lstm_search_space,
    derive_lstm_comparison_config,
    load_and_validate_comparison,
)
from vedant_infra.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = tuple(sorted((ROOT / "configs/comparisons").glob("lstm_*_synthetic_v1.json")))


class LSTMConfigMappingTests(unittest.TestCase):
    def test_exactly_three_smoke_comparisons_validate_against_source(self):
        self.assertEqual(len(COMPARISONS), 3)
        for path in COMPARISONS:
            comparison, source = load_and_validate_comparison(path)
            self.assertEqual(comparison["model"], source["model"])
            self.assertEqual(
                comparison["mapped_training"],
                {field: source[field] for field in MAPPED_TRAINING_FIELDS},
            )
            self.assertEqual(comparison["dataset_sha256"], sha256_file(source["fixture_path"]))
            self.assertEqual(comparison["split_sha256"], sha256_file(source["synthetic_split_path"]))
            self.assertEqual(
                comparison["source_gru_checkpoint_hash"],
                sha256_file(comparison["source_gru_checkpoint_ref"]),
            )

    def test_derivation_changes_family_and_cell_not_meaningful_settings(self):
        source_path = ROOT / "configs/smoke/gru_recovery_phase5_v1.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        comparison = derive_lstm_comparison_config(
            source,
            source_gru_config_ref=source_path.relative_to(ROOT).as_posix(),
            source_gru_config_hash=sha256_file(source_path),
            source_gru_checkpoint_ref="test-only-source-checkpoint.pt",
            source_gru_checkpoint_hash="a" * 64,
            lstm_run_id="test-only-derived-lstm",
            output_directory="artifacts/test-only-derived-lstm",
        )
        self.assertEqual(comparison["model_family"], "lstm")
        self.assertEqual(comparison["recurrent_cell"], "lstm")
        self.assertEqual(comparison["model"], source["model"])
        self.assertEqual(
            comparison["mapped_training"],
            {field: source[field] for field in MAPPED_TRAINING_FIELDS},
        )
        self.assertEqual(comparison["unmapped_parameters"], {})

    def test_no_candidate_arrays_or_search_settings_are_allowed(self):
        for path in COMPARISONS:
            comparison = json.loads(path.read_text(encoding="utf-8"))
            assert_no_lstm_search_space(comparison)
            contaminated = dict(comparison)
            contaminated["n_trials"] = 30
            with self.assertRaises(LSTMComparisonConfigError):
                assert_no_lstm_search_space(contaminated)
            contaminated = dict(comparison)
            contaminated["learning_rates"] = [0.001, 0.01]
            with self.assertRaises(LSTMComparisonConfigError):
                assert_no_lstm_search_space(contaminated)


if __name__ == "__main__":
    unittest.main()
