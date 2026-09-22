import csv
import json
import unittest
from pathlib import Path

import torch

from data.dataset import CanonicalTensorDataset, SyntheticTensorizationPolicy
from data.schema import FeatureSchemaReference
from training.class_weights import SupportClassWeight
from vedant_infra.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = tuple(sorted((ROOT / "configs/comparisons").glob("lstm_*_synthetic_v1.json")))


class LSTMSensitivityGovernanceTests(unittest.TestCase):
    def test_original_smoke_configs_remain_non_scientific_after_stage3(self):
        for path in COMPARISONS:
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(config["source_selected_gru_run_id"])
            self.assertFalse(config["scientific_sensitivity"])
            self.assertTrue(config["sensitivity_only"])
            self.assertFalse(config["test_examples_allowed"])
            self.assertFalse(config["participates_in_serving_selection"])

    def test_no_lstm_search_implementation_exists_and_lstm_is_not_selected(self):
        prohibited = (
            ROOT / "src/training/lstm_search.py",
            ROOT / "configs/search/lstm",
        )
        self.assertEqual([path for path in prohibited if path.exists()], [])
        selected = json.loads(
            (ROOT / "artifacts/models/selected_models_v1.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("lstm", {item["family"] for item in selected["tasks"].values()})

    def test_split_subjects_and_tensor_preprocessing_are_shared(self):
        for comparison_path in COMPARISONS:
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            source = json.loads(
                Path(comparison["source_gru_config_ref"]).read_text(encoding="utf-8")
            )
            schema = FeatureSchemaReference(
                version=comparison["feature_schema_version"],
                feature_names=tuple(comparison["feature_names"]),
                status=comparison["feature_schema_status"],
            )
            policy = SyntheticTensorizationPolicy(comparison["synthetic_missing_fill_value"])
            for partition in ("train", "validation"):
                gru_data = CanonicalTensorDataset.from_fixture(
                    source["fixture_path"],
                    schema,
                    partition,
                    policy,
                    source["synthetic_split_path"],
                )
                lstm_data = CanonicalTensorDataset.from_fixture(
                    comparison["fixture_path"],
                    schema,
                    partition,
                    policy,
                    comparison["synthetic_split_path"],
                )
                self.assertEqual(len(gru_data), len(lstm_data))
                for index in range(len(gru_data)):
                    self.assertEqual(
                        gru_data[index]["identifiers"]["subject_id"],
                        lstm_data[index]["identifiers"]["subject_id"],
                    )
                    for field in ("sequence", "padding_mask", "observation_mask"):
                        self.assertTrue(
                            torch.equal(gru_data[index][field], lstm_data[index][field])
                        )

    def test_task_preprocessing_artifacts_are_reused_not_refit(self):
        by_task = {
            json.loads(path.read_text(encoding="utf-8"))["task"]: json.loads(
                path.read_text(encoding="utf-8")
            )
            for path in COMPARISONS
        }
        recovery = by_task["recovery"]["preprocessing"]
        support = by_task["organ_support"]["preprocessing"]
        self.assertEqual(recovery["target_scaler_sha256"], sha256_file(recovery["target_scaler_ref"]))
        self.assertEqual(support["class_weight_sha256"], sha256_file(support["class_weight_ref"]))
        weight = SupportClassWeight.load(support["class_weight_ref"])
        self.assertEqual(weight.source_partition, "train")

    def test_registered_lstm_smokes_are_parent_linked_when_present(self):
        registry = ROOT / "experiments/registry.csv"
        with registry.open("r", encoding="utf-8", newline="") as handle:
            rows = {row["run_id"]: row for row in csv.DictReader(handle)}
        for path in COMPARISONS:
            config = json.loads(path.read_text(encoding="utf-8"))
            row = rows[config["run_id"]]
            self.assertEqual(row["model_family"], "lstm")
            self.assertEqual(row["parent_run_id"], config["source_gru_run_id"])
            self.assertIn("NOT SERVING CANDIDATE", row["notes"])


if __name__ == "__main__":
    unittest.main()
