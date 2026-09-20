import json
import unittest
from pathlib import Path

from data.schema import (
    FeatureSchemaReference,
    deserialize_canonical_dataset,
    serialize_canonical_dataset,
    temporal_information_views,
    unflatten_temporal,
)
from vedant_infra.hashing import sha256_bytes, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "canonical" / "canonical_synthetic_v1.json"
CONTRACT_PATH = REPOSITORY_ROOT / "configs" / "tensor_contract_v1.json"


def load_payload_dataset():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    raw_schema = payload["synthetic_feature_schema"]
    schema = FeatureSchemaReference(
        version=raw_schema["version"],
        feature_names=tuple(raw_schema["feature_names"]),
        status=raw_schema["status"],
    )
    dataset = deserialize_canonical_dataset(FIXTURE_PATH.read_bytes(), schema)
    return payload, dataset, schema


def example_with_tag(dataset, tag):
    return next(item for item in dataset.examples if tag in item.case_tags)


class PaddingAndBoundaryFixtureTests(unittest.TestCase):
    def test_earliest_cutoff_has_four_padding_bins_with_no_clinical_values(self):
        _, dataset, _ = load_payload_dataset()
        example = example_with_tag(dataset, "earliest_legal_cutoff")
        self.assertEqual(example.icu_elapsed_hours, 24)
        self.assertEqual(example.padding_mask, (True, True, True, True, False, False, False, False))
        for index in range(4):
            self.assertTrue(all(value is None for value in example.history_values[index]))
            self.assertTrue(all(value is False for value in example.observation_mask[index]))

    def test_full_48h_history_has_no_padding(self):
        _, dataset, _ = load_payload_dataset()
        example = example_with_tag(dataset, "full_48h_icu_history")
        self.assertEqual(example.padding_mask, (False,) * 8)

    def test_padding_and_clinical_missingness_are_distinct_states(self):
        _, dataset, _ = load_payload_dataset()
        padded = example_with_tag(dataset, "earliest_legal_cutoff")
        missing = example_with_tag(dataset, "padding_vs_missingness")
        self.assertTrue(padded.padding_mask[0])
        self.assertFalse(padded.observation_mask[0][0])
        self.assertFalse(missing.padding_mask[0])
        self.assertFalse(missing.observation_mask[0][1])
        self.assertIsNone(missing.history_values[0][1])

    def test_no_observation_feature_remains_masked_and_tslo_blocked(self):
        _, dataset, _ = load_payload_dataset()
        example = example_with_tag(dataset, "no_observation_inside_lookback")
        self.assertTrue(all(not row[1] for row in example.observation_mask))
        self.assertTrue(all(row[1] is None for row in example.history_values))
        self.assertIsNone(example.tslo_hours)
        self.assertIn("BLOCKED", example.tslo_status)

    def test_left_open_right_closed_boundary_metadata_is_exact(self):
        payload, _, _ = load_payload_dataset()
        boundary = payload["fixture_semantics"]["boundary_assertions"]
        self.assertEqual(boundary["at_t_minus_48h"], {"offset_seconds": -172800, "inside_lookback": False, "bin_index": None})
        self.assertTrue(boundary["immediately_after_t_minus_48h"]["inside_lookback"])
        self.assertEqual(boundary["immediately_after_t_minus_48h"]["bin_index"], 0)
        self.assertEqual(boundary["at_t"], {"offset_seconds": 0, "inside_lookback": True, "bin_index": 7})


class TaskAndPopulationFixtureTests(unittest.TestCase):
    def test_required_task_cases_are_present(self):
        _, dataset, _ = load_payload_dataset()
        tags = {tag for item in dataset.examples for tag in item.case_tags}
        required = {
            "recovery24_and_recovery48_available",
            "recovery24_available_recovery48_unavailable",
            "both_recovery_horizons_unavailable",
            "icu_time_available_recovery_censored",
            "organ_support_eligible_positive",
            "organ_support_eligible_negative",
            "organ_support_censored_ineligible",
        }
        self.assertTrue(required.issubset(tags))

    def test_organ_support_positive_negative_and_censored_contracts(self):
        _, dataset, _ = load_payload_dataset()
        positive = example_with_tag(dataset, "organ_support_eligible_positive")
        negative = example_with_tag(dataset, "organ_support_eligible_negative")
        censored = example_with_tag(dataset, "organ_support_censored_ineligible")
        self.assertTrue(positive.eligibility.organ_support_eligible)
        self.assertEqual(positive.targets.organ_support_target, 1)
        self.assertTrue(negative.eligibility.organ_support_eligible)
        self.assertEqual(negative.targets.organ_support_target, 0)
        self.assertFalse(censored.eligibility.organ_support_eligible)
        self.assertIsNone(censored.targets.organ_support_target)

    def test_stays_have_different_snapshot_counts(self):
        _, dataset, _ = load_payload_dataset()
        counts = {}
        for item in dataset.examples:
            counts[item.stay_id] = counts.get(item.stay_id, 0) + 1
        self.assertEqual(counts["SYNTH-STAY-A"], 2)
        self.assertEqual(counts["SYNTH-STAY-B"], 3)
        self.assertEqual(counts["SYNTH-STAY-C"], 1)

    def test_multiple_subjects_and_phase2_splits_are_present(self):
        _, dataset, _ = load_payload_dataset()
        self.assertEqual({item.split for item in dataset.examples}, {"train", "validation", "test"})
        self.assertEqual(len({item.subject_id for item in dataset.examples}), 3)

    def test_out_of_order_source_is_canonicalized(self):
        payload, dataset, _ = load_payload_dataset()
        self.assertEqual(payload["examples"][0]["subject_id"], "SYNTH-C")
        ordered_keys = [(item.subject_id, item.stay_id, item.prediction_time) for item in dataset.examples]
        self.assertEqual(ordered_keys, sorted(ordered_keys))


class InformationParityAndReproducibilityTests(unittest.TestCase):
    def test_gru_xgboost_temporal_information_is_exactly_reversible(self):
        _, dataset, schema = load_payload_dataset()
        for example in dataset.examples:
            views = temporal_information_views(example)
            self.assertEqual(
                unflatten_temporal(views["xgboost"]["history_values"], schema.feature_dim),
                views["gru"]["history_values"],
            )
            self.assertEqual(
                unflatten_temporal(views["xgboost"]["observation_mask"], schema.feature_dim),
                views["gru"]["observation_mask"],
            )
            self.assertEqual(views["xgboost"]["padding_mask"], views["gru"]["padding_mask"])
            self.assertEqual(views["xgboost"]["tslo_hours"], views["gru"]["tslo_hours"])
            self.assertEqual(views["xgboost"]["static_features"], views["gru"]["static_features"])

    def test_fixture_serialization_is_deterministic_and_hash_sensitive(self):
        _, dataset, schema = load_payload_dataset()
        first = serialize_canonical_dataset(dataset, schema)
        second = serialize_canonical_dataset(dataset, schema)
        self.assertEqual(first, second)
        self.assertEqual(sha256_bytes(first), sha256_bytes(second))
        self.assertNotEqual(sha256_bytes(first), sha256_bytes(first + b"changed"))

    def test_contract_and_fixture_have_no_machine_absolute_paths(self):
        for path in (CONTRACT_PATH, FIXTURE_PATH):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", text)
            self.assertNotIn("/mnt/data", text)

    def test_fixture_is_explicitly_nonclinical_and_placeholder_only(self):
        payload, dataset, _ = load_payload_dataset()
        semantics = payload["fixture_semantics"]
        self.assertFalse(semantics["contains_real_clinical_data"])
        self.assertFalse(semantics["feature_names_are_final"])
        self.assertFalse(semantics["feature_dimension_is_final"])
        self.assertIn("SYNTHETIC", dataset.dataset_kind)
        self.assertTrue(all(name.startswith("feature_") for name in dataset.feature_schema.feature_names))

    def test_exact_contract_and_fixture_file_hashes_are_stable_calls(self):
        self.assertEqual(sha256_file(CONTRACT_PATH), sha256_file(CONTRACT_PATH))
        self.assertEqual(sha256_file(FIXTURE_PATH), sha256_file(FIXTURE_PATH))
        contract_bytes = CONTRACT_PATH.read_bytes()
        self.assertNotEqual(
            sha256_bytes(contract_bytes), sha256_bytes(contract_bytes + b"changed")
        )


if __name__ == "__main__":
    unittest.main()
