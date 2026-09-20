import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

from data.schema import (
    HISTORY_DTYPE,
    STATIC_UNRESOLVED_STATUS,
    TSLO_UNRESOLVED_STATUS,
    CanonicalDataset,
    FeatureSchemaReference,
    TensorContractError,
    deserialize_canonical_dataset,
    serialize_canonical_dataset,
    validate_canonical_dataset,
    validate_canonical_example,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "canonical" / "canonical_synthetic_v1.json"
CONTRACT_PATH = REPOSITORY_ROOT / "configs" / "tensor_contract_v1.json"


def load_dataset():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    schema_data = payload["synthetic_feature_schema"]
    schema = FeatureSchemaReference(
        version=schema_data["version"],
        feature_names=tuple(schema_data["feature_names"]),
        status=schema_data["status"],
    )
    return deserialize_canonical_dataset(FIXTURE_PATH.read_bytes(), schema), schema


class ShapeAndSchemaTests(unittest.TestCase):
    def test_valid_8_by_f_examples_are_accepted(self):
        dataset, schema = load_dataset()
        validate_canonical_dataset(dataset, schema)
        self.assertTrue(all(len(item.history_values) == 8 for item in dataset.examples))
        self.assertTrue(
            all(len(row) == schema.feature_dim for item in dataset.examples for row in item.history_values)
        )

    def test_seven_and_nine_bins_are_rejected(self):
        dataset, schema = load_dataset()
        example = dataset.examples[0]
        for history in (example.history_values[:7], example.history_values + (example.history_values[-1],)):
            with self.subTest(length=len(history)), self.assertRaises(TensorContractError):
                validate_canonical_example(replace(example, history_values=history), schema)

    def test_wrong_feature_dimension_is_rejected(self):
        dataset, schema = load_dataset()
        example = dataset.examples[0]
        wrong = tuple(row[:1] for row in example.history_values)
        with self.assertRaises(TensorContractError):
            validate_canonical_example(replace(example, history_values=wrong), schema)

    def test_feature_order_or_version_mismatch_is_rejected(self):
        dataset, schema = load_dataset()
        reversed_schema = replace(schema, feature_names=tuple(reversed(schema.feature_names)))
        with self.assertRaises(TensorContractError):
            validate_canonical_dataset(dataset, reversed_schema)
        with self.assertRaises(TensorContractError):
            validate_canonical_example(
                replace(dataset.examples[0], feature_schema_version="wrong"), schema
            )

    def test_padding_observation_and_tslo_shapes_or_status_are_enforced(self):
        dataset, schema = load_dataset()
        example = dataset.examples[0]
        with self.assertRaises(TensorContractError):
            validate_canonical_example(replace(example, padding_mask=(True,) * 7), schema)
        with self.assertRaises(TensorContractError):
            validate_canonical_example(
                replace(example, observation_mask=example.observation_mask[:7]), schema
            )
        fake_tslo = tuple((0.0, 0.0) for _ in range(8))
        with self.assertRaises(TensorContractError):
            validate_canonical_example(replace(example, tslo_hours=fake_tslo), schema)
        self.assertEqual(example.tslo_status, TSLO_UNRESOLVED_STATUS)

    def test_static_features_cannot_be_invented_while_contract_is_blocked(self):
        dataset, schema = load_dataset()
        example = dataset.examples[0]
        with self.assertRaises(TensorContractError):
            validate_canonical_example(
                replace(
                    example,
                    static_features=(1.0,),
                    static_feature_names=("invented_static",),
                ),
                schema,
            )
        self.assertEqual(example.static_status, STATIC_UNRESOLVED_STATUS)

    def test_missing_identifier_is_rejected(self):
        dataset, schema = load_dataset()
        with self.assertRaises(TensorContractError):
            validate_canonical_example(replace(dataset.examples[0], subject_id=""), schema)

    def test_duplicate_canonical_key_is_rejected(self):
        dataset, schema = load_dataset()
        duplicate = replace(dataset, examples=dataset.examples + (dataset.examples[0],))
        with self.assertRaises(TensorContractError):
            validate_canonical_dataset(duplicate, schema)

    def test_version_mismatches_are_rejected(self):
        dataset, schema = load_dataset()
        with self.assertRaises(TensorContractError):
            validate_canonical_dataset(replace(dataset, tensor_contract_version="v0"), schema)
        with self.assertRaises(TensorContractError):
            validate_canonical_example(
                replace(dataset.examples[0], timestamp_spec_version="wrong"), schema
            )

    def test_invalid_dtype_and_nan_are_rejected_without_conversion(self):
        dataset, schema = load_dataset()
        example = dataset.examples[0]
        with self.assertRaises(TensorContractError):
            validate_canonical_example(replace(example, history_dtype="float64"), schema)
        history = [list(row) for row in example.history_values]
        valid_bin = next(index for index, padded in enumerate(example.padding_mask) if not padded)
        history[valid_bin][0] = math.nan
        with self.assertRaises(TensorContractError):
            validate_canonical_example(
                replace(example, history_values=tuple(tuple(row) for row in history)), schema
            )


class TaskAndSerializationTests(unittest.TestCase):
    def test_independent_task_eligibility_combination_is_valid(self):
        dataset, schema = load_dataset()
        example = next(
            item
            for item in dataset.examples
            if "independent_task_masks" in item.case_tags
        )
        self.assertTrue(example.eligibility.recovery24_eligible)
        self.assertFalse(example.eligibility.recovery48_eligible)
        self.assertTrue(example.eligibility.icu_time_eligible)
        self.assertFalse(example.eligibility.organ_support_eligible)
        validate_canonical_example(example, schema)

    def test_ineligible_target_cannot_enter_training_ready_contract(self):
        dataset, schema = load_dataset()
        example = next(item for item in dataset.examples if not item.eligibility.organ_support_eligible)
        with self.assertRaises(TensorContractError):
            validate_canonical_example(
                replace(
                    example,
                    targets=replace(example.targets, organ_support_target=0),
                ),
                schema,
            )

    def test_explicit_eligibility_is_not_inferred_from_target_nullness(self):
        dataset, schema = load_dataset()
        example = next(item for item in dataset.examples if item.eligibility.recovery24_eligible)
        with self.assertRaises(TensorContractError):
            validate_canonical_example(
                replace(
                    example,
                    targets=replace(example.targets, recovery24_delta_sofa=None),
                ),
                schema,
            )

    def test_serialization_is_deterministic_and_roundtrip_equivalent(self):
        dataset, schema = load_dataset()
        first = serialize_canonical_dataset(dataset, schema)
        second = serialize_canonical_dataset(dataset, schema)
        self.assertEqual(first, second)
        restored = deserialize_canonical_dataset(first, schema)
        self.assertEqual(restored, dataset)

    def test_contract_declares_symbolic_f_and_blocked_upstream_items(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(contract["sequence"]["batch_shape"], ["B", 8, "F"])
        self.assertIsNone(contract["sequence"]["feature_dimension"]["value"])
        self.assertIn("BLOCKED", contract["sequence"]["feature_dimension"]["status"])
        self.assertIsNone(contract["tslo"]["no_observation_representation"])
        self.assertIn("BLOCKED", contract["tslo"]["status"])
        self.assertEqual(contract["tslo"]["example_shape"], [8, "F"])
        self.assertEqual(
            contract["tslo"]["feature_alignment"],
            "identical to feature_schema_v1 order",
        )
        self.assertIsNone(contract["static_features"]["feature_order"])
        self.assertIn("BLOCKED", contract["static_features"]["status"])


if __name__ == "__main__":
    unittest.main()
