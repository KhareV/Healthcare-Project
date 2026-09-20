import json
import tempfile
import unittest
from pathlib import Path

from data.split import (
    SUPPORTED_ERA_TO_SPLIT,
    CohortSubjectRecord,
    SplitIntegrityError,
    SplitValidationError,
    SubjectSplit,
    assert_split_hash,
    audit_cohort_records,
    era_to_split,
    generate_subject_split,
    generate_subject_split_from_mappings,
    serialize_split_csv,
    split_content_sha256,
    subject_sets,
    validate_subject_isolation,
    verify_split_artifacts,
    write_split_artifacts,
)
from vedant_infra.hashing import sha256_bytes, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "splits"
    / "cohort_rows_synthetic_v1.json"
)
EXPECTED_SPLIT_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "splits"
    / "expected_split_synthetic_v1.csv"
)
SPEC_PATH = REPOSITORY_ROOT / "configs" / "split_spec_v1.yaml"
FINAL_SPLIT_PATH = REPOSITORY_ROOT / "artifacts" / "splits" / "split_v1.csv"


def fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FrozenEraMappingTests(unittest.TestCase):
    def test_each_frozen_era_maps_exactly(self):
        self.assertEqual(era_to_split("2008-2010"), "train")
        self.assertEqual(era_to_split("2011-2013"), "train")
        self.assertEqual(era_to_split("2014-2016"), "validation")
        self.assertEqual(era_to_split("2017-2019"), "test")

    def test_unknown_or_space_normalized_era_is_rejected(self):
        for era in ("2020-2022", "2008 - 2010", " 2008-2010 "):
            with self.subTest(era=era), self.assertRaises(SplitValidationError):
                era_to_split(era)


class SubjectContractTests(unittest.TestCase):
    def test_all_eras_produce_exact_expected_mapping(self):
        result = generate_subject_split_from_mappings(fixture()["rows"])
        actual = {
            row.subject_id: (row.anchor_year_group, row.split) for row in result.rows
        }
        self.assertEqual(
            actual,
            {
                "A": ("2008-2010", "train"),
                "B": ("2011-2013", "train"),
                "C": ("2014-2016", "validation"),
                "D": ("2017-2019", "test"),
            },
        )

    def test_subject_sets_are_pairwise_disjoint(self):
        result = generate_subject_split_from_mappings(fixture()["rows"])
        sets = subject_sets(result.rows)
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])

    def test_identical_duplicate_subject_rows_deduplicate_and_report(self):
        result = generate_subject_split_from_mappings(fixture()["rows"])
        self.assertEqual(len(result.rows), 4)
        self.assertEqual(result.qa.duplicate_input_count, 1)

    def test_conflicting_subject_era_is_hard_failure(self):
        records = [
            CohortSubjectRecord("A", "2008-2010", "stay-A"),
            CohortSubjectRecord("A", "2017-2019", "stay-A"),
        ]
        audit = audit_cohort_records(records)
        self.assertEqual(audit.qa.conflicting_subject_count, 1)
        with self.assertRaises(SplitValidationError):
            generate_subject_split(records)

    def test_conflicting_retained_stay_is_hard_failure(self):
        records = [
            CohortSubjectRecord("A", "2008-2010", "stay-A1"),
            CohortSubjectRecord("A", "2008-2010", "stay-A2"),
        ]
        with self.assertRaises(SplitValidationError):
            generate_subject_split(records)

    def test_null_subject_fails_and_is_counted(self):
        records = [CohortSubjectRecord(None, "2008-2010")]
        audit = audit_cohort_records(records)
        self.assertEqual(audit.qa.missing_subject_count, 1)
        with self.assertRaises(SplitValidationError):
            generate_subject_split(records)

    def test_null_era_fails_and_is_counted(self):
        records = [CohortSubjectRecord("A", None)]
        audit = audit_cohort_records(records)
        self.assertEqual(audit.qa.missing_era_count, 1)
        with self.assertRaises(SplitValidationError):
            generate_subject_split(records)

    def test_unknown_era_fails_and_is_counted(self):
        records = [CohortSubjectRecord("A", "2020-2022")]
        audit = audit_cohort_records(records)
        self.assertEqual(audit.qa.unsupported_era_count, 1)
        with self.assertRaises(SplitValidationError):
            generate_subject_split(records)

    def test_timestamp_multiplicity_does_not_change_subject_assignment(self):
        repeated_rows = [
            {
                "subject_id": "A",
                "anchor_year_group": "2008-2010",
                "stay_id": "stay-A",
                "prediction_time": "t{}".format(index),
            }
            for index in range(12)
        ]
        result = generate_subject_split_from_mappings(repeated_rows)
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0].split, "train")
        self.assertEqual(result.qa.duplicate_input_count, 11)

    def test_duplicate_output_subject_is_rejected_even_in_same_split(self):
        rows = [
            SubjectSplit("A", "train", "2008-2010"),
            SubjectSplit("A", "train", "2008-2010"),
        ]
        with self.assertRaises(SplitValidationError):
            validate_subject_isolation(rows)


class DeterminismAndLeakageTests(unittest.TestCase):
    def test_reordered_input_has_identical_rows_bytes_and_hash(self):
        rows = fixture()["rows"]
        forward = generate_subject_split_from_mappings(rows)
        reverse = generate_subject_split_from_mappings(reversed(rows))
        self.assertEqual(forward.rows, reverse.rows)
        self.assertEqual(serialize_split_csv(forward.rows), serialize_split_csv(reverse.rows))
        self.assertEqual(split_content_sha256(forward.rows), split_content_sha256(reverse.rows))

    def test_repeated_generation_needs_no_seed_and_is_identical(self):
        rows = fixture()["rows"]
        self.assertEqual(
            generate_subject_split_from_mappings(rows),
            generate_subject_split_from_mappings(rows),
        )

    def test_labels_outcomes_missingness_and_timestamps_are_ignored(self):
        rows_a = fixture()["rows"]
        rows_b = []
        for row in rows_a:
            changed = dict(row)
            changed.update(
                {
                    "future_sofa": -999,
                    "icu_time_target": 123456,
                    "organ_support_label": "changed",
                    "missingness": 1.0,
                    "prediction_time": "changed-future-time",
                    "model_performance": 0.999,
                    "test_result": "changed",
                }
            )
            rows_b.append(changed)
        self.assertEqual(
            generate_subject_split_from_mappings(rows_a),
            generate_subject_split_from_mappings(rows_b),
        )

    def test_canonical_fixture_serialization_is_exact(self):
        result = generate_subject_split_from_mappings(fixture()["rows"])
        self.assertEqual(
            serialize_split_csv(result.rows), EXPECTED_SPLIT_PATH.read_bytes()
        )

    def test_hash_is_stable_and_changed_content_changes_it(self):
        result = generate_subject_split_from_mappings(fixture()["rows"])
        content = serialize_split_csv(result.rows)
        self.assertEqual(split_content_sha256(result.rows), sha256_bytes(content))
        self.assertNotEqual(sha256_bytes(content), sha256_bytes(content + b"changed"))


class StructuralQATests(unittest.TestCase):
    def test_structural_qa_matches_hand_counted_fixture(self):
        expected = fixture()["expected"]
        qa = generate_subject_split_from_mappings(fixture()["rows"]).qa
        self.assertEqual(qa.total_input_rows, expected["total_input_rows"])
        self.assertEqual(qa.total_subjects, expected["total_subjects"])
        self.assertEqual(qa.train_subjects, expected["train_subjects"])
        self.assertEqual(qa.validation_subjects, expected["validation_subjects"])
        self.assertEqual(qa.test_subjects, expected["test_subjects"])
        self.assertEqual(qa.duplicate_input_count, expected["duplicate_input_count"])
        self.assertEqual(dict(qa.counts_by_anchor_year_group), expected["counts_by_anchor_year_group"])
        self.assertEqual(qa.missing_subject_count, 0)
        self.assertEqual(qa.missing_era_count, 0)
        self.assertEqual(qa.unsupported_era_count, 0)
        self.assertEqual(qa.conflicting_subject_count, 0)


class ArtifactIntegrityTests(unittest.TestCase):
    def test_write_load_and_verify_synthetic_artifacts(self):
        result = generate_subject_split_from_mappings(fixture()["rows"])
        source_hash = "a" * 64
        spec_hash = sha256_file(SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "split_synthetic_v1.csv"
            metadata_path = Path(directory) / "split_synthetic_v1.metadata.json"
            metadata = write_split_artifacts(
                result,
                split_path,
                metadata_path,
                source_cohort_version="synthetic_fixture_v1",
                source_cohort_sha256=source_hash,
                split_spec_sha256=spec_hash,
                created_at_utc="2026-09-17T00:00:00Z",
            )
            self.assertEqual(metadata["split_file_sha256"], sha256_file(split_path))
            verified = verify_split_artifacts(split_path, metadata_path)
            self.assertEqual(verified["holdout_description"], "coarse temporal holdout")

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            path.write_bytes(b"original")
            expected = sha256_file(path)
            path.write_bytes(b"changed")
            with self.assertRaises(SplitIntegrityError):
                assert_split_hash(path, expected)

    def test_metadata_tampering_fails_closed(self):
        result = generate_subject_split_from_mappings(fixture()["rows"])
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "split_synthetic_v1.csv"
            metadata_path = Path(directory) / "split_synthetic_v1.metadata.json"
            write_split_artifacts(
                result,
                split_path,
                metadata_path,
                source_cohort_version="synthetic_fixture_v1",
                source_cohort_sha256="a" * 64,
                split_spec_sha256=sha256_file(SPEC_PATH),
                created_at_utc="2026-09-17T00:00:00Z",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["era_mapping"]["2017-2019"] = "train"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(SplitIntegrityError):
                verify_split_artifacts(split_path, metadata_path)

    def test_final_real_split_is_not_fabricated(self):
        self.assertFalse(FINAL_SPLIT_PATH.exists())

    def test_spec_matches_frozen_mapping_and_non_random_policy(self):
        spec = SPEC_PATH.read_text(encoding="utf-8")
        required_lines = {
            "spec_version: split_spec_v1",
            "split_unit: subject_id",
            "holdout_description: coarse_temporal_holdout",
            'train_anchor_year_groups: ["2008-2010", "2011-2013"]',
            'validation_anchor_year_groups: ["2014-2016"]',
            'test_anchor_year_groups: ["2017-2019"]',
            "randomness_used: false",
            "row_level_splitting_allowed: false",
            "stratification_allowed: false",
            "subject_overlap_allowed: false",
        }
        self.assertTrue(required_lines.issubset(set(spec.splitlines())))
        self.assertEqual(
            SUPPORTED_ERA_TO_SPLIT,
            {
                "2008-2010": "train",
                "2011-2013": "train",
                "2014-2016": "validation",
                "2017-2019": "test",
            },
        )


if __name__ == "__main__":
    unittest.main()
