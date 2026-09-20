import json
import unittest
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.timestamps import (
    EARLIEST_OFFSET_HOURS,
    K_MAX,
    K_MIN,
    MAX_EXAMPLES_PER_STAY,
    SPACING_HOURS,
    PredictionTimestamp,
    RetainedICUStay,
    TimestampValidationError,
    CandidateCutoff,
    calculate_horizon_availability,
    generate_candidate_cutoffs,
    generate_prediction_rows_for_stay,
    generate_prediction_timestamps,
    retain_legal_cutoffs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "timestamps" / "cases.json"
SPEC_PATH = REPOSITORY_ROOT / "configs" / "timestamp_spec_v1.yaml"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def stay_from_fixture(name):
    data = load_fixture()["stays"][name]
    return RetainedICUStay(
        subject_id=data["subject_id"],
        stay_id=data["stay_id"],
        intime=datetime.fromisoformat(data["intime"]),
        outtime=datetime.fromisoformat(data["outtime"]),
    )


class TimestampGridTests(unittest.TestCase):
    def test_earliest_cutoff_and_exact_outtime_boundary_are_legal(self):
        stay = stay_from_fixture("earliest_and_boundary_equal")
        rows = generate_prediction_rows_for_stay(stay)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].prediction_time, stay.intime + timedelta(hours=24))
        self.assertEqual(rows[0].prediction_time, stay.outtime - timedelta(hours=6))
        self.assertEqual(rows[0].grid_index, 0)

    def test_six_hour_spacing_and_exact_grid_formula(self):
        stay = stay_from_fixture("exactly_12")
        rows = generate_prediction_rows_for_stay(stay)
        for previous, current in zip(rows, rows[1:]):
            self.assertEqual(current.prediction_time - previous.prediction_time, timedelta(hours=6))
        for row in rows:
            expected = stay.intime + timedelta(
                hours=EARLIEST_OFFSET_HOURS + SPACING_HOURS * row.grid_index
            )
            self.assertEqual(row.prediction_time, expected)
            self.assertGreaterEqual(row.grid_index, K_MIN)
            self.assertLessEqual(row.grid_index, K_MAX)

    def test_exactly_twelve_cutoffs(self):
        data = load_fixture()["stays"]["exactly_12"]
        rows = generate_prediction_rows_for_stay(stay_from_fixture("exactly_12"))
        self.assertEqual(len(rows), 12)
        self.assertEqual([row.icu_elapsed_hours for row in rows], data["expected_offsets_hours"])
        self.assertEqual(rows[-1].prediction_time, stay_from_fixture("exactly_12").outtime - timedelta(hours=6))

    def test_long_stay_is_capped_without_post_cap_sampling(self):
        stay = stay_from_fixture("long_stay_capped")
        candidates = generate_candidate_cutoffs(stay)
        rows = generate_prediction_rows_for_stay(stay)
        self.assertEqual(len(candidates), MAX_EXAMPLES_PER_STAY)
        self.assertEqual(len(rows), MAX_EXAMPLES_PER_STAY)
        self.assertEqual(rows[-1].grid_index, 11)
        self.assertNotIn(stay.intime + timedelta(hours=96), [row.prediction_time for row in rows])

    def test_shorter_legal_stay_has_manually_expected_count(self):
        data = load_fixture()["stays"]["four_cutoffs"]
        rows = generate_prediction_rows_for_stay(stay_from_fixture("four_cutoffs"))
        self.assertEqual(len(rows), data["expected_count"])
        self.assertEqual([row.icu_elapsed_hours for row in rows], [24, 30, 36, 42])

    def test_one_microsecond_beyond_boundary_is_rejected(self):
        stay = stay_from_fixture("no_legal_cutoff_one_microsecond_short")
        candidates = generate_candidate_cutoffs(stay)
        self.assertEqual(candidates[0].prediction_time, stay.intime + timedelta(hours=24))
        self.assertGreater(candidates[0].prediction_time, stay.outtime - timedelta(hours=6))
        self.assertEqual(retain_legal_cutoffs(stay, candidates), ())
        self.assertEqual(generate_prediction_rows_for_stay(stay), ())

    def test_grid_is_relative_to_exact_unrounded_intime(self):
        stay = stay_from_fixture("earliest_and_boundary_equal")
        first = generate_prediction_rows_for_stay(stay)[0]
        self.assertEqual((first.prediction_time.hour, first.prediction_time.minute), (3, 17))

    def test_repeated_calls_are_identical(self):
        stays = [stay_from_fixture("four_cutoffs"), stay_from_fixture("exactly_12")]
        self.assertEqual(generate_prediction_timestamps(stays), generate_prediction_timestamps(stays))

    def test_timezone_aware_input_preserves_exact_relative_clock(self):
        india = timezone(timedelta(hours=5, minutes=30))
        intime = datetime(2026, 1, 1, 3, 17, tzinfo=india)
        stay = RetainedICUStay(201, 2001, intime, intime + timedelta(hours=30))
        row = generate_prediction_rows_for_stay(stay)[0]
        self.assertEqual(row.prediction_time, intime + timedelta(hours=24))
        self.assertEqual((row.prediction_time.hour, row.prediction_time.minute), (3, 17))

    def test_multiple_stays_are_independent_and_deterministically_ordered(self):
        stay_later_id = stay_from_fixture("four_cutoffs")
        stay_earlier_id = stay_from_fixture("exactly_12")
        reversed_input = [stay_later_id, stay_earlier_id]
        rows = generate_prediction_timestamps(reversed_input)
        self.assertEqual(rows[0].subject_id, stay_earlier_id.subject_id)
        counts = {
            stay_id: sum(row.stay_id == stay_id for row in rows)
            for stay_id in (stay_later_id.stay_id, stay_earlier_id.stay_id)
        }
        self.assertEqual(counts[stay_later_id.stay_id], 4)
        self.assertEqual(counts[stay_earlier_id.stay_id], 12)


class HorizonAvailabilityTests(unittest.TestCase):
    def test_case_a_24h_and_48h_available_at_exact_48h_equality(self):
        row = generate_prediction_rows_for_stay(
            stay_from_fixture("horizon_48h_exact_equality")
        )[0]
        self.assertTrue(row.recovery24_followup_available)
        self.assertTrue(row.recovery48_followup_available)
        self.assertTrue(row.support24_full_followup_available)
        self.assertTrue(row.icu_time_temporally_eligible)

    def test_case_b_24h_available_and_48h_unavailable(self):
        row = generate_prediction_rows_for_stay(
            stay_from_fixture("horizon_24h_exact_equality")
        )[0]
        self.assertTrue(row.recovery24_followup_available)
        self.assertFalse(row.recovery48_followup_available)
        self.assertTrue(row.support24_full_followup_available)

    def test_case_c_both_recovery_horizons_and_support_followup_unavailable(self):
        row = generate_prediction_rows_for_stay(
            stay_from_fixture("both_recovery_horizons_unavailable")
        )[0]
        self.assertFalse(row.recovery24_followup_available)
        self.assertFalse(row.recovery48_followup_available)
        self.assertFalse(row.support24_full_followup_available)
        self.assertTrue(row.icu_time_temporally_eligible)

    def test_horizon_one_microsecond_after_outtime_is_unavailable(self):
        cutoff = datetime(2026, 10, 1, 0, 0, 0, 1)
        outtime = cutoff + timedelta(hours=24) - timedelta(microseconds=1)
        availability = calculate_horizon_availability(cutoff, outtime)
        self.assertFalse(availability.recovery24_followup_available)
        self.assertFalse(availability.support24_full_followup_available)


class ValidationAndScopeTests(unittest.TestCase):
    def test_duplicate_stay_input_fails_instead_of_duplicating_rows(self):
        stay = stay_from_fixture("earliest_and_boundary_equal")
        with self.assertRaises(TimestampValidationError):
            generate_prediction_timestamps([stay, stay])

    def test_malformed_candidate_cannot_bypass_frozen_grid(self):
        stay = stay_from_fixture("exactly_12")
        malformed = CandidateCutoff(
            grid_index=0,
            prediction_time=stay.intime + timedelta(hours=25),
        )
        with self.assertRaises(TimestampValidationError):
            retain_legal_cutoffs(stay, [malformed])

    def test_multiple_stays_for_one_subject_fail_upstream_contract(self):
        stay = stay_from_fixture("earliest_and_boundary_equal")
        second_stay = RetainedICUStay(
            subject_id=stay.subject_id,
            stay_id="different-stay",
            intime=stay.intime,
            outtime=stay.outtime,
        )
        with self.assertRaises(TimestampValidationError):
            generate_prediction_timestamps([stay, second_stay])

    def test_invalid_non_positive_duration_fails(self):
        data = load_fixture()["invalid"]["non_positive_duration"]
        stay = RetainedICUStay(
            subject_id=data["subject_id"],
            stay_id=data["stay_id"],
            intime=datetime.fromisoformat(data["intime"]),
            outtime=datetime.fromisoformat(data["outtime"]),
        )
        with self.assertRaises(TimestampValidationError):
            generate_prediction_rows_for_stay(stay)

    def test_missing_or_invalid_time_fails(self):
        valid = datetime(2026, 1, 1)
        with self.assertRaises(TimestampValidationError):
            generate_prediction_rows_for_stay(RetainedICUStay(1, 1, None, valid))
        with self.assertRaises(TimestampValidationError):
            generate_prediction_rows_for_stay(RetainedICUStay(1, 1, valid, None))

    def test_output_schema_contains_time_flags_but_no_labels(self):
        names = {field.name for field in fields(PredictionTimestamp)}
        self.assertEqual(
            names,
            {
                "subject_id",
                "stay_id",
                "prediction_time",
                "grid_index",
                "icu_elapsed_hours",
                "recovery24_followup_available",
                "recovery48_followup_available",
                "support24_full_followup_available",
                "icu_time_temporally_eligible",
                "timestamp_spec_version",
            },
        )
        forbidden = {
            "sofa",
            "delta_sofa24",
            "delta_sofa48",
            "icu_time_target",
            "organ_support_eligible",
            "organ_support_label",
            "hospital_dischtime",
        }
        self.assertTrue(names.isdisjoint(forbidden))

    def test_future_clinical_payload_cannot_change_timestamp_grid(self):
        stay = stay_from_fixture("four_cutoffs")
        future_clinical_data_a = {"future_lab": 1, "future_sofa": 2, "support_event": False}
        future_clinical_data_b = {"future_lab": 999, "future_sofa": 24, "support_event": True}
        before = generate_prediction_timestamps([stay])
        self.assertNotEqual(future_clinical_data_a, future_clinical_data_b)
        after = generate_prediction_timestamps([stay])
        self.assertEqual(before, after)

    def test_spec_constants_match_frozen_implementation(self):
        spec = SPEC_PATH.read_text(encoding="utf-8")
        required_lines = {
            "spec_version: timestamp_spec_v1",
            "earliest_offset_hours: 24",
            "spacing_hours: 6",
            "k_min: 0",
            "k_max: 11",
            "max_examples_per_stay: 12",
            "minimum_time_before_outtime_hours: 6",
            "post_cap_sampling: false",
            "random_sampling: false",
            'lookback_interval_semantics: "(t-48h, t]"',
            "pre_ICU_policy: padding",
            "support_followup_is_final_eligibility: false",
            "label_calculation_in_scope: false",
        }
        self.assertTrue(required_lines.issubset(set(spec.splitlines())))


if __name__ == "__main__":
    unittest.main()
