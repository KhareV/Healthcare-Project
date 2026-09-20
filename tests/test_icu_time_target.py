import unittest
from dataclasses import replace
from datetime import datetime, timedelta

import torch

from models.gru import GRUEncoder
from models.icu_time_target import (
    current_icu_remaining_hours,
    log1p_remaining_icu_hours,
    validate_log1p_icu_time_targets,
)
from test_gru_base import legal_config, validation_batch


class ICUTimeTargetTests(unittest.TestCase):
    def test_log1p_known_values_and_inverse(self):
        hours = torch.tensor([0.0, 1.0, 10.0, 47.5])
        transformed = log1p_remaining_icu_hours(hours)
        self.assertEqual(transformed[0].item(), 0.0)
        self.assertAlmostEqual(transformed[1].item(), torch.log(torch.tensor(2.0)).item())
        self.assertTrue(torch.allclose(torch.expm1(transformed), hours))
        validate_log1p_icu_time_targets(transformed)

    def test_invalid_target_values_are_rejected(self):
        for values in (
            torch.tensor([-1.0]),
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
        ):
            with self.assertRaises(ValueError):
                log1p_remaining_icu_hours(values)
        with self.assertRaises(ValueError):
            current_icu_remaining_hours(
                datetime(2026, 1, 2), datetime(2026, 1, 1)
            )

    def test_current_icu_outtime_not_hospital_dischtime_defines_target(self):
        prediction_time = datetime(2026, 1, 1, 12)
        icu_outtime = datetime(2026, 1, 2, 12)
        hospital_dischtime_a = datetime(2026, 1, 3, 12)
        hospital_dischtime_b = datetime(2026, 2, 1, 12)
        target_a = current_icu_remaining_hours(prediction_time, icu_outtime)
        target_b = current_icu_remaining_hours(prediction_time, icu_outtime)
        self.assertNotEqual(hospital_dischtime_a, hospital_dischtime_b)
        self.assertEqual(target_a, 24.0)
        self.assertEqual(target_a, target_b)

    def test_outtime_or_target_change_cannot_change_model_input(self):
        batch = validation_batch()
        encoder = GRUEncoder(legal_config())
        prepared_before, lengths_before = encoder.assemble_temporal_input(batch)
        prediction_time = datetime(2026, 1, 1, 12)
        earlier_outtime = datetime(2026, 1, 2, 12)
        later_outtime = datetime(2026, 1, 3, 12)
        earlier_target = log1p_remaining_icu_hours(
            torch.tensor(
                [current_icu_remaining_hours(prediction_time, earlier_outtime)]
            )
        )
        later_target = log1p_remaining_icu_hours(
            torch.tensor(
                [current_icu_remaining_hours(prediction_time, later_outtime)]
            )
        )
        self.assertFalse(torch.equal(earlier_target, later_target))
        changed_targets = dict(batch.targets)
        changed_targets["icu_time"] = batch.targets["icu_time"].clone()
        changed_targets["icu_time"][0] = later_target[0]
        target_changed_batch = replace(batch, targets=changed_targets)
        prepared_after, lengths_after = encoder.assemble_temporal_input(
            target_changed_batch
        )
        self.assertTrue(torch.equal(prepared_before, prepared_after))
        self.assertTrue(torch.equal(lengths_before, lengths_after))
        self.assertNotIn("outtime", batch.identifiers)
        self.assertNotIn("hospital_dischtime", batch.identifiers)


if __name__ == "__main__":
    unittest.main()
