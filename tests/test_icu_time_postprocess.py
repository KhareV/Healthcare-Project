import unittest

import torch

from models.icu_time_postprocess import (
    ICU_TIME_POSTPROCESS_VERSION,
    remaining_icu_hours_from_log_prediction,
)


class ICUTimePostprocessTests(unittest.TestCase):
    def test_frozen_scalar_and_vector_cases(self):
        raw = torch.tensor([-1.0, 0.0, torch.log(torch.tensor(2.0)), torch.log(torch.tensor(11.0))])
        expected = torch.tensor([0.0, 0.0, 1.0, 10.0])
        actual = remaining_icu_hours_from_log_prediction(raw)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.all(actual >= 0))

    def test_positive_value_is_expm1_and_negative_is_zero(self):
        positive = torch.tensor([2.0])
        self.assertTrue(
            torch.equal(
                remaining_icu_hours_from_log_prediction(positive),
                torch.expm1(positive),
            )
        )
        self.assertEqual(
            remaining_icu_hours_from_log_prediction(torch.tensor([-20.0])).item(),
            0.0,
        )

    def test_validation_inference_and_serving_use_same_function(self):
        raw = torch.tensor([[-2.0], [1.25]])
        validation = remaining_icu_hours_from_log_prediction(raw)
        inference = remaining_icu_hours_from_log_prediction(raw)
        serving = remaining_icu_hours_from_log_prediction(raw)
        self.assertTrue(torch.equal(validation, inference))
        self.assertTrue(torch.equal(inference, serving))
        self.assertEqual(ICU_TIME_POSTPROCESS_VERSION, "icu_time_nonnegative_log_expm1_v1")

    def test_nonfinite_values_fail_closed(self):
        for value in (float("nan"), float("inf"), 1000.0):
            with self.assertRaises(ValueError):
                remaining_icu_hours_from_log_prediction(torch.tensor([value]))


if __name__ == "__main__":
    unittest.main()
