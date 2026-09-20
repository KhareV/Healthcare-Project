import unittest

import torch

from models.gru_icu_time import ICU_TIME_RAW_OUTPUT
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from models.lstm import ICUTimeLSTM
from training.tasks.icu_time import ICUTimeTaskAdapter
from test_gru_base import legal_config, validation_batch


class ICUTimeLSTMTests(unittest.TestCase):
    def test_output_is_one_log_value(self):
        model = ICUTimeLSTM(legal_config())
        model.eval()
        self.assertEqual(tuple(model(validation_batch()).shape), (3, 1))
        self.assertEqual(model.raw_output_meaning, ICU_TIME_RAW_OUTPUT)

    def test_shared_postprocessor_clamps_then_expm1(self):
        raw = torch.tensor([[-2.0], [0.0], [2.0]])
        hours = remaining_icu_hours_from_log_prediction(raw)
        self.assertTrue(torch.equal(hours[:2], torch.zeros((2, 1))))
        self.assertTrue(torch.allclose(hours[2], torch.expm1(torch.tensor([2.0]))))
        self.assertTrue(torch.all(hours >= 0))

    def test_shared_adapter_uses_unchanged_eligibility(self):
        batch = validation_batch()
        result = ICUTimeTaskAdapter().compute_loss(torch.zeros((3, 1)), batch)
        self.assertEqual(result.eligible_count, int(batch.eligibility["icu_time"].sum()))


if __name__ == "__main__":
    unittest.main()
