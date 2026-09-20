import unittest

import torch

from data.collate import create_dataloader
from models.gru_support import uncalibrated_support_probability
from models.lstm import OrganSupportLSTM
from phase7_helpers import support_dataset
from test_gru_base import legal_config
from training.tasks.organ_support import OrganSupportTaskAdapter


class SupportLSTMTests(unittest.TestCase):
    def test_output_is_one_raw_logit_and_sigmoid_is_external(self):
        batch = next(
            iter(
                create_dataloader(
                    support_dataset("validation"), batch_size=3, shuffle=False, seed=1
                )
            )
        )
        model = OrganSupportLSTM(legal_config())
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(2.0)
        model.eval()
        logits = model(batch)
        probability = uncalibrated_support_probability(logits)
        self.assertEqual(tuple(logits.shape), (3, 1))
        self.assertTrue(torch.equal(logits, torch.full((3, 1), 2.0)))
        self.assertTrue(torch.equal(probability, torch.sigmoid(logits)))

    def test_censored_rows_remain_outside_shared_bce_loss(self):
        batch = next(
            iter(
                create_dataloader(
                    support_dataset("validation"), batch_size=3, shuffle=False, seed=1
                )
            )
        )
        result = OrganSupportTaskAdapter(pos_weight=1.0).compute_loss(
            torch.zeros((3, 1)), batch
        )
        self.assertEqual(result.eligible_count, int(batch.eligibility["organ_support"].sum()))
        changed = batch.targets["organ_support"].clone()
        changed[~batch.eligibility["organ_support"]] = float("nan")
        from dataclasses import replace

        changed_batch = replace(
            batch, targets={**batch.targets, "organ_support": changed}
        )
        changed_result = OrganSupportTaskAdapter(pos_weight=1.0).compute_loss(
            torch.zeros((3, 1)), changed_batch
        )
        self.assertTrue(torch.equal(result.loss, changed_result.loss))


if __name__ == "__main__":
    unittest.main()
