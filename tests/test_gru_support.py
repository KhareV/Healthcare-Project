import unittest

import torch

from data.collate import create_dataloader
from models.gru import GRUEncoder
from models.gru_support import OrganSupportGRU, uncalibrated_support_probability
from phase7_helpers import support_dataset
from test_gru_base import legal_config


class OrganSupportGRUTests(unittest.TestCase):
    def test_reuses_encoder_and_returns_one_raw_logit(self):
        batch = next(
            iter(create_dataloader(support_dataset("validation"), batch_size=3, shuffle=False, seed=1))
        )
        model = OrganSupportGRU(legal_config())
        model.eval()
        logits = model(batch)
        self.assertIsInstance(model.encoder, GRUEncoder)
        self.assertEqual(tuple(logits.shape), (3, 1))

    def test_batch_size_one_and_external_sigmoid_only(self):
        batch = next(
            iter(create_dataloader(support_dataset("train"), batch_size=1, shuffle=False, seed=1))
        )
        model = OrganSupportGRU(legal_config())
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(2.0)
        model.eval()
        logits = model(batch)
        probability = uncalibrated_support_probability(logits)
        self.assertTrue(torch.equal(logits, torch.tensor([[2.0]])))
        self.assertTrue(torch.equal(probability, torch.sigmoid(logits)))
        self.assertFalse(torch.equal(probability, torch.sigmoid(probability)))
        self.assertTrue(torch.all((probability >= 0) & (probability <= 1)))

    def test_shared_encoder_rejects_invalid_shape_and_static_input(self):
        batch = next(
            iter(create_dataloader(support_dataset("train"), batch_size=2, shuffle=False, seed=1))
        )
        model = OrganSupportGRU(legal_config())
        from dataclasses import replace
        with self.assertRaises(ValueError):
            model(replace(batch, sequence=batch.sequence[:, :7]))
        with self.assertRaises(ValueError):
            model(replace(batch, static_features=torch.zeros((2, 1))))


if __name__ == "__main__":
    unittest.main()
