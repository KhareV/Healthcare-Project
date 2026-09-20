import io
import unittest

import torch

from models.gru_recovery import RECOVERY_HORIZON_ORDER, RecoveryGRU
from models.recovery_output import reconstruct_absolute_sofa
from test_gru_base import legal_config, validation_batch


class RecoveryGRUTests(unittest.TestCase):
    def test_output_shape_for_one_and_multiple_examples(self):
        model = RecoveryGRU(legal_config())
        model.eval()
        batch = validation_batch()
        self.assertEqual(tuple(model(batch).shape), (3, 2))
        one = type(batch)(
            identifiers={key: value[:1] for key, value in batch.identifiers.items()},
            sequence=batch.sequence[:1],
            padding_mask=batch.padding_mask[:1],
            observation_mask=batch.observation_mask[:1],
            tslo=None,
            static_features=None,
            targets={key: value[:1] for key, value in batch.targets.items()},
            eligibility={key: value[:1] for key, value in batch.eligibility.items()},
            versions=batch.versions,
            feature_names=batch.feature_names,
        )
        self.assertEqual(tuple(model(one).shape), (1, 2))

    def test_horizon_order_is_explicit_and_survives_state_roundtrip(self):
        model = RecoveryGRU(legal_config())
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.copy_(torch.tensor([24.0, 48.0]))
        model.eval()
        expected = model(validation_batch())
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = RecoveryGRU(legal_config())
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        restored.eval()
        self.assertEqual(RECOVERY_HORIZON_ORDER, ("delta_sofa_24h", "delta_sofa_48h"))
        self.assertTrue(torch.equal(restored(validation_batch()), expected))
        self.assertTrue(torch.equal(expected[0], torch.tensor([24.0, 48.0])))

    def test_independent_reconstruction_and_display_only_clipping(self):
        baseline = torch.tensor([10.0])
        raw_delta = torch.tensor([[2.0, -3.0]])
        reconstructed = reconstruct_absolute_sofa(baseline, raw_delta)
        self.assertTrue(torch.equal(reconstructed, torch.tensor([[12.0, 7.0]])))
        extreme = torch.tensor([[30.0, -30.0]])
        self.assertTrue(
            torch.equal(
                reconstruct_absolute_sofa(baseline, extreme, clip_for_display=True),
                torch.tensor([[24.0, 0.0]]),
            )
        )
        self.assertTrue(torch.equal(extreme, torch.tensor([[30.0, -30.0]])))


if __name__ == "__main__":
    unittest.main()
