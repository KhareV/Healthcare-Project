import io
import unittest

import torch

from models.gru import GRUEncoder
from models.gru_icu_time import ICU_TIME_RAW_OUTPUT, ICUTimeGRU
from test_gru_base import legal_config, validation_batch


class ICUTimeGRUTests(unittest.TestCase):
    def test_reuses_shared_encoder_and_outputs_one_log_value(self):
        model = ICUTimeGRU(legal_config())
        model.eval()
        output = model(validation_batch())
        self.assertIsInstance(model.encoder, GRUEncoder)
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertEqual(model.raw_output_meaning, ICU_TIME_RAW_OUTPUT)

    def test_batch_size_one_and_state_roundtrip(self):
        batch = validation_batch()
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
        model = ICUTimeGRU(legal_config())
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(2.0)
        model.eval()
        expected = model(one)
        payload = io.BytesIO()
        torch.save(model.state_dict(), payload)
        payload.seek(0)
        restored = ICUTimeGRU(legal_config())
        restored.load_state_dict(torch.load(payload, weights_only=True))
        restored.eval()
        self.assertEqual(tuple(expected.shape), (1, 1))
        self.assertTrue(torch.equal(restored(one), expected))


if __name__ == "__main__":
    unittest.main()
