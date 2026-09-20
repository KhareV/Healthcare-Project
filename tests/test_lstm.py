import unittest
from dataclasses import replace

import torch

from models.gru import ModelContractError
from models.lstm import LSTMEncoder, trainable_parameter_count
from models.gru import GRUEncoder
from test_gru_base import legal_config, validation_batch


class LSTMEncoderTests(unittest.TestCase):
    def test_shape_for_one_and_multiple_examples(self):
        batch = validation_batch()
        encoder = LSTMEncoder(legal_config())
        encoder.eval()
        self.assertEqual(tuple(encoder(batch).shape), (3, 64))
        one = replace(
            batch,
            identifiers={key: value[:1] for key, value in batch.identifiers.items()},
            sequence=batch.sequence[:1],
            padding_mask=batch.padding_mask[:1],
            observation_mask=batch.observation_mask[:1],
            targets={key: value[:1] for key, value in batch.targets.items()},
            eligibility={key: value[:1] for key, value in batch.eligibility.items()},
        )
        self.assertEqual(tuple(encoder(one).shape), (1, 64))

    def test_input_assembly_is_bitwise_identical_to_gru(self):
        batch = validation_batch()
        gru = GRUEncoder(legal_config())
        lstm = LSTMEncoder(legal_config())
        gru_input, gru_lengths = gru.assemble_temporal_input(batch)
        lstm_input, lstm_lengths = lstm.assemble_temporal_input(batch)
        self.assertTrue(torch.equal(gru_input, lstm_input))
        self.assertTrue(torch.equal(gru_lengths, lstm_lengths))

    def test_rejects_wrong_time_feature_and_static_contracts(self):
        batch = validation_batch()
        encoder = LSTMEncoder(legal_config())
        for sequence in (
            batch.sequence[:, :7],
            torch.cat((batch.sequence, torch.zeros((3, 1, 2))), dim=1),
            torch.zeros((3, 8, 3)),
        ):
            with self.assertRaises(ModelContractError):
                encoder(replace(batch, sequence=sequence))
        with self.assertRaises(ModelContractError):
            encoder(replace(batch, static_features=torch.zeros((3, 1))))

    def test_hidden_size_is_not_changed_to_parameter_match(self):
        gru = GRUEncoder(legal_config())
        lstm = LSTMEncoder(legal_config())
        self.assertEqual(gru.config.hidden_dim, lstm.config.hidden_dim)
        self.assertGreater(trainable_parameter_count(lstm), trainable_parameter_count(gru))

    def test_targets_and_eligibility_are_not_model_inputs(self):
        batch = validation_batch()
        changed = replace(
            batch,
            targets={key: torch.full_like(value, 9999.0) for key, value in batch.targets.items()},
            eligibility={
                key: torch.logical_not(value) for key, value in batch.eligibility.items()
            },
        )
        encoder = LSTMEncoder(legal_config())
        encoder.eval()
        self.assertTrue(torch.equal(encoder(batch), encoder(changed)))


if __name__ == "__main__":
    unittest.main()
