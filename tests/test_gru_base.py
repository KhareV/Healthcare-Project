import unittest
from dataclasses import replace

import torch

from data.collate import create_dataloader
from models.gru import GRUEncoder, GRUEncoderConfig, ModelContractError
from phase4_helpers import tensor_dataset


def legal_config(**changes):
    values = dict(
        feature_dim=2,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        include_observation_mask=True,
        include_tslo=False,
        static_dim=0,
        bidirectional=False,
    )
    values.update(changes)
    return GRUEncoderConfig(**values)


def validation_batch():
    return next(
        iter(
            create_dataloader(
                tensor_dataset("validation"), batch_size=3, shuffle=False, seed=3
            )
        )
    )


class GRUEncoderTests(unittest.TestCase):
    def test_encoder_accepts_frozen_batch_and_explicit_observation_mask(self):
        batch = validation_batch()
        encoder = GRUEncoder(legal_config())
        encoder.eval()
        output = encoder(batch)
        self.assertEqual(tuple(output.shape), (3, 64))
        self.assertEqual(encoder.temporal_input_dim, 4)

    def test_prepared_gru_information_is_exact_canonical_value_and_mask_data(self):
        batch = next(
            iter(create_dataloader(tensor_dataset("train"), batch_size=1, shuffle=False, seed=3))
        )
        encoder = GRUEncoder(legal_config())
        prepared, lengths = encoder.assemble_temporal_input(batch)
        length = int(lengths[0].item())
        source = torch.cat(
            (batch.sequence, batch.observation_mask.to(batch.sequence.dtype)), dim=-1
        )
        self.assertTrue(torch.equal(prepared[0, :length], source[0, -length:]))
        self.assertTrue(torch.equal(prepared[0, length:], torch.zeros_like(prepared[0, length:])))

    def test_batch_size_one_is_supported(self):
        loader = create_dataloader(
            tensor_dataset("train"), batch_size=1, shuffle=False, seed=3
        )
        output = GRUEncoder(legal_config())(next(iter(loader)))
        self.assertEqual(tuple(output.shape), (1, 64))

    def test_seven_nine_and_wrong_feature_dimension_fail(self):
        batch = validation_batch()
        encoder = GRUEncoder(legal_config())
        with self.assertRaises(ModelContractError):
            encoder(replace(batch, sequence=batch.sequence[:, :7]))
        extra = torch.zeros((3, 1, 2))
        with self.assertRaises(ModelContractError):
            encoder(replace(batch, sequence=torch.cat((batch.sequence, extra), dim=1)))
        with self.assertRaises(ModelContractError):
            GRUEncoder(legal_config(feature_dim=3))(batch)

    def test_static_input_is_explicitly_dimensioned_and_concatenated(self):
        batch = validation_batch()
        with self.assertRaises(ModelContractError):
            GRUEncoder(legal_config())(
                replace(batch, static_features=torch.zeros((3, 2)))
            )
        configured = GRUEncoder(legal_config(static_dim=2))
        output = configured(replace(batch, static_features=torch.zeros((3, 2))))
        self.assertEqual(tuple(output.shape), (3, 66))
        with self.assertRaises(ModelContractError):
            configured(batch)

    def test_bidirectional_and_out_of_search_configuration_fail(self):
        for changes in (
            {"bidirectional": True},
            {"hidden_dim": 32},
            {"num_layers": 3},
            {"dropout": 0.5},
            {"include_observation_mask": False},
        ):
            with self.assertRaises(ModelContractError):
                legal_config(**changes).validate()


if __name__ == "__main__":
    unittest.main()
