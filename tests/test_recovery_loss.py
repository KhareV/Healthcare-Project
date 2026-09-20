import unittest

import torch

from preprocess.target_scaler import RecoveryTargetScaler
from training.tasks.recovery import masked_recovery_huber_loss


def identity_scaler():
    return RecoveryTargetScaler(
        mean_24=0.0,
        scale_24=1.0,
        mean_48=0.0,
        scale_48=1.0,
        eligible_count_24=2,
        eligible_count_48=2,
        source_partition="train",
        split_hash="a" * 64,
        tensor_contract_version="tensor_contract_v1",
        feature_schema_version="synthetic_feature_schema_v1",
    )


class RecoveryLossTests(unittest.TestCase):
    def test_independent_mask_cases(self):
        predictions = torch.zeros((2, 2))
        targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        cases = (
            (torch.tensor([[1, 1], [1, 1]], dtype=torch.bool), 4),
            (torch.tensor([[1, 0], [1, 1]], dtype=torch.bool), 3),
            (torch.tensor([[1, 0], [0, 0]], dtype=torch.bool), 1),
        )
        for mask, count in cases:
            result = masked_recovery_huber_loss(
                predictions, targets, mask, identity_scaler()
            )
            self.assertEqual(result.eligible_count, count)
            self.assertTrue(torch.isfinite(result.loss))

    def test_no_eligible_pair_has_no_fake_signal(self):
        result = masked_recovery_huber_loss(
            torch.zeros((2, 2)),
            torch.full((2, 2), float("nan")),
            torch.zeros((2, 2), dtype=torch.bool),
            identity_scaler(),
        )
        self.assertEqual(result.eligible_count, 0)
        self.assertIsNone(result.loss)

    def test_absurd_ineligible_target_cannot_change_loss_or_gradient(self):
        mask = torch.tensor([[True, False], [True, True]])
        ordinary = torch.tensor([[1.0, 7.0], [2.0, 3.0]])
        attacked = ordinary.clone()
        attacked[0, 1] = 999999.0

        def loss_and_gradient(targets):
            predictions = torch.nn.Parameter(torch.tensor([[0.2, 0.3], [0.4, 0.5]]))
            result = masked_recovery_huber_loss(
                predictions, targets, mask, identity_scaler()
            )
            result.loss.backward()
            return result.loss.detach(), predictions.grad.detach().clone()

        normal_loss, normal_gradient = loss_and_gradient(ordinary)
        attack_loss, attack_gradient = loss_and_gradient(attacked)
        self.assertTrue(torch.equal(normal_loss, attack_loss))
        self.assertTrue(torch.equal(normal_gradient, attack_gradient))

        def optimizer_update(targets):
            predictions = torch.nn.Parameter(torch.tensor([[0.2, 0.3], [0.4, 0.5]]))
            optimizer = torch.optim.SGD((predictions,), lr=0.1)
            optimizer.zero_grad()
            result = masked_recovery_huber_loss(
                predictions, targets, mask, identity_scaler()
            )
            result.loss.backward()
            optimizer.step()
            return predictions.detach().clone()

        self.assertTrue(torch.equal(optimizer_update(ordinary), optimizer_update(attacked)))

    def test_equal_horizon_mean_not_total_batch_denominator(self):
        predictions = torch.zeros((2, 2))
        targets = torch.tensor([[1.0, 2.0], [1.0, 999999.0]])
        mask = torch.tensor([[True, True], [True, False]])
        result = masked_recovery_huber_loss(
            predictions, targets, mask, identity_scaler()
        )
        # Huber(0,1)=0.5 and Huber(0,2)=1.5; equal horizon mean=(0.5+1.5)/2.
        self.assertAlmostEqual(result.loss.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
