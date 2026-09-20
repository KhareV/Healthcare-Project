import unittest

import torch

from training.tasks.icu_time import masked_icu_time_huber_loss


class ICUTimeLossTests(unittest.TestCase):
    def test_absurd_ineligible_target_has_no_effect(self):
        mask = torch.tensor([True, False, True])
        ordinary = torch.tensor([2.0, 7.0, 4.0])
        attacked = torch.tensor([2.0, -999.0, 4.0])

        def loss_gradient_and_update(targets):
            prediction = torch.nn.Parameter(torch.tensor([1.0, 100.0, 3.0]))
            optimizer = torch.optim.SGD((prediction,), lr=0.1)
            optimizer.zero_grad()
            result = masked_icu_time_huber_loss(
                prediction, targets, mask, delta=1.0
            )
            result.loss.backward()
            gradient = prediction.grad.detach().clone()
            optimizer.step()
            return result.loss.detach(), gradient, prediction.detach().clone()

        expected = loss_gradient_and_update(ordinary)
        attack = loss_gradient_and_update(attacked)
        for left, right in zip(expected, attack):
            self.assertTrue(torch.equal(left, right))

    def test_hand_calculated_eligible_only_huber(self):
        result = masked_icu_time_huber_loss(
            torch.tensor([1.0, 100.0, 3.0]),
            torch.tensor([2.0, -999.0, 4.0]),
            torch.tensor([True, False, True]),
        )
        self.assertEqual(result.eligible_count, 2)
        self.assertAlmostEqual(result.loss.item(), 0.5)

    def test_all_ineligible_returns_no_signal(self):
        result = masked_icu_time_huber_loss(
            torch.tensor([1.0, 2.0]),
            torch.tensor([float("nan"), -999.0]),
            torch.tensor([False, False]),
        )
        self.assertEqual(result.eligible_count, 0)
        self.assertIsNone(result.loss)


if __name__ == "__main__":
    unittest.main()
