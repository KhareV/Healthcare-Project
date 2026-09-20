import unittest

import torch

from training.tasks.organ_support import masked_support_bce_with_logits_loss


class SupportLossTests(unittest.TestCase):
    def test_censored_absurd_target_cannot_affect_loss_gradient_or_update(self):
        eligibility = torch.tensor([True, False, True])
        ordinary = torch.tensor([1.0, 0.0, 0.0])
        attacked = torch.tensor([1.0, 999999.0, 0.0])

        def execute(labels):
            logits = torch.nn.Parameter(torch.tensor([0.1, 20.0, -0.4]))
            optimizer = torch.optim.SGD((logits,), lr=0.1)
            optimizer.zero_grad()
            result = masked_support_bce_with_logits_loss(
                logits, labels, eligibility, pos_weight=2.0
            )
            result.loss.backward()
            gradient = logits.grad.detach().clone()
            optimizer.step()
            return result.loss.detach(), gradient, logits.detach().clone()

        expected = execute(ordinary)
        attack = execute(attacked)
        for left, right in zip(expected, attack):
            self.assertTrue(torch.equal(left, right))

    def test_all_censored_has_no_fake_signal(self):
        result = masked_support_bce_with_logits_loss(
            torch.tensor([0.0, 1.0]),
            torch.tensor([float("nan"), 999999.0]),
            torch.tensor([False, False]),
            pos_weight=1.0,
        )
        self.assertEqual(result.eligible_count, 0)
        self.assertIsNone(result.loss)

    def test_malformed_eligible_binary_label_is_rejected(self):
        with self.assertRaises(ValueError):
            masked_support_bce_with_logits_loss(
                torch.tensor([0.0]),
                torch.tensor([2.0]),
                torch.tensor([True]),
                pos_weight=1.0,
            )


if __name__ == "__main__":
    unittest.main()
