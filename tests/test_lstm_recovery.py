import unittest

import torch

from models.gru_recovery import RECOVERY_HORIZON_ORDER
from models.lstm import RecoveryLSTM
from models.recovery_output import reconstruct_absolute_sofa
from pathlib import Path

from preprocess.target_scaler import RecoveryTargetScaler
from training.tasks.recovery import RecoveryTaskAdapter
from test_gru_base import legal_config, validation_batch


class RecoveryLSTMTests(unittest.TestCase):
    def test_output_contract_and_explicit_horizon_order(self):
        model = RecoveryLSTM(legal_config())
        model.eval()
        self.assertEqual(tuple(model(validation_batch()).shape), (3, 2))
        self.assertEqual(model.horizon_order, RECOVERY_HORIZON_ORDER)
        self.assertEqual(model.horizon_order, ("delta_sofa_24h", "delta_sofa_48h"))

    def test_independent_reconstruction_is_shared_with_phase5(self):
        baseline = torch.tensor([10.0])
        deltas = torch.tensor([[2.0, -3.0]])
        self.assertTrue(
            torch.equal(reconstruct_absolute_sofa(baseline, deltas), torch.tensor([[12.0, 7.0]]))
        )

    def test_shared_adapter_preserves_targets_and_independent_masks(self):
        batch = validation_batch()
        scaler = RecoveryTargetScaler.load(
            Path(__file__).resolve().parents[1]
            / "artifacts/smoke/phase5_synthetic_recovery_gru_v1/recovery_target_scaler.json"
        )
        adapter = RecoveryTaskAdapter(scaler)
        predictions = torch.zeros((len(batch.sequence), 2))
        result = adapter.compute_loss(predictions, batch)
        expected = int(batch.eligibility["recovery24"].sum()) + int(
            batch.eligibility["recovery48"].sum()
        )
        self.assertEqual(result.eligible_count, expected)


if __name__ == "__main__":
    unittest.main()
