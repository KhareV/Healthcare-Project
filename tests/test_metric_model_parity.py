import unittest

from evaluation.metrics import PredictionRecord, evaluate_organ_support, evaluate_recovery


class MetricModelParityTests(unittest.TestCase):
    def test_recovery_metrics_do_not_branch_by_model_family(self):
        rows = [
            PredictionRecord("A", "t0", 2.0, 3.0, True),
            PredictionRecord("B", "t0", -2.0, -1.0, True),
        ]
        gru = evaluate_recovery(rows, rows, metadata={"model_family": "gru"})
        xgb = evaluate_recovery(rows, rows, metadata={"model_family": "xgboost"})
        self.assertEqual(gru["24h"].metrics, xgb["24h"].metrics)
        self.assertEqual(gru["24h"].counts, xgb["24h"].counts)

    def test_classification_metrics_do_not_branch_by_model_family(self):
        rows = [
            PredictionRecord("A", "t0", 1.0, 0.8, True),
            PredictionRecord("B", "t0", 0.0, 0.2, True),
        ]
        gru = evaluate_organ_support(rows, metadata={"model_family": "gru"})
        xgb = evaluate_organ_support(rows, metadata={"model_family": "xgboost"})
        self.assertEqual(gru.metrics, xgb.metrics)
        self.assertEqual(gru.counts, xgb.counts)


if __name__ == "__main__":
    unittest.main()
