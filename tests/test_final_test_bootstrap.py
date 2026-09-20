from evaluation.bootstrap import grouped_bootstrap
from evaluation.metrics import evaluate_recovery_horizon
from final_test_helpers import record


def test_final_bootstrap_preserves_stay_copies_and_skips_invalid_values():
    rows = (record("A", 1, 0, 1), record("A", 2, 0, 1), record("B", 3, 0, 3))
    result = grouped_bootstrap(
        rows,
        metric_fn=lambda values: evaluate_recovery_horizon(values, horizon="24h").metrics["mae"],
        metric_name="mae",
        task="recovery",
        horizon="24h",
        n_bootstrap=1,
        seed=1,
        forced_draws=(("A", "A"),),
    )
    assert result.sampled_cluster_sequences == (("A", "A"),)
    assert result.n_original_stays == 2
    assert result.metadata["rows_sampled_independently"] is False


def test_invalid_metric_replicate_is_counted_without_fake_value():
    rows = (record("A", 1, 0, 0.1), record("B", 2, 1, 0.9))
    result = grouped_bootstrap(
        rows,
        metric_fn=lambda values: float("nan"),
        metric_name="auroc",
        task="organ_support",
        n_bootstrap=2,
        seed=2,
    )
    assert result.n_valid_replicates == 0
    assert result.n_invalid_replicates == 2
    assert result.bootstrap_distribution == ()
