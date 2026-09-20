import csv
from pathlib import Path

from evaluation.ablations import (
    derive_feature_manifest,
    evaluate_recovery_ablation,
    write_ablation_table,
)
from evaluation.metrics import PredictionRecord, evaluate_recovery_horizon
from evaluation.selection_validation import sha256_file
from ablation_helpers import contract, variant


def predictions(horizon):
    return (
        PredictionRecord("A", "2020-01-01", 2.0, 1.0, True),
        PredictionRecord("A", "2020-01-02", 4.0, 2.0, True),
        PredictionRecord("B", "2020-01-01", -2.0, -1.0, True),
    )


def test_ablation_reuses_phase9_horizon_metrics_and_generates_table(tmp_path):
    candidate = variant(tmp_path)
    manifest = derive_feature_manifest(candidate, contract())
    run_artifact = Path(tmp_path) / "ablation.model"
    run_artifact.write_bytes(b"synthetic ablation model")
    records24 = predictions("24h")
    records48 = predictions("48h")
    result = evaluate_recovery_ablation(
        variant=candidate,
        manifest=manifest,
        run_id="synthetic-ablation-run",
        run_artifact_ref=str(run_artifact),
        run_artifact_sha256=sha256_file(run_artifact),
        recovery24=records24,
        recovery48=records48,
        source_partition="validation",
        test_accessed=False,
    )
    authoritative = evaluate_recovery_horizon(records24, horizon="24h")
    assert result.recovery24 == authoritative
    table, table_hash = write_ablation_table(tmp_path / "ablation_table.csv", (result,))
    assert sha256_file(table) == table_hash
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["MAE24"] == str(authoritative.metrics["mae"])
    assert rows[0]["removed_feature_groups"] == "tslo_features"
