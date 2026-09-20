import json
from dataclasses import replace
from pathlib import Path

from evaluation.metrics import METRIC_IMPLEMENTATION_VERSION
from evaluation.selection_validation import (
    CandidateSummary,
    SearchEvidence,
    sha256_file,
)
from experiments.search_governance import (
    CandidateResult,
    InformationManifest,
    canonical_sha256,
)


PRIMARY = {
    "recovery": "validation_recovery24_mae",
    "icu_stay_time": "validation_median_absolute_error_hours",
    "organ_support": "validation_auprc",
}


def information(family, **updates):
    values = {
        "family": family,
        "row_keys_hash": "rows-v1",
        "feature_schema_version": "feature_schema_v1",
        "temporal_bins": 8,
        "dynamic_features": ("heart_rate", "creatinine"),
        "observation_masks": ("heart_rate_observed", "creatinine_observed"),
        "tslo_features": ("heart_rate_tslo", "creatinine_tslo"),
        "static_features": ("age",),
        "preprocessing_hash": "preprocess-v1",
        "split_hash": "split-v1",
        "eligibility_hash": "eligible-v1",
        "representation": "flattened_8xf" if family == "xgboost" else "sequence_8xf",
    }
    values.update(updates)
    return InformationManifest(**values)


def candidate(tmp_path, task, family, primary, *, secondary=None, winner_index=0):
    prefix = "{}_{}".format(task, family)
    declared = tuple(
        {
            "candidate_id": "{}_{:02d}".format(prefix, index),
            "config": {"synthetic_parameter": index},
            "config_hash": canonical_sha256({"synthetic_parameter": index}),
        }
        for index in range(30)
    )
    declared_by_id = {item["candidate_id"]: item["config_hash"] for item in declared}
    winner_id = "{}_{:02d}".format(prefix, winner_index)
    results = []
    for index, candidate_id in enumerate(declared_by_id):
        if task == "organ_support":
            value = primary if index == winner_index else primary - 0.01 - index / 1000.0
        else:
            value = primary if index == winner_index else primary + 1.0 + index / 1000.0
        results.append(
            CandidateResult(
                candidate_id=candidate_id,
                run_id="run-" + candidate_id,
                status="COMPLETE",
                metrics={PRIMARY[task]: value},
                probability_type=("raw_uncalibrated" if task == "organ_support" else None),
            )
        )
    search = SearchEvidence(
        task=task,
        family=family,
        run_type="synthetic_dry_run",
        search_version="search-v1",
        search_space_hash="space-" + family,
        candidate_list_hash=canonical_sha256(declared),
        declared_candidates=declared,
        results=tuple(results),
        referenced_best_candidate_id=winner_id,
    )
    metrics = {PRIMARY[task]: primary}
    metrics.update(secondary or {})
    counts = {"N_examples": 2000, "N_ICU_stays": 500}
    if task == "organ_support":
        counts.update({"N_positive_examples": 140, "N_positive_stays": 80})
    artifact = Path(tmp_path) / (prefix + ".model")
    artifact.write_bytes(("synthetic model " + prefix).encode("utf-8"))
    metadata = Path(str(artifact) + ".metadata.json")
    metadata.write_text(
        json.dumps(
            {
                "task": task,
                "model_family": family,
                "feature_schema_version": "feature_schema_v1",
                "split_hash": "split-v1",
                "preprocessing_hash": "preprocess-v1",
                "label_version": "labels-v1",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return CandidateSummary(
        task=task,
        family=family,
        candidate_id=winner_id,
        run_id="run-" + winner_id,
        status="COMPLETE",
        config_hash=declared_by_id[winner_id],
        split_hash="split-v1",
        feature_version="feature_schema_v1",
        label_version="labels-v1",
        preprocessing_hash="preprocess-v1",
        code_commit="synthetic-commit",
        artifact_ref=str(artifact),
        artifact_sha256=sha256_file(artifact),
        artifact_metadata_ref=str(metadata),
        metrics=metrics,
        counts=counts,
        metric_implementation_version=METRIC_IMPLEMENTATION_VERSION,
        source_partition="validation",
        test_accessed=False,
        information_manifest=information(family),
        search=search,
        probability_type=("raw_uncalibrated" if task == "organ_support" else None),
        validation_prediction_ref=(
            "synthetic-validation-predictions.json" if task == "organ_support" else None
        ),
    )


def pair(tmp_path, task, xgb_primary, gru_primary, *, xgb_secondary=None, gru_secondary=None):
    return (
        candidate(tmp_path, task, "xgboost", xgb_primary, secondary=xgb_secondary),
        candidate(tmp_path, task, "gru", gru_primary, secondary=gru_secondary),
    )


def replace_candidate(candidate_value, **updates):
    return replace(candidate_value, **updates)
