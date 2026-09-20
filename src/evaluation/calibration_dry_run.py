"""Ephemeral synthetic Phase-13 end-to-end check; writes no repository artifacts."""

import json
import tempfile
from pathlib import Path

from evaluation.calibrate import fit_support_calibrator, save_calibrator_artifact
from evaluation.calibration_validation import (
    IsotonicPolicy,
    PREDICTION_ARTIFACT_VERSION,
    SelectedSupportModel,
    canonical_json_bytes,
    sha256_bytes,
)
from evaluation.selection_validation import sha256_file
from evaluation.threshold import (
    ThresholdSearchPolicy,
    choose_support_threshold,
    save_threshold_artifact,
)


def run_synthetic_dry_run():
    with tempfile.TemporaryDirectory(prefix="phase13_synthetic_") as directory:
        root = Path(directory)
        model_path = root / "selected_support.model"
        model_path.write_bytes(b"synthetic selected support model")
        model_hash = sha256_file(model_path)
        prediction_path = root / "validation_predictions.json"
        rows = [
            {"row_id": "r1", "stay_id": "s1", "organ_support_eligible": True, "raw_probability": 0.1, "label": 0},
            {"row_id": "r2", "stay_id": "s1", "organ_support_eligible": True, "raw_probability": 0.6, "label": 1},
            {"row_id": "r3", "stay_id": "s2", "organ_support_eligible": True, "raw_probability": 0.3, "label": 0},
            {"row_id": "r4", "stay_id": "s2", "organ_support_eligible": True, "raw_probability": 0.9, "label": 1},
        ]
        payload = {
            "artifact_version": PREDICTION_ARTIFACT_VERSION,
            "task": "organ_support",
            "family": "gru",
            "run_id": "synthetic-selected-support",
            "candidate_id": "synthetic-candidate",
            "config_hash": "synthetic-config-hash",
            "model_sha256": model_hash,
            "split_hash": "synthetic-split",
            "feature_version": "synthetic-features",
            "label_version": "synthetic-labels",
            "probability_type": "raw_uncalibrated",
            "partition": "validation",
            "test_accessed": False,
            "rows": rows,
        }
        prediction_path.write_bytes(canonical_json_bytes(payload))
        selected = SelectedSupportModel(
            task="organ_support",
            family="gru",
            run_id="synthetic-selected-support",
            candidate_id="synthetic-candidate",
            config_hash="synthetic-config-hash",
            artifact_ref=str(model_path),
            artifact_sha256=model_hash,
            split_hash="synthetic-split",
            feature_version="synthetic-features",
            label_version="synthetic-labels",
            preprocessing_hash="synthetic-preprocessor",
            code_commit="synthetic-only",
            selection_metric="validation_auprc",
            probability_type="raw_uncalibrated",
            validation_prediction_ref=str(prediction_path),
            validation_prediction_sha256=sha256_bytes(prediction_path.read_bytes()),
            selection_mode="synthetic",
            test_accessed=False,
        )
        calibrator = fit_support_calibrator(
            selected,
            policy=IsotonicPolicy(
                policy_version="synthetic_only_isotonic_v1",
                out_of_bounds="clip",
                sample_weighting="stay_balanced",
            ),
        )
        calibrator_ref = save_calibrator_artifact(root / "calibrator.json", calibrator)
        threshold = choose_support_threshold(
            selected,
            calibrator,
            calibrator_ref=calibrator_ref["artifact_ref"],
            calibrator_sha256=calibrator_ref["artifact_sha256"],
            policy=ThresholdSearchPolicy(
                policy_version="synthetic_only_threshold_v1",
                candidate_method="unique_calibrated_probabilities",
                tie_policy="lowest_threshold",
            ),
        )
        threshold_ref = save_threshold_artifact(root / "threshold.json", threshold)
        return {
            "run_type": "synthetic_dry_run_not_scientific",
            "selected_family": selected.family,
            "calibrator_sha256": calibrator_ref["artifact_sha256"],
            "threshold_sha256": threshold_ref["artifact_sha256"],
            "threshold": threshold.threshold,
            "validation_f1": threshold.validation_metrics["f1"],
            "test_accessed": False,
            "repository_artifacts_written": False,
        }


def main():
    print(json.dumps(run_synthetic_dry_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
