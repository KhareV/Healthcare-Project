import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.metrics import PredictionRecord
from experiments.search_governance import canonical_sha256
from vedant_infra.hashing import sha256_file


def record(stay, row, target, prediction):
    return PredictionRecord(
        stay_id=stay,
        prediction_time="2020-01-{:02d}T00:00:00".format(row),
        target=float(target),
        prediction=float(prediction),
        eligible=True,
    )


def fake_calibrator():
    return SimpleNamespace(
        x_thresholds=(0.0, 1.0),
        y_thresholds=(0.0, 1.0),
        policy=SimpleNamespace(out_of_bounds="clip"),
    )


def write_phase20_config(root: Path, marker: Path):
    bootstrap = root / "configs/evaluation/bootstrap-final.json"
    bootstrap.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.write_text(
        json.dumps(
            {
                "n_bootstrap": 4,
                "seed": 17,
                "final_test_access_allowed": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    slices = root / "configs/error_analysis/final.json"
    slices.parent.mkdir(parents=True, exist_ok=True)
    slices.write_text(
        json.dumps(
            {
                "spec_version": "synthetic-final-slices-v1",
                "status": "FROZEN_FOR_FINAL_TEST",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = root / "configs/final-test.json"
    payload = {
        "config_version": "final_test_evaluation_v1",
        "g3_marker_ref": str(marker.relative_to(root)),
        "required_g3_scope": "synthetic_test",
        "bootstrap_config_ref": str(bootstrap.relative_to(root)),
        "bootstrap_config_sha256": sha256_file(bootstrap),
        "retry_policy": {
            "approved_before_test": True,
            "policy_version": "synthetic-retry-v1",
        },
        "final_evaluation_plan": {
            "status": "FROZEN_BEFORE_TEST",
            "selected_models_only": True,
            "evaluate_naive": False,
            "evaluate_nonselected_families": False,
            "evaluate_lstm": False,
            "icu_error_percentiles": [],
        },
        "error_analysis_spec_ref": str(slices.relative_to(root)),
        "g4_schema": {
            "status": "FROZEN_BEFORE_TEST",
            "schema_version": "synthetic-g4-v1",
        },
        "pretest_audit_ref": "artifacts/governance/pretest.json",
        "real_test_loader_entrypoint": "synthetic.module:loader",
        "real_test_evaluator_entrypoint": "synthetic.module:evaluator",
    }
    payload["config_sha256"] = canonical_sha256(payload)
    config.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return config
