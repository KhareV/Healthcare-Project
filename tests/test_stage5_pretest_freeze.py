"""Stage-5 Part A: prove every remaining evaluation parameter was frozen
before any final-test access, using TRAIN/VALIDATION data only. Never opens
the test partition.
"""

import json
from pathlib import Path

import pytest

from evaluation.final_test import pretest_audit
from evaluation.naive_baseline import NaiveBaselineArtifact, load_or_fit_naive_baselines
from experiments.stage5_final_test import (
    ROOT,
    build_slice_specification,
    load_and_infer,
    stage5_evaluate,
)
from vedant_infra.g3 import validate_g3_marker


def test_pretest_audit_passes_before_any_test_access():
    report = pretest_audit(ROOT)
    assert report.overall_status == "PASS", report.blockers
    assert report.test_data_accessed is False
    assert report.test_loader_calls == 0


def test_g3_still_active_and_unchanged():
    marker = validate_g3_marker(ROOT / "artifacts/governance/g3_freeze.json", ROOT, expected_scope="real")
    assert marker["status"] == "G3_ACTIVE"
    assert marker["test_accessed"] is False


def test_stage4_prerequisites_present():
    for ref in (
        "artifacts/acceptance/stage4_serving_integration_freeze_v1.json",
        "artifacts/serving/final_serving_bundle_v1.json",
    ):
        assert (ROOT / ref).is_file()


def test_bootstrap_config_frozen_to_authorized_values():
    config = json.loads((ROOT / "configs/evaluation/bootstrap_v1.json").read_text())
    assert config["n_bootstrap"] == 2000
    assert config["seed"] == 20260921
    assert config["final_test_access_allowed"] is True
    assert config["ci_quantiles"] == [0.025, 0.975]


def test_naive_baseline_reuses_frozen_artifact_and_is_train_only():
    artifact, path, digest = load_or_fit_naive_baselines(
        ROOT,
        artifact_ref="artifacts/final_test/naive_baseline/naive_baseline_v1.json",
        train_ref="artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl",
    )
    assert isinstance(artifact, NaiveBaselineArtifact)
    assert artifact.source_partition == "train"
    assert artifact.recovery24_constant_delta == 0.0
    assert artifact.recovery48_constant_delta == 0.0
    assert 0.0 < artifact.organ_support_prevalence < 1.0
    assert artifact.icu_time_log1p_median > 0.0
    # Re-fitting must reproduce the identical frozen artifact (idempotent,
    # deterministic, no hidden randomness).
    refit = load_or_fit_naive_baselines(
        ROOT, artifact_ref="artifacts/final_test/naive_baseline/naive_baseline_v1.json",
        train_ref="artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl",
    )
    assert refit[2] == digest


def test_error_analysis_slices_frozen_before_test():
    spec, spec_hash = build_slice_specification(ROOT)
    assert spec.final_test_authorized is True
    names = {item.slice_name for item in spec.slices}
    assert names == {
        "baseline_sofa", "remaining_icu_duration", "observation_density",
        "cutoff_timing", "cardiac_subtype", "complete_component_sofa",
    }
    for item in spec.slices:
        assert item.status == "FROZEN"


def test_final_test_plan_artifact_frozen_pre_access():
    plan = json.loads((ROOT / "artifacts/governance/stage5_final_test_plan_v1.json").read_text())
    assert plan["status"] == "FINAL_TEST_PLAN_FROZEN_PRE_ACCESS"
    assert plan["test_accessed"] is False
    assert plan["bootstrap"]["n_bootstrap"] == 2000
    assert plan["bootstrap"]["seed"] == 20260921
    assert plan["model_exposure_plan"]["evaluate_lstm"] is False
    assert plan["model_exposure_plan"]["evaluate_nonselected_families"] is False


def test_final_test_access_state_untouched():
    """The *current* authorized exposure must be untouched: state is
    AUTHORIZED_NOT_RUN and no consumption event has happened since the most
    recent G3_FREEZE_CREATED. History may legitimately retain an earlier
    consumption event from a prior, formally-reset freeze cycle (see
    artifacts/governance/stage5_final_test_access_incident_v1.json) — that
    permanent audit trail is the point of preserving evidence, not a defect.
    """
    state = json.loads((ROOT / "artifacts/governance/test_access_state.json").read_text())
    assert state["state"] == "AUTHORIZED_NOT_RUN"
    history = state["history"]
    freeze_indices = [i for i, event in enumerate(history) if event.get("event") == "G3_FREEZE_CREATED"]
    assert freeze_indices, "no G3_FREEZE_CREATED event in history"
    current_freeze_index = freeze_indices[-1]
    assert not any(
        event.get("event") == "FINAL_TEST_ACCESS_CONSUMED"
        for event in history[current_freeze_index + 1:]
    )


def test_loader_and_evaluator_pipeline_works_on_validation_never_test():
    """Exercises the exact loader/evaluator machinery Part B will invoke,
    against the validation partition — proving correctness before the one
    authorized test exposure, without consuming it."""

    loaded = load_and_infer(ROOT, partition="validation")
    assert loaded["partition"] == "validation"
    evaluated = stage5_evaluate(loaded, root=ROOT, n_bootstrap=10, seed=1)
    assert evaluated["processed"].recovery["24h"].metrics["mae"] > 0
    assert 0 <= evaluated["processed"].support_calibrated.metrics["f1"] <= 1
    assert len(evaluated["error_rows"]) > 0


def test_stage5_loader_refuses_non_test_partition_for_evaluator():
    from experiments.stage5_final_test import stage5_evaluator

    loaded = load_and_infer(ROOT, partition="validation")
    with pytest.raises(Exception):
        stage5_evaluator(loaded)
