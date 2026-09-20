import json
from dataclasses import replace

import pytest

from evaluation.select import (
    build_selection_stage_manifest,
    select_task,
    validate_selection_stage_manifest,
)
from evaluation.selection_validation import SelectionValidationError
from experiments.search_governance import canonical_sha256
from selection_helpers import pair, replace_candidate


def mixed_results(tmp_path):
    rx, rg = pair(tmp_path, "recovery", 1.0, 2.0)
    ix, ig = pair(tmp_path, "icu_stay_time", 12.0, 10.0)
    sx, sg = pair(tmp_path, "organ_support", 0.4, 0.3)
    return {
        "recovery": select_task(task="recovery", xgboost=rx, gru=rg),
        "icu_stay_time": select_task(task="icu_stay_time", xgboost=ix, gru=ig),
        "organ_support": select_task(task="organ_support", xgboost=sx, gru=sg),
    }


def rehash(manifest):
    content = dict(manifest)
    content.pop("manifest_sha256", None)
    content["manifest_sha256"] = canonical_sha256(content)
    return content


def test_mixed_family_manifest_and_explanation_routing(tmp_path):
    manifest = build_selection_stage_manifest(mixed_results(tmp_path))
    assert manifest["tasks"]["recovery"]["explanation_method"] == "tree_shap"
    assert manifest["tasks"]["icu_stay_time"]["explanation_method"] == "integrated_gradients"
    assert manifest["tasks"]["organ_support"]["explanation_method"] == "tree_shap"
    validate_selection_stage_manifest(manifest)


def test_support_calibration_and_threshold_must_remain_pending(tmp_path):
    manifest = build_selection_stage_manifest(mixed_results(tmp_path))
    support = manifest["tasks"]["organ_support"]
    assert support["calibrator"] is None
    assert support["threshold"] is None
    attacked = json.loads(json.dumps(manifest))
    attacked["tasks"]["organ_support"]["threshold"] = 0.5
    with pytest.raises(SelectionValidationError, match="calibrator or threshold"):
        validate_selection_stage_manifest(rehash(attacked))


def test_wrong_explanation_route_is_rejected(tmp_path):
    attacked = json.loads(json.dumps(build_selection_stage_manifest(mixed_results(tmp_path))))
    attacked["tasks"]["recovery"]["explanation_method"] = "integrated_gradients"
    with pytest.raises(SelectionValidationError, match="explanation routing"):
        validate_selection_stage_manifest(rehash(attacked))


def test_artifact_hash_task_and_family_swaps_fail_closed(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.0, 2.0)
    with open(xgb.artifact_ref, "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(SelectionValidationError, match="SHA-256"):
        select_task(task="recovery", xgboost=xgb, gru=gru)

    xgb, gru = pair(tmp_path, "recovery", 1.0, 2.0)
    metadata = json.loads(open(xgb.artifact_metadata_ref, encoding="utf-8").read())
    metadata["task"] = "icu_stay_time"
    open(xgb.artifact_metadata_ref, "w", encoding="utf-8").write(json.dumps(metadata))
    with pytest.raises(SelectionValidationError, match="task"):
        select_task(task="recovery", xgboost=xgb, gru=gru)

    xgb, gru = pair(tmp_path, "recovery", 1.0, 2.0)
    metadata = json.loads(open(xgb.artifact_metadata_ref, encoding="utf-8").read())
    metadata["model_family"] = "gru"
    open(xgb.artifact_metadata_ref, "w", encoding="utf-8").write(json.dumps(metadata))
    with pytest.raises(SelectionValidationError, match="family"):
        select_task(task="recovery", xgboost=xgb, gru=gru)


def test_manifest_revalidates_artifact_after_manifest_creation(tmp_path):
    manifest = build_selection_stage_manifest(mixed_results(tmp_path))
    artifact = manifest["tasks"]["recovery"]["artifact_ref"]
    with open(artifact, "ab") as handle:
        handle.write(b"post-manifest-tamper")
    with pytest.raises(SelectionValidationError, match="SHA-256"):
        validate_selection_stage_manifest(manifest)


def test_manifest_content_and_hash_are_deterministic(tmp_path):
    first = build_selection_stage_manifest(mixed_results(tmp_path))
    second = build_selection_stage_manifest(mixed_results(tmp_path))
    assert first == second
    validate_selection_stage_manifest(first)


def test_real_selection_rejects_synthetic_search_evidence(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.0, 2.0)
    with pytest.raises(SelectionValidationError, match="REAL VALIDATION SEARCH RESULTS REQUIRED"):
        select_task(
            task="recovery",
            xgboost=xgb,
            gru=gru,
            selection_mode="real",
        )


def test_phase_12_manifest_cannot_claim_g3(tmp_path):
    attacked = json.loads(json.dumps(build_selection_stage_manifest(mixed_results(tmp_path))))
    attacked["status"] = "G3_SELECTION_CALIBRATION_FREEZE"
    with pytest.raises(SelectionValidationError, match="cannot claim G3"):
        validate_selection_stage_manifest(rehash(attacked))
