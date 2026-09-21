import copy
import json
from pathlib import Path

import pytest

from experiments.search_governance import GovernanceError, ValidationOnlyAccess
from experiments.search_registry import record_search_attempt
from experiments.xgb_phase12 import (
    MASTER_SEARCH_SEED,
    MAX_ATTEMPTS,
    RankedCandidate,
    assert_lifecycle_transition,
    candidate_manifest,
    canonical_sha256,
    choose_best,
    derived_seed,
    materialize_candidates,
    ranking_key,
    validate_candidate_list,
    validate_candidate_manifest,
    validate_search_space,
)


ROOT = Path(__file__).resolve().parents[1]
SPACE_PATH = ROOT / "configs/search_space_xgb_phase12_v1.json"


def space():
    return json.loads(SPACE_PATH.read_text())


def test_master_and_task_seed_derivation_are_fixed_and_not_python_hash_based():
    assert MASTER_SEARCH_SEED == 20260921
    assert derived_seed("recovery") == int.from_bytes(
        __import__("hashlib").sha256(b"20260921|phase12|xgb|recovery").digest()[:4], "big"
    )
    assert len({derived_seed(task) for task in ("recovery", "icu_time", "organ_support")}) == 3


@pytest.mark.parametrize("task", ("recovery", "icu_time", "organ_support"))
def test_lhs_is_deterministic_exact_30_unique_in_bounds_and_candidate31_absent(task):
    source = space()
    first = materialize_candidates(source, task)
    second = materialize_candidates(source, task)
    assert first == second
    assert len(first) == len({item["config_hash"] for item in first}) == 30
    assert first[-1]["candidate_id"].endswith("030")
    assert not any(item["candidate_id"].endswith("031") for item in first)
    validate_candidate_list(first, task, source)
    with pytest.raises(GovernanceError, match="exactly 30"):
        validate_candidate_list(first[:29], task, source)
    extra = copy.deepcopy(first[0]); extra["candidate_id"] = first[0]["candidate_id"][:-3] + "031"
    with pytest.raises(GovernanceError):
        validate_candidate_list(first + (extra,), task, source)


def test_lhs_each_continuous_dimension_uses_every_stratum_and_log_transforms():
    source = space()
    candidates = materialize_candidates(source, "recovery")
    for name in ("subsample", "colsample_bytree", "reg_alpha"):
        spec = source["parameters"][name]
        low, high = spec["minimum"], spec["maximum"]
        strata = {min(int((item["config"][name] - low) / (high - low) * 30), 29) for item in candidates}
        assert strata == set(range(30))
    import math
    for name in ("learning_rate", "reg_lambda"):
        spec = source["parameters"][name]
        low, high = math.log(spec["minimum"]), math.log(spec["maximum"])
        strata = {min(int((math.log(item["config"][name]) - low) / (high - low) * 30), 29) for item in candidates}
        assert strata == set(range(30))


def test_candidate_manifest_is_deterministic_and_binds_exact_space():
    source = space()
    first = candidate_manifest(source, "recovery", implementation_commit="a" * 40)
    second = candidate_manifest(source, "recovery", implementation_commit="a" * 40)
    assert canonical_sha256(first) == canonical_sha256(second)
    validate_candidate_manifest(first, source)
    attacked = copy.deepcopy(first); attacked["candidates"][0]["config"]["max_depth"] = 9
    with pytest.raises(GovernanceError):
        validate_candidate_manifest(attacked, source)


def test_search_space_posthoc_mutations_fail():
    source = space(); validate_search_space(source)
    for path, value in (
        (("parameters", "max_depth", "maximum"), 9),
        (("fixed_execution", "early_stopping_rounds"), 51),
    ):
        attacked = copy.deepcopy(source)
        target = attacked
        for key in path[:-1]: target = target[key]
        target[path[-1]] = value
        with pytest.raises(GovernanceError):
            validate_search_space(attacked)


def test_started_search_rejects_candidate_seed_tie_and_space_mutations():
    before = {
        "status": "RUNNING", "search_version": "v", "family": "xgboost", "tasks": [],
        "master_search_seed": 1, "sampler": "lhs", "task_sampler_seeds": {},
        "search_space_sha256": "a", "candidate_manifests": {}, "split_sha256": "b",
        "feature_schema_sha256": "c", "flat_feature_map_sha256": "d", "preprocessor_sha256": "e",
        "target_contract_sha256": "f", "recovery_target_scaler_sha256": "g",
        "support_class_weight_sha256": "h", "validation_objectives": {},
        "tie_break_policies": {}, "early_stopping": {}, "max_attempts": MAX_ATTEMPTS,
        "test_access_policy": "blocked", "implementation_commit": "i",
        "environment_sha256": "j", "train_subject_set_sha256": "k",
        "validation_subject_set_sha256": "l",
        "validation_row_keys_sha256": "m",
        "selected_models_v1_presearch": {"present": False},
        "test_gate_presearch": {"phase10_test_opened": False},
    }
    for field in ("search_space_sha256", "task_sampler_seeds", "candidate_manifests", "tie_break_policies"):
        after = copy.deepcopy(before); after[field] = "ATTACK"
        with pytest.raises(GovernanceError, match="cannot mutate"):
            assert_lifecycle_transition(before, after)


def test_test_access_is_hard_blocked():
    with pytest.raises(GovernanceError, match="TEST ACCESS FORBIDDEN"):
        ValidationOnlyAccess.require("test")


def test_scientific_retry_keeps_candidate_config_and_model_seed_and_stops_at_three(tmp_path):
    source = space()
    manifest = candidate_manifest(source, "recovery", implementation_commit="a" * 40)
    manifest.update({
        "task": "recovery", "code_commit": "a" * 40,
        "search_space_ref": "configs/search_space_xgb_phase12_v1.json",
        "split_hash": "b" * 64, "feature_version": "schema",
        "label_version": "targets", "validation_objective": "mae24",
    })
    candidate = manifest["candidates"][0]
    registry = tmp_path / "registry.csv"
    first = record_search_attempt(
        registry, manifest=manifest, candidates=manifest["candidates"],
        candidate_id=candidate["candidate_id"], run_id="attempt-1", attempt_number=1,
        attempt_status_detail="FAILED_SOFTWARE", timestamp_utc="2026-09-22T00:00:00Z",
        seed=candidate["model_seed"], run_type="scientific",
    )
    assert first["config_hash"] == candidate["config_hash"]
    with pytest.raises(GovernanceError, match="seed"):
        record_search_attempt(
            registry, manifest=manifest, candidates=manifest["candidates"],
            candidate_id=candidate["candidate_id"], run_id="bad-seed", attempt_number=2,
            attempt_status_detail="FAILED_SOFTWARE", timestamp_utc="2026-09-22T00:01:00Z",
            seed=candidate["model_seed"] + 1, retry_of_run_id="attempt-1", run_type="scientific",
        )
    second = record_search_attempt(
        registry, manifest=manifest, candidates=manifest["candidates"],
        candidate_id=candidate["candidate_id"], run_id="attempt-2", attempt_number=2,
        attempt_status_detail="FAILED_SOFTWARE", timestamp_utc="2026-09-22T00:02:00Z",
        seed=candidate["model_seed"], retry_of_run_id="attempt-1", run_type="scientific",
    )
    record_search_attempt(
        registry, manifest=manifest, candidates=manifest["candidates"],
        candidate_id=candidate["candidate_id"], run_id="attempt-3", attempt_number=3,
        attempt_status_detail="FAILED_SCIENTIFIC", timestamp_utc="2026-09-22T00:03:00Z",
        seed=candidate["model_seed"], retry_of_run_id="attempt-2", run_type="scientific",
    )
    assert second["config_hash"] == first["config_hash"] and second["seed"] == first["seed"]
    with pytest.raises(GovernanceError, match="maximum"):
        record_search_attempt(
            registry, manifest=manifest, candidates=manifest["candidates"],
            candidate_id=candidate["candidate_id"], run_id="attempt-4", attempt_number=4,
            attempt_status_detail="FAILED_SCIENTIFIC", timestamp_utc="2026-09-22T00:04:00Z",
            seed=candidate["model_seed"], retry_of_run_id="attempt-3", run_type="scientific",
        )


def test_recovery_primary_and_exact_tie_hierarchy_are_frozen():
    better_48_worse_24 = RankedCandidate("A", "COMPLETE", {"mae24": 2, "mae48": 0, "rmse24": 0})
    better_24 = RankedCandidate("B", "COMPLETE", {"mae24": 1, "mae48": 99, "rmse24": 99})
    assert choose_best((better_48_worse_24, better_24), "recovery").candidate_id == "B"
    exact_a = RankedCandidate("A", "COMPLETE", {"mae24": 1, "mae48": 2, "rmse24": 3})
    exact_b = RankedCandidate("B", "COMPLETE", {"mae24": 1, "mae48": 2, "rmse24": 3})
    assert ranking_key(exact_a, "recovery") < ranking_key(exact_b, "recovery")


def test_icu_primary_beats_rmse_and_support_primary_beats_auroc():
    icu_a = RankedCandidate("A", "COMPLETE", {"median_absolute_error_hours": 5, "mae_hours": 9, "rmse_hours": 99})
    icu_b = RankedCandidate("B", "COMPLETE", {"median_absolute_error_hours": 6, "mae_hours": 1, "rmse_hours": 1})
    assert choose_best((icu_a, icu_b), "icu_time").candidate_id == "A"
    support_a = RankedCandidate("A", "COMPLETE", {"auprc": .4, "brier": .4, "auroc": .99}, "raw_uncalibrated")
    support_b = RankedCandidate("B", "COMPLETE", {"auprc": .5, "brier": .9, "auroc": .5}, "raw_uncalibrated")
    assert choose_best((support_a, support_b), "organ_support").candidate_id == "B"
    with pytest.raises(GovernanceError, match="uncalibrated"):
        choose_best((RankedCandidate("C", "COMPLETE", {"auprc": .9, "brier": .1, "auroc": .9}, "calibrated"),), "organ_support")
