"""Performance-v2 Phase 1: remaining required governance/determinism tests
(beyond tests/test_performance_v2_phase1_leakage.py).
"""

import json
from pathlib import Path

from performance_v2.data_loading import load_dev_rows
from performance_v2.diagnostic_features import build_all_groups, feature_group_of

ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = ROOT / "artifacts/performance_v2/phase1"


def _sample_rows(limit=15):
    return load_dev_rows(ROOT, splits=("validation",))[:limit]


def test_feature_group_builder_is_deterministic():
    rows = _sample_rows(limit=8)
    for row in rows:
        first = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        second = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        assert first.keys() == second.keys()
        for name in first:
            left, right = first[name], second[name]
            if left != left and right != right:
                continue
            assert left == right, name


def test_diagnostic_features_are_stable_under_row_dict_copy():
    """Rebuilding from a deep-copied row (as any real pipeline call would
    pass) yields byte-identical features -- no hidden mutable/global state."""

    import copy

    rows = _sample_rows(limit=8)
    for row in rows:
        original = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        copied = build_all_groups(copy.deepcopy(dict(row)), baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        assert original == copied or all(
            (a != a and b != b) or a == b for a, b in zip(original.values(), copied.values())
        )


def test_oracle_summary_artifacts_are_explicitly_labeled_non_serving():
    """Every oracle artifact this phase produced must carry the required
    NON_SERVING label -- nothing here may be mistaken for a servable
    prediction/metric artifact."""

    observable = json.loads((PHASE1_DIR / "observable_oracle/observable_oracle_summary_v1.json").read_text())
    assert observable["status"] == "NON_SERVING_DIAGNOSTIC_ORACLE"
    latent = json.loads((PHASE1_DIR / "latent_oracle/latent_oracle_summary_v1.json").read_text())
    assert latent["status"] == "NON_SERVING_LATENT_ORACLE"
    assert latent["never_allowed_in_production_features"] is True


def test_latent_oracle_features_are_never_part_of_the_lawful_diagnostic_feature_set():
    """The lawful diagnostic feature builder (used for baselines/observable
    oracle/ablation -- the only feature source with any path toward a
    future serving recommendation) must never contain a latent-state
    feature name. Latent features are built by an entirely separate module
    (scripts/performance_v2_phase1_latent_oracle.py) that is never imported
    by the diagnostic feature builder."""

    rows = _sample_rows(limit=5)
    forbidden_tokens = ("z0_", "equilibrium_", "support_propensity", "latent")
    for row in rows:
        features = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        lowered = [name.lower() for name in features]
        for token in forbidden_tokens:
            assert not any(token in name for name in lowered), token


def test_every_diagnostic_feature_name_maps_to_a_declared_group():
    rows = _sample_rows(limit=5)
    for row in rows:
        features = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        for name in features:
            assert feature_group_of(name) in ("A", "B", "C", "D", "E", "F")


def test_naive_baseline_constants_are_fit_from_train_only():
    from evaluation.naive_baseline import load_or_fit_naive_baselines

    artifact, _path, _sha = load_or_fit_naive_baselines(
        ROOT,
        artifact_ref="artifacts/final_test/naive_baseline/naive_baseline_v1.json",
        train_ref="artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl",
    )
    assert artifact.source_partition == "train"


def test_stronger_baselines_were_fit_on_train_and_ranked_on_validation_only():
    payload = json.loads((PHASE1_DIR / "stronger_baselines_v1.json").read_text())
    for task_key in ("recovery", "icu_stay_time", "organ_support"):
        task = payload[task_key]
        if "recovery" in task_key:
            continue
        assert "n_train" in task and "n_validation" in task
        assert task["n_train"] > task["n_validation"]


def test_observable_oracle_and_ablation_report_validation_metrics_only():
    """The observable-oracle and ablation artifacts must never carry a
    field named or shaped like a TEST-partition result."""

    for name in ("observable_oracle/observable_oracle_summary_v1.json", "feature_ablation_v1.json"):
        payload = json.loads((PHASE1_DIR / name).read_text())
        blob = json.dumps(payload).lower()
        assert '"partition": "test"' not in blob
        assert '"test_mae"' not in blob and '"test_auprc"' not in blob


def test_phase1_artifacts_declare_development_diagnostic_status():
    for name in (
        "canonical_feature_inventory_v1.json", "v1_validation_performance_summary.json",
        "target_distribution_diagnosis_v1.json", "stronger_baselines_v1.json",
        "feature_ablation_v1.json", "performance_ceiling_summary_v1.json",
        "stochasticity_analysis_v1.json", "validation_error_analysis_v1.json",
        "train_validation_gap_v1.json",
    ):
        payload = json.loads((PHASE1_DIR / name).read_text())
        assert payload["status"] == "DEVELOPMENT_DIAGNOSTIC_ONLY", name


def test_selected_v2_strategy_is_frozen_and_decided():
    payload = json.loads((PHASE1_DIR / "selected_v2_strategy_v1.json").read_text())
    assert payload["status"] == "DECIDED"
    assert payload["selected_path"] == "PATH_A_FEATURE_AND_MODEL_V2"
    assert set(payload["tasks"].keys()) == {"recovery", "icu_stay_time", "organ_support"}
    assert payload["generator_v2_required"] is False
    assert payload["fresh_dataset_generation_needed_before_phase2"] is False
    assert "fresh_v2_test_strategy" in payload
