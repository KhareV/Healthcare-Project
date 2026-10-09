import csv
import json
import random
from pathlib import Path

import pytest

from performance_v3 import data as v3data
from performance_v3.splits import (
    V3_CROSSFIT_VERSION,
    clone_safe_folds,
    crossfit_fold_for_subject,
    learning_curve_subsample,
    nested_subject_subsets,
    subject_rank_key,
)

ROOT = Path(__file__).resolve().parents[1]
V2_FOLD_CSV = ROOT / "artifacts/performance_v2/phase3/dev_crossfit_fold_assignment_v1.csv"
V2_FOLD_VERSION = "performance_v2_phase3_dev_crossfit_fold_v1"

TEMPORAL = [
    "pao2__latest", "fio2__latest", "platelet_count__latest", "bilirubin_total__latest",
    "mean_arterial_pressure__latest", "glasgow_coma_scale__latest", "creatinine__latest",
    "urine_output_volume__sum", "heart_rate__latest", "systolic_blood_pressure__latest",
    "diastolic_blood_pressure__latest", "respiratory_rate__latest", "oxygen_saturation__latest",
    "temperature__latest", "lactate__latest", "vasopressor_on", "invasive_ventilation_on",
    "norepinephrine_rate", "epinephrine_rate", "dopamine_rate", "dobutamine_rate",
]


def _ids(n, prefix="SYN-V3-D1-S-"):
    return [f"{prefix}{i:08d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------- splits
@pytest.mark.skipif(not V2_FOLD_CSV.exists(), reason="V2 fold assignment artifact absent")
def test_fold_function_reproduces_v2_assignment_exactly():
    with V2_FOLD_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1700
    for row in rows:
        assert crossfit_fold_for_subject(row["subject_id"], 5, V2_FOLD_VERSION) == int(row["crossfit_fold_5"])


def test_v3_fold_default_version_and_balance():
    assert V3_CROSSFIT_VERSION == "performance_v3_dev_crossfit_v1"
    ids = _ids(2000)
    folds = [crossfit_fold_for_subject(s) for s in ids]
    assert set(folds) == {0, 1, 2, 3, 4}
    assert all(300 < folds.count(f) < 500 for f in range(5))
    assert folds == [crossfit_fold_for_subject(s, 5, V3_CROSSFIT_VERSION) for s in ids]


def test_nested_subsets_are_prefixes_and_deterministic():
    ids = _ids(500)
    shuffled = ids[:]
    random.Random(7).shuffle(shuffled)
    a = nested_subject_subsets(ids, [50, 100, 250, 500], salt="lc")
    b = nested_subject_subsets(shuffled, [500, 250, 100, 50], salt="lc")
    assert a == b
    sizes = sorted(a)
    for small, large in zip(sizes, sizes[1:]):
        assert a[large][: len(a[small])] == a[small]
    assert len(set(a[500])) == 500
    assert nested_subject_subsets(ids, [50], salt="other")[50] != a[50]
    assert subject_rank_key("x", "s") == subject_rank_key("x", "s")


def test_nested_subsets_refuse_off_grid_and_oversize():
    ids = _ids(100)
    with pytest.raises(ValueError):
        nested_subject_subsets(ids, [10, 40], salt="s", allowed_prefix_sizes=[10, 20])
    with pytest.raises(ValueError):
        nested_subject_subsets(ids, [101], salt="s")
    with pytest.raises(ValueError):
        nested_subject_subsets(ids + ids[:1], [10], salt="s")


def test_learning_curve_repeats_are_independent_and_reproducible():
    ids = _ids(400)
    r0 = learning_curve_subsample(ids, 100, 0, salt="lc")
    r1 = learning_curve_subsample(ids, 100, 1, salt="lc")
    assert r0 == learning_curve_subsample(ids, 100, 0, salt="lc")
    assert len(r0) == len(set(r0)) == 100
    assert r0 != r1
    assert len(set(r0) & set(r1)) < 60  # expected overlap ~25


def test_clone_safe_folds_share_fold_of_lexicographic_first_member():
    ids = _ids(60)
    groups = [[ids[10], ids[3], ids[50]], [ids[20], ids[21]]]
    folds = clone_safe_folds(ids, groups, 5, V3_CROSSFIT_VERSION)
    first = min(groups[0])
    assert {folds[m] for m in groups[0]} == {crossfit_fold_for_subject(first)}
    assert folds[ids[20]] == folds[ids[21]] == crossfit_fold_for_subject(ids[20])
    assert folds[ids[0]] == crossfit_fold_for_subject(ids[0])
    mapping = {ids[10]: "g", ids[3]: "g", ids[50]: "g"}
    assert clone_safe_folds(ids, mapping)[ids[50]] == folds[ids[50]]


def test_clone_groups_overlapping_fail_closed():
    ids = _ids(10)
    with pytest.raises(ValueError):
        clone_safe_folds(ids, [[ids[0], ids[1]], [ids[1], ids[2]]])


# ---------------------------------------------------------------- data
def _fixture_row(subject_id, stay_id, grid_index=0, hour_offset=24):
    intime = "2108-09-08T00:00:00.000000Z"
    return (
        {
            "subject_id": subject_id, "stay_id": stay_id, "grid_index": grid_index,
            "prediction_time": f"2108-09-09T{hour_offset % 24:02d}:00:00.000000Z" if hour_offset < 48 else "2108-09-10T06:00:00.000000Z",
            "history_values": [[0.0] * 21 for _ in range(8)], "observation_mask": [[False] * 21 for _ in range(8)],
            "tslo_hours": [[0.0] * 21 for _ in range(8)], "padding_mask": [True] * 8,
            "temporal_feature_names": TEMPORAL, "static_feature_names": ["age_years", "sex_category", "cardiac_condition_group"],
            "static_features": [50, "MALE", "SYNTHETIC_HF"], "baseline_sofa": None,
            "icu_time_eligible": True, "icu_time_log1p": 2.0,
        },
        {"stay_id": stay_id, "subject_id": subject_id, "intime": intime, "age_years": 50, "sex_category": "MALE", "cardiac_condition_group": "SYNTHETIC_HF"},
    )


def _write_cohort(base: Path, cohort_name: str, subjects, rows_per=2):
    cdir = base / cohort_name
    (cdir / "pre_split").mkdir(parents=True)
    (cdir / "timeline").mkdir(parents=True)
    with (cdir / "pre_split/pre_split_scientific_package.jsonl").open("w") as pre, (cdir / "timeline/canonical_statics.jsonl").open("w") as st:
        for i, subject in enumerate(subjects):
            stay = subject.replace("-S-", "-E-")
            for k in range(rows_per):
                row, static = _fixture_row(subject, stay, grid_index=k, hour_offset=24 + 6 * k)
                pre.write(json.dumps(row) + "\n")
            st.write(json.dumps(static) + "\n")
    return cdir


def test_load_cohort_rows_enrichment_and_tags(tmp_path):
    cdir = _write_cohort(tmp_path, "SYN-V3-D1", _ids(3))
    rows = v3data.load_cohort_rows(tmp_path, cdir, cohort_id="SYN-V3-D1", role="dev")
    assert len(rows) == 6
    assert all(r["split"] == "dev" and r["cohort_id"] == "SYN-V3-D1" for r in rows)
    r = rows[1]
    assert r["cutoff_index"] == 1
    assert r["elapsed_episode_hours_at_t"] == pytest.approx(30.0)
    assert r["hours_since_first_eligible_cutoff"] == pytest.approx(6.0)
    assert r["age_years"] == 50 and r["intime"].startswith("2108-09-08")
    compact = v3data.load_cohort_rows(tmp_path, cdir, cohort_id="SYN-V3-D1", role="lce", compact=True)
    assert all(not (set(v3data.HEAVY_FIELDS) & set(c)) for c in compact)
    assert all(c["split"] == "lce" for c in compact)


@pytest.mark.parametrize("role", ["test", "final", "sealed", "validation", "train", ""])
def test_non_development_roles_refused(tmp_path, role):
    cdir = _write_cohort(tmp_path, "SYN-V3-D1", _ids(1))
    with pytest.raises(v3data.CohortAccessError):
        v3data.load_cohort_rows(tmp_path, cdir, cohort_id="SYN-V3-D1", role=role)


@pytest.mark.parametrize("cohort_id", ["SYN-V3-T", "SYN-V3-TEST", "SYN-V2", "SYN-V2-X"])
def test_sealed_or_exposed_cohort_ids_refused(tmp_path, cohort_id):
    cdir = _write_cohort(tmp_path, "SYN-V3-D1", _ids(1))
    with pytest.raises(v3data.CohortAccessError):
        v3data.load_cohort_rows(tmp_path, cdir, cohort_id=cohort_id, role="dev")


@pytest.mark.parametrize("dirname", ["final_test_cohort", "fresh_test_cohort", "SYN-V3-T", "sealed_cohort"])
def test_sealed_looking_directories_refused(tmp_path, dirname):
    cdir = _write_cohort(tmp_path, dirname, _ids(1))
    with pytest.raises(v3data.CohortAccessError):
        v3data.load_cohort_rows(tmp_path, cdir, cohort_id="SYN-V3-D1", role="dev")


def test_test_namespace_subjects_refused_even_in_neutral_directory(tmp_path):
    cdir = _write_cohort(tmp_path, "SYN-V3-D2", _ids(1, prefix="SYN-V3-T-S-"))
    with pytest.raises(v3data.CohortAccessError):
        v3data.load_cohort_rows(tmp_path, cdir, cohort_id="SYN-V3-D2", role="dev")
    cdir2 = _write_cohort(tmp_path, "SYN-V3-D3", _ids(1, prefix="SYN-V2-S-"))
    with pytest.raises(v3data.CohortAccessError):
        v3data.load_cohort_rows(tmp_path, cdir2, cohort_id="SYN-V3-D3", role="dev")


def test_assemble_dev_pool_uniqueness_ordering_and_compact(tmp_path):
    d1 = _write_cohort(tmp_path, "SYN-V3-D1", _ids(3, "SYN-V3-D1-S-"))
    d2 = _write_cohort(tmp_path, "SYN-V3-D2", _ids(2, "SYN-V3-D2-S-"))
    pool = v3data.assemble_dev_pool(tmp_path, [("SYN-V3-D1", d1), ("SYN-V3-D2", d2)], include_v1=False)
    assert [r["cohort_id"] for r in pool] == ["SYN-V3-D1"] * 6 + ["SYN-V3-D2"] * 4
    again = v3data.assemble_dev_pool(tmp_path, {"SYN-V3-D1": d1, "SYN-V3-D2": d2}, include_v1=False)
    assert [(r["subject_id"], r["grid_index"]) for r in pool] == [(r["subject_id"], r["grid_index"]) for r in again]
    small = v3data.assemble_dev_pool(tmp_path, [("SYN-V3-D1", d1)], include_v1=False, compact=True)
    assert all("history_values" not in r for r in small)
    dup = _write_cohort(tmp_path, "SYN-V3-D4", _ids(1, "SYN-V3-D1-S-"))
    with pytest.raises(ValueError):
        v3data.assemble_dev_pool(tmp_path, [("SYN-V3-D1", d1), ("SYN-V3-D4", dup)], include_v1=False)
    with pytest.raises(ValueError):
        v3data.assemble_dev_pool(tmp_path, [("SYN-V3-D1", d1), ("SYN-V3-D1", d1)], include_v1=False)


V1_PRE = ROOT / "artifacts/data/synthetic/pre_split/final/phase9_final_v1/pre_split_scientific_package.jsonl"
V1_STAT = ROOT / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_statics.jsonl"


@pytest.mark.skipif(not (V1_PRE.exists() and V1_STAT.exists()), reason="V1 final artifacts absent")
def test_enrichment_matches_v2_load_dev_rows_and_v1_tagging():
    from performance_v2.data_loading import load_jsonl

    v2_rows = {(r["stay_id"], r["prediction_time"]): r for r in v3data.load_dev_rows(ROOT)}
    assert v2_rows, "V1 dev rows expected"
    statics = {r["stay_id"]: r for r in load_jsonl(V1_STAT)}
    checked = 0
    for raw in load_jsonl(V1_PRE):
        key = (raw["stay_id"], raw["prediction_time"])
        if key not in v2_rows:
            continue
        mine = v3data._enrich(raw, statics[raw["stay_id"]])
        ref = v2_rows[key]
        for field in ("elapsed_episode_hours_at_t", "cutoff_index", "hours_since_first_eligible_cutoff", "age_years", "sex_category", "cardiac_condition_group", "intime"):
            assert mine[field] == ref[field]
        checked += 1
        if checked >= 40:
            break
    assert checked == 40
    first = next(v3data.iter_v1_dev_rows(ROOT, compact=True))
    assert first["cohort_id"] == "SYN" and first["role"] == "dev" and first["split"] in ("train", "validation")
    assert "history_values" not in first
