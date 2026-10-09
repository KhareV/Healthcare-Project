import copy
import json
import random
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from evaluation.weights import compute_stay_weights
from performance_v2.context_normalization import fit_normalization
from performance_v2.context_v2 import VARIANT_KEYS
from performance_v2.diagnostic_features import feature_group_of
from performance_v2.feature_matrix import build_feature_matrix_fast
from performance_v3 import features as f3

ROOT = Path(__file__).resolve().parents[1]
REAL_PRE = ROOT / "artifacts/performance_v2/phase3/fresh_test_cohort/pre_split/pre_split_scientific_package.jsonl"
REAL_STATICS = ROOT / "artifacts/performance_v2/phase3/fresh_test_cohort/timeline/canonical_statics.jsonl"

TEMPORAL = [
    "pao2__latest", "fio2__latest", "platelet_count__latest", "bilirubin_total__latest",
    "mean_arterial_pressure__latest", "glasgow_coma_scale__latest", "creatinine__latest",
    "urine_output_volume__sum", "heart_rate__latest", "systolic_blood_pressure__latest",
    "diastolic_blood_pressure__latest", "respiratory_rate__latest", "oxygen_saturation__latest",
    "temperature__latest", "lactate__latest", "vasopressor_on", "invasive_ventilation_on",
    "norepinephrine_rate", "epinephrine_rate", "dopamine_rate", "dobutamine_rate",
]
GROUPS = ["SYNTHETIC_HF", "SYNTHETIC_ISCHEMIC", "SYNTHETIC_ARRHYTHMIA"]


def synthetic_rows(n_stays=24, cutoffs=4, seed=3):
    rng = random.Random(seed)
    rows = []
    for s in range(n_stays):
        stay = f"SYN-V3-D1-E-{s:08d}"
        subject = f"SYN-V3-D1-S-{s:08d}"
        group = GROUPS[s % 3]
        for k in range(cutoffs):
            values, mask, tslo = [], [], []
            for b in range(8):
                v, m, t = [], [], []
                for c in range(21):
                    observed = rng.random() < 0.6
                    binary = TEMPORAL[c] in ("vasopressor_on", "invasive_ventilation_on")
                    val = None
                    if observed or binary:
                        val = float(rng.random() < 0.3) if binary else rng.gauss(50, 10)
                    v.append(val)
                    m.append(bool(observed))
                    t.append(float(rng.randint(0, 48)))
                values.append(v)
                mask.append(m)
                tslo.append(t)
            elapsed = 24.0 + 6.0 * k
            rows.append({
                "subject_id": subject, "stay_id": stay, "grid_index": k, "cohort_id": "SYN-V3-D1", "split": "dev",
                "prediction_time": f"2108-09-{9 + k // 4:02d}T{(6 * k) % 24:02d}:00:00.000000Z",
                "history_values": values, "observation_mask": mask, "tslo_hours": tslo, "padding_mask": [False] * 8,
                "temporal_feature_names": list(TEMPORAL),
                "static_feature_names": ["age_years", "sex_category", "cardiac_condition_group"],
                "static_features": [30 + s, "MALE" if s % 2 else "FEMALE", group],
                "elapsed_episode_hours_at_t": elapsed, "cutoff_index": k, "hours_since_first_eligible_cutoff": elapsed - 24.0,
                "baseline_sofa": float(rng.randint(0, 12)),  # D1: current SOFA(t) exists for every row
                "delta_sofa_24": rng.gauss(0, 1) if k < 3 else None, "delta_sofa_48": rng.gauss(0, 1) if k < 2 else None,
                "icu_time_log1p": rng.uniform(1, 4), "organ_support_label": float(rng.random() < 0.4) if k < 3 else None,
                "recovery24_eligible": k < 3, "recovery48_eligible": k < 2, "icu_time_eligible": True, "organ_support_eligible": k < 3,
                "support_audit_disposition": "X", "outtime": "2108-09-12T00:00:00Z",
            })
    return rows


def real_sample(n=200):
    if not (REAL_PRE.exists() and REAL_STATICS.exists()):
        return []
    from performance_v2.data_loading import _utc, load_jsonl

    statics = {r["stay_id"]: r for r in load_jsonl(REAL_STATICS)}
    out = []
    for row in load_jsonl(REAL_PRE):
        s = statics[row["stay_id"]]
        elapsed = (_utc(row["prediction_time"]) - _utc(s["intime"])).total_seconds() / 3600.0
        r = dict(row)
        r.update({"elapsed_episode_hours_at_t": elapsed, "cutoff_index": int(row["grid_index"]),
                  "hours_since_first_eligible_cutoff": elapsed - 24.0, "cohort_id": "SYN-V3-D1", "split": "dev"})
        if r.get("baseline_sofa") is None:
            continue  # D1: null-baseline rows have no current SOFA without a SOFA(t) cache; covered by dedicated tests
        out.append(r)
        if len(out) >= n:
            break
    return out


@pytest.fixture(scope="module")
def rows():
    return synthetic_rows()


@pytest.fixture(scope="module")
def trends(rows):
    rng = random.Random(11)
    out = {}
    for r in rows[::2]:
        trend = {k: float(rng.randint(0, 10)) for k in ("t_minus_6h", "t_minus_12h", "t_minus_24h", "t_minus_48h")}
        trend["t"] = float(r["baseline_sofa"])
        out[(r["stay_id"], r["prediction_time"])] = trend
    return out


# ------------------------------------------------------------ equality with V2
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cache_equals_v2_fast_matrix_synthetic(rows, trends, n_jobs):
    cache = f3.build_feature_cache(rows, sofa_trends=trends, n_jobs=n_jobs, chunk_size=17)
    ref, ref_names = build_feature_matrix_fast(rows, trends)
    assert list(cache.names) == list(ref_names)
    assert cache.matrix.dtype == np.float32 and cache.matrix.shape == ref.shape
    np.testing.assert_allclose(cache.matrix, ref, equal_nan=True, rtol=0, atol=0)
    assert cache.meta["stay_id"].shape == (len(rows),)


@pytest.mark.skipif(not real_sample(1), reason="V2 fresh-cohort sample absent (gitignored artifact)")
def test_cache_equals_v2_fast_matrix_real_sample():
    sample = real_sample(200)
    cache = f3.build_feature_cache(sample, n_jobs=2, chunk_size=50)
    ref, ref_names = build_feature_matrix_fast(sample, {})
    assert list(cache.names) == list(ref_names)
    np.testing.assert_allclose(cache.matrix, ref, equal_nan=True, rtol=0, atol=0)


def test_streaming_callable_source_and_supplied_names(rows):
    full = f3.build_feature_cache(rows, n_jobs=1)
    streamed = f3.build_feature_cache(lambda: (r for r in rows), n_jobs=1, names=full.names, chunk_size=9)
    np.testing.assert_array_equal(np.nan_to_num(streamed.matrix, nan=-1), np.nan_to_num(full.matrix, nan=-1))
    assert streamed.input_sha256 == full.input_sha256
    with pytest.raises(f3.FeatureCacheError):
        f3.build_feature_cache(rows, n_jobs=1, names=full.names[:-5])  # unseen names fail closed


def test_save_load_roundtrip_and_tamper_detection(rows, tmp_path):
    cache = f3.build_feature_cache(rows, n_jobs=1)
    npz, sidecar = f3.save_feature_cache(cache, tmp_path / "c")
    loaded = f3.load_feature_cache(tmp_path / "c")
    assert loaded.names == cache.names and loaded.input_sha256 == cache.input_sha256
    np.testing.assert_array_equal(np.nan_to_num(loaded.matrix, nan=-1), np.nan_to_num(cache.matrix, nan=-1))
    for k in cache.meta:
        np.testing.assert_array_equal(loaded.meta[k], cache.meta[k])
    meta = json.loads(sidecar.read_text())
    assert meta["builder_version"] == "performance_v3_feature_cache_v1" and len(meta["matrix_sha256"]) == 64
    meta["matrix_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(f3.FeatureCacheError):
        f3.load_feature_cache(tmp_path / "c")


# ------------------------------------------------------------ cutoff safety
def test_post_cutoff_fields_cannot_change_any_feature(rows, trends):
    base = f3.build_feature_cache(rows, sofa_trends=trends, n_jobs=1)
    poisoned = copy.deepcopy(rows)
    rng = random.Random(5)
    for r in poisoned:
        r["delta_sofa_24"], r["delta_sofa_48"] = 99.0, -99.0
        r["icu_time_log1p"] = rng.uniform(100, 200)
        r["organ_support_label"] = 1.0 - (r["organ_support_label"] or 0.0)
        r["recovery24_eligible"] = not r["recovery24_eligible"]
        r["recovery48_eligible"] = not r["recovery48_eligible"]
        r["organ_support_eligible"] = not r["organ_support_eligible"]
        r["support_audit_disposition"] = "POISON"
        r["outtime"] = "1999-01-01T00:00:00Z"
    after = f3.build_feature_cache(poisoned, sofa_trends=trends, names=base.names, n_jobs=1)
    np.testing.assert_array_equal(np.nan_to_num(after.matrix, nan=-7e9), np.nan_to_num(base.matrix, nan=-7e9))
    for forbidden in ("delta_sofa_24", "icu_time_log1p"):
        assert forbidden not in base.names


def test_fields_deleted_entirely_do_not_break_or_change_features(rows):
    base = f3.build_feature_cache(rows, n_jobs=1)
    stripped = [{k: v for k, v in r.items() if k not in ("delta_sofa_24", "delta_sofa_48", "icu_time_log1p", "organ_support_label", "outtime", "support_audit_disposition", "recovery24_eligible", "recovery48_eligible", "icu_time_eligible", "organ_support_eligible")} for r in rows]
    after = f3.build_feature_cache(stripped, names=base.names, n_jobs=1)
    np.testing.assert_array_equal(np.nan_to_num(after.matrix, nan=-7e9), np.nan_to_num(base.matrix, nan=-7e9))


def test_ninth_bin_rejected_fail_closed(rows):
    bad = copy.deepcopy(rows[:2])
    bad[0]["history_values"].append([0.0] * 21)
    with pytest.raises(Exception):
        f3.build_feature_cache(bad, n_jobs=1)


def test_no_prohibited_tokens_in_feature_names(rows):
    cache = f3.build_feature_cache(rows, n_jobs=1)
    for name in cache.names:
        assert not any(tok in name.lower() for tok in f3.forbidden_name_tokens()), name
    assert {feature_group_of(n) for n in cache.names} == {"A", "B", "C", "D", "E", "F"}


# ------------------------------------------------------------ variants
def test_variant_columns_exact_context_keys(rows):
    cache = f3.build_feature_cache(rows, n_jobs=1)
    names = cache.names

    def selected(variant, task=None, **kw):
        mask = f3.variant_columns(names, variant, task, **kw)
        return [n for n, m in zip(names, mask) if m]

    a_count = sum(feature_group_of(n) == "A" for n in names)
    ctrl_rec = selected("V3_CTRL", "recovery24")
    assert len(ctrl_rec) == a_count + 2 and set(ctrl_rec) - {n for n in names if feature_group_of(n) == "A"} == set(VARIANT_KEYS["B_MIN"])
    assert len(selected("V3_CTRL", "icu_stay_time")) == a_count + 12
    assert len(selected("V3_CTRL", "organ_support")) == a_count + 4
    assert len(selected("V3_B")) == a_count + 4
    assert len(selected("V3_BF")) == a_count + 12
    n_c = sum(feature_group_of(n) == "C" for n in names)
    n_d = sum(feature_group_of(n) == "D" for n in names)
    n_e = sum(feature_group_of(n) == "E" for n in names)
    assert len(selected("V3_BC")) == a_count + 4 + n_c
    assert len(selected("V3_BD")) == a_count + 4 + n_d
    assert len(selected("V3_BE")) == a_count + 4 + n_e
    assert len(selected("V3_FULL")) == a_count + 12 + n_c + n_d + n_e
    with pytest.raises(ValueError):
        f3.variant_columns(names, "V3_CTRL")
    with pytest.raises(ValueError):
        f3.variant_columns(names, "V9")
    with pytest.raises(NotImplementedError):
        f3.variant_columns(names, "V3_BFS")
    with pytest.raises(NotImplementedError):
        f3.variant_columns(names, "V3_FULL", include_s=True)


def test_sofa_component_group_and_bfs_variant(rows):
    comps = {}
    for r in rows:
        comps[(r["stay_id"], r["prediction_time"])] = {c: (i % 5, bool(i % 2)) for i, c in enumerate(f3._component_names())}
    cache = f3.build_feature_cache(rows, sofa_components=comps, n_jobs=1)
    s_names = [n for n in cache.names if feature_group_of(n) == "S"]
    assert len(s_names) == 12 and tuple(sorted(s_names)) == tuple(sorted(f3.sofa_component_feature_names()))
    mask = f3.variant_columns(cache.names, "V3_BFS")
    assert sum(feature_group_of(n) == "S" for n, m in zip(cache.names, mask) if m) == 12
    assert all(np.isfinite(cache.matrix[:, cache.column_index()[n]]).all() for n in s_names)
    assert all(np.isnan(v) for v in f3.sofa_component_features(None).values())


# ------------------------------------------------------------ normalization / weights
@pytest.mark.parametrize("variant", ["V0", "B_MIN", "B_FULL", "B_PLUS_F"])
def test_context_normalization_matches_v2_fit(rows, variant):
    cache = f3.build_feature_cache(rows, n_jobs=1)
    train_mask = np.asarray([int(s[-3:]) % 3 != 0 for s in cache.meta["subject_id"]])
    train_rows = [r for r, m in zip(rows, train_mask) if m]
    ref = fit_normalization(train_rows, variant)
    got = f3.fit_context_normalization(cache, variant, row_mask=train_mask)
    assert got["feature_names"] == ref["feature_names"]
    np.testing.assert_allclose(got["mean"], ref["mean"], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(got["std"], ref["std"], rtol=1e-6)
    assert got["fit_subject_hash"] == ref["fit_subject_hash"]
    assert f3.fit_context_normalization(train_rows, variant)["fit_subject_hash"] == ref["fit_subject_hash"]
    applied = f3.apply_context_normalization(cache, got)
    assert applied.shape == (cache.n_rows, len(ref["feature_names"])) and np.isfinite(applied).all()


def test_stay_balanced_weights_and_task_masks(rows):
    cache = f3.build_feature_cache(rows, n_jobs=1)
    for task, (elig, target) in f3.TASK_COLUMNS.items():
        w = f3.weights_stay_balanced(cache.meta, task)
        mask = f3.task_mask(cache.meta, task)
        assert (w[~mask] == 0).all() and (w[mask] > 0).all()
        stays = cache.meta["stay_id"]
        for stay in np.unique(stays[mask]):
            assert w[(stays == stay) & mask].sum() == pytest.approx(1.0)
        ref = compute_stay_weights([str(s) for s in stays], [bool(m) for m in mask])
        np.testing.assert_allclose(w, ref.weights)
    assert f3.canonical_task("icu_stay_time") == "icu_time"
    with pytest.raises(ValueError):
        f3.weights_stay_balanced(cache.meta, "mortality")


# ------------------------------------------------------------ memory sanity
def test_memory_is_dominated_by_the_float32_matrix(rows):
    big = rows * 6  # ~576 rows
    tracemalloc.start()
    cache = f3.build_feature_cache(big, n_jobs=1, chunk_size=32)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert cache.matrix.dtype == np.float32
    matrix_bytes = cache.matrix.nbytes
    # peak holds the concatenated matrix + chunk list (<= ~3x) and only one chunk of per-row dicts
    assert peak < 6 * matrix_bytes + 50_000_000
