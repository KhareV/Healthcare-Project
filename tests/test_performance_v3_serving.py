import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from performance_v2.diagnostic_features import build_all_groups
from performance_v3 import final_models as fm
from performance_v3 import serving as sv
from performance_v3.calibration import fit_calibrator
from test_performance_v3_features import synthetic_rows

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("A", "B", "F")
FAMILY = {"recovery24": "xgboost", "recovery48": "lightgbm", "icu_stay_time": "ridge", "organ_support": "xgboost"}


def _features(row):
    return build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})


def _matrix(rows, names):
    index = {n: i for i, n in enumerate(names)}
    onehot = np.asarray(["==" in n for n in names])
    X = np.full((len(rows), len(names)), np.nan, dtype=np.float32)
    X[:, onehot] = 0.0
    for r, row in enumerate(rows):
        for k, v in _features(row).items():
            if k in index:
                X[r, index[k]] = v
    return X


@pytest.fixture(scope="module")
def rows():
    return synthetic_rows(n_stays=30, cutoffs=4, seed=9)


@pytest.fixture(scope="module")
def names(rows):
    return tuple(sorted({n for r in rows for n in _features(r) if n.split("__", 1)[0] in GROUPS}))


def make_bundle_dir(root: Path, rows, names, *, seed=0, recovery24_seed=None):
    X = _matrix(rows, names)
    rng = np.random.default_rng(seed)
    y = {
        "recovery24": np.nan_to_num(X[:, 0]) * 0.01 + rng.normal(size=len(rows)),
        "recovery48": np.nan_to_num(X[:, 1]) * 0.01 + rng.normal(size=len(rows)),
        "icu_stay_time": np.abs(rng.normal(3.0, 0.5, size=len(rows))),
        "organ_support": (rng.random(len(rows)) < 0.4).astype(float),
    }
    w = np.ones(len(rows))
    entries = {}
    for task, family in FAMILY.items():
        cfg = {"objective": "binary" if task == "organ_support" else "squarederror", "weight_mode": "none", "class_weight": "none",
               "max_depth": 3, "learning_rate": 0.2, "num_leaves": 8, "min_data_in_leaf": 3, "alpha": 1.0}
        s = recovery24_seed if (task == "recovery24" and recovery24_seed is not None) else 0
        model = fm.fit_final(family, cfg, X, y[task], w, task=task, feature_columns=names, n_estimators=15, nthread=1, seed=s)
        d = root / task
        model.save(d)
        (d / "feature_names.json").write_text(json.dumps(list(names)))
        entries[task] = {
            "family": family, "feature_variant": "V3_BF", "model_ref": f"{task}/{fm.META_NAME}",
            "model_sha256": sv._sha256_file(d / fm.META_NAME), "feature_names_ref": f"{task}/feature_names.json",
            "feature_names_sha256": sv._sha256_file(d / "feature_names.json"), "n_estimators": model.n_estimators,
            "postprocess": sv.ICU_POSTPROCESS if task == "icu_stay_time" else "none",
        }
    raw = fm.load_fitted_model(root / "organ_support").predict(X)
    params = fit_calibrator("platt", raw, y["organ_support"], w)
    manifest = {
        "artifact_version": "performance_v3_selected_models_v1", "bundle_id": "toy", "tasks": entries,
        "calibrator": {"method": "platt", "params": params, "sha256": sv.params_sha256(params)},
        "threshold": {"value": 0.4, "comparator": ">="},
    }
    path = root / "selected_models_v3.json"
    path.write_text(json.dumps(manifest))
    return path, X


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory, rows, names):
    root = tmp_path_factory.mktemp("bundle")
    path, _ = make_bundle_dir(root, rows, names)
    return root, path


@pytest.fixture(scope="module")
def bundle(bundle_dir):
    root, path = bundle_dir
    return sv.V3Bundle.load(root, path)


def test_predict_rows_output_schema_and_domains(bundle, rows):
    out = bundle.predict_rows(rows[:20])
    assert len(out) == 20
    for row, p in zip(rows[:20], out):
        assert p["stay_id"] == row["stay_id"] and p["prediction_time"] == row["prediction_time"]
        assert p["icu_remaining_hours"] == pytest.approx(np.expm1(max(p["icu_raw_log_prediction"], 0.0)))
        assert 0.0 <= p["support_calibrated_probability"] <= 1.0
        assert p["support_alert"] == (p["support_calibrated_probability"] >= 0.4)
        assert set(p["model_hashes"]) == {"recovery24", "recovery48", "icu_stay_time", "organ_support", "calibrator"}
        if p["current_sofa"] is not None:
            assert p["recovery24_sofa"] == pytest.approx(min(max(p["current_sofa"] + p["recovery24_delta"], 0.0), 24.0))
            assert p["recovery48_sofa"] == pytest.approx(min(max(p["current_sofa"] + p["recovery48_delta"], 0.0), 24.0))
        else:
            assert p["recovery24_sofa"] is None and p["recovery48_sofa"] is None


def test_batch_equals_per_row(bundle, rows):
    batch = bundle.predict_rows(rows[:12])
    single = [bundle.predict_rows([r])[0] for r in rows[:12]]
    for a, b in zip(batch, single):
        for key in ("recovery24_delta", "recovery48_delta", "icu_raw_log_prediction", "support_raw_probability", "support_calibrated_probability"):
            assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-12)  # BLAS matmul may differ by 1 ULP across batch sizes (ridge)


def test_future_and_label_fields_cannot_change_predictions(bundle, rows):
    base = bundle.predict_rows(rows[:10])
    mutated = []
    for r in rows[:10]:
        m = copy.deepcopy(r)
        m.update({"delta_sofa_24": 99.0, "delta_sofa_48": -99.0, "icu_time_log1p": 0.0, "organ_support_label": 1.0 - float(r.get("organ_support_label") or 0),
                  "recovery24_eligible": False, "recovery48_eligible": False, "icu_time_eligible": False, "organ_support_eligible": False,
                  "outtime": "2200-01-01T00:00:00Z", "support_audit_disposition": "FUTURE"})
        mutated.append(m)
    after = bundle.predict_rows(mutated)
    for a, b in zip(base, after):
        for key in ("recovery24_delta", "recovery48_delta", "icu_raw_log_prediction", "support_raw_probability", "support_calibrated_probability"):
            assert a[key] == b[key]


def test_illegal_extra_bin_is_rejected_not_read_as_future(bundle, rows):
    bad = copy.deepcopy(rows[0])
    bad["history_values"].append(list(bad["history_values"][-1]))
    bad["observation_mask"].append(list(bad["observation_mask"][-1]))
    bad["tslo_hours"].append(list(bad["tslo_hours"][-1]))
    with pytest.raises(Exception):
        bundle.predict_rows([bad])


def test_hash_mismatch_fails_closed(tmp_path, rows, names):
    path, _ = make_bundle_dir(tmp_path, rows, names)
    # 1. tamper with the model file
    f = tmp_path / "recovery24" / "model.json"
    original = f.read_bytes()
    f.write_bytes(original + b" ")
    with pytest.raises((sv.BundleError, fm.HashMismatchError)):
        sv.V3Bundle.load(tmp_path, path)
    f.write_bytes(original)
    sv.V3Bundle.load(tmp_path, path)  # restored -> loads
    # 2. tamper with the manifest-recorded meta hash
    manifest = json.loads(path.read_text())
    manifest["tasks"]["icu_stay_time"]["model_sha256"] = "0" * 64
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps(manifest))
    with pytest.raises(sv.BundleError):
        sv.V3Bundle.load(tmp_path, bad)
    # 3. tamper with the calibrator params
    manifest = json.loads(path.read_text())
    manifest["calibrator"]["params"]["slope"] = float(manifest["calibrator"]["params"]["slope"]) + 1.0
    bad.write_text(json.dumps(manifest))
    with pytest.raises(sv.BundleError):
        sv.V3Bundle.load(tmp_path, bad)
    # 4. wrong threshold comparator
    manifest = json.loads(path.read_text())
    manifest["threshold"]["comparator"] = ">"
    bad.write_text(json.dumps(manifest))
    with pytest.raises(sv.BundleError):
        sv.V3Bundle.load(tmp_path, bad)


def test_feature_names_file_must_match_model_columns(tmp_path, rows, names):
    path, _ = make_bundle_dir(tmp_path, rows, names)
    names_file = tmp_path / "organ_support" / "feature_names.json"
    names_file.write_text(json.dumps(list(reversed(names))))
    manifest = json.loads(path.read_text())
    manifest["tasks"]["organ_support"]["feature_names_sha256"] = sv._sha256_file(names_file)
    path.write_text(json.dumps(manifest))
    with pytest.raises(sv.BundleError):
        sv.V3Bundle.load(tmp_path, path)


def test_recovery_horizons_are_independent(tmp_path_factory, rows, names, bundle):
    other_root = tmp_path_factory.mktemp("bundle_b")
    other_path, _ = make_bundle_dir(other_root, rows, names, recovery24_seed=77)
    other = sv.V3Bundle.load(other_root, other_path)
    # retrain the 24h model with different parameters so its output differs
    manifest = json.loads(other_path.read_text())
    a = bundle.predict_rows(rows[:15])
    b = other.predict_rows(rows[:15])
    # 48h outputs depend only on the 48h model
    for x, y in zip(a, b):
        assert x["recovery48_delta"] == y["recovery48_delta"]
        assert x["recovery48_sofa"] == y["recovery48_sofa"]
    # and assemble_predictions never mixes the horizons
    out = sv.assemble_predictions(
        [{"stay_id": "s", "prediction_time": "t", "baseline_sofa": 10.0}], recovery24=[-3.0], recovery48=[20.0], icu_raw=[1.0],
        support_raw=[0.5], support_calibrated=[0.5], threshold=0.4, model_hashes={}, bundle_id="x")[0]
    assert out["recovery24_sofa"] == 7.0 and out["recovery48_sofa"] == 24.0  # clipped, not chained


def test_missing_sofa_inputs_for_c_and_s_columns_fail_closed(bundle):
    assert bundle._required_inputs() == (False, False)
    cols = ["A__bin0__x__value", "C__sofa__slope"]
    stub = sv.V3Bundle(bundle_id="s", models={t: SimpleNamespace(feature_columns=cols) for t in sv.TASKS}, calibrator={"method": "none"},
                       threshold=0.5, hashes={}, variants={})
    with pytest.raises(sv.FeatureContractError):
        stub.row_features([{"stay_id": "a", "prediction_time": "t"}], sofa_trends=None)


def test_predict_stay_at_uses_runtime_truncated_row(bundle, rows):
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def predict(self, stay_id, prediction_time):
            self.calls.append((stay_id, prediction_time))
            row = next(r for r in rows if r["stay_id"] == stay_id and r["prediction_time"] == prediction_time)
            return SimpleNamespace(feature_row=row)

    runtime = FakeRuntime()
    r = rows[5]
    via_runtime = bundle.predict_stay_at(runtime, r["stay_id"], r["prediction_time"])
    direct = bundle.predict_rows([r])[0]
    assert runtime.calls == [(r["stay_id"], r["prediction_time"])]
    assert via_runtime["support_raw_probability"] == direct["support_raw_probability"]
    assert via_runtime["recovery24_delta"] == direct["recovery24_delta"]


def test_explain_route_mapping():
    for family in ("xgboost", "lightgbm", "catboost"):
        assert sv.explain_route("organ_support", family) == "tree_shap"
    assert sv.explain_route("icu_stay_time", "gru") == "integrated_gradients"
    with pytest.raises(ValueError):
        sv.explain_route("nonsense", "xgboost")


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "catboost"])
def test_tree_shap_additive_on_raw_margin(family, tmp_path, rows, names):
    X = _matrix(rows, names)
    rng = np.random.default_rng(1)
    y = np.nan_to_num(X[:, 0]) * 0.02 + rng.normal(size=len(rows))
    cfg = {"objective": "squarederror", "weight_mode": "none", "max_depth": 3, "learning_rate": 0.2, "num_leaves": 8, "min_data_in_leaf": 3, "depth": 4}
    model = fm.fit_final(family, cfg, X, y, np.ones(len(rows)), task="recovery24", feature_columns=names, n_estimators=12, nthread=1)
    local = sv.explain_local(model, X[0], top_k=5)
    assert local["additivity_check_passed"] and local["output_domain"] == sv.OUTPUT_DOMAIN
    assert len(local["positive"]) <= 5 and all(v > 0 for _, v in local["positive"]) and all(v < 0 for _, v in local["negative"])
    assert all(n in names for n, _ in local["positive"] + local["negative"])
    g = sv.explain_global(model, X[:40], top_k=7)
    values = [v for _, v in g["mean_abs_shap"]]
    assert values == sorted(values, reverse=True) and len(values) == 7


# ----------------------------------------------------------------------------
# Frozen V2 comparator identity (read-only on the EXPOSED, consumed V2 cohort)
# ----------------------------------------------------------------------------

PRED_DIR = ROOT / "artifacts/performance_v2/phase4/predictions"


@pytest.mark.skipif(not (PRED_DIR / "organ_support_predictions_v1.jsonl").exists(), reason="V2 phase-4 predictions not present")
def test_v2_comparator_reproduces_stored_v2_predictions():
    from performance_v2.fresh_test_access import load_fresh_test_rows

    def load(name):
        out = {}
        for line in (PRED_DIR / name).read_text().splitlines():
            r = json.loads(line)
            out[(r["stay_id"], r["prediction_time"])] = r
        return out

    comparator = sv.V2Comparator.load(ROOT)
    fresh = load_fresh_test_rows(ROOT)  # state FINAL_V2_RUN_COMPLETED: read-only identity check only
    stored = {t: load(f"{t}_predictions_v1.jsonl") for t in ("recovery24", "recovery48", "icu_stay_time", "organ_support")}
    sample = [r for r in fresh if (r["stay_id"], r["prediction_time"]) in stored["organ_support"]][:300]
    assert len(sample) >= 100
    out = comparator.predict_rows(sample)
    checked = {"recovery24": 0, "recovery48": 0, "icu_stay_time": 0, "organ_support": 0}
    for row, p in zip(sample, out):
        key = (row["stay_id"], row["prediction_time"])
        if key in stored["organ_support"]:
            s = stored["organ_support"][key]
            assert p["support_raw_probability"] == pytest.approx(s["raw_probability"], abs=1e-6)
            assert p["support_calibrated_probability"] == pytest.approx(s["calibrated_probability"], abs=1e-6)
            assert p["support_alert"] == s["predicted_alert"]
            checked["organ_support"] += 1
        if key in stored["recovery24"]:
            assert p["recovery24_delta"] == pytest.approx(stored["recovery24"][key]["prediction_delta24"], abs=1e-6)
            checked["recovery24"] += 1
        if key in stored["recovery48"]:
            rec = stored["recovery48"][key]
            field = [k for k in rec if k.startswith("prediction_delta")][0]
            assert p["recovery48_delta"] == pytest.approx(rec[field], abs=1e-6)
            checked["recovery48"] += 1
        if key in stored["icu_stay_time"]:
            assert p["icu_remaining_hours"] == pytest.approx(stored["icu_stay_time"][key]["prediction_hours"], abs=1e-5)
            checked["icu_stay_time"] += 1
    assert all(v > 0 for v in checked.values()), checked
