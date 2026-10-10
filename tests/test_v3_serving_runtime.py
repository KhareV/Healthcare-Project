"""Gate-0 V3 serving tests: parity with the offline frozen bundle, defect-D1 regression, cutoff safety,
fail-closed loading, and absence of V2 models from the V3 path."""
import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")
OUT_KEYS = ("recovery24_delta", "recovery48_delta", "icu_raw_log_prediction", "support_raw_probability", "support_calibrated_probability")


def _dt(v):
    return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(timezone.utc)


@pytest.fixture(scope="module")
def runtime():
    from serving.v3.runtime import V3ServingRuntime

    return V3ServingRuntime(ROOT)


@pytest.fixture(scope="module")
def demo():
    return json.loads((ROOT / "configs/performance_v2/v2_demo_manifest_v1.json").read_text())["demo_subjects"]


def test_runtime_loaded_the_four_selected_v3_heads(runtime):
    fam = {t: runtime.bundle.models[t].family for t in TASKS}
    assert fam == {"recovery24": "xgboost", "recovery48": "catboost", "icu_stay_time": "xgboost", "organ_support": "xgboost"}
    assert runtime.release_info["selected_models_sha256"] == "af9014a33d0a219a3f64b713c6812add46b07f44efcb4d212674ea47b7011961"
    assert runtime.bundle.threshold == 0.3021480061849302 and runtime.bundle.calibrator["method"] == "platt"
    assert not hasattr(runtime, "calibration_knots") and not hasattr(runtime, "group_a_names")  # no V2 model state


def test_parity_with_offline_frozen_bundle_including_recovery24_ineligible_cutoffs(runtime, demo):
    """D1 regression: cutoffs with <24h follow-up (recovery24-ineligible) get a finite, non-null-SOFA prediction
    identical to the offline V3 bundle applied to the pre-split development rows."""
    from performance_v3 import data as v3data

    trends = {tuple(k.split("||", 1)): v for k, v in json.loads((ROOT / "artifacts/performance_v3/cache/sofa_V1.json").read_text())["trends"].items()}
    want = {(s["stay_id"]) for s in demo}
    rows = {(r["stay_id"], _dt(r["prediction_time"])): r for r in v3data.iter_v1_dev_rows(ROOT) if r["stay_id"] in want}
    checked = ineligible = 0
    max_diff = 0.0
    for s in demo:
        outtime = _dt(runtime._statics_by_stay[s["stay_id"]]["outtime"])
        for t in s["legal_cutoffs"]:
            key = (s["stay_id"], _dt(t))
            if key not in rows:
                continue
            pred = runtime.predict(s["stay_id"], t)
            offline = runtime.bundle.predict_rows([rows[key]], sofa_trends=trends)[0]
            for k in OUT_KEYS:
                max_diff = max(max_diff, abs(getattr(pred, k) - offline[k]))
            assert np.isfinite(pred.current_sofa) and np.isfinite(pred.icu_remaining_hours)
            assert pred.feature_row["baseline_sofa"] is not None
            assert not np.isnan(pred.matrices["recovery24"][0, pred.feature_names["recovery24"].index("B__current_sofa")])
            if (_dt(t) - outtime).total_seconds() / 3600 + 24 > 0:  # t + 24h > outtime
                ineligible += 1
                assert rows[key]["baseline_sofa"] is None  # offline harness has the D1 null here; V3 does not
            checked += 1
    assert checked >= 20 and ineligible >= 3
    assert max_diff <= 1e-5, max_diff


def test_future_rows_cannot_change_a_prediction(runtime, demo):
    s = demo[0]
    sid = s["stay_id"]; t = s["legal_cutoffs"][3]; cutoff = _dt(t)
    statics = dict(runtime._statics_by_stay[sid])
    events = [dict(e) for e in runtime._events_by_stay[sid]]
    supports = [dict(r) for r in runtime._supports_by_stay.get(sid, [])]

    def register(new_id, events, supports):
        row = dict(statics, stay_id=new_id)
        for e in events:
            e["stay_id"] = new_id
        for r in supports:
            r["stay_id"] = new_id
            r["support_event_id"] = f'{r["support_event_id"]}-{new_id}'  # unique refs inside the shared support provider
        runtime.register_ephemeral_stay(stay_id=new_id, subject_id=statics["subject_id"], statics_row=row, events=events,
                                        legal_cutoffs=tuple(s["legal_cutoffs"]), owner_user_id="pytest-owner")
        if supports:
            runtime.register_ephemeral_support(stay_id=new_id, support_rows=supports)

    base_events = [dict(e) for e in events]
    mutated = []
    for e in events:
        e = dict(e)
        if _dt(e["event_time"]) > cutoff:
            if hash(e.get("event_id")) % 3 == 0:
                continue  # delete
            e["value_numeric"] = float(e["value_numeric"]) * 3.0 + 7.0
        mutated.append(e)
    base_supports = [r for r in supports]
    mut_supports = [r for r in supports if _dt(r["interval_start"]) <= cutoff]
    register("CUSTOM-PERTURB-A", base_events, [dict(r) for r in base_supports])
    register("CUSTOM-PERTURB-B", mutated, [dict(r) for r in mut_supports])
    a = runtime.predict("CUSTOM-PERTURB-A", t)
    b = runtime.predict("CUSTOM-PERTURB-B", t)
    for k in OUT_KEYS + ("current_sofa", "icu_remaining_hours"):
        assert getattr(a, k) == pytest.approx(getattr(b, k), abs=1e-9), k
    assert any(_dt(e["event_time"]) > cutoff for e in events)


def test_v3_path_never_loads_v2_models(runtime, demo, monkeypatch):
    import xgboost as xgb

    original = xgb.Booster.load_model
    seen = []

    def guarded(self, fname, *a, **k):
        seen.append(str(fname))
        if "performance_v2" in str(fname):
            raise AssertionError("V2 model loaded on the V3 path")
        return original(self, fname, *a, **k)

    monkeypatch.setattr(xgb.Booster, "load_model", guarded)
    s = demo[1]
    runtime.predict(s["stay_id"], s["legal_cutoffs"][0])
    assert not any("performance_v2" in x for x in seen)


def test_unknown_and_sealed_style_stays_are_rejected(runtime):
    from serving.v2.guard import UnknownDemoStayError

    for sid in ("SYN-V3-T-E-00000001", "SYN-V2-E-00000001", "SYN-E-00001999", "CUSTOM-NOT-REGISTERED"):
        with pytest.raises(UnknownDemoStayError):
            runtime.predict(sid, "2100-01-01T00:00:00Z")


def test_illegal_cutoff_rejected(runtime, demo):
    from serving.v2.runtime import IllegalCutoffError

    with pytest.raises(IllegalCutoffError):
        runtime.predict(demo[0]["stay_id"], "2100-01-01T00:00:00Z")


def _clone_tree(tmp_path):
    for rel in ("artifacts/performance_v3/phase3", "artifacts/performance_v3/governance/v3_model_freeze_v1.json",
                "artifacts/performance_v3/product_handoff_v1.json"):
        src, dst = ROOT / rel, tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy(src, dst)
    return tmp_path


def test_corrupted_model_file_fails_closed(tmp_path):
    from serving.v3.runtime import V3ServingError, V3ServingRuntime

    root = _clone_tree(tmp_path)
    model = root / "artifacts/performance_v3/phase3/v3/models/recovery24/model.json"
    model.write_bytes(model.read_bytes()[:-8] + b"corrupt!")
    with pytest.raises(V3ServingError):
        V3ServingRuntime(root)


def test_manifest_hash_mismatch_fails_closed_and_product_app_does_not_start(tmp_path):
    from api.product_app import build_product_app
    from serving.v3.runtime import V3ServingError

    root = _clone_tree(tmp_path)
    manifest = root / "artifacts/performance_v3/phase3/selected_models_v3.json"
    doc = json.loads(manifest.read_text()); doc["bundle_id"] = "tampered"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(V3ServingError):
        build_product_app(root)
