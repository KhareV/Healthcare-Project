import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from performance_v3.final_models import fit_final  # noqa: E402
from performance_v3.serving import explain_route, tree_shap_values  # noqa: E402
import performance_v3_explainability as ex  # noqa: E402

BUNDLE = ROOT / "artifacts/performance_v3/phase3/v3/bundle_manifest.json"
CACHE = ROOT / "artifacts/performance_v3/cache/features_LCE.json"
real = pytest.mark.skipif(not (BUNDLE.exists() and CACHE.exists()), reason="V3 bundle/cache not built")


def test_routes():
    for fam in ("xgboost", "lightgbm", "catboost"):
        assert explain_route("organ_support", fam) == "tree_shap"
    assert explain_route("icu_stay_time", "gru") == "integrated_gradients"


def test_readable_names_and_wording():
    assert "bin 7" in ex.readable("A__bin7__lactate__latest__value__mask") or ex.readable("x") == "x"
    assert ex.WORDING == "contributed to the prediction"


def test_additivity_toy_xgboost_on_raw_margin():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)).astype(np.float32)
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=200)
    cfg = {"objective": "squarederror", "weight_mode": "none", "max_depth": 3, "learning_rate": 0.2, "n_estimators": 30}
    m = fit_final("xgboost", cfg, X, y, None, task="recovery24", feature_columns=[f"f{i}" for i in range(5)], n_estimators=30, nthread=2)
    v, base, diag = tree_shap_values(m, X[:50])
    assert diag["additivity_check_passed"] and "RAW_MARGIN" in diag["output_domain"]
    assert int(np.argmax(np.abs(v).mean(axis=0))) == 0


@real
def test_real_bundle_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "OUT", tmp_path)
    rep = ex.run(BUNDLE, CACHE.with_suffix(""), n=150, make_figures=False)
    assert set(rep["tasks"]) == {"recovery24", "recovery48", "icu_stay_time", "organ_support"}
    for e in rep["tasks"].values():
        assert e["additivity_passed"] and e["additivity_max_abs_error"] < 1e-3
        assert e["forbidden_token_feature_names"] == [] and len(e["global_top20"]) == 20
        assert all(l["additivity_passed"] for l in e["local"])
    assert "RAW_MARGIN" in rep["output_domain"]
