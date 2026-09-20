import importlib.util
from pathlib import Path

import numpy as np
import pytest

from explainability.tree_shap import TreeShapAdapter
from explainability.router import lint_descriptive_explanation_text
from serving.prediction_schema import lint_contract_text
from tree_shap_helpers import (
    FEATURE_VERSION,
    FLATTENING_HASH,
    FLATTENING_VERSION,
    MODEL_HASH,
    SyntheticXGBContract,
    config,
    context,
    identities,
    prepared,
    target,
)


def _real_dependencies():
    if importlib.util.find_spec("shap") is None or importlib.util.find_spec("xgboost") is None:
        return None, None, "SHAP or XGBoost package is absent"
    try:
        import shap
        import xgboost
    except Exception as error:
        return None, None, "SHAP/XGBoost runtime unavailable: " + str(error)
    return shap, xgboost, None


def test_actual_synthetic_xgboost_treeexplainer_additivity_when_governed_runtime_available():
    _, xgb, blocker = _real_dependencies()
    if blocker:
        pytest.skip("BLOCKED — " + blocker)
    X = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
    model = xgb.XGBRegressor(n_estimators=2, max_depth=2, learning_rate=1.0, random_state=17, n_jobs=1)
    model.fit(X, y)
    model.task = "icu_stay_time"
    model.family = "xgboost"
    model.artifact_sha256 = MODEL_HASH
    model.feature_schema_version = FEATURE_VERSION
    model.flattening_version = FLATTENING_VERSION
    model.flattening_sha256 = FLATTENING_HASH
    model.output_names = ("remaining_icu_log1p_hours",)
    model.flattened_feature_names = ("A_value_oldest", "A_value_newest")
    model.input_dim = 2
    model.xgboost_version = xgb.__version__
    selected_identities = identities()[:2]
    row = prepared()
    row = type(row)(
        values=np.asarray([[1.0, 0.0]], dtype=np.float32),
        feature_identities=selected_identities,
        feature_schema_version=row.feature_schema_version,
        flattening_version=row.flattening_version,
        flattening_sha256=row.flattening_sha256,
        prediction_time=row.prediction_time,
        task=row.task,
    )
    raw = float(model.predict(row.values)[0])
    result = TreeShapAdapter(config=config(), targets={"icu_stay_time": target()}, synthetic=True).explain(
        context(model, prepared_input=row, raw_output=raw)
    )
    assert result.details["base_value"] + sum(result.details["raw_signed_shap_values"]) == pytest.approx(raw, abs=1e-6)


def test_production_adapter_has_no_training_preprocessing_or_api_code():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/explainability/tree_shap.py").read_text().lower()
    prohibited = (".fit(", ".train(", "hyperparameter", "build_features", "raw history", "final_test", "isotonic", "threshold optimization", "fastapi")
    for term in prohibited:
        assert term not in source
    assert "shap.treeexplainer" in source
    assert (root / "api/main.py").is_file()
    assert (root / "dashboard/app.py").is_file()


def test_noncausal_terminology_and_phase10_config_are_committed():
    root = Path(__file__).resolve().parents[1]
    assert lint_contract_text("Feature A contributed to the prediction") == ()
    assert lint_descriptive_explanation_text("Feature A caused deterioration")
    assert (root / "configs/explainability/tree_shap_synthetic_development_v1.json").is_file()


def test_adapter_requires_no_true_label_or_raw_timeline():
    model = SyntheticXGBContract()
    explanation_context = context(model)
    assert not hasattr(explanation_context, "label")
    assert not hasattr(explanation_context.prepared_input, "raw_history")
