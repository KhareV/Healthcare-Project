"""V3 inference sidecar (production V2 untouched) and the frozen V2 comparator.

Manifest schema (`selected_models_v3.json`-shaped; hashes are SHA-256):

    {
      "artifact_version": "performance_v3_selected_models_v1",
      "bundle_id": "...",
      "tasks": {
        "recovery24" | "recovery48" | "icu_stay_time" | "organ_support": {
          "family": "xgboost|lightgbm|catboost|ridge|logistic",
          "feature_variant": "V3_*",
          "model_ref": "<path to model_meta.json written by final_models.FittedModel.save>",
          "model_sha256": "<sha256 of that meta file>",
          "feature_names_ref": "<json list of ordered column names>",
          "feature_names_sha256": "<sha256 of that file>",
          "n_estimators": int,
          "postprocess": "none" | "expm1(clamp_min(raw,0))",
          "normalisation_ref": optional, "normalisation_sha256": optional   # V3 trees consume RAW context columns
        },
        # organ_support additionally:
        "calibrator": {"method": "...", "params": {frozen performance_v3.calibration dict}, "sha256": "<sha256 of canonical params json>"},
        "threshold": {"value": float, "comparator": ">="}
      }
    }

`predict_rows` consumes canonical rows (pre_split package rows enriched by
`performance_v3.data` / `performance_v2.data_loading`) and computes features with
`performance_v2.diagnostic_features.build_all_groups` -- the same function the
training feature cache used -- then selects the stored ordered column list
(column mismatch fails closed).

Recovery SOFA reconstruction: SOFA_hat_h = clip(SOFA(t) + Delta_h, 0, 24), each
horizon from its own independent model; 48h never uses the 24h prediction.

Explanations: `explain_route` maps tree families -> TreeSHAP, gru -> Integrated
Gradients. TreeSHAP here explains the RAW model margin (pre-calibration, pre-
postprocess); it is NOT a decomposition of the calibrated support probability.
The governed `explainability.tree_shap` adapter is tied to the V1 flattened-name
contract and its production gating, so it is not used for V3 development fixtures;
`shap.TreeExplainer` is used directly with an explicit output-domain label.

Development/serving code only; never reads sealed cohorts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from performance_v2.diagnostic_features import build_all_groups
from performance_v3.calibration import apply_calibrator
from performance_v3.features import current_sofa_at_t, sofa_component_features
from performance_v3.final_models import FittedModel, load_fitted_model, names_sha256

SERVING_VERSION = "performance_v3_serving_v1"
TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")
ICU_POSTPROCESS = "expm1(clamp_min(raw,0))"


class BundleError(RuntimeError):
    pass


class FeatureContractError(BundleError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def params_sha256(params: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise BundleError(f"{label}: file missing: {path}")
    if _sha256_file(path) != expected:
        raise BundleError(f"{label}: sha256 mismatch for {path}")


# --------------------------------------------------------------------------
# shared output assembly
# --------------------------------------------------------------------------

def _reconstruct_sofa(current: Optional[float], delta: float) -> Optional[float]:
    if current is None or (isinstance(current, float) and math.isnan(current)):
        return None
    return float(np.clip(float(current) + float(delta), 0.0, 24.0))


def assemble_predictions(rows: Sequence[Mapping[str, object]], *, recovery24, recovery48, icu_raw, support_raw,
                         support_calibrated, threshold: float, model_hashes: Mapping[str, str], bundle_id: str) -> List[Dict[str, object]]:
    out = []
    icu_hours = np.expm1(np.clip(np.asarray(icu_raw, dtype=np.float64), 0.0, None))
    if not np.all(np.isfinite(icu_hours)):
        raise BundleError("postprocessed remaining ICU hours must be finite")
    for i, row in enumerate(rows):
        current = row.get("baseline_sofa")
        out.append({
            "stay_id": row["stay_id"], "prediction_time": row["prediction_time"], "grid_index": int(row.get("grid_index", -1)),
            "current_sofa": None if current is None else float(current),
            "recovery24_delta": float(recovery24[i]), "recovery48_delta": float(recovery48[i]),
            "recovery24_sofa": _reconstruct_sofa(current, recovery24[i]),
            "recovery48_sofa": _reconstruct_sofa(current, recovery48[i]),
            "icu_raw_log_prediction": float(icu_raw[i]), "icu_remaining_hours": float(icu_hours[i]),
            "support_raw_probability": float(support_raw[i]), "support_calibrated_probability": float(support_calibrated[i]),
            "support_threshold": float(threshold), "support_alert": bool(float(support_calibrated[i]) >= float(threshold)),
            "model_hashes": dict(model_hashes), "bundle_id": bundle_id, "serving_version": SERVING_VERSION,
        })
    return out


# --------------------------------------------------------------------------
# V3 bundle
# --------------------------------------------------------------------------

class V3Bundle:
    def __init__(self, *, bundle_id: str, models: Mapping[str, FittedModel], calibrator: Mapping[str, object],
                 threshold: float, hashes: Mapping[str, str], variants: Mapping[str, str]):
        self.bundle_id = bundle_id
        self.models = dict(models)
        self.calibrator = dict(calibrator)
        self.threshold = float(threshold)
        self.model_hashes = dict(hashes)
        self.variants = dict(variants)

    @classmethod
    def load(cls, root, manifest_path) -> "V3Bundle":
        root = Path(root)
        manifest_path = Path(manifest_path)
        manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks = manifest.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
            raise BundleError("manifest must define exactly the four tasks " + repr(TASKS))
        models: Dict[str, FittedModel] = {}
        hashes: Dict[str, str] = {}
        variants: Dict[str, str] = {}
        for task in TASKS:
            entry = tasks[task]
            meta_path = root / entry["model_ref"]
            _verify(meta_path, entry["model_sha256"], f"{task} model meta")
            names_path = root / entry["feature_names_ref"]
            _verify(names_path, entry["feature_names_sha256"], f"{task} feature names")
            model = load_fitted_model(meta_path)  # verifies every model file hash + names hash
            stored = json.loads(names_path.read_text(encoding="utf-8"))
            if tuple(stored) != model.feature_columns:
                raise BundleError(f"{task}: feature_names_ref differs from the model's stored feature_columns")
            if model.family != entry["family"] or model.task != task:
                raise BundleError(f"{task}: manifest family/task disagrees with the stored model")
            if int(entry.get("n_estimators", model.n_estimators)) != model.n_estimators:
                raise BundleError(f"{task}: n_estimators disagrees with the stored model")
            if entry.get("postprocess") not in (None, "none", ICU_POSTPROCESS):
                raise BundleError(f"{task}: unknown postprocess {entry.get('postprocess')!r}")
            if "normalisation_ref" in entry:
                _verify(root / entry["normalisation_ref"], entry["normalisation_sha256"], f"{task} normalisation")
            models[task] = model
            hashes[task] = entry["model_sha256"]
            variants[task] = entry["feature_variant"]
        cal = manifest["calibrator"]
        if params_sha256(cal["params"]) != cal["sha256"]:
            raise BundleError("calibrator params sha256 mismatch")
        if cal["params"].get("method") != cal["method"]:
            raise BundleError("calibrator method disagrees with its params")
        thr = manifest["threshold"]
        if thr.get("comparator") != ">=":
            raise BundleError("threshold comparator must be '>='")
        hashes["calibrator"] = cal["sha256"]
        return cls(bundle_id=str(manifest.get("bundle_id", manifest_path.stem)), models=models, calibrator=cal["params"],
                   threshold=float(thr["value"]), hashes=hashes, variants=variants)

    # ---- features ------------------------------------------------------
    def _required_inputs(self) -> Tuple[bool, bool]:
        names = [n for m in self.models.values() for n in m.feature_columns]
        return any(n.startswith("C__sofa") for n in names), any(n.startswith("S__") for n in names)

    def feature_matrix(self, task: str, feature_dicts: Sequence[Mapping[str, float]]) -> np.ndarray:
        columns = self.models[task].feature_columns
        index = {name: i for i, name in enumerate(columns)}
        onehot = np.asarray(["==" in n for n in columns], dtype=bool)
        matrix = np.full((len(feature_dicts), len(columns)), np.nan, dtype=np.float32)
        matrix[:, onehot] = 0.0
        for r, features in enumerate(feature_dicts):
            for name, value in features.items():
                j = index.get(name)
                if j is not None:
                    matrix[r, j] = value
        return matrix

    def row_features(self, rows: Sequence[Mapping[str, object]], sofa_trends=None, sofa_components=None) -> List[Dict[str, float]]:
        need_trend, need_comp = self._required_inputs()
        if need_trend and sofa_trends is None:
            raise FeatureContractError("bundle uses C__sofa trend columns; supply sofa_trends keyed (stay_id, prediction_time)")
        if need_comp and sofa_components is None:
            raise FeatureContractError("bundle uses S__ SOFA component columns; supply sofa_components keyed (stay_id, prediction_time)")
        out = []
        for row in rows:
            key = (row["stay_id"], row["prediction_time"])
            trend = (sofa_trends or {}).get(key, {})
            features = build_all_groups(row, baseline_sofa=current_sofa_at_t(row, trend), sofa_trend=trend)
            if sofa_components is not None:
                features.update(sofa_component_features(sofa_components.get(key)))
            out.append(features)
        return out

    # ---- prediction ----------------------------------------------------
    def predict_rows(self, rows: Sequence[Mapping[str, object]], *, sofa_trends=None, sofa_components=None) -> List[Dict[str, object]]:
        rows = list(rows)
        if not rows:
            return []
        feats = self.row_features(rows, sofa_trends, sofa_components)
        rows = [dict(r, baseline_sofa=current_sofa_at_t(r, (sofa_trends or {}).get((r["stay_id"], r["prediction_time"]), {}))) for r in rows]
        raw = {}
        for task in TASKS:
            model = self.models[task]
            raw[task] = model.predict(self.feature_matrix(task, feats), model.feature_columns)
        calibrated = apply_calibrator(self.calibrator, raw["organ_support"])
        return assemble_predictions(
            rows, recovery24=raw["recovery24"], recovery48=raw["recovery48"], icu_raw=raw["icu_stay_time"],
            support_raw=raw["organ_support"], support_calibrated=calibrated, threshold=self.threshold,
            model_hashes=self.model_hashes, bundle_id=self.bundle_id,
        )

    def predict_stay_at(self, runtime, stay_id: str, prediction_time: str) -> Dict[str, object]:
        """Raw-history-truncated prediction through the EXISTING V2 serving runtime.

        `runtime` is a `serving.v2.runtime.V2ServingRuntime`. Its `predict()` is the
        only public entry that performs guard + legal-cutoff validation + cutoff-
        truncated canonical feature construction + current-SOFA scoring; the canonical
        row it built is returned as `V2Prediction.feature_row`. V3 models are then
        applied to that row (V2 models also run inside predict(); their output is ignored).
        Past-cutoff SOFA inputs for C__/S__ columns use the runtime's frozen SOFA provider
        at cutoffs <= t only.
        """

        prediction = runtime.predict(stay_id, prediction_time)
        row = dict(prediction.feature_row)
        trends = components = None
        need_trend, need_comp = self._required_inputs()
        if need_trend or need_comp:
            trends, components = past_sofa_inputs(runtime, row)
        return self.predict_rows([row], sofa_trends=trends, sofa_components=components)[0]


def past_sofa_inputs(runtime, row: Mapping[str, object]):
    """SOFA trend (t, t-6h, t-12h, t-24h, t-48h clamped at intime) + components at t from the
    runtime's frozen provider; mirrors performance_v2.sofa_trend semantics, never reads > t."""

    from datetime import datetime, timedelta, timezone

    from data.synthetic.sofa import sofa_at
    from experiments.stage5_final_test import _z
    from performance_v2.sofa_trend import _LAGS_HOURS

    provider = runtime.sofa_provider
    stay_id = row["stay_id"]
    record = provider.stays[stay_id]
    intime = datetime.fromisoformat(str(record["intime"]).replace("Z", "+00:00")).astimezone(timezone.utc)
    t = datetime.fromisoformat(str(row["prediction_time"]).replace("Z", "+00:00")).astimezone(timezone.utc)
    key = (stay_id, row["prediction_time"])
    trend: Dict[str, float] = {}
    comps: Dict[str, Tuple[int, bool]] = {}
    for label, lag in _LAGS_HOURS.items():
        cutoff = t - timedelta(hours=lag)
        if cutoff < intime:
            continue
        try:
            result = sofa_at(provider.history, record, _z(cutoff), spec=provider.spec, bindings=provider.bindings, support_provider=provider.support_provider)
        except Exception:
            continue
        trend[label] = float(result.total_score)
        if lag == 0:
            comps = {c.component_name: (c.score, c.observed) for c in result.components}
    return {key: trend}, {key: comps}


# --------------------------------------------------------------------------
# Frozen V2 comparator (apply-only; nothing is refit or recalibrated)
# --------------------------------------------------------------------------

class V2Comparator:
    SELECTED = "artifacts/performance_v2/phase3/selected_models_v2.json"

    def __init__(self, *, models, group_a_names, normalizations, variants, knots, threshold, hashes):
        self.models = models
        self.group_a_names = group_a_names
        self.normalizations = normalizations
        self.variants = variants
        self.knots = knots
        self.threshold = float(threshold)
        self.model_hashes = hashes

    @classmethod
    def load(cls, root, *, group_a_names: Optional[Mapping[str, Tuple[str, ...]]] = None) -> "V2Comparator":
        import xgboost as xgb

        from performance_v2.final_evaluation import (
            TASK_MODEL_FILE, TASK_VARIANT, load_frozen_normalization,
        )

        root = Path(root)
        selected_path = root / cls.SELECTED
        selected = json.loads(selected_path.read_text(encoding="utf-8"))["tasks"]
        models, hashes, normalizations, variants = {}, {}, {}, {}
        for task in TASKS:
            entry = selected[task]
            model_path = root / TASK_MODEL_FILE[task]
            if entry["model_ref"] != TASK_MODEL_FILE[task]:
                raise BundleError(f"{task}: V2 model_ref disagrees with performance_v2.final_evaluation.TASK_MODEL_FILE")
            _verify(model_path, entry["model_sha256"], f"V2 {task} model")
            _verify(root / entry["preprocessor_ref"], entry["preprocessor_sha256"], f"V2 {task} preprocessor")
            if entry["feature_variant"] != TASK_VARIANT[task]:
                raise BundleError(f"{task}: frozen feature variant disagrees with TASK_VARIANT")
            model = xgb.XGBClassifier() if task == "organ_support" else xgb.XGBRegressor()
            model.load_model(str(model_path))
            models[task] = model
            hashes[task] = entry["model_sha256"]
            variants[task] = entry["feature_variant"]
            normalizations[task] = load_frozen_normalization(root, entry["feature_variant"])
        support = selected["organ_support"]
        _verify(root / support["isotonic_calibrator_ref"], support["isotonic_calibrator_sha256"], "V2 isotonic calibrator")
        _verify(root / support["threshold_ref"], support["threshold_sha256"], "V2 threshold")
        knots = json.loads((root / support["isotonic_calibrator_ref"]).read_text(encoding="utf-8"))["calibration_knots"]
        threshold = float(json.loads((root / support["threshold_ref"]).read_text(encoding="utf-8"))["threshold_value"])
        hashes["calibrator"] = support["isotonic_calibrator_sha256"]
        if group_a_names is None:
            group_a_names = reconstruct_v2_group_a_names(root)
        return cls(models=models, group_a_names=dict(group_a_names), normalizations=normalizations, variants=variants,
                   knots=knots, threshold=threshold, hashes=hashes)

    def predict_rows(self, rows: Sequence[Mapping[str, object]], *, current_sofa_trends=None) -> List[Dict[str, object]]:
        """`current_sofa_trends=None` -> V2_AS_EVALUATED (pre-split baseline_sofa, nulls preserved, the V2
        offline harness). Passing {(stay_id, prediction_time): {"t": SOFA(t), ...}} -> V2_AS_SERVED
        (SOFA(t) supplied for every row, exactly what the V2 serving runtime feeds the models)."""

        from performance_v2.final_evaluation import build_inference_matrix, isotonic_apply

        rows = list(rows)
        if not rows:
            return []
        if current_sofa_trends is not None:
            rows = [dict(r, baseline_sofa=current_sofa_at_t(r, current_sofa_trends.get((r["stay_id"], r["prediction_time"]), {}))) for r in rows]
        raw = {}
        for task in TASKS:
            matrix = build_inference_matrix(rows, group_a_names=self.group_a_names[task], normalization=self.normalizations[task])
            model = self.models[task]
            raw[task] = model.predict_proba(matrix)[:, 1] if task == "organ_support" else model.predict(matrix)
        calibrated = isotonic_apply(raw["organ_support"], self.knots)
        return assemble_predictions(
            rows, recovery24=raw["recovery24"], recovery48=raw["recovery48"], icu_raw=raw["icu_stay_time"],
            support_raw=raw["organ_support"], support_calibrated=calibrated, threshold=self.threshold,
            model_hashes=self.model_hashes, bundle_id="performance_v2_frozen",
        )


def reconstruct_v2_group_a_names(root) -> Dict[str, Tuple[str, ...]]:
    """Same pure function of the frozen V1 DEV rows that Phase 3/4 used, loaded once."""

    from performance_v2.data_loading import load_dev_rows
    from performance_v2.targets import icu_rows, recovery_targets, support_rows
    from performance_v2.v2_features import group_a_feature_names

    rows = load_dev_rows(Path(root), splits=("train",)) + load_dev_rows(Path(root), splits=("validation",))
    return {
        "recovery24": group_a_feature_names(recovery_targets(rows, "24")),
        "recovery48": group_a_feature_names(recovery_targets(rows, "48")),
        "icu_stay_time": group_a_feature_names(icu_rows(rows)),
        "organ_support": group_a_feature_names(support_rows(rows)),
    }


# --------------------------------------------------------------------------
# Explanation routing + TreeSHAP (raw margin)
# --------------------------------------------------------------------------

TREE_FAMILIES = ("xgboost", "lightgbm", "catboost")
OUTPUT_DOMAIN = "RAW_MARGIN_PRE_CALIBRATION_NOT_CALIBRATED_PROBABILITY"


def explain_route(task: str, family: str) -> str:
    if task not in TASKS:
        raise ValueError("unknown task: " + repr(task))
    if family in TREE_FAMILIES:
        return "tree_shap"
    if family == "gru":
        return "integrated_gradients"
    if family in ("ridge", "logistic"):
        return "linear_contributions"
    raise ValueError("unknown family: " + repr(family))


def _raw_margin(model: FittedModel, X: np.ndarray) -> np.ndarray:
    if model.family == "xgboost":
        import xgboost as xgb

        return model._model.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)), iteration_range=(0, model.n_estimators), output_margin=True)
    if model.family == "lightgbm":
        return model._model.predict(np.asarray(X, dtype=np.float64), num_iteration=model.n_estimators, raw_score=True)
    return model._model.predict(np.asarray(X, dtype=np.float64), prediction_type="RawFormulaVal")


def tree_shap_values(model: FittedModel, X) -> Tuple[np.ndarray, float, Dict[str, object]]:
    """(shap [n,d], base_value, diagnostics) for a tree FittedModel on the RAW margin."""

    if model.family not in TREE_FAMILIES:
        raise ValueError("tree_shap requires a tree family, got " + repr(model.family))
    try:
        import shap
    except ImportError as error:  # pragma: no cover
        raise BundleError("BLOCKED - shap not installed; no fallback attribution is permitted") from error
    X = np.asarray(X, dtype=np.float64 if model.family != "xgboost" else np.float32)
    explainer = shap.TreeExplainer(model._model, feature_perturbation="tree_path_dependent", model_output="raw")
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=np.float64)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
    margin = np.asarray(_raw_margin(model, X), dtype=np.float64)
    additive = bool(np.allclose(values.sum(axis=1) + base, margin, atol=1e-3))
    return values, base, {
        "method": "TreeSHAP (shap.TreeExplainer, tree_path_dependent, model_output=raw)",
        "output_domain": OUTPUT_DOMAIN,
        "additivity_check_passed": additive,
        "base_value": base,
    }


def explain_local(model: FittedModel, x_row, *, top_k: int = 8) -> Dict[str, object]:
    values, base, diag = tree_shap_values(model, np.asarray(x_row).reshape(1, -1))
    pairs = list(zip(model.feature_columns, values[0].tolist()))
    positive = sorted((p for p in pairs if p[1] > 0), key=lambda p: -p[1])[:top_k]
    negative = sorted((p for p in pairs if p[1] < 0), key=lambda p: p[1])[:top_k]
    return {"positive": positive, "negative": negative, **diag}


def explain_global(model: FittedModel, X, *, top_k: int = 20) -> Dict[str, object]:
    values, base, diag = tree_shap_values(model, X)
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_k]
    return {"mean_abs_shap": [(model.feature_columns[i], float(mean_abs[i])) for i in order], **diag}
