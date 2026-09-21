"""Phase-11 canonical XGBoost candidate bundles and native artifact loading."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from data.xgb_canonical import TaskMatrix, XGBCanonicalContractError
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from preprocess.target_scaler import RecoveryTargetScaler
from training.class_weights import SupportClassWeight


XGB_BUNDLE_VERSION = "canonical_xgb_candidate_bundle_v1"
RECOVERY_HORIZON_ORDER = ("delta_sofa_24h", "delta_sofa_48h")
TASKS = ("recovery", "icu_stay_time", "organ_support")
RUN_ENGINEERING_SMOKE = "ENGINEERING_XGB_SMOKE"
RUN_PHASE12_SCIENTIFIC = "PHASE12_SCIENTIFIC"


class XGBCandidateError(ValueError):
    """Raised for unsafe candidate configuration, data, or artifacts."""


@dataclass(frozen=True)
class XGBCandidateConfig:
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    reg_alpha: float
    reg_lambda: float
    n_estimators: int

    def validate(self) -> None:
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int) or not 3 <= self.max_depth <= 8:
            raise XGBCandidateError("max_depth must be an integer in 3..8")
        if not 0.01 <= float(self.learning_rate) <= 0.20:
            raise XGBCandidateError("learning_rate must be in 0.01..0.20")
        if not 0.60 <= float(self.subsample) <= 1.00:
            raise XGBCandidateError("subsample must be in 0.60..1.00")
        if not 0.60 <= float(self.colsample_bytree) <= 1.00:
            raise XGBCandidateError("colsample_bytree must be in 0.60..1.00")
        if not 1.0 <= float(self.min_child_weight) <= 10.0:
            raise XGBCandidateError("min_child_weight must be in 1..10")
        if not 0.0 <= float(self.reg_alpha) <= 1.0:
            raise XGBCandidateError("reg_alpha must be in 0..1")
        if not 0.1 <= float(self.reg_lambda) <= 10.0:
            raise XGBCandidateError("reg_lambda must be in 0.1..10")
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, int) or not 1 <= self.n_estimators <= 1000:
            raise XGBCandidateError("n_estimators must be an integer in 1..1000")


@dataclass(frozen=True)
class XGBLineage:
    feature_schema_version: str
    feature_schema_sha256: str
    flat_feature_map_version: str
    flat_feature_map_sha256: str
    preprocessor_version: str
    preprocessor_sha256: str
    split_version: str
    split_sha256: str
    target_schema_version: str
    target_schema_sha256: str
    recovery_target_scaler_sha256: Optional[str]
    support_class_weight_sha256: Optional[str]
    event_dictionary_sha256: str
    code_commit: str


def canonical_config(config: XGBCandidateConfig, *, task: str, seed: int, run_type: str) -> Mapping[str, object]:
    config.validate()
    if task not in TASKS:
        raise XGBCandidateError("unsupported XGBoost task")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise XGBCandidateError("XGBoost seed must be an explicit integer")
    if run_type not in (RUN_ENGINEERING_SMOKE, RUN_PHASE12_SCIENTIFIC):
        raise XGBCandidateError("unsupported XGBoost run type")
    return {
        "task": task, "hyperparameters": asdict(config), "random_state": seed,
        "tree_method": "hist", "device": "cpu", "n_jobs": 1,
        "early_stopping_rounds": 3 if run_type == RUN_ENGINEERING_SMOKE else 50,
        "serialization": "native_json", "numeric_dtype": "float32",
        "objectives": (
            {"24h": "reg:squarederror", "48h": "reg:squarederror"}
            if task == "recovery" else
            {"task": "reg:squarederror" if task == "icu_stay_time" else "binary:logistic"}
        ),
    }


def config_sha256(config: XGBCandidateConfig, *, task: str, seed: int, run_type: str) -> str:
    return hashlib.sha256(canonical_json_bytes(canonical_config(config, task=task, seed=seed, run_type=run_type))).hexdigest()


def _xgb():
    try:
        import xgboost
    except Exception as error:
        raise XGBCandidateError("XGBoost 2.1.4 runtime is unavailable") from error
    if xgboost.__version__ != "2.1.4":
        raise XGBCandidateError("XGBoost version must match observed project version 2.1.4")
    return xgboost


def _validate_matrix(matrix: TaskMatrix, expected_partition: str, expected_task: str) -> None:
    if matrix.partition != expected_partition or matrix.task_view != expected_task:
        raise XGBCandidateError("task matrix partition/task mismatch")
    if matrix.X.ndim != 2 or matrix.y.ndim != 1 or len(matrix.X) != len(matrix.y) or len(matrix.row_keys) != len(matrix.y):
        raise XGBCandidateError("task matrix shape mismatch")
    if not np.isfinite(matrix.X).all() or not np.isfinite(matrix.y).all():
        raise XGBCandidateError("task matrix must be finite")


def _estimator_parameters(config: XGBCandidateConfig, normalized: Mapping[str, object], *, objective: str, eval_metric: str, scale_pos_weight: Optional[float] = None) -> Mapping[str, object]:
    values = dict(asdict(config))
    values.update({
        "objective": objective, "eval_metric": eval_metric,
        "random_state": normalized["random_state"], "tree_method": normalized["tree_method"],
        "device": normalized["device"], "n_jobs": normalized["n_jobs"],
        "early_stopping_rounds": normalized["early_stopping_rounds"],
    })
    if scale_pos_weight is not None:
        values["scale_pos_weight"] = float(scale_pos_weight)
    return values


def _best_iteration(model) -> int:
    try:
        return int(model.best_iteration)
    except (AttributeError, TypeError):
        return int(model.n_estimators) - 1


def _save_model(model, path: Path) -> Mapping[str, object]:
    model.save_model(path)
    return {"path": path.name, "sha256": sha256_file(path), "best_iteration": _best_iteration(model)}


def _require_bound_artifact(path: Optional[Path], expected_sha256: Optional[str], name: str) -> None:
    if path is None or expected_sha256 is None:
        raise XGBCandidateError(name + " requires a lineage-bound artifact path")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise XGBCandidateError(name + " artifact hash mismatch")


class RecoveryXGBBundle:
    family = "xgboost"
    task = "recovery"
    output_names = RECOVERY_HORIZON_ORDER

    def __init__(self, model24, model48, scaler: RecoveryTargetScaler, metadata, feature_names: Sequence[str]):
        self.model24, self.model48, self.scaler, self.metadata = model24, model48, scaler, metadata
        self.flattened_feature_names = tuple(feature_names)
        self.input_dim = len(self.flattened_feature_names)
        self.xgboost_version = metadata["xgboost_version"]

    def predict_standardized(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack((self.model24.predict(X), self.model48.predict(X))).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        standardized = self.predict_standardized(X)
        means = np.asarray((self.scaler.mean_24, self.scaler.mean_48), dtype=np.float32)
        scales = np.asarray((self.scaler.scale_24, self.scaler.scale_48), dtype=np.float32)
        return standardized * scales + means


class ICUTimeXGBBundle:
    family = "xgboost"
    task = "icu_stay_time"

    def __init__(self, model, metadata, feature_names: Sequence[str]):
        self.model, self.metadata = model, metadata
        self.flattened_feature_names = tuple(feature_names); self.input_dim = len(self.flattened_feature_names)
        self.xgboost_version = metadata["xgboost_version"]

    def predict_log(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.float32)

    def predict_hours(self, X: np.ndarray) -> np.ndarray:
        return remaining_icu_hours_from_log_prediction(torch.from_numpy(self.predict_log(X))).numpy()


class SupportXGBBundle:
    family = "xgboost"
    task = "organ_support"
    probability_type = "raw_uncalibrated"
    calibrator = None
    threshold = None

    def __init__(self, model, metadata, feature_names: Sequence[str]):
        self.model, self.metadata = model, metadata
        self.flattened_feature_names = tuple(feature_names); self.input_dim = len(self.flattened_feature_names)
        self.xgboost_version = metadata["xgboost_version"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(self.model.predict_proba(X)[:, 1], dtype=np.float32)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise XGBCandidateError("support probability is invalid")
        return probabilities


def train_xgb_candidate(
    *, task: str, candidate_id: str, config: XGBCandidateConfig,
    train_matrices: Mapping[str, TaskMatrix], validation_matrices: Mapping[str, TaskMatrix],
    output_directory: Path, lineage: XGBLineage, feature_names: Sequence[str],
    seed: int, run_type: str, recovery_scaler: Optional[RecoveryTargetScaler] = None,
    support_class_weight: Optional[SupportClassWeight] = None,
    recovery_scaler_path: Optional[Path] = None,
    support_class_weight_path: Optional[Path] = None,
    phase12_authorization: Optional[str] = None,
):
    """Train one bundle; Phase 11 authorizes fixture-only engineering smoke."""
    if run_type == RUN_PHASE12_SCIENTIFIC and phase12_authorization != "PHASE12_FROZEN_CANDIDATE_MANIFEST":
        raise XGBCandidateError("scientific XGBoost training requires Phase-12 authorization")
    if run_type == RUN_ENGINEERING_SMOKE and not candidate_id.startswith("engineering_xgb_smoke_"):
        raise XGBCandidateError("engineering smoke candidate IDs must be explicitly non-scientific")
    normalized = canonical_config(config, task=task, seed=seed, run_type=run_type)
    digest = config_sha256(config, task=task, seed=seed, run_type=run_type)
    xgb = _xgb()
    output_directory.mkdir(parents=True, exist_ok=False)
    model_entries = {}
    models = {}
    required = ("recovery24", "recovery48") if task == "recovery" else (task,)
    for view in required:
        _validate_matrix(train_matrices[view], "train", view)
        _validate_matrix(validation_matrices[view], "validation", view)
    if task == "recovery":
        if recovery_scaler is None or lineage.recovery_target_scaler_sha256 is None:
            raise XGBCandidateError("recovery bundle requires frozen target scaler lineage")
        _require_bound_artifact(
            recovery_scaler_path,
            lineage.recovery_target_scaler_sha256,
            "recovery target scaler",
        )
        for view, suffix in (("recovery24", "24h"), ("recovery48", "48h")):
            model = xgb.XGBRegressor(**_estimator_parameters(config, normalized, objective="reg:squarederror", eval_metric="rmse"))
            model.fit(train_matrices[view].X, train_matrices[view].y, eval_set=[(validation_matrices[view].X, validation_matrices[view].y)], verbose=False)
            path = output_directory / f"recovery_{suffix}.json"
            model_entries[view] = {**_save_model(model, path), "train_eligible_count": len(train_matrices[view].y), "validation_eligible_count": len(validation_matrices[view].y)}
            models[view] = model
    elif task == "icu_stay_time":
        model = xgb.XGBRegressor(**_estimator_parameters(config, normalized, objective="reg:squarederror", eval_metric="rmse"))
        model.fit(train_matrices[task].X, train_matrices[task].y, eval_set=[(validation_matrices[task].X, validation_matrices[task].y)], verbose=False)
        path = output_directory / "icu_stay_time.json"
        model_entries[task] = {**_save_model(model, path), "train_eligible_count": len(train_matrices[task].y), "validation_eligible_count": len(validation_matrices[task].y)}
        models[task] = model
    elif task == "organ_support":
        if support_class_weight is None or lineage.support_class_weight_sha256 is None:
            raise XGBCandidateError("support bundle requires frozen class-weight lineage")
        _require_bound_artifact(
            support_class_weight_path,
            lineage.support_class_weight_sha256,
            "support class weight",
        )
        model = xgb.XGBClassifier(**_estimator_parameters(config, normalized, objective="binary:logistic", eval_metric="logloss", scale_pos_weight=support_class_weight.pos_weight))
        model.fit(train_matrices[task].X, train_matrices[task].y, eval_set=[(validation_matrices[task].X, validation_matrices[task].y)], verbose=False)
        path = output_directory / "organ_support.json"
        model_entries[task] = {**_save_model(model, path), "train_eligible_count": len(train_matrices[task].y), "validation_eligible_count": len(validation_matrices[task].y)}
        models[task] = model
    else:
        raise XGBCandidateError("unsupported task")
    metadata = {
        "bundle_version": XGB_BUNDLE_VERSION, "task": task, "family": "xgboost",
        "candidate_id": candidate_id, "run_type": run_type,
        "scientific_candidate": run_type == RUN_PHASE12_SCIENTIFIC,
        "counts_toward_30_config_budget": run_type == RUN_PHASE12_SCIENTIFIC,
        "config": normalized, "config_sha256": digest, "lineage": asdict(lineage),
        "seed": seed, "xgboost_version": xgb.__version__, "serialization": "native_json",
        "feature_names": list(feature_names), "input_dimension": len(feature_names),
        "models": model_entries, "metrics_computed": False, "test_accessed": False,
        "calibration": None, "threshold": None,
    }
    metadata_path = output_directory / "bundle.metadata.json"
    metadata_path.write_bytes(canonical_json_bytes(metadata))
    if task == "recovery":
        return RecoveryXGBBundle(models["recovery24"], models["recovery48"], recovery_scaler, metadata, feature_names)
    if task == "icu_stay_time":
        return ICUTimeXGBBundle(models[task], metadata, feature_names)
    return SupportXGBBundle(models[task], metadata, feature_names)


def load_xgb_bundle(
    directory: Path, *, expected_task: str, expected_lineage: XGBLineage,
    expected_feature_names: Sequence[str], recovery_scaler: Optional[RecoveryTargetScaler] = None,
    support_class_weight: Optional[SupportClassWeight] = None,
    recovery_scaler_path: Optional[Path] = None,
    support_class_weight_path: Optional[Path] = None,
):
    metadata_path = directory / "bundle.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("bundle_version") != XGB_BUNDLE_VERSION or metadata.get("family") != "xgboost" or metadata.get("task") != expected_task:
        raise XGBCandidateError("XGBoost bundle family/task/version mismatch")
    if metadata.get("lineage") != asdict(expected_lineage):
        raise XGBCandidateError("XGBoost bundle lineage mismatch")
    if tuple(metadata.get("feature_names", ())) != tuple(expected_feature_names) or metadata.get("input_dimension") != len(expected_feature_names):
        raise XGBCandidateError("XGBoost flat feature map/order mismatch")
    if metadata.get("config_sha256") != hashlib.sha256(canonical_json_bytes(metadata.get("config"))).hexdigest():
        raise XGBCandidateError("XGBoost candidate config hash mismatch")
    xgb = _xgb()
    loaded = {}
    for view, entry in metadata["models"].items():
        path = directory / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise XGBCandidateError("XGBoost native model hash mismatch")
        model = xgb.XGBClassifier() if expected_task == "organ_support" else xgb.XGBRegressor()
        model.load_model(path)
        loaded[view] = model
    if expected_task == "recovery":
        if recovery_scaler is None or expected_lineage.recovery_target_scaler_sha256 is None:
            raise XGBCandidateError("recovery scaler is required")
        _require_bound_artifact(
            recovery_scaler_path,
            expected_lineage.recovery_target_scaler_sha256,
            "recovery target scaler",
        )
        return RecoveryXGBBundle(loaded["recovery24"], loaded["recovery48"], recovery_scaler, metadata, expected_feature_names)
    if expected_task == "icu_stay_time":
        return ICUTimeXGBBundle(loaded["icu_stay_time"], metadata, expected_feature_names)
    if support_class_weight is None or expected_lineage.support_class_weight_sha256 is None:
        raise XGBCandidateError("support class weight is required")
    _require_bound_artifact(
        support_class_weight_path,
        expected_lineage.support_class_weight_sha256,
        "support class weight",
    )
    return SupportXGBBundle(loaded["organ_support"], metadata, expected_feature_names)


def xgb_registry_fields(metadata: Mapping[str, object], *, metadata_ref: str) -> Mapping[str, str]:
    """Map a bundle into the existing shared registry without a second registry."""
    if metadata.get("family") != "xgboost" or metadata.get("bundle_version") != XGB_BUNDLE_VERSION:
        raise XGBCandidateError("registry mapping requires a canonical XGBoost bundle")
    lineage = metadata["lineage"]
    model_details = {
        "bundle_metadata_ref": metadata_ref,
        "xgboost_version": metadata["xgboost_version"],
        "serialization": metadata["serialization"],
        "best_iterations": {name: item["best_iteration"] for name, item in metadata["models"].items()},
        "eligibility_counts": {
            name: {"train": item["train_eligible_count"], "validation": item["validation_eligible_count"]}
            for name, item in metadata["models"].items()
        },
        "recovery_target_scaler_sha256": lineage.get("recovery_target_scaler_sha256"),
        "support_class_weight_sha256": lineage.get("support_class_weight_sha256"),
    }
    return {
        "task": str(metadata["task"]), "model_family": "xgboost",
        "candidate_id": str(metadata["candidate_id"]), "seed": str(metadata["seed"]),
        "config_hash": str(metadata["config_sha256"]), "split_hash": str(lineage["split_sha256"]),
        "feature_version": str(lineage["feature_schema_version"]),
        "label_version": str(lineage["target_schema_version"]),
        "preprocessor_sha256": str(lineage["preprocessor_sha256"]),
        "event_dict_version": "synthetic_event_dict_v2",
        "derived_feature_hash": str(lineage["flat_feature_map_sha256"]),
        "code_commit": str(lineage["code_commit"]),
        "run_type": "smoke" if metadata["run_type"] == RUN_ENGINEERING_SMOKE else "scientific",
        "notes": json.dumps(model_details, sort_keys=True, separators=(",", ":")),
    }
