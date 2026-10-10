"""V3 product serving runtime: the frozen, hash-verified V3 selected-model bundle on top of the
shared canonical-history store.

Pipeline per request (single implementation of every step, nothing duplicated):
  authorize stay -> validate legal cutoff -> truncate raw history at t -> canonical features
  (features.synthetic builder) -> SOFA(t) from the frozen provider for EVERY legal cutoff (defect D1)
  -> exact V3 feature columns (diagnostic_features groups) -> frozen models by hash -> ICU/SOFA
  postprocessing -> Platt calibrator + frozen threshold.

FAIL CLOSED: any bundle/hash/load problem raises V3ServingError. There is deliberately no fallback to
the defective V2 serving models; V2 is never imported or loaded on this path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from performance_v3.serving import BundleError, V3Bundle
from serving.v2.runtime import CanonicalHistoryRuntime, IllegalCutoffError  # noqa: F401  (re-exported)

SELECTED_MODELS_REL = "artifacts/performance_v3/phase3/selected_models_v3.json"
HANDOFF_REL = "artifacts/performance_v3/product_handoff_v1.json"
FREEZE_REL = "artifacts/performance_v3/governance/v3_model_freeze_v1.json"
FINAL_FREEZE_REL = "artifacts/performance_v3/governance/v3_final_evaluation_freeze_v1.json"
THRESHOLD_VALUE = 0.3021480061849302
TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")


class V3ServingError(RuntimeError):
    """Raised for any V3 bundle load / verification failure (never silently degraded)."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class V3Prediction:
    stay_id: str
    prediction_time: str
    grid_index: int
    icu_elapsed_hours: int
    current_sofa: float
    recovery24_delta: float
    recovery48_delta: float
    recovery24_sofa: float
    recovery48_sofa: float
    icu_raw_log_prediction: float
    icu_remaining_hours: float
    support_raw_probability: float
    support_calibrated_probability: float
    support_threshold: float
    support_alert: bool
    model_hashes: Mapping[str, str]
    feature_row: Mapping[str, object]
    matrices: Mapping[str, np.ndarray]
    feature_names: Mapping[str, Tuple[str, ...]]
    data_revision: str = ""
    bundle_id: str = ""


class V3ServingRuntime(CanonicalHistoryRuntime):
    def __init__(self, root: Path, manifest_rel: str = SELECTED_MODELS_REL):
        root = Path(root).resolve()
        # Verify and load the scientific bundle FIRST (fail fast, before any heavy store loads).
        self.bundle, self.release_info = self._load_bundle(root, manifest_rel)
        super().__init__(root)
        if self.bundle._required_inputs() != (False, False):
            raise V3ServingError("selected V3 bundle needs past-SOFA trend/component inputs, which the product runtime does not supply")
        self.model_hashes = dict(self.bundle.model_hashes)
        self.models = self.bundle.models

    @staticmethod
    def _load_bundle(root: Path, manifest_rel: str):
        manifest = root / manifest_rel
        try:
            actual = _sha256(manifest)
            handoff = json.loads((root / HANDOFF_REL).read_text(encoding="utf-8"))
            freeze = json.loads((root / FREEZE_REL).read_text(encoding="utf-8"))
            expected = {
                "product_handoff_v1.recommended_bundle.manifest_sha256": handoff["recommended_bundle"]["manifest_sha256"],
                "v3_model_freeze_v1.selected_models_v3_sha256": freeze["selected_models_v3_sha256"],
            }
            for label, value in expected.items():
                if value != actual:
                    raise V3ServingError(f"selected_models_v3.json sha256 {actual[:12]}… does not match {label} {str(value)[:12]}…")
            v31 = root / "artifacts/performance_v3_1/product_handoff.json"
            if v31.exists():
                effective = json.loads(v31.read_text(encoding="utf-8"))["effective_release"]["manifest_sha256"]
                if effective != actual:
                    raise V3ServingError("V3.1 handoff effective release does not match the loaded V3 manifest")
            bundle = V3Bundle.load(root, manifest_rel)
            if abs(bundle.threshold - THRESHOLD_VALUE) > 1e-12:
                raise V3ServingError("frozen support threshold mismatch")
            if bundle.calibrator.get("method") != "platt":
                raise V3ServingError("frozen support calibrator is not Platt")
            if set(bundle.models) != set(TASKS):
                raise V3ServingError("bundle does not define exactly the four tasks")
            manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
        except V3ServingError:
            raise
        except (BundleError, OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise V3ServingError(f"V3 bundle failed verification: {type(error).__name__}: {error}") from error
        info = {
            "bundle_id": bundle.bundle_id, "release": "V3", "status": "CURRENT",
            "selected_models_sha256": actual, "selected_models_ref": manifest_rel,
            "model_freeze_sha256": _sha256(root / FREEZE_REL),
            "final_evaluation_freeze_ref": FINAL_FREEZE_REL,
            "final_evaluation_freeze_sha256": _sha256(root / FINAL_FREEZE_REL) if (root / FINAL_FREEZE_REL).exists() else None,
            "tasks": {t: {"family": bundle.models[t].family, "feature_variant": bundle.variants[t], "model_sha256": bundle.model_hashes[t],
                          "n_features": len(bundle.models[t].feature_columns), "n_estimators": bundle.models[t].n_estimators}
                      for t in TASKS},
            "calibrator": {"method": bundle.calibrator.get("method"), "params_sha256": bundle.model_hashes.get("calibrator")},
            "threshold": {"value": bundle.threshold, "comparator": ">="},
            "protocol_sha256": manifest_doc.get("protocol_sha256"), "amendment_sha256": manifest_doc.get("amendment_sha256"),
        }
        return bundle, info

    def predict(self, stay_id: str, prediction_time: str) -> V3Prediction:
        ctx = self.canonical_row(stay_id, prediction_time)
        row = dict(ctx.row)
        if row.get("baseline_sofa") is None or not np.isfinite(float(row["baseline_sofa"])):
            raise V3ServingError("current SOFA(t) unavailable; refusing to predict (defect D1 guard)")
        feats = self.bundle.row_features([row])
        matrices = {t: self.bundle.feature_matrix(t, feats) for t in TASKS}
        names = {t: tuple(self.bundle.models[t].feature_columns) for t in TASKS}
        if any(np.isnan(matrices[t][0, names[t].index("B__current_sofa")]) for t in TASKS if "B__current_sofa" in names[t]):
            raise V3ServingError("B__current_sofa is missing in a model matrix (defect D1 guard)")
        out = self.bundle.predict_rows([row])[0]
        return V3Prediction(
            stay_id=stay_id, prediction_time=prediction_time, grid_index=ctx.grid_index, icu_elapsed_hours=ctx.icu_elapsed_hours,
            current_sofa=float(out["current_sofa"]), recovery24_delta=out["recovery24_delta"], recovery48_delta=out["recovery48_delta"],
            recovery24_sofa=out["recovery24_sofa"], recovery48_sofa=out["recovery48_sofa"],
            icu_raw_log_prediction=out["icu_raw_log_prediction"], icu_remaining_hours=out["icu_remaining_hours"],
            support_raw_probability=out["support_raw_probability"], support_calibrated_probability=out["support_calibrated_probability"],
            support_threshold=out["support_threshold"], support_alert=out["support_alert"], model_hashes=dict(self.model_hashes),
            feature_row=row, matrices=matrices, feature_names=names, data_revision=self.data_revision(stay_id), bundle_id=self.bundle.bundle_id,
        )
