"""Real TaskPredictor / FrozenPreprocessor adapters for the frozen Stage-3 models.

Each predictor loads through the existing tested model-loading utilities
(``models.xgb_canonical.load_xgb_bundle``, ``models.gru_icu_time.load_icu_time_bundle``)
and exposes exactly the seams ``serving.pipeline.PredictionPipeline`` and the
explanation router already require. No fitting, training, or calibration
happens here; every artifact is loaded read-only from paths whose hashes were
already verified by the Stage-4 resolver before this module is invoked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch

from data.collate import CanonicalBatch
from explainability.tree_shap import FlattenedFeatureIdentity, PreparedTreeInput
from models.gru_icu_time import load_icu_time_bundle
from models.gru_recovery import RECOVERY_HORIZON_ORDER
from models.xgb_canonical import RecoveryXGBBundle, SupportXGBBundle, XGBLineage, load_xgb_bundle
from preprocess.synthetic_phase10 import FrozenSyntheticFeaturePreprocessor
from preprocess.target_scaler import RecoveryTargetScaler
from serving.interfaces import ModelIdentity
from serving.recovery import ORIGINAL_DELTA_DOMAIN, OriginalUnitRecoveryDeltas
from training.class_weights import SupportClassWeight
from vedant_infra.hashing import sha256_file


PREDICTION_TIME_KEY = "_stage4_prediction_time"


class RealPredictorError(RuntimeError):
    pass


@dataclass(frozen=True)
class FlatFeatureMap:
    payload: Mapping[str, object]
    sha256: str

    @property
    def flat_dimension(self) -> int:
        return int(self.payload["flat_dimension"])

    @property
    def map_version(self) -> str:
        return str(self.payload["map_version"])

    @property
    def flat_names(self) -> Tuple[str, ...]:
        return tuple(item["flat_name"] for item in self.payload["entries"])

    @property
    def entries(self) -> Tuple[Mapping[str, object], ...]:
        return tuple(self.payload["entries"])


def load_flat_feature_map(path: Path) -> FlatFeatureMap:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FlatFeatureMap(payload=payload, sha256=sha256_file(path))


def build_feature_identities(flat_map: FlatFeatureMap) -> Tuple[FlattenedFeatureIdentity, ...]:
    """Reverse-map every flat index to its original bin/feature identity.

    Sourced directly from the frozen ``synthetic_xgb_flat_feature_map_v1``
    entries (``information_type`` -> ``channel_type``, ``canonical_feature_name``
    -> ``base_feature``, ``bin_index`` unchanged); relative bin hours are
    recomputed with the exact formula ``FlattenedFeatureIdentity.validate``
    enforces so the identity is provably self-consistent.
    """

    identities = []
    for entry in flat_map.entries:
        channel = entry["information_type"]
        bin_index = entry["bin_index"]
        temporal = channel in ("value", "observation_mask", "tslo", "padding")
        start = end = None
        if temporal:
            start = -48 + 6 * int(bin_index)
            end = start + 6
        base_feature = entry["canonical_feature_name"]
        if base_feature is None:
            base_feature = "structural_padding"
        identity = FlattenedFeatureIdentity(
            flat_name=entry["flat_name"],
            base_feature=base_feature,
            channel_type=channel,
            bin_index=bin_index,
            relative_start_hours=start,
            relative_end_hours=end,
            aggregation=entry.get("aggregation"),
        )
        identity.validate()
        identities.append(identity)
    return tuple(identities)


def _xgb_flat_vector(transformed: Mapping[str, object], *, flat_dimension: int) -> np.ndarray:
    """Concatenate a transformed xgboost-family view exactly as flatten_row does.

    ``data.xgb_canonical.flatten_row`` concatenates
    ``values, observation_mask, tslo_hours, padding_mask, statics`` in that
    literal order (the frozen flat feature map's block_order). The canonical
    builder's "xgboost" family view is already the row-major flatten of the
    "gru" [8,F] view, so this reproduces the identical vector without
    re-synthesizing a training-row dict.
    """

    values = np.asarray(transformed["history_values"], dtype=np.float32)
    mask = np.asarray(transformed["observation_mask"], dtype=np.float32)
    tslo = np.asarray(transformed["tslo_hours"], dtype=np.float32)
    padding = np.asarray(transformed["padding_mask"], dtype=np.float32)
    statics = np.asarray(transformed["static_features"], dtype=np.float32)
    flat = np.concatenate((values, mask, tslo, padding, statics)).astype(np.float32, copy=False)
    if flat.shape != (flat_dimension,) or not np.isfinite(flat).all():
        raise RealPredictorError("flattened XGBoost serving row has invalid dimension or values")
    return flat.reshape(1, -1)


def _canonical_batch(
    transformed: Mapping[str, object],
    *,
    feature_names: Sequence[str],
    feature_schema_version: str,
    prediction_time: str,
) -> CanonicalBatch:
    sequence = torch.tensor([transformed["history_values"]], dtype=torch.float32)
    observation_mask = torch.tensor([transformed["observation_mask"]], dtype=torch.bool)
    padding_mask = torch.tensor([transformed["padding_mask"]], dtype=torch.bool)
    tslo = transformed.get("tslo_hours")
    tslo_tensor = None if tslo is None else torch.tensor([tslo], dtype=torch.float32)
    static = transformed.get("static_features")
    static_tensor = None if static is None else torch.tensor([static], dtype=torch.float32)
    return CanonicalBatch(
        identifiers={"prediction_time": (prediction_time,)},
        sequence=sequence,
        padding_mask=padding_mask,
        observation_mask=observation_mask,
        tslo=tslo_tensor,
        static_features=static_tensor,
        targets={},
        eligibility={},
        versions={"feature_schema_version": feature_schema_version},
        feature_names=tuple(feature_names),
    )


class RealFrozenPreprocessor:
    """Adapt ``FrozenSyntheticFeaturePreprocessor`` to one task/family.

    For the ``gru`` family the output is a batch-of-one ``CanonicalBatch``
    (the exact type both prediction and the Integrated Gradients adapter
    require); for ``xgboost`` it is a ``PreparedTreeInput`` binding the
    flattened ``[1,520]`` row to its explicit feature identities (the exact
    type both prediction and TreeSHAP require). Both read
    ``PREDICTION_TIME_KEY`` off the supplied view — added by
    ``serving.real.history.Stage4CanonicalInputProvider`` one layer above the
    frozen, unmodified ``CanonicalHistoryInputProvider`` — since the shared
    family view itself carries no cutoff.
    """

    def __init__(
        self,
        *,
        task: str,
        family: str,
        path: Path,
        flat_map: FlatFeatureMap,
        feature_schema_version: str,
        temporal_feature_names: Sequence[str] = (),
        feature_identities: Tuple[FlattenedFeatureIdentity, ...] = (),
    ) -> None:
        self.task = task
        self.family = family
        self._impl = FrozenSyntheticFeaturePreprocessor(path)
        self.artifact_sha256 = self._impl.artifact_sha256
        self._flat_map = flat_map
        self._feature_schema_version = feature_schema_version
        self._temporal_feature_names = tuple(temporal_feature_names)
        self._feature_identities = feature_identities
        self.feature_metadata: Mapping[str, object] = {}

    def transform(self, prepared_input: Mapping[str, object]) -> object:
        prediction_time = prepared_input[PREDICTION_TIME_KEY]
        view = {key: value for key, value in prepared_input.items() if key != PREDICTION_TIME_KEY}
        transformed = self._impl.transform(view)
        if self.family == "gru":
            self.feature_metadata = {
                "static_feature_names": tuple(transformed.get("static_feature_names", ()))
            }
            return _canonical_batch(
                transformed,
                feature_names=self._temporal_feature_names,
                feature_schema_version=self._feature_schema_version,
                prediction_time=prediction_time,
            )
        flat = _xgb_flat_vector(transformed, flat_dimension=self._flat_map.flat_dimension)
        prepared = PreparedTreeInput(
            values=flat,
            feature_identities=self._feature_identities,
            feature_schema_version=self._feature_schema_version,
            flattening_version=self._flat_map.map_version,
            flattening_sha256=self._flat_map.sha256,
            prediction_time=prediction_time,
            task=self.task,
        )
        prepared.validate()
        return prepared


class RecoveryXGBoostTaskPredictor:
    """Loads xgb-recovery-014 once; predicts both frozen horizons together."""

    task = "recovery"
    family = "xgboost"

    def __init__(
        self,
        *,
        identity: ModelIdentity,
        bundle_dir: Path,
        scaler_path: Path,
        flat_map: FlatFeatureMap,
        feature_schema_version: str,
    ) -> None:
        self.model_version = identity.model_version
        self.artifact_sha256 = identity.artifact_sha256
        metadata = json.loads((bundle_dir / "bundle.metadata.json").read_text(encoding="utf-8"))
        lineage = XGBLineage(**metadata["lineage"])
        scaler = RecoveryTargetScaler.load(scaler_path)
        bundle: RecoveryXGBBundle = load_xgb_bundle(
            bundle_dir,
            expected_task="recovery",
            expected_lineage=lineage,
            expected_feature_names=flat_map.flat_names,
            recovery_scaler=scaler,
            recovery_scaler_path=scaler_path,
        )
        self._bundle = bundle
        self.scaler = scaler
        self.horizon_order = RECOVERY_HORIZON_ORDER
        # Raw sklearn boosters, exposed for TreeSHAP: shap.TreeExplainer must
        # receive an actual xgboost model object, not this wrapper.
        self.tree_shap_models = {
            RECOVERY_HORIZON_ORDER[0]: bundle.model24,
            RECOVERY_HORIZON_ORDER[1]: bundle.model48,
        }
        self.xgboost_version = bundle.xgboost_version
        self.flattened_feature_names = flat_map.flat_names
        self.input_dim = flat_map.flat_dimension
        self.feature_schema_version = feature_schema_version
        self.flattening_version = flat_map.map_version
        self.flattening_sha256 = flat_map.sha256

    def predict(self, prepared_input: PreparedTreeInput) -> OriginalUnitRecoveryDeltas:
        if not isinstance(prepared_input, PreparedTreeInput):
            raise RealPredictorError("recovery predictor requires a PreparedTreeInput")
        original = self._bundle.predict(prepared_input.values)[0]
        return OriginalUnitRecoveryDeltas(
            delta_24h=float(original[0]),
            delta_48h=float(original[1]),
            output_domain=ORIGINAL_DELTA_DOMAIN,
            horizon_order=RECOVERY_HORIZON_ORDER,
            transform_provenance="frozen_inverse_transform_applied_once",
        )


class SupportXGBoostTaskPredictor:
    """Loads xgb-support-024 once; predicts the raw uncalibrated probability."""

    task = "organ_support"
    family = "xgboost"

    def __init__(
        self,
        *,
        identity: ModelIdentity,
        bundle_dir: Path,
        class_weight_path: Path,
        flat_map: FlatFeatureMap,
        feature_schema_version: str,
    ) -> None:
        self.model_version = identity.model_version
        self.artifact_sha256 = identity.artifact_sha256
        metadata = json.loads((bundle_dir / "bundle.metadata.json").read_text(encoding="utf-8"))
        lineage = XGBLineage(**metadata["lineage"])
        weight = SupportClassWeight.load(class_weight_path)
        bundle: SupportXGBBundle = load_xgb_bundle(
            bundle_dir,
            expected_task="organ_support",
            expected_lineage=lineage,
            expected_feature_names=flat_map.flat_names,
            support_class_weight=weight,
            support_class_weight_path=class_weight_path,
        )
        self._bundle = bundle
        self.tree_shap_models = {"organ_support_probability": bundle.model}
        self.xgboost_version = bundle.xgboost_version
        self.flattened_feature_names = flat_map.flat_names
        self.input_dim = flat_map.flat_dimension
        self.feature_schema_version = feature_schema_version
        self.flattening_version = flat_map.map_version
        self.flattening_sha256 = flat_map.sha256

    def predict(self, prepared_input: PreparedTreeInput) -> float:
        if not isinstance(prepared_input, PreparedTreeInput):
            raise RealPredictorError("support predictor requires a PreparedTreeInput")
        probability = self._bundle.predict(prepared_input.values)[0]
        return float(probability)


class ICUTimeGRUTaskPredictor(torch.nn.Module):
    """Loads gru-icu-time-026 once; wraps it as the single object used for
    both prediction (via ``predict``) and Integrated Gradients (this module's
    own differentiable ``forward``), satisfying the router's identity check.
    """

    task = "icu_stay_time"
    family = "gru"

    def __init__(
        self,
        *,
        identity: ModelIdentity,
        checkpoint_path: Path,
        temporal_feature_names: Sequence[str],
        feature_schema_version: str,
    ) -> None:
        super().__init__()
        self.model_version = identity.model_version
        self.artifact_sha256 = identity.artifact_sha256
        sidecar = json.loads(
            checkpoint_path.with_name(checkpoint_path.name + ".metadata.json").read_text(encoding="utf-8")
        )
        model, metadata = load_icu_time_bundle(
            checkpoint_path,
            expected_tensor_contract_version=sidecar["tensor_contract_version"],
            expected_feature_schema_version=sidecar["feature_schema_version"],
        )
        self.inner = model
        self.metadata = metadata
        # Set eval mode on this wrapper (not just the inner module): the IG
        # adapter reads/restores training mode from whatever object it was
        # given (``context.model``, i.e. this wrapper), and nn.Module starts
        # in training mode by default. Without this, the adapter's
        # `model.train(was_training)` restores *training* mode (dropout
        # enabled) after every explanation, corrupting later predictions.
        self.eval()

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        return self.inner(batch)

    def predict(self, prepared_input: CanonicalBatch) -> float:
        if not isinstance(prepared_input, CanonicalBatch):
            raise RealPredictorError("ICU-time predictor requires a prepared CanonicalBatch")
        with torch.no_grad():
            output = self.inner(prepared_input)
        return float(output.reshape(()).item())
