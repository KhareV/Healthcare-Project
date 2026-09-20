"""Hash-first selected-manifest resolution and serving compatibility checks."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Tuple

from evaluation.selection_validation import EXPLANATION_METHOD, SERVING_FAMILIES, TASKS
from experiments.lineage import read_artifact_index
from experiments.search_governance import canonical_sha256
from serving.interfaces import FrozenPreprocessor, ModelIdentity, SupportCalibrator, TaskPredictor
from vedant_infra.hashing import is_sha256, sha256_file


SYNTHETIC_MANIFEST_VERSION = "selected_models_synthetic_phase6_v1"
SYNTHETIC_STATUS = "SYNTHETIC_NON_SCIENTIFIC_NON_SERVING_NON_G3_AUTHORIZING"
REAL_MANIFEST_VERSION = "selected_models_calibrated_v1"


class ServingArtifactError(RuntimeError):
    """Base error for fail-closed serving initialization."""


class ManifestInvalidError(ServingArtifactError):
    pass


class ArtifactMissingError(ServingArtifactError):
    pass


class ArtifactHashMismatchError(ServingArtifactError):
    pass


class ArtifactCompatibilityError(ServingArtifactError):
    pass


class UnsupportedServingFamilyError(ServingArtifactError):
    pass


class SyntheticArtifactProhibitedError(ServingArtifactError):
    pass


class UnregisteredArtifactError(ServingArtifactError):
    pass


ModelLoader = Callable[[str, Path, Mapping[str, object], ModelIdentity], TaskPredictor]
PreprocessorLoader = Callable[[str, Path, Mapping[str, object], ModelIdentity], FrozenPreprocessor]
CalibratorLoader = Callable[[Path, Mapping[str, object]], SupportCalibrator]


@dataclass(frozen=True)
class ResolvedTaskBundle:
    identity: ModelIdentity
    predictor: TaskPredictor
    preprocessor: FrozenPreprocessor


@dataclass(frozen=True)
class ServingBundle:
    scope: str
    manifest_version: str
    manifest_ref: str
    manifest_file_sha256: str
    split_version: str
    tasks: Tuple[ResolvedTaskBundle, ...]
    support_calibrator: SupportCalibrator
    support_calibrator_sha256: str
    support_threshold: float
    support_threshold_sha256: str

    def task(self, name: str) -> ResolvedTaskBundle:
        for item in self.tasks:
            if item.identity.task == name:
                return item
        raise ArtifactCompatibilityError("resolved serving bundle lacks task: " + name)


def _safe_path(root: Path, reference: object, field: str) -> Path:
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ManifestInvalidError(field + " must be a repository-relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / reference).resolve()
    if resolved_root not in resolved.parents:
        raise ManifestInvalidError(field + " escapes repository root")
    return resolved


def _verify_file(root: Path, reference: object, expected_hash: object, field: str) -> Path:
    path = _safe_path(root, reference, field)
    if not isinstance(expected_hash, str) or not is_sha256(expected_hash):
        raise ManifestInvalidError(field + " SHA-256 is malformed")
    if not path.is_file():
        raise ArtifactMissingError(field + " is missing: " + str(reference))
    if sha256_file(path) != expected_hash:
        raise ArtifactHashMismatchError(field + " SHA-256 mismatch")
    return path


def _read_verified_json(path: Path, field: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactCompatibilityError(field + " is not valid JSON metadata") from error
    if not isinstance(value, dict):
        raise ArtifactCompatibilityError(field + " metadata must be an object")
    return MappingProxyType(value)


def _require_nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestInvalidError(field + " is required")
    return value


def _verify_manifest_integrity(manifest: Mapping[str, object]) -> None:
    supplied = manifest.get("manifest_sha256")
    content = dict(manifest)
    content.pop("manifest_sha256", None)
    if not isinstance(supplied, str) or supplied != canonical_sha256(content):
        raise ArtifactHashMismatchError("selected manifest content hash mismatch")


def _validate_scope(manifest: Mapping[str, object], scope: str) -> None:
    if scope not in ("synthetic", "real"):
        raise ManifestInvalidError("serving scope must be synthetic or real")
    if scope == "synthetic":
        if (
            manifest.get("manifest_version") != SYNTHETIC_MANIFEST_VERSION
            or manifest.get("selection_mode") != "synthetic"
            or manifest.get("status") != SYNTHETIC_STATUS
            or manifest.get("serving_ready") is not False
        ):
            raise ManifestInvalidError("synthetic serving manifest classification is invalid")
    else:
        if manifest.get("selection_mode") != "real":
            raise SyntheticArtifactProhibitedError("synthetic manifest cannot authorize real serving")
        if manifest.get("manifest_version") != REAL_MANIFEST_VERSION:
            raise ManifestInvalidError("real serving requires selected_models_calibrated_v1")
        if manifest.get("serving_ready") is not True:
            raise ManifestInvalidError("real selected manifest is not serving-ready")
    if manifest.get("test_accessed") is not False:
        raise ManifestInvalidError("serving manifest must not depend on test access")


def _task_identity(task: str, entry: Mapping[str, object], split_hash: str) -> ModelIdentity:
    family = entry.get("family")
    if family not in SERVING_FAMILIES:
        raise UnsupportedServingFamilyError("unsupported serving family for " + task)
    if entry.get("explanation_method") != EXPLANATION_METHOD[family]:
        raise ArtifactCompatibilityError("explanation route mismatch for " + task)
    if entry.get("split_hash") != split_hash:
        raise ArtifactCompatibilityError("split provenance mismatch for " + task)
    transform_required = entry.get("target_transform_required") is True
    transform_ref = entry.get("target_transform_ref")
    transform_hash = entry.get("target_transform_sha256")
    if transform_required:
        transform_ref = _require_nonempty(transform_ref, task + ".target_transform_ref")
        transform_hash = _require_nonempty(transform_hash, task + ".target_transform_sha256")
    elif transform_ref is not None or transform_hash is not None:
        raise ManifestInvalidError(task + " target-transform fields are inconsistent")
    return ModelIdentity(
        task=task,
        family=family,
        model_version=_require_nonempty(entry.get("model_version"), task + ".model_version"),
        artifact_ref=_require_nonempty(entry.get("artifact_ref"), task + ".artifact_ref"),
        artifact_sha256=_require_nonempty(entry.get("artifact_sha256"), task + ".artifact_sha256"),
        preprocessor_ref=_require_nonempty(entry.get("preprocessor_ref"), task + ".preprocessor_ref"),
        preprocessor_sha256=_require_nonempty(entry.get("preprocessor_sha256"), task + ".preprocessor_sha256"),
        feature_version=_require_nonempty(entry.get("feature_version"), task + ".feature_version"),
        label_version=_require_nonempty(entry.get("label_version"), task + ".label_version"),
        split_hash=split_hash,
        explanation_method=entry["explanation_method"],
        target_transform_ref=transform_ref,
        target_transform_sha256=transform_hash,
    )


def _validate_task_metadata(
    task: str,
    identity: ModelIdentity,
    model_metadata: Mapping[str, object],
    preprocessor_metadata: Mapping[str, object],
    *,
    scope: str,
) -> None:
    expected_model = {
        "task": task,
        "family": identity.family,
        "model_version": identity.model_version,
        "artifact_sha256": identity.artifact_sha256,
        "feature_version": identity.feature_version,
        "label_version": identity.label_version,
        "split_hash": identity.split_hash,
        "preprocessor_sha256": identity.preprocessor_sha256,
        "target_transform_required": identity.target_transform_ref is not None,
    }
    for field, expected in expected_model.items():
        if model_metadata.get(field) != expected:
            raise ArtifactCompatibilityError(task + " model metadata mismatch: " + field)
    expected_output = {
        "recovery": "independent_delta_24h_delta_48h",
        "icu_stay_time": "log1p_remaining_current_icu_hours",
        "organ_support": "uncalibrated_probability",
    }[task]
    if model_metadata.get("output_contract") != expected_output:
        raise ArtifactCompatibilityError(task + " model output contract mismatch")
    if identity.target_transform_ref is not None and model_metadata.get(
        "target_transform_sha256"
    ) != identity.target_transform_sha256:
        raise ArtifactCompatibilityError(task + " target-transform binding mismatch")
    expected_preprocessor = {
        "task": task,
        "feature_version": identity.feature_version,
        "split_hash": identity.split_hash,
        "fit_partition": "train",
    }
    for field, expected in expected_preprocessor.items():
        if preprocessor_metadata.get(field) != expected:
            raise ArtifactCompatibilityError(task + " preprocessor mismatch: " + field)
    synthetic_flags = (
        model_metadata.get("synthetic"), preprocessor_metadata.get("synthetic")
    )
    if scope == "synthetic" and synthetic_flags != (True, True):
        raise ArtifactCompatibilityError("synthetic task artifacts must be explicitly marked")
    if scope == "synthetic" and (
        model_metadata.get("serving_authorized") is not False
        or preprocessor_metadata.get("serving_authorized") is not False
    ):
        raise ArtifactCompatibilityError("synthetic task artifacts cannot claim serving authorization")
    if scope == "real" and any(synthetic_flags):
        raise SyntheticArtifactProhibitedError("synthetic task artifact used in real serving")
    if scope == "real" and (
        model_metadata.get("serving_authorized") is not True
        or preprocessor_metadata.get("serving_authorized") is not True
    ):
        raise ArtifactCompatibilityError("real task artifact is not serving-authorized")
    if scope == "real" and any(
        str(value).startswith("SYNTHETIC_")
        for value in (
            identity.model_version,
            identity.feature_version,
            identity.label_version,
        )
    ):
        raise SyntheticArtifactProhibitedError("synthetic task version used in real serving")


def _validate_support_chain(
    support: ModelIdentity,
    calibrator: Mapping[str, object],
    calibrator_hash: str,
    threshold: Mapping[str, object],
    *,
    scope: str,
) -> float:
    calibrator_expected = {
        "selected_family": support.family,
        "selected_model_sha256": support.artifact_sha256,
        "feature_version": support.feature_version,
        "label_version": support.label_version,
        "split_hash": support.split_hash,
        "preprocessor_sha256": support.preprocessor_sha256,
    }
    for field, expected in calibrator_expected.items():
        if calibrator.get(field) != expected:
            raise ArtifactCompatibilityError("support calibrator/model mismatch: " + field)
    threshold_expected = {
        "selected_model_sha256": support.artifact_sha256,
        "calibrator_sha256": calibrator_hash,
        "split_hash": support.split_hash,
    }
    for field, expected in threshold_expected.items():
        if threshold.get(field) != expected:
            raise ArtifactCompatibilityError("support threshold/calibrator mismatch: " + field)
    value = threshold.get("threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ArtifactCompatibilityError("support threshold must be finite within [0,1]")
    if threshold.get("comparator") != "greater_than_or_equal":
        raise ArtifactCompatibilityError("support threshold comparator is incompatible")
    flags = (calibrator.get("synthetic"), threshold.get("synthetic"))
    if scope == "synthetic" and flags != (True, True):
        raise ArtifactCompatibilityError("synthetic support artifacts must be explicitly marked")
    if scope == "synthetic" and (
        calibrator.get("serving_authorized") is not False
        or threshold.get("serving_authorized") is not False
    ):
        raise ArtifactCompatibilityError("synthetic support artifacts cannot claim serving authorization")
    if scope == "real" and any(flags):
        raise SyntheticArtifactProhibitedError("synthetic calibration used in real serving")
    if scope == "real" and (
        calibrator.get("serving_authorized") is not True
        or threshold.get("serving_authorized") is not True
    ):
        raise ArtifactCompatibilityError("real support artifact is not serving-authorized")
    return float(value)


def _validate_real_registry(
    root: Path,
    manifest_ref: str,
    manifest_hash: str,
    identities: Tuple[ModelIdentity, ...],
    calibration_refs: Tuple[Tuple[str, str, str], ...],
) -> None:
    records = read_artifact_index(root / "experiments/artifacts.csv")
    by_path = {item.artifact_path: item for item in records}
    requirements = [(manifest_ref, manifest_hash, "selected_model_manifest")]
    for identity in identities:
        requirements.extend(
            (
                (identity.artifact_ref, identity.artifact_sha256, "model_checkpoint"),
                (identity.preprocessor_ref, identity.preprocessor_sha256, "preprocessor"),
            )
        )
    requirements.extend(calibration_refs)
    for reference, digest, artifact_type in requirements:
        record = by_path.get(reference)
        if (
            record is None
            or record.artifact_sha256 != digest
            or record.artifact_type != artifact_type
            or record.run_type != "scientific"
            or record.status != "registered"
        ):
            raise UnregisteredArtifactError("real serving artifact is not traceable: " + reference)


def load_serving_bundle(
    root: Path,
    *,
    manifest_ref: str,
    expected_manifest_sha256: str,
    scope: str,
    model_loaders: Mapping[str, ModelLoader],
    preprocessor_loader: PreprocessorLoader,
    calibrator_loader: CalibratorLoader,
) -> ServingBundle:
    """Validate every dependency before invoking any deserializing loader."""

    root = root.resolve()
    manifest_path = _verify_file(root, manifest_ref, expected_manifest_sha256, "selected manifest")
    manifest = _read_verified_json(manifest_path, "selected manifest")
    _verify_manifest_integrity(manifest)
    _validate_scope(manifest, scope)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASKS):
        raise ManifestInvalidError("selected manifest requires exactly three known tasks")
    split_hash = manifest.get("split_hash")
    if not isinstance(split_hash, str) or not is_sha256(split_hash):
        raise ManifestInvalidError("manifest split_hash is malformed")
    split_version = _require_nonempty(manifest.get("split_version"), "split_version")
    if scope == "real" and split_version.startswith("SYNTHETIC_"):
        raise SyntheticArtifactProhibitedError("synthetic split version used in real serving")

    identities = tuple(_task_identity(task, tasks[task], split_hash) for task in TASKS)
    if scope == "real":
        recovery_identity = next(item for item in identities if item.task == "recovery")
        if recovery_identity.family == "gru" and recovery_identity.target_transform_ref is None:
            raise ArtifactCompatibilityError(
                "real recovery GRU requires frozen target-transform statistics"
            )
    verified = {}
    for identity in identities:
        entry = tasks[identity.task]
        model = _verify_file(root, identity.artifact_ref, identity.artifact_sha256, identity.task + " model")
        model_meta = _verify_file(root, entry.get("artifact_metadata_ref"), entry.get("artifact_metadata_sha256"), identity.task + " model metadata")
        preprocessor = _verify_file(root, identity.preprocessor_ref, identity.preprocessor_sha256, identity.task + " preprocessor")
        target_transform = None
        if identity.target_transform_ref is not None:
            target_transform = _verify_file(
                root,
                identity.target_transform_ref,
                identity.target_transform_sha256,
                identity.task + " target transform",
            )
        verified[identity.task] = (model, model_meta, preprocessor, target_transform)

    support_entry = tasks["organ_support"]
    calibrator_ref = _require_nonempty(support_entry.get("calibrator_ref"), "support.calibrator_ref")
    calibrator_hash = _require_nonempty(support_entry.get("calibrator_sha256"), "support.calibrator_sha256")
    threshold_ref = _require_nonempty(support_entry.get("threshold_ref"), "support.threshold_ref")
    threshold_hash = _require_nonempty(support_entry.get("threshold_sha256"), "support.threshold_sha256")
    calibrator_path = _verify_file(root, calibrator_ref, calibrator_hash, "support calibrator")
    threshold_path = _verify_file(root, threshold_ref, threshold_hash, "support threshold")

    metadata = {}
    for identity in identities:
        _, model_meta_path, preprocessor_path, _ = verified[identity.task]
        model_meta = _read_verified_json(model_meta_path, identity.task + " model metadata")
        preprocessor_meta = _read_verified_json(preprocessor_path, identity.task + " preprocessor")
        _validate_task_metadata(identity.task, identity, model_meta, preprocessor_meta, scope=scope)
        metadata[identity.task] = (model_meta, preprocessor_meta)
    calibrator_meta = _read_verified_json(calibrator_path, "support calibrator")
    threshold_meta = _read_verified_json(threshold_path, "support threshold")
    threshold_value = _validate_support_chain(
        next(item for item in identities if item.task == "organ_support"),
        calibrator_meta,
        calibrator_hash,
        threshold_meta,
        scope=scope,
    )

    if scope == "real":
        _validate_real_registry(
            root,
            manifest_ref,
            expected_manifest_sha256,
            identities,
            (
                (calibrator_ref, calibrator_hash, "calibrator"),
                (threshold_ref, threshold_hash, "threshold"),
            ),
        )

    resolved = []
    for identity in identities:
        if identity.family not in model_loaders:
            raise UnsupportedServingFamilyError("no loader is registered for " + identity.family)
        model_path, _, preprocessor_path, _ = verified[identity.task]
        model_meta, preprocessor_meta = metadata[identity.task]
        predictor = model_loaders[identity.family](identity.task, model_path, model_meta, identity)
        preprocessor = preprocessor_loader(identity.task, preprocessor_path, preprocessor_meta, identity)
        if (
            predictor.task != identity.task
            or predictor.family != identity.family
            or predictor.model_version != identity.model_version
            or predictor.artifact_sha256 != identity.artifact_sha256
        ):
            raise ArtifactCompatibilityError("model loader returned incompatible identity")
        if preprocessor.task != identity.task or preprocessor.artifact_sha256 != identity.preprocessor_sha256:
            raise ArtifactCompatibilityError("preprocessor loader returned incompatible identity")
        resolved.append(ResolvedTaskBundle(identity, predictor, preprocessor))
    calibrator = calibrator_loader(calibrator_path, calibrator_meta)
    if calibrator.artifact_sha256 != calibrator_hash:
        raise ArtifactCompatibilityError("calibrator loader returned incompatible identity")
    return ServingBundle(
        scope=scope,
        manifest_version=str(manifest["manifest_version"]),
        manifest_ref=manifest_ref,
        manifest_file_sha256=expected_manifest_sha256,
        split_version=split_version,
        tasks=tuple(resolved),
        support_calibrator=calibrator,
        support_calibrator_sha256=calibrator_hash,
        support_threshold=threshold_value,
        support_threshold_sha256=threshold_hash,
    )
