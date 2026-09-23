"""Stage-4 real serving bundle resolver.

Binds the exact frozen Stage-3 outputs (``artifacts/models/selected_models_v1.json``
under an ACTIVE G3 freeze) into a ``serving.artifacts.ServingBundle`` and a
``serving.pipeline.PredictionPipeline``, without going through
``serving.artifacts.load_serving_bundle`` — that function's ``scope="real"``
path was built for a manifest shape
(``serving_ready=True``, per-task ``serving_authorized``/``output_contract``
sidecar metadata) that the actually-accepted, G3-frozen
``selected_models_calibrated_v1`` manifest deliberately does not and must not
have (``serving_ready`` stays ``false`` so its hash — and G3 — never change).

Every dependency hash this resolver trusts is independently re-verified here
by re-reading ``validate_g3_marker``'s live-state comparison (G1, Phase-14,
Stage-2 selection, LSTM sensitivity, split, preprocessor, recovery scaler,
support class weight, selected_models_v1, support calibrator, support
threshold, code commit) plus this module's own per-model-file hash checks.
Nothing here mutates any G3-bound artifact.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

from data.synthetic.history_provider import CanonicalTimelineHistoryProvider
from data.synthetic.provenance import sha256_file as synthetic_sha256_file
from data.synthetic.sofa_provider import PulkitStateSOFASupportProvider, SyntheticCurrentSOFAProvider
from data.synthetic.validation import load_jsonl
from evaluation.calibrate import apply_support_calibrator, load_calibrator_artifact
from evaluation.selection_validation import TASKS
from explainability.router import ExplanationRouter
from features.synthetic import SyntheticCanonicalFeatureBuilder
from labels.phase9_final import _normalized
from labels.support_state import ExecutionMode
from labels.synthetic_profile import load_synthetic_event_dictionary
from serving.artifacts import ResolvedTaskBundle, ServingBundle
from serving.history import SyntheticPointEventHistoryTruncator
from serving.interfaces import ModelIdentity
from serving.pipeline import PredictionPipeline
from serving.postprocessing import CanonicalICUTimeServingPostprocessor
from serving.preprocessing import CanonicalHistoryInputProvider
from serving.real.explanations import (
    RECOVERY_EXPLANATION_TARGET,
    SUPPORT_OUTPUT_NAME,
    build_real_ig_adapter,
    build_real_tree_shap_adapter,
)
from serving.real.history import (
    NonTestHistoryProvider,
    Stage4CanonicalInputProvider,
    Stage4CurrentSOFAProvider,
    load_stay_split_index,
)
from serving.real.predictors import (
    FlatFeatureMap,
    ICUTimeGRUTaskPredictor,
    RealFrozenPreprocessor,
    RecoveryXGBoostTaskPredictor,
    SupportXGBoostTaskPredictor,
    build_feature_identities,
    load_flat_feature_map,
)
from serving.recovery import CurrentSOFAProvider, ExplicitOriginalUnitDeltaAdapter, RecoveryServingPostprocessor
from vedant_infra.g3 import G3FreezeError, validate_g3_marker
from vedant_infra.hashing import sha256_file


G3_PATH = "artifacts/governance/g3_freeze.json"
SELECTED_MODELS_PATH = "artifacts/models/selected_models_v1.json"


class Stage4BundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class RealSupportCalibrator:
    artifact_sha256: str
    _impl: object

    def transform(self, raw_probability: float) -> float:
        return apply_support_calibrator(self._impl, [raw_probability])[0]


def _json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(task: str, entry: Mapping[str, object]) -> ModelIdentity:
    transform_ref = entry.get("recovery_target_scaler_ref")
    transform_hash = entry.get("recovery_target_scaler_sha256")
    return ModelIdentity(
        task=task,
        family=entry["family"],
        model_version=entry["candidate_id"],
        artifact_ref=entry["artifact_ref"],
        artifact_sha256=entry["artifact_sha256"],
        preprocessor_ref=entry["preprocessor_ref"],
        preprocessor_sha256=entry["preprocessor_sha256"],
        feature_version=entry["feature_version"],
        label_version=entry["label_version"],
        split_hash=entry["split_hash"],
        explanation_method=entry["explanation_method"],
        target_transform_ref=transform_ref,
        target_transform_sha256=transform_hash,
    )


def _verify(root: Path, ref: str, expected_sha256: str, label: str) -> Path:
    path = (root / ref).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise Stage4BundleError(label + " is missing: " + ref)
    if sha256_file(path) != expected_sha256:
        raise Stage4BundleError(label + " hash mismatch: " + ref)
    return path


def _git_commit(root: Path) -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()


@dataclass(frozen=True)
class Stage4Resolution:
    bundle: ServingBundle
    pipeline: PredictionPipeline
    manifest: Mapping[str, object]
    manifest_path: Path
    manifest_sha256: str
    g3_marker: Mapping[str, object]
    g3_path: Path
    g3_sha256: str
    flat_feature_map: FlatFeatureMap
    tslo_no_observation_value: float
    sofa_spec_version: str
    code_commit: str
    current_sofa_provider: object
    split_by_stay: Mapping[object, str]


def resolve_stage4_bundle(root: Path) -> Stage4Resolution:
    """Fail closed on any hash/state mismatch; load nothing until verified."""

    root = root.resolve()
    g3_path = root / G3_PATH
    try:
        marker = validate_g3_marker(g3_path, root, expected_scope="real")
    except G3FreezeError as error:
        raise Stage4BundleError("G3 is not a currently valid active real freeze: " + str(error)) from error
    if marker.get("test_accessed") is not False:
        raise Stage4BundleError("G3 marker records test access; refusing to serve")

    manifest_ref = marker["selected_models_manifest"]
    manifest_sha256 = marker["selected_models_sha256"]
    manifest_path = _verify(root, manifest_ref, manifest_sha256, "selected_models_v1")
    manifest = _json(manifest_path)
    if manifest.get("status") != "SELECTED_MODELS_FROZEN_PRE_TEST" or manifest.get("serving_ready") is not False:
        raise Stage4BundleError("selected_models_v1 is not in the expected pre-test frozen state")
    if manifest.get("test_accessed") is not False:
        raise Stage4BundleError("selected_models_v1 records test access; refusing to serve")
    tasks = manifest["tasks"]
    if set(tasks) != set(TASKS):
        raise Stage4BundleError("selected_models_v1 does not declare exactly the three known tasks")
    for task, entry in tasks.items():
        if entry.get("family") == "lstm":
            raise Stage4BundleError("LSTM is not serving-eligible")

    bindings = manifest["governance_bindings"]
    feature_schema_path = _verify(root, bindings["feature_schema"]["ref"], bindings["feature_schema"]["sha256"], "feature schema")
    preprocessor_path = _verify(root, bindings["preprocessor"]["ref"], bindings["preprocessor"]["sha256"], "preprocessor")
    split_path = _verify(root, bindings["split"]["ref"], bindings["split"]["sha256"], "split")
    scaler_path = _verify(root, bindings["recovery_target_scaler"]["ref"], bindings["recovery_target_scaler"]["sha256"], "recovery target scaler")
    class_weight_path = _verify(root, bindings["support_class_weight"]["ref"], bindings["support_class_weight"]["sha256"], "support class weight")

    flat_map_ref = "artifacts/features/synthetic_xgb_flat_feature_map_v1.json"
    flat_map_sha256 = tasks["recovery"]["flat_feature_map_sha256"]
    flat_map_path = _verify(root, flat_map_ref, flat_map_sha256, "flat feature map")
    flat_map = load_flat_feature_map(flat_map_path)
    feature_identities = build_feature_identities(flat_map)

    calibrator_ref = tasks["organ_support"]["calibrator"]["artifact_ref"]
    calibrator_sha256 = tasks["organ_support"]["calibrator"]["artifact_sha256"]
    calibrator_path = _verify(root, calibrator_ref, calibrator_sha256, "support calibrator")
    threshold_ref = tasks["organ_support"]["threshold"]["artifact_ref"]
    threshold_sha256 = tasks["organ_support"]["threshold"]["artifact_sha256"]
    threshold_path = _verify(root, threshold_ref, threshold_sha256, "support threshold")
    threshold_payload = _json(threshold_path)
    threshold_value = float(threshold_payload["threshold"])
    if tasks["organ_support"]["threshold"]["comparator"] != "greater_than_or_equal":
        raise Stage4BundleError("support threshold comparator is incompatible")
    if threshold_value != tasks["organ_support"]["threshold"]["value"]:
        raise Stage4BundleError("support threshold value differs between manifest and artifact")

    feature_schema_version = tasks["recovery"]["feature_version"]

    identities = {task: _identity(task, tasks[task]) for task in TASKS}

    recovery_entry = tasks["recovery"]
    recovery_bundle_dir = (root / recovery_entry["artifact_ref"]).resolve().parent
    for name, item in recovery_entry["model_artifacts"].items():
        _verify(root, str((Path(recovery_entry["artifact_ref"]).parent / item["path"])), item["sha256"], "recovery " + name + " model")
    recovery_predictor = RecoveryXGBoostTaskPredictor(
        identity=identities["recovery"],
        bundle_dir=recovery_bundle_dir,
        scaler_path=scaler_path,
        flat_map=flat_map,
        feature_schema_version=feature_schema_version,
    )

    support_entry = tasks["organ_support"]
    support_bundle_dir = (root / support_entry["artifact_ref"]).resolve().parent
    for name, item in support_entry["model_artifacts"].items():
        _verify(root, str((Path(support_entry["artifact_ref"]).parent / item["path"])), item["sha256"], "support " + name + " model")
    support_predictor = SupportXGBoostTaskPredictor(
        identity=identities["organ_support"],
        bundle_dir=support_bundle_dir,
        class_weight_path=class_weight_path,
        flat_map=flat_map,
        feature_schema_version=feature_schema_version,
    )

    icu_entry = tasks["icu_stay_time"]
    icu_checkpoint_path = _verify(root, icu_entry["artifact_ref"], icu_entry["artifact_sha256"], "ICU checkpoint")

    schema_payload = _json(feature_schema_path)
    temporal_feature_names = tuple(item["canonical_feature_name"] for item in flat_map.entries[: len(schema_payload["temporal_features"])]) if "temporal_features" in schema_payload else None
    if temporal_feature_names is None:
        # Fall back to the flat map's own temporal block, which is already
        # verified to equal the feature schema's exact bin0 ordering.
        f = flat_map.payload["temporal_feature_count"]
        temporal_feature_names = tuple(entry["canonical_feature_name"] for entry in flat_map.entries[:f])
    tslo_no_observation_value = float(schema_payload["tslo"]["no_observation_sentinel"])

    icu_predictor = ICUTimeGRUTaskPredictor(
        identity=identities["icu_stay_time"],
        checkpoint_path=icu_checkpoint_path,
        temporal_feature_names=temporal_feature_names,
        feature_schema_version=feature_schema_version,
    )

    preprocessors = {
        "recovery": RealFrozenPreprocessor(
            task="recovery", family="xgboost", path=preprocessor_path, flat_map=flat_map,
            feature_schema_version=feature_schema_version, feature_identities=feature_identities,
        ),
        "organ_support": RealFrozenPreprocessor(
            task="organ_support", family="xgboost", path=preprocessor_path, flat_map=flat_map,
            feature_schema_version=feature_schema_version, feature_identities=feature_identities,
        ),
        "icu_stay_time": RealFrozenPreprocessor(
            task="icu_stay_time", family="gru", path=preprocessor_path, flat_map=flat_map,
            feature_schema_version=feature_schema_version, temporal_feature_names=temporal_feature_names,
        ),
    }
    predictors = {"recovery": recovery_predictor, "organ_support": support_predictor, "icu_stay_time": icu_predictor}

    resolved: Tuple[ResolvedTaskBundle, ...] = tuple(
        ResolvedTaskBundle(identities[task], predictors[task], preprocessors[task]) for task in TASKS
    )

    calibrator_impl = load_calibrator_artifact(calibrator_path, expected_sha256=calibrator_sha256)
    calibrator = RealSupportCalibrator(artifact_sha256=calibrator_sha256, _impl=calibrator_impl)

    split_version = "synthetic_split_v2"
    bundle = ServingBundle(
        scope="real",
        manifest_version=str(manifest["manifest_version"]),
        manifest_ref=manifest_ref,
        manifest_file_sha256=manifest_sha256,
        split_version=split_version,
        tasks=resolved,
        support_calibrator=calibrator,
        support_calibrator_sha256=calibrator_sha256,
        support_threshold=threshold_value,
        support_threshold_sha256=threshold_sha256,
    )

    router = ExplanationRouter(
        bundle,
        {
            "tree_shap": build_real_tree_shap_adapter(),
            "integrated_gradients": build_real_ig_adapter(
                feature_schema_version=feature_schema_version,
                tslo_no_observation_value=tslo_no_observation_value,
            ),
        },
    )

    timeline_manifest_ref = "artifacts/data/synthetic/timelines/final/phase9_final_v1/synthetic_processed_manifest_v1.json"
    timeline_manifest_path = root / timeline_manifest_ref
    provider = CanonicalTimelineHistoryProvider(timeline_manifest_path, root)
    timeline_manifest = _json(timeline_manifest_path)
    statics = load_jsonl(root / next(x["repository_relative_path"] for x in timeline_manifest["artifacts"] if x["logical_name"] == "canonical_statics"))
    builder = SyntheticCanonicalFeatureBuilder(
        schema_path=feature_schema_path, root=root, statics_by_stay={row["stay_id"]: row for row in statics}
    )
    split_by_stay = load_stay_split_index(
        retained_cohort_path=root / "artifacts/data/synthetic/cohorts/final/phase9_final_v1/retained_cohort.jsonl",
        split_path=split_path,
    )
    guarded_provider = NonTestHistoryProvider(provider, split_by_stay=split_by_stay)
    canonical = Stage4CanonicalInputProvider(
        CanonicalHistoryInputProvider(
            history_provider=guarded_provider,
            history_truncator=SyntheticPointEventHistoryTruncator(),
            build_features=builder,
            feature_schema=builder.feature_schema,
        )
    )

    raw_manifest = _json(root / "artifacts/data/synthetic/final/phase9_final_v1/synthetic_dataset_manifest_v1.json")
    support_path = root / next(x["repository_relative_path"] for x in raw_manifest["artifacts"] if x["logical_name"] == "support_intervals")
    supports = load_jsonl(support_path)
    event_dict_path = root / "configs/event_dict_v2.yaml"
    dictionary = load_synthetic_event_dictionary(event_dict_path)
    _vaso, vent, exposures = _normalized(supports, dictionary)
    support_process_path = root / "configs/synthetic/support_process_v1.yaml"
    support_provider = PulkitStateSOFASupportProvider(
        event_dictionary=dictionary,
        execution_mode=ExecutionMode.SYNTHETIC,
        ventilation_intervals=vent,
        vasoactive_exposures=exposures,
        vasoactive_dose_contract_version="synthetic_support_process_v1",
        vasoactive_dose_contract_sha256=synthetic_sha256_file(support_process_path),
        vasoactive_coverage_known=True,
    )
    sofa_spec_path = root / "configs/synthetic/sofa_spec_v1.json"
    sofa = SyntheticCurrentSOFAProvider(
        manifest_path=timeline_manifest_path, root=root, sofa_spec_path=sofa_spec_path,
        support_provider=support_provider,
    )
    sofa_adapter: CurrentSOFAProvider = Stage4CurrentSOFAProvider(sofa)

    recovery_provider = RecoveryServingPostprocessor(
        current_sofa_provider=sofa_adapter,
        delta_adapter=ExplicitOriginalUnitDeltaAdapter(),
        expected_sofa_version=sofa.spec.version,
    )
    postprocessor = CanonicalICUTimeServingPostprocessor(recovery_provider)

    pipeline = PredictionPipeline(
        bundle,
        input_provider=canonical,
        postprocessor=postprocessor,
        explanation_router=router,
        explanation_targets={
            "recovery": RECOVERY_EXPLANATION_TARGET,
            "organ_support": SUPPORT_OUTPUT_NAME,
            "icu_stay_time": "icu_stay_time_hours",
        },
    )

    return Stage4Resolution(
        bundle=bundle,
        pipeline=pipeline,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        g3_marker=marker,
        g3_path=g3_path,
        g3_sha256=sha256_file(g3_path),
        flat_feature_map=flat_map,
        tslo_no_observation_value=tslo_no_observation_value,
        sofa_spec_version=sofa.spec.version,
        code_commit=_git_commit(root),
        current_sofa_provider=sofa_adapter,
        split_by_stay=split_by_stay,
    )
